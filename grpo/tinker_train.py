"""Tinker SDK GRPO training launcher for the role-beam agent.

Mirrors the paper's headline GRPO recipe: rank-16 LoRA on a Qwen3 base, with
a rule-based reward derived from Search-R1 (token-level F1 minus an
action-cost vector), no learned reward model. K=4 rollouts per prompt yield
group-relative advantages.

Hyperparameters and SDK call shape follow paper §A.7. Inputs and outputs:

- **Input** (one training-prompt parquet shard from ``grpo.build_train_data``):
  rows with ``question_id``, ``prompt`` (chat-format messages), ``gold_answer``,
  optional ``reward_model`` metadata.
- **Output**: LoRA adapter checkpoints per ``save_every`` steps under
  ``--output_dir``; one wandb run log; one JSONL of per-step training stats.

Environment expects ``TINKER_API_KEY`` and (optional) ``WANDB_API_KEY``
pre-exported. The trainer calls a caller-supplied ``RuntimeFn`` (see the
``--runtime_module`` flag) that executes a single agent-loop trajectory
given a prompt and returns the per-turn assistant emissions plus the gold
answer so the reward function can score them. The default rollout lives in
``grpo.role_beam_agent_loop.run_rollout``; for SDK-only smoke tests, a stub
runtime that produces a single commit per prompt is enough.
"""
from __future__ import annotations

import argparse
import dataclasses
import json
import logging
import os
import time
from pathlib import Path
from typing import Any, Callable, Dict, Iterable, List, Optional, Sequence

import pyarrow.parquet as pq

# The Tinker SDK is imported lazily so module-level help still works without it
# installed; users without Tinker access can still inspect this file as a recipe.

logger = logging.getLogger(__name__)


@dataclasses.dataclass
class Rollout:
    """One full agent-loop trajectory on one prompt."""

    prompt_id: str
    assistant_token_ids: List[int]
    assistant_text: str
    gold_answer: str
    n_lm_calls: int
    extra: Dict[str, Any] = dataclasses.field(default_factory=dict)


RuntimeFn = Callable[[Dict[str, Any], Any, int], Rollout]
"""Signature: ``(row, policy_handle, seed) -> Rollout``.

``row`` is one parquet row. ``policy_handle`` is the Tinker client used
to sample assistant turns. ``seed`` differs across rollouts within a group.
"""


# ---------------------------------------------------------------------------
# Reward
# ---------------------------------------------------------------------------


def _import_reward(path: Optional[str]) -> Callable[[Rollout], float]:
    """Resolve the reward callable.

    Default points at :mod:`grpo.reward`. Override with ``--reward_module`` to
    swap in a custom reward; it must expose a ``compute_score(rollout)``
    function that takes a :class:`Rollout` and returns a scalar.
    """
    from importlib import import_module

    mod = import_module(path or "grpo.reward")
    if not hasattr(mod, "compute_score"):
        raise AttributeError(f"reward module {mod.__name__} lacks compute_score(rollout)")
    return mod.compute_score


# ---------------------------------------------------------------------------
# Tinker client wrapper
# ---------------------------------------------------------------------------


def make_policy(base_model: str, rank: int) -> Any:
    """Create a Tinker LoRA training client.

    Mirrors paper §A.7: default scope
    ``train_attn=True, train_mlp=True, train_unembed=True`` (Tinker SDK default).
    """
    from tinker import create_lora_training_client  # type: ignore

    client = create_lora_training_client(base_model=base_model, rank=rank)
    return client


def step_policy(
    client: Any,
    batch_data: Sequence[Any],
    learning_rate: float,
) -> Dict[str, float]:
    """One PPO step: forward_backward + optim_step.

    Paper §A.7 uses ``loss_fn="ppo"`` with Tinker defaults (clip range
    [0.8, 1.2], no explicit KL-to-reference penalty) and AdamW with
    ``beta_1=0.9, beta_2=0.95, eps=1e-12``.
    """
    from tinker import AdamParams  # type: ignore

    fb = client.forward_backward(batch_data, loss_fn="ppo", loss_fn_config=None)
    client.optim_step(AdamParams(learning_rate=learning_rate))
    return {"loss": float(getattr(fb, "loss", 0.0))}


# ---------------------------------------------------------------------------
# Rollout grouping + GRPO advantage
# ---------------------------------------------------------------------------


def group_relative_advantages(rewards: Sequence[float]) -> List[float]:
    """GRPO advantage: ``A_i = r_i - mean(r)`` within the group."""
    if not rewards:
        return []
    mu = sum(rewards) / len(rewards)
    return [r - mu for r in rewards]


# ---------------------------------------------------------------------------
# Main training loop
# ---------------------------------------------------------------------------


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--train_parquet", required=True, help="Path to the training shard from build_train_data.py.")
    p.add_argument("--output_dir", required=True, help="Where checkpoints and the stats log are written.")
    p.add_argument("--base_model", default="Qwen/Qwen3-8B", help="HF model id Tinker should adapt.")
    p.add_argument("--rank", type=int, default=16, help="LoRA rank (paper headline: 16).")
    p.add_argument("--learning_rate", type=float, default=1e-4)
    p.add_argument("--rollouts_per_prompt", type=int, default=4, help="K in paper notation.")
    p.add_argument("--prompts_per_batch", type=int, default=10)
    p.add_argument("--steps", type=int, default=200)
    p.add_argument("--save_every", type=int, default=25)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--reward_module", default=None, help="Dotted module path implementing compute_score.")
    p.add_argument(
        "--runtime_module",
        default="grpo.role_beam_agent_loop",
        help=(
            "Dotted module exposing ``run_rollout(row, policy_handle, seed) -> Rollout``. "
            "The default is the helper in ``role_beam_agent_loop.py``; swap in a custom "
            "module for headless smoke tests."
        ),
    )
    p.add_argument("--wandb_project", default=os.environ.get("WANDB_PROJECT", "telemetry-agent"))
    return p.parse_args()


def load_runtime(runtime_module: str) -> RuntimeFn:
    from importlib import import_module

    mod = import_module(runtime_module)
    if not hasattr(mod, "run_rollout"):
        raise AttributeError(f"runtime {mod.__name__} lacks run_rollout(row, policy, seed)")
    return mod.run_rollout


def load_prompts(path: Path) -> List[Dict[str, Any]]:
    table = pq.read_table(str(path))
    return table.to_pylist()


def main() -> None:
    args = parse_args()
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")

    rows = load_prompts(Path(args.train_parquet))
    logger.info("Loaded %d training rows from %s", len(rows), args.train_parquet)

    reward_fn = _import_reward(args.reward_module)
    runtime = load_runtime(args.runtime_module)

    client = make_policy(args.base_model, args.rank)
    logger.info("Tinker LoRA client ready: base=%s rank=%d", args.base_model, args.rank)

    try:
        import wandb

        wandb.init(project=args.wandb_project, config=vars(args))
    except Exception as exc:  # noqa: BLE001
        logger.warning("wandb disabled (%s)", exc)
        wandb = None  # type: ignore

    stats_path = out_dir / "train_stats.jsonl"
    n_prompts = len(rows)
    rng_seed = args.seed

    with stats_path.open("w", encoding="utf-8") as stats_writer:
        for step in range(args.steps):
            t0 = time.time()
            batch_rows = [rows[(step * args.prompts_per_batch + i) % n_prompts] for i in range(args.prompts_per_batch)]

            advantages: List[float] = []
            rollouts: List[Rollout] = []
            for prompt_idx, row in enumerate(batch_rows):
                group_rewards: List[float] = []
                group_rollouts: List[Rollout] = []
                for k in range(args.rollouts_per_prompt):
                    seed = rng_seed + step * args.prompts_per_batch * args.rollouts_per_prompt + prompt_idx * args.rollouts_per_prompt + k
                    roll = runtime(row, client, seed)
                    group_rollouts.append(roll)
                    group_rewards.append(reward_fn(roll))
                group_adv = group_relative_advantages(group_rewards)
                advantages.extend(group_adv)
                rollouts.extend(group_rollouts)

            # Forward / backward with the rolled-out trajectories. The exact
            # batch container shape is dictated by the Tinker SDK; this
            # placeholder leaves the integration point clear.
            batch_data = list(zip(rollouts, advantages))
            metrics = step_policy(client, batch_data, args.learning_rate)
            metrics.update({
                "step": step,
                "mean_advantage": sum(advantages) / max(len(advantages), 1),
                "mean_reward": sum(reward_fn(r) for r in rollouts) / len(rollouts),
                "wall_time_s": time.time() - t0,
            })
            stats_writer.write(json.dumps(metrics) + "\n")
            stats_writer.flush()
            if wandb is not None:
                wandb.log(metrics)

            if (step + 1) % args.save_every == 0:
                ckpt_dir = out_dir / f"ckpt-step{step + 1}"
                ckpt_dir.mkdir(parents=True, exist_ok=True)
                try:
                    client.save_lora(str(ckpt_dir))  # type: ignore[attr-defined]
                    logger.info("Saved checkpoint to %s", ckpt_dir)
                except Exception as exc:  # noqa: BLE001
                    logger.warning("Checkpoint save failed at step %d: %s", step, exc)

            logger.info(
                "step=%d mean_reward=%.3f mean_advantage=%+.3f wall_s=%.1f",
                step,
                metrics["mean_reward"],
                metrics["mean_advantage"],
                metrics["wall_time_s"],
            )

    if wandb is not None:
        wandb.finish()


if __name__ == "__main__":
    main()
