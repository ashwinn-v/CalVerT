"""Calibrated verifier telemetry for action-efficient LLM agents.

Companion code for the paper "Calibrated Verifier Telemetry: A Framework- and
Model-Agnostic Signal for Action-Efficient LLM Agents".

Three open-weight reproducibility paths:

- ``telemetry_agent.runners.hotpotqa_role_beam`` — role-beam evaluation on
  HotpotQA-distractor with DINCO + MiniCheck telemetry on a vLLM substrate.
- ``telemetry_agent.runners.witqa_role_beam`` — single-turn role-beam on WiTQA
  (closed-book vs retrieve-then-answer decision).
- ``telemetry_agent.runners.triviaqa_closed_book`` — closed-book DINCO
  evaluation on TriviaQA ``rc.nocontext`` (used for the calibration appendix).

The Tinker GRPO training setup lives in the top-level ``grpo`` package.
"""

__version__ = "0.1.0"
