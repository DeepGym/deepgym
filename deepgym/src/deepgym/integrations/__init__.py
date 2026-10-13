"""DeepGym framework integrations.

Available integrations:
- axolotl: Axolotl GRPO reward functions and config helpers
- dapo: Thin DAPO reward/config helpers for verl-style training recipes
- trl: HuggingFace TRL GRPOTrainer reward functions
- verl: ByteDance verl compute_score and batch reward functions
- verl_tool: verl multi-turn tool backed by a live DeepGym session
- verl_mask: verl single-turn agent loop that keeps unscored samples out of training
- openrlhf: OpenRLHF reward server FastAPI router
- reward: Universal RewardFunction and AsyncRewardFunction
- hf: HuggingFace Hub push/pull for environments and results
- lm_eval: lm-evaluation-harness task adapter
"""
