"""DeepGym framework integrations.

Available integrations:
- axolotl: Axolotl GRPO reward functions, PRM dataset generation, config helpers
- dapo: Thin DAPO reward/config helpers for verl-style training recipes
- trl: HuggingFace TRL GRPOTrainer reward functions
- verl: ByteDance verl compute_score and batch reward functions
- openrlhf: OpenRLHF reward server FastAPI router
- reward: Universal RewardFunction and AsyncRewardFunction
- hf: HuggingFace Hub push/pull for environments and results
- lm_eval: lm-evaluation-harness task adapter
"""
