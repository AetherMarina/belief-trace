# Model Training Experiments

This directory logs the systematic hyperparameter tuning matrix for `BeliefTrace`.


## Master Leaderboard

| Exp ID | Configuration | Micro F1 | Macro F1 | Samples F1 | Hallucinations | Report |
| :--- | :--- | :---: | :---: | :---: | :---: | :---: |
| `exp01` | **Baseline** (Llama 3.1 8B) | 0.000 | 0.000 | 0.000 | 20 | [View Report](exp1_baseline.md) |
| `exp02` | `lora_cloud_r8_lr1e-4` | 0.310 | 0.226 | 0.306 | 1 | [View Report](exp1_lora_r8_lr1e-4.md) |
| `exp03` | `lora_cloud_r8_lr2e-4` 🏆 | **0.489** | **0.405** | **0.458** | **0** | [View Report](exp1_lora_r8_lr2e-4.md) |
| `exp04` | `lora_cloud_r16_lr1e-4` | 0.419 | 0.325 | 0.392 | 0 | [View Report](exp1_lora_r16_lr1e-4.md) |
| `exp05` | `lora_cloud_r16_lr2e-4` | 0.373 | 0.306 | 0.354 | 0 | [View Report](exp1_lora_r16_lr2e-4.md) |