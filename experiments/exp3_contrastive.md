# Experiment 3 — Contrastive Boundary Augmentation

## Goal

Experiment 3 tested whether introducing contrastive synthetic examples could improve the model's ability to distinguish between semantically similar and commonly confused core beliefs. The motivation was that the baseline model struggled with fine-grained multi-label decision boundaries, often bleeding predictions across overlapping psychological concepts (e.g., confusing "worthless" with "defective").
## Data Strategy

* Base dataset: Experiment 1 dataset
* Added data: Additional 78 synthetic examples split into isolated contrastive variants
* Generation method: Controlled target vs. confuser single-label pair generation
* Purpose: Improve boundary learning between similar belief classes

### The Disambiguation Variant Mechanism

For every target-confuser pairing, the script created exactly 3 examples of two structural variants:
* **Target Positive** (target_positive): Narratives generated explicitly reflecting only the target belief.
* **Confuser Only** (confuser_only): Narratives generated explicitly reflecting only the confounding neighbor belief.

Unlike standard multi-label generation, these items were kept as pure single-labels to draw clear semantic lines in the training space.

## Training Configurations

| Configuration | Macro F1 | Micro F1 | Samples F1 |
| ------------- | -------: | -------: | ---------: |
| r16_lr1e-4    | 0.369965 | 0.418079 |   0.388486 |
| r16_lr2e-4    | 0.385877 | 0.445783 |   0.413443 |
| r8_lr1e-4     | 0.409551 | 0.451977 |   0.425379 |
| r8_lr2e-4     | 0.406592 | 0.459016 |   0.446724 |

## Best Run

`r8_lr2e-4`

```text id="wio8d7"
Macro F1:   0.406592
Micro F1:   0.459016
Samples F1: 0.446724
```

## Interpretation

Experiment 3 demonstrated a fascinating hyperparameter interaction and provided strong evidence that contrastive dataset design is a viable tuning tool:

* **Massive Lift in Lower Configurations**: In configurations where the baseline model performed poorly due to low capacity or sub-optimal learning rates, Experiment 3 acted as a major stabilizer. In the r8_lr1e-4 runtime, contrastive data shot the Macro F1 from a dismal 0.226783 up to 0.409551, outperforming the baseline by nearly 18 percentage points while simultaneously boosting Micro F1 from 0.310 to 0.451.
* **The Strong Baseline Ceiling**: While Experiment 3 proved highly effective at correcting errors in weaker setups, it could not beat the absolute champion performance of Experiment 1 + r8_lr2e-4 (Macro: 0.405712, Micro: 0.489362). In that specific configuration, Experiment 3 achieved a comparable Macro F1 (0.406592) but dropped roughly 3 points on Micro F1.

The fallback in global Micro F1 at the highest baseline tier indicates that isolating highly correlated beliefs into strict single-label training instances fixes class confusion locally, but limits the model's capacity to naturally handle highly integrated multi-label expressions when it is already optimally tuned.

## Decision

Diagnostic only.

Experiment 3 was not selected as the final dataset because it did not outperform Experiment 1 on overall multi-label recovery.

## Summary

Experiment 3 showed that contrastive augmentation is useful for studying label boundaries, but it should not replace the original training distribution. It improved selected class-level behavior but reduced overall Micro F1 and Samples F1 compared with the Experiment 1 champion.
