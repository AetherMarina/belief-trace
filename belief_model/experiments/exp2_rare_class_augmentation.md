# Experiment 2 — Targeted Rare-Class Augmentation

## Goal

Experiment 2 tested whether adding more synthetic examples for underrepresented core-belief classes could improve class-level performance, especially Macro F1. The motivation was that several belief labels had low support in the original CBT-Bench-derived training data, making them difficult for the model to learn reliably.

## Data Strategy

* Base dataset: Experiment 1 dataset
* Added data: Additional 250 synthetic examples for rare or underrepresented belief classes
* Generation method: Targeted synthetic generation using selected belief labels as generation targets
* Purpose: Improve recall and representation for minority belief classes

## Training Configurations

| Configuration | Macro F1 | Micro F1 | Samples F1 |
| ------------- | -------: | -------: | ---------: |
| r16_lr1e-4    | 0.375141 | 0.354286 |   0.367388 |
| r16_lr2e-4    | 0.324685 | 0.364706 |   0.368364 |
| r8_lr1e-4     | 0.336916 | 0.380435 |   0.368397 |
| r8_lr2e-4     | 0.424379 | 0.422222 |   0.427735 |

## Best Run

`r8_lr2e-4`

```text
Macro F1:   0.424379
Micro F1:   0.422222
Samples F1: 0.427735
```

## Interpretation

Experiment 2 produced the highest Macro F1 among all experiments with the `r8_lr2e-4` configuration. This suggests that targeted rare-class augmentation improved class-level balance and helped the model pay more attention to minority labels.

However, compared with the Experiment 1 champion, Experiment 2 reduced Micro F1 and Samples F1. This means that although class-level balance improved, overall multi-label belief-set recovery became worse.

## Decision

Diagnostic only.

Experiment 2 is highly valuable because it establishes the upper bound for individual minority class recognition (Macro F1). However, it was not selected as the final training dataset because it compromised overall prediction quality and multi-label synergy compared with Experiment 1.

## Summary

Experiment 2 proved that targeted rare-class augmentation is highly effective at rescuing weak, underrepresented classes. However, the results show a clear trade-off: improving minority-label behavior can easily disturb the broader multi-label distribution. Future rare-class curation must be applied with tighter contextual boundaries to prevent localized noise from harming the global multi-label ecosystem.