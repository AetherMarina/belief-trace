# Experiment 4 — Targeted Label-Set-Preserving Augmentation

## Goal

Experiment 4 tested whether generating new narratives while preserving existing multi-label belief configurations could improve model generalization without distorting the underlying class combinations. The motivation was that prior attempts at balancing minority classes (such as Experiment 2) introduced isolated labels that disrupted the co-occurrence synergy of real CBT-style belief sets.
## Data Strategy

* Base dataset: Experiment 1 dataset
* Added data: Small batch of targeted synthetic examples bounded by dynamic allocation quotas (typically 60–90 total examples depending on multi-label overlap deductions)
* Generation method: New narratives were generated while preserving selected multi-label belief sets
* Purpose: Improve weak class representation without disrupting realistic belief co-occurrence patterns

Unlike Experiment 3, which separated target and confuser labels, Experiment 4 preserved multi-label combinations. The goal was to strengthen weak classes while keeping the model exposed to realistic belief constellations.

## Training Configurations

| Configuration | Macro F1 | Micro F1 | Samples F1 |
| ------------- | -------: | -------: | ---------: |
| r16_lr1e-4    | 0.385254 | 0.422680 |   0.412557 |
| r16_lr2e-4    | 0.358718 | 0.431373 |   0.408693 |
| r8_lr1e-4     | 0.369116 | 0.443299 |   0.387547 |
| r8_lr2e-4     | 0.388860 | 0.446809 |   0.444582 |

## Best Run

`r8_lr2e-4`

```text id="4x6927"
Macro F1:   0.388860
Micro F1:   0.446809
Samples F1: 0.444582
```

## Interpretation

Experiment 4 provided critical insights into how network capacity interacts with data syntax:

* Strong Performance in Rank-16 Topologies: In the higher-capacity r16_lr1e-4 run, Experiment 4 successfully outpaced all other strategies. It elevated Macro F1 from 0.325 to 0.385 and pushed Micro F1 from 0.419 to 0.422. This demonstrates that preserving the exact structural multi-label blueprints provides high-quality signal that fits larger model topologies cleanly.

* Diminishing Returns on the Champion Run: Despite its success in rank-16 setups, Experiment 4 could not unseat the Experiment 1 + r8_lr2e-4 baseline champion (Macro: 0.405, Micro: 0.489). In the rank-8 environment, it underperformed the baseline across all metrics.

This behavior suggests that when the LoRA capacity is constricted ($r=8$), the model lacks the parameter space to learn from subtle narrative variations of the same multi-label schemas. It benefits more from the cleaner, broader data profile of Experiment 1 than it does from slight contextual variations of fixed label sets.

## Decision

Diagnostic only.

Experiment 4 was not selected as the final dataset because it did not outperform Experiment 1 on the strongest configuration.

## Summary

Experiment 4 showed that label-set-preserving augmentation is a more realistic strategy than pure contrastive examples, but the generated examples still did not improve overall multi-label recovery. The result supports the broader conclusion that the Experiment 1 data distribution remains the strongest current training signal.
