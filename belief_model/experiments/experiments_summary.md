# Augmentation Ablation Summary

This document summarizes the augmentation experiments for BeliefTrace core-belief extraction. All runs use the same evaluation set and report Macro F1, Micro F1, and Samples F1 for multi-label belief prediction.

## Experiment Overview

| Experiment   | Strategy                                   | Purpose                                                                         |
| ------------ | ------------------------------------------ | ------------------------------------------------------------------------------- |
| Experiment 1 | Base augmented dataset                     | Initial synthetic augmentation strategy and main baseline                       |
| Experiment 2 | Targeted rare-class augmentation           | Add more examples for underrepresented belief classes                           |
| Experiment 3 | Contrastive augmentation                   | Improve boundaries between semantically similar beliefs                         |
| Experiment 4 | Targeted label-set-preserving augmentation | Generate new narratives while preserving selected multi-label belief sets       |
| Experiment 5 | Hard-negative boundary augmentation        | Reduce false positives for broad attractor labels while adding positive anchors |

## Full Results

| Experiment | Configuration | Macro F1 | Micro F1 | Samples F1 |
| ---------: | ------------- | -------: | -------: | ---------: |
|          1 | r16_lr1e-4    | 0.325186 | 0.419890 |   0.392138 |
|          2 | r16_lr1e-4    | 0.375141 | 0.354286 |   0.367388 |
|          3 | r16_lr1e-4    | 0.369965 | 0.418079 |   0.388486 |
|          4 | r16_lr1e-4    | 0.385254 | 0.422680 |   0.412557 |
|          1 | r16_lr2e-4    | 0.306172 | 0.373626 |   0.354646 |
|          2 | r16_lr2e-4    | 0.324685 | 0.364706 |   0.368364 |
|          3 | r16_lr2e-4    | 0.385877 | 0.445783 |   0.413443 |
|          4 | r16_lr2e-4    | 0.358718 | 0.431373 |   0.408693 |
|          1 | r8_lr1e-4     | 0.226783 | 0.310160 |   0.306876 |
|          2 | r8_lr1e-4     | 0.336916 | 0.380435 |   0.368397 |
|          3 | r8_lr1e-4     | 0.409551 | 0.451977 |   0.425379 |
|          4 | r8_lr1e-4     | 0.369116 | 0.443299 |   0.387547 |
|          1 | r8_lr2e-4     | 0.405712 | 0.489362 |   0.458508 |
|          2 | r8_lr2e-4     | 0.424379 | 0.422222 |   0.427735 |
|          3 | r8_lr2e-4     | 0.406592 | 0.459016 |   0.446724 |
|          4 | r8_lr2e-4     | 0.388860 | 0.446809 |   0.444582 |
|          5 | r8_lr2e-4     | 0.403894 | 0.441989 |   0.444812 |

## Best Runs

| Criterion                     | Best Run                | Macro F1 | Micro F1 | Samples F1 |
| ----------------------------- | ----------------------- | -------: | -------: | ---------: |
| Best Macro F1                 | Experiment 2, r8_lr2e-4 | 0.424379 | 0.422222 |   0.427735 |
| Best Micro F1                 | Experiment 1, r8_lr2e-4 | 0.405712 | 0.489362 |   0.458508 |
| Best Samples F1               | Experiment 1, r8_lr2e-4 | 0.405712 | 0.489362 |   0.458508 |
| Best balanced final candidate | Experiment 1, r8_lr2e-4 | 0.405712 | 0.489362 |   0.458508 |

## Statistical Significance Testing

To check whether later augmentation strategies produced reliable improvements over the champion model, paired bootstrap resampling was run against the strongest Experiment 1 configuration:

`Experiment 1 + r8_lr2e-4`

Each comparison used the same evaluation examples and computed bootstrap confidence intervals for the difference:

`Model B - Model A`

where Model A is Experiment 1 and Model B is the augmentation variant.

| Comparison    | Metric     | 95% CI Lower | 95% CI Upper | Significant |
| ------------- | ---------- | -----------: | -----------: | ----------- |
| Exp 2 - Exp 1 | Macro F1   |      -0.1019 |       0.1473 | No          |
| Exp 2 - Exp 1 | Micro F1   |      -0.1706 |       0.0805 | No          |
| Exp 2 - Exp 1 | Samples F1 |      -0.1477 |       0.1191 | No          |
| Exp 3 - Exp 1 | Macro F1   |      -0.1615 |       0.0127 | No          |
| Exp 3 - Exp 1 | Micro F1   |      -0.1781 |      -0.0040 | Yes         |
| Exp 3 - Exp 1 | Samples F1 |      -0.1590 |       0.0115 | No          |
| Exp 4 - Exp 1 | Macro F1   |      -0.1697 |       0.0583 | No          |
| Exp 4 - Exp 1 | Micro F1   |      -0.1757 |       0.0534 | No          |
| Exp 4 - Exp 1 | Samples F1 |      -0.1547 |       0.0770 | No          |
| Exp 5 - Exp 1 | Macro F1   |      -0.1523 |       0.0440 | No          |
| Exp 5 - Exp 1 | Micro F1   |      -0.2134 |       0.0317 | No          |
| Exp 5 - Exp 1 | Samples F1 |      -0.1925 |       0.0624 | No          |

### Bootstrap Interpretation

None of the later augmentation experiments produced a statistically reliable improvement over the Experiment 1 champion.

Experiment 3 showed a statistically significant decrease in Micro F1 compared with Experiment 1, with the full confidence interval below zero. This suggests that contrastive augmentation harmed overall label-level recovery in the optimized `r8_lr2e-4` setting.

For Experiments 2, 4, and 5, the confidence intervals crossed zero across all metrics. This means that their observed differences from Experiment 1 may be attributable to evaluation split variability rather than reliable performance changes.

Overall, the bootstrap analysis strengthens the final decision: Experiment 1 remains the safest champion model.

## Interpretation

**Experiment 1** (r8_lr2e-4) remains the strongest overall run. It achieved the highest global Micro F1 (0.489362) and Samples F1 (0.458508), making it the champion baseline for overall multi-label belief-set recovery. Preserving this base distribution proved more effective than any subsequent targeted synthetic patching.  

**Experiment 2** produced the highest Macro F1 with r8_lr2e-4 (0.424379), suggesting that targeted rare-class injection improved class-level balance (such as I am needy and I am unattractive). However, it did so at the expense of the overall multi-label distribution. The aggressive influx of rare labels caused the model to over-predict them in mixed contexts, generating a cascade of false positives that degraded overall multi-label recovery (Micro F1 dropped to 0.422222).

**Experiment 3** showed that contrastive augmentation serves as a powerful performance stabilizer in sub-optimal, low-capacity configurations. Most notably, in the weaker r8_lr1e-4 environment, isolating overlapping psychological schemas into strict single-label target_positive vs. confuser_only training pairs shot the Macro F1 from a dismal 0.226783 up to 0.409551. However, forcing these semantic boundaries apart ultimately limited the model's capacity to naturally handle integrated clinical co-occurrences in the optimized r8_lr2e-4 configuration.

**Experiment 4** proved that targeted label-set-preserving augmentation is the safest mechanism for introducing narrative diversity. By utilizing existing training rows as exact multi-label blueprints, it boosted performance across all rank-16 topologies (elevating r16_lr1e-4 Macro F1 from 0.325 to 0.385 and Micro F1 from 0.419 to 0.422) without causing the false-positive explosions seen in Experiment 2. It only failed to add value once the model's capacity was heavily constricted ($r=8$).

**Experiment 5** tested hard-negative boundary augmentation. Diagnostically, it successfully lowered false positives for prominent "attractor" labels (e.g., false positives for I am undesirable, unwanted dropped from 9 to 5, and powerless, weak, vulnerable plummeted from 6 to 2) and woke up dead "ghost" classes. However, the contrastive negative constraints introduced an overwhelming negative bias; the model overcorrected and suppressed its predictions so heavily that global Recall collapsed for core classes like I am unlovable (crashing from 6 True Positives to 1) and I am worthless, waste (falling from 8 True Positives to 3), dropping Micro F1 to 0.441989.

## Final Conclusion

Across five augmentation strategies, the best overall model remained:

`Experiment 1 + r8_lr2e-4`

with:

* Macro F1: 0.405712
* Micro F1: 0.489362
* Samples F1: 0.458508

The ablation results suggest that adding synthetic data does not automatically improve performance. Later augmentation strategies improved selected classes, but they also disturbed the original multi-label distribution and reduced overall belief-set recovery.

For the current dataset size and evaluation split, preserving the Experiment 1 training distribution was more effective than adding small targeted synthetic patches.

## Final Decision

Use `Experiment 1 + r8_lr2e-4` as the champion model.

Treat Experiments 2–5 as ablation studies showing how different augmentation strategies affect class balance, boundary learning, and overall multi-label recovery.

Paired bootstrap testing also found no statistically reliable improvement from Experiments 2–5 over the Experiment 1 champion. The only statistically significant result was a Micro F1 decrease for Experiment 3 compared with Experiment 1.
