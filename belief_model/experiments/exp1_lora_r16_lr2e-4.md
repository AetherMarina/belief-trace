# Experiment 05: High Capacity, High Learning Rate (r=16, lr=2e-4)

- **Model:** `meta-llama/Llama-3.1-8B-Instruct`
- **Configuration:** LoRA Rank = 16, LoRA Alpha = 16, Learning Rate = 2e-4
- **Context:** The most aggressive configuration in the matrix, combining maximum adapter capacity with the higher learning rate.
- **Objective:** Test if combining a large parameter rank with an intense optimization speed yields faster convergence and superior semantic mapping across all categories.

## Dataset Provenance
- **Dataset:** Phase 1 Augmented (CBT-Bench + Baseline Synthetic)
- **Generator Version:** `git tag phase1` (Commit: `e95af56a31ccb9ce90257017214ccf515e8b001d`)
- **Generation Date:** 2026-06-21
- **Total Training Examples:** 276
- **Rare-Class Balancing:** class-aware synthetic augmentation, with 3 examples generated for rare classes and 1 for other classes
- **Manifest:** [`experiments/dataset_manifests/phase1_manifest.json`](./dataset_manifests/phase1_manifest.json)

## Summary Metrics

| Metric | Value | Description |
| :--- | :---: | :--- |
| **Exact Match Accuracy** | 0.0% | The structural complexity of complete multi-label alignment remains a persistent baseline issue. |
| **Micro F1-Score** | 0.373 | Sharp decline from Exp 03 (0.489) and Exp 04 (0.419), indicating a clear performance regression. |
| **Macro F1-Score** | 0.306 | Poor generalization, driven by severe drops in accuracy across mid-to-high support classes. |
| **Samples F1-Score** | 0.354  | Reflects the overfitting penalty; the model's ability to seamlessly reconstruct a full set of overlapping beliefs for an individual narrative degraded significantly.  |
| **Hallucination Count** | 0 | Perfect structural execution. The syntax formatting engine remains entirely stable. |

## Detailed Classification Report

```json
{
    "exact_match_accuracy": 0.0,
    "macro_avg": {
        "precision": 0.33625950073318495,
        "recall": 0.30944976076555014,
        "f1_score": 0.30617204432993905
    },
    "micro_avg": {
        "precision": 0.3655913978494624,
        "recall": 0.38202247191011235,
        "f1_score": 0.37362637362637363
    },
    "samples_avg": {
        "precision": 0.395297619047619,
        "recall": 0.3846428571428572,
        "f1_score": 0.3546456321456321
    },
    "hallucination_count": 0,
    "detailed_report": {
        "exact_match_accuracy": 0.0,
        "macro_avg": {
            "precision": 0.33625950073318495,
            "recall": 0.30944976076555014,
            "f1_score": 0.30617204432993905
        },
        "micro_avg": {
            "precision": 0.3655913978494624,
            "recall": 0.38202247191011235,
            "f1_score": 0.37362637362637363
        },
        "samples_avg": {
            "precision": 0.395297619047619,
            "recall": 0.3846428571428572,
            "f1_score": 0.3546456321456321
        },
        "hallucination_count": 0,
        "detailed_report": {
            "I am a failure, loser": {
                "precision": 0.3333333333333333,
                "recall": 0.16666666666666666,
                "f1-score": 0.2222222222222222,
                "support": 6.0
            },
            "I am a victim": {
                "precision": 0.25,
                "recall": 0.4,
                "f1-score": 0.3076923076923077,
                "support": 5.0
            },
            "I am bad - dangerous, toxic, evil": {
                "precision": 1.0,
                "recall": 0.6666666666666666,
                "f1-score": 0.8,
                "support": 3.0
            },
            "I am bound to be abandoned": {
                "precision": 0.0,
                "recall": 0.0,
                "f1-score": 0.0,
                "support": 3.0
            },
            "I am bound to be alone": {
                "precision": 0.75,
                "recall": 0.375,
                "f1-score": 0.5,
                "support": 8.0
            },
            "I am bound to be rejected": {
                "precision": 0.5,
                "recall": 0.3333333333333333,
                "f1-score": 0.4,
                "support": 3.0
            },
            "I am defective": {
                "precision": 0.3333333333333333,
                "recall": 0.4,
                "f1-score": 0.36363636363636365,
                "support": 5.0
            },
            "I am helpless": {
                "precision": 0.5714285714285714,
                "recall": 0.6666666666666666,
                "f1-score": 0.6153846153846154,
                "support": 6.0
            },
            "I am immoral": {
                "precision": 1.0,
                "recall": 0.6666666666666666,
                "f1-score": 0.8,
                "support": 3.0
            },
            "I am incompetent": {
                "precision": 0.0,
                "recall": 0.0,
                "f1-score": 0.0,
                "support": 3.0
            },
            "I am needy": {
                "precision": 0.0,
                "recall": 0.0,
                "f1-score": 0.0,
                "support": 1.0
            },
            "I am out of control": {
                "precision": 0.0,
                "recall": 0.0,
                "f1-score": 0.0,
                "support": 4.0
            },
            "I am powerless, weak, vulnerable": {
                "precision": 0.5384615384615384,
                "recall": 0.875,
                "f1-score": 0.6666666666666666,
                "support": 8.0
            },
            "I am trapped": {
                "precision": 0.18181818181818182,
                "recall": 0.5,
                "f1-score": 0.26666666666666666,
                "support": 4.0
            },
            "I am unattractive": {
                "precision": 0.0,
                "recall": 0.0,
                "f1-score": 0.0,
                "support": 1.0
            },
            "I am undesirable, unwanted": {
                "precision": 0.0,
                "recall": 0.0,
                "f1-score": 0.0,
                "support": 4.0
            },
            "I am unlovable": {
                "precision": 0.375,
                "recall": 0.375,
                "f1-score": 0.375,
                "support": 8.0
            },
            "I am worthless, waste": {
                "precision": 0.5555555555555556,
                "recall": 0.45454545454545453,
                "f1-score": 0.5,
                "support": 11.0
            },
            "I don't deserve to live": {
                "precision": 0.0,
                "recall": 0.0,
                "f1-score": 0.0,
                "support": 3.0
            },
            "micro avg": {
                "precision": 0.3655913978494624,
                "recall": 0.38202247191011235,
                "f1-score": 0.37362637362637363,
                "support": 89.0
            },
            "macro avg": {
                "precision": 0.33625950073318495,
                "recall": 0.30944976076555014,
                "f1-score": 0.30617204432993905,
                "support": 89.0
            },
            "weighted avg": {
                "precision": 0.4043967892282499,
                "recall": 0.38202247191011235,
                "f1-score": 0.3739582514863414,
                "support": 89.0
            },
            "samples avg": {
                "precision": 0.395297619047619,
                "recall": 0.3846428571428572,
                "f1-score": 0.3546456321456321,
                "support": 89.0
            }
        }
    }
}
```

## Key Findings & Qualitative Analysis

1. **Classic Overfitting Trap:**  This run presents a textbook case of overfitting caused by over-parameterization on a highly specialized dataset. Giving the model twice as many trainable dimensions ($r=16$) combined with a high learning rate (`2e-4`) caused it to memorize specific training phrasings rather than abstracting the underlying cognitive patterns.
2. **Degradation of Core Classes:** The clearest evidence of overfitting is the performance collapse on high-support anchor classes. `"I am worthless, waste"` dropped from an F1 of 0.666 in Exp 03 down to **0.500** here. Similarly, `"I am bound to be alone"` dropped to **0.500**, and `"I am bound to be abandoned"` crashed completely to **0.000** (support: 3.0).
3. **Anomalous Spikes in Specific Categories:** Interestingly, this aggressive pushing caused the model to excel uniquely at specific outlier categories. It achieved perfect precision (**1.000**) and an F1 of **0.800** on both `"I am bad - dangerous, toxic, evil"` and `"I am immoral"`. The model became overly sensitized to these dark moral themes at the expense of general clinical schema tracking.
4. **Verdict:** This configuration is highly unstable and unsuited for production. It confirms that for structured qualitative tasks, pushing both capacity and learning rate simultaneously leads to over-optimization that destroys the model's ability to generalize on unseen evaluation narratives.