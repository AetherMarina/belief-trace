# Experiment 04: Increased Capacity, Conservative Learning (r=16, lr=1e-4)

- **Model:** `meta-llama/Llama-3.1-8B-Instruct`
- **Configuration:** LoRA Rank = 16, LoRA Alpha = 16, Learning Rate = 1e-4
- **Context:** Transitioning to a higher-capacity adapter while dropping back to the conservative learning rate used in Experiment 02.
- **Objective:** Determine if expanding the trainable rank parameters allows the model to capture more complex latent representations without overshooting on a stable learning rate.

## Dataset Provenance
- **Dataset:** Phase 1 Augmented (CBT-Bench + Baseline Synthetic)
- **Generator Version:** `git tag phase1` (Commit: `e95af56a31ccb9ce90257017214ccf515e8b001d`)
- **Generation Date:** 2026-06-21
- **Total Training Examples:** 276
- **Rare-Class Balancing:** class-aware synthetic augmentation, with 3 examples generated for rare classes and 1 for other classes
- **Manifest:** [`experiments/dataset_manifests/phase1_manifest.json`](./dataset_manifests/phase1_manifest.json)

## Summary Metrics

| Metric | Value | Description                                                                                                                    |
| :--- | :---: |:-------------------------------------------------------------------------------------------------------------------------------|
| **Exact Match Accuracy** | 0.0% | The complexity of predicting exact multi-label structures remains unresolved.                                                  |
| **Micro F1-Score** | 0.419 | Solid improvement over Exp 02 (0.310), but underperforms compared to Exp 03 (0.489).                                           |
| **Macro F1-Score** | 0.325 | Moderate generalization across classes, showing a healthier balance than lower-rank conservative runs.                         |
| **Samples F1-Score** | 0.392 | Demonstrates that the higher rank provides a safer structural fallback for complex narratives when optimization speeds are low.|
| **Hallucination Count** | 0 | Perfect structural alignment. The syntax formatting constraint remains fully solved.                                           |

## Detailed Classification Report

```json
{
    "exact_match_accuracy": 0.0,
    "macro_avg": {
        "precision": 0.3530701754385965,
        "recall": 0.34489633173843703,
        "f1_score": 0.32518586168471747
    },
    "micro_avg": {
        "precision": 0.41304347826086957,
        "recall": 0.42696629213483145,
        "f1_score": 0.4198895027624309
    },
    "samples_avg": {
        "precision": 0.4333333333333333,
        "recall": 0.38619047619047614,
        "f1_score": 0.39213841713841713
    },
    "hallucination_count": 0,
    "detailed_report": {
        "exact_match_accuracy": 0.0,
        "macro_avg": {
            "precision": 0.3530701754385965,
            "recall": 0.34489633173843703,
            "f1_score": 0.32518586168471747
        },
        "micro_avg": {
            "precision": 0.41304347826086957,
            "recall": 0.42696629213483145,
            "f1_score": 0.4198895027624309
        },
        "samples_avg": {
            "precision": 0.4333333333333333,
            "recall": 0.38619047619047614,
            "f1_score": 0.39213841713841713
        },
        "hallucination_count": 0,
        "detailed_report": {
            "I am a failure, loser": {
                "precision": 0.25,
                "recall": 0.16666666666666666,
                "f1-score": 0.2,
                "support": 6.0
            },
            "I am a victim": {
                "precision": 0.375,
                "recall": 0.6,
                "f1-score": 0.46153846153846156,
                "support": 5.0
            },
            "I am bad - dangerous, toxic, evil": {
                "precision": 0.5,
                "recall": 0.3333333333333333,
                "f1-score": 0.4,
                "support": 3.0
            },
            "I am bound to be abandoned": {
                "precision": 0.5,
                "recall": 0.6666666666666666,
                "f1-score": 0.5714285714285714,
                "support": 3.0
            },
            "I am bound to be alone": {
                "precision": 0.5714285714285714,
                "recall": 0.5,
                "f1-score": 0.5333333333333333,
                "support": 8.0
            },
            "I am bound to be rejected": {
                "precision": 0.0,
                "recall": 0.0,
                "f1-score": 0.0,
                "support": 3.0
            },
            "I am defective": {
                "precision": 0.3333333333333333,
                "recall": 0.4,
                "f1-score": 0.36363636363636365,
                "support": 5.0
            },
            "I am helpless": {
                "precision": 0.6666666666666666,
                "recall": 0.3333333333333333,
                "f1-score": 0.4444444444444444,
                "support": 6.0
            },
            "I am immoral": {
                "precision": 0.5,
                "recall": 0.3333333333333333,
                "f1-score": 0.4,
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
                "precision": 0.5,
                "recall": 0.5,
                "f1-score": 0.5,
                "support": 8.0
            },
            "I am trapped": {
                "precision": 0.3333333333333333,
                "recall": 0.5,
                "f1-score": 0.4,
                "support": 4.0
            },
            "I am unattractive": {
                "precision": 0.0,
                "recall": 0.0,
                "f1-score": 0.0,
                "support": 1.0
            },
            "I am undesirable, unwanted": {
                "precision": 0.16666666666666666,
                "recall": 0.5,
                "f1-score": 0.25,
                "support": 4.0
            },
            "I am unlovable": {
                "precision": 0.42857142857142855,
                "recall": 0.75,
                "f1-score": 0.5454545454545454,
                "support": 8.0
            },
            "I am worthless, waste": {
                "precision": 0.5833333333333334,
                "recall": 0.6363636363636364,
                "f1-score": 0.6086956521739131,
                "support": 11.0
            },
            "I don't deserve to live": {
                "precision": 1.0,
                "recall": 0.3333333333333333,
                "f1-score": 0.5,
                "support": 3.0
            },
            "micro avg": {
                "precision": 0.41304347826086957,
                "recall": 0.42696629213483145,
                "f1-score": 0.4198895027624309,
                "support": 89.0
            },
            "macro avg": {
                "precision": 0.3530701754385965,
                "recall": 0.34489633173843703,
                "f1-score": 0.32518586168471747,
                "support": 89.0
            },
            "weighted avg": {
                "precision": 0.41526217228464424,
                "recall": 0.42696629213483145,
                "f1-score": 0.3992447383263211,
                "support": 89.0
            },
            "samples avg": {
                "precision": 0.4333333333333333,
                "recall": 0.38619047619047614,
                "f1-score": 0.39213841713841713,
                "support": 89.0
            }
        }
    }
}
```

## Key Findings & Qualitative Analysis

1. **Rank as an Underfitting Mitigator:** Comparing this to Experiment 02 (which used the exact same `1e-4` learning rate but $r=8$), doubling the rank to 16 caused a substantial performance leap (Micro F1 from 0.310 to 0.419). This proves that the extra capacity allowed the model to absorb finer textual nuances even with a slower optimization speed.
2. **Shift in Error Profiles (Recall over Precision):** Unlike the champion run (Exp 03), this model prioritizes broad retrieval overcautious correctness. For instance, on `"I am unlovable"` (support: 8.0), it achieved an exceptional **Recall of 0.750**, but with a lower Precision of 0.428. It is flag-happy—willing to tag beliefs generously, which introduces false positives.  
3. **Improved Performance on Mid-Support Classes:** The increased rank helped stabilize learning on intermediately represented categories. `"I am a victim"` reached an F1-score of **0.461** (support: 5.0), whereas it scored 0.000 in both previous low-rank attempts.
4. **Verdict:**  A strong runner-up config. It demonstrates that if computational memory allows, higher ranks can compensate for low learning rates, though it ultimately falls short of the sharper precision found in Rank 8 at `2e-4`.  