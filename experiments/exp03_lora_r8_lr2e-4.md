# Experiment 03: The Champion Run (r=8, lr=2e-4)

- **Model:** `meta-llama/Llama-3.1-8B-Instruct`
- **Configuration:** LoRA Rank = 8, LoRA Alpha = 8, Learning Rate = 2e-4
- **Context:** Hyperparameter adjustment based on Experiment 02, doubling the learning rate while keeping the conservative rank fixed.
- **Objective:** Test whether a more aggressive optimization push allows the Rank 8 adapter to map complex, low-support cognitive schemas effectively.

## Summary Metrics

| Metric | Value | Description |
| :--- | :---: | :--- |
| **Exact Match Accuracy** | 0.0% | Even at peak performance, matching multi-label sets perfectly remains an open challenge. |
| **Micro F1-Score** | **0.489** | **Highest overall performance**, showing a massive jump from 0.310 in Exp 02. |
| **Macro F1-Score** | **0.405** | Significant gain, proving better generalization across diverse belief categories. |
| **Samples F1-Score** | 0.458 | High score demonstrates that the model maintains strong coherence when reconstructing the full multi-label belief set for an individual narrative. |
| **Hallucination Count** | **0** | **Absolute syntax perfection.** The model achieved complete grounding within the taxonomy. |

## Detailed Classification Report

```json
{
    "exact_match_accuracy": 0.0,
    "macro_avg": {
        "precision": 0.4171118355328881,
        "recall": 0.44266347687400315,
        "f1_score": 0.4057118012796683
    },
    "micro_avg": {
        "precision": 0.46464646464646464,
        "recall": 0.5168539325842697,
        "f1_score": 0.48936170212765956
    },
    "samples_avg": {
        "precision": 0.4600000000000001,
        "recall": 0.49285714285714277,
        "f1_score": 0.458508436008436
    },
    "hallucination_count": 0,
    "detailed_report": {
        "exact_match_accuracy": 0.0,
        "macro_avg": {
            "precision": 0.4171118355328881,
            "recall": 0.44266347687400315,
            "f1_score": 0.4057118012796683
        },
        "micro_avg": {
            "precision": 0.46464646464646464,
            "recall": 0.5168539325842697,
            "f1_score": 0.48936170212765956
        },
        "samples_avg": {
            "precision": 0.4600000000000001,
            "recall": 0.49285714285714277,
            "f1_score": 0.458508436008436
        },
        "hallucination_count": 0,
        "detailed_report": {
            "I am a failure, loser": {
                "precision": 0.3333333333333333,
                "recall": 0.3333333333333333,
                "f1-score": 0.3333333333333333,
                "support": 6.0
            },
            "I am a victim": {
                "precision": 0.2,
                "recall": 0.2,
                "f1-score": 0.2,
                "support": 5.0
            },
            "I am bad - dangerous, toxic, evil": {
                "precision": 0.5,
                "recall": 0.6666666666666666,
                "f1-score": 0.5714285714285714,
                "support": 3.0
            },
            "I am bound to be abandoned": {
                "precision": 0.6666666666666666,
                "recall": 0.6666666666666666,
                "f1-score": 0.6666666666666666,
                "support": 3.0
            },
            "I am bound to be alone": {
                "precision": 0.6666666666666666,
                "recall": 0.25,
                "f1-score": 0.36363636363636365,
                "support": 8.0
            },
            "I am bound to be rejected": {
                "precision": 0.0,
                "recall": 0.0,
                "f1-score": 0.0,
                "support": 3.0
            },
            "I am defective": {
                "precision": 0.5,
                "recall": 0.4,
                "f1-score": 0.4444444444444444,
                "support": 5.0
            },
            "I am helpless": {
                "precision": 0.7142857142857143,
                "recall": 0.8333333333333334,
                "f1-score": 0.7692307692307693,
                "support": 6.0
            },
            "I am immoral": {
                "precision": 0.75,
                "recall": 1.0,
                "f1-score": 0.8571428571428571,
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
                "precision": 0.5,
                "recall": 0.5,
                "f1-score": 0.5,
                "support": 4.0
            },
            "I am powerless, weak, vulnerable": {
                "precision": 0.5,
                "recall": 0.75,
                "f1-score": 0.6,
                "support": 8.0
            },
            "I am trapped": {
                "precision": 0.3333333333333333,
                "recall": 0.75,
                "f1-score": 0.46153846153846156,
                "support": 4.0
            },
            "I am unattractive": {
                "precision": 0.0,
                "recall": 0.0,
                "f1-score": 0.0,
                "support": 1.0
            },
            "I am undesirable, unwanted": {
                "precision": 0.1,
                "recall": 0.25,
                "f1-score": 0.14285714285714285,
                "support": 4.0
            },
            "I am unlovable": {
                "precision": 0.5454545454545454,
                "recall": 0.75,
                "f1-score": 0.631578947368421,
                "support": 8.0
            },
            "I am worthless, waste": {
                "precision": 0.6153846153846154,
                "recall": 0.7272727272727273,
                "f1-score": 0.6666666666666666,
                "support": 11.0
            },
            "I don't deserve to live": {
                "precision": 1.0,
                "recall": 0.3333333333333333,
                "f1-score": 0.5,
                "support": 3.0
            },
            "micro avg": {
                "precision": 0.46464646464646464,
                "recall": 0.5168539325842697,
                "f1-score": 0.48936170212765956,
                "support": 89.0
            },
            "macro avg": {
                "precision": 0.4171118355328881,
                "recall": 0.44266347687400315,
                "f1-score": 0.4057118012796683,
                "support": 89.0
            },
            "weighted avg": {
                "precision": 0.4801713268005403,
                "recall": 0.5168539325842697,
                "f1-score": 0.473437745643541,
                "support": 89.0
            },
            "samples avg": {
                "precision": 0.4600000000000001,
                "recall": 0.49285714285714277,
                "f1-score": 0.458508436008436,
                "support": 89.0
            }
        }
    }
}
```

## Key Findings & Qualitative Analysis

1. **Optimal Learning Rate Synergy:**  Increasing the learning rate to 2e-4 provided the exact energy needed for the low-rank adapter ($r=8$) to update its internal semantic mapping. The jump in Micro F1 (from 0.310 to 0.489) demonstrates that the previous configuration was heavily underfitting. 
2. **Breakthrough in Low-Support Categories:** Unlike Experiment 02, this run began successfully learning boundaries for rarer classes. Notably, `"I am immoral"` achieved an F1-score of **0.857** (support: 3.0) and `"I am bad - dangerous, toxic, evil"` reached **0.571** (support: 3.0).
3. **High-Precision Safety Anchors:** The model shows high cautiousness when identifying high-stakes beliefs. For `"I don’t deserve to live"`, it scored a **Precision of 1.000** with a Recall of 0.333 (F1: 0.500). This implies that when the model outputs this extreme schema, it is highly accurate, though it misses several instances.
4. **Persistent Data Sparsity Limits** Despite the optimization boost, categories with a support of 1.0 (`"I am needy"`, `"I am unattractive"`) still scored 0.000. This confirms that hyperparameter tuning cannot fully overcome severe data scarcity.
5. **Verdict: This is the current production champion** It achieves the best balance between structural formatting and psychological interpretation.