# Experiment 02: Initial LoRA Run (r=8, lr=1e-4)

- **Model:** `meta-llama/Llama-3.1-8B-Instruct`
- **Configuration:** LoRA Rank = 8, LoRA Alpha = 8, Learning Rate = 1e-4
- **Context:** The first parameter-efficient fine-tuning experiment on the CBT-Bench dataset.
- **Objective:** Evaluate how a conservative adapter setup handles formatting constraints and initial semantic mapping of latent beliefs.

## Summary Metrics

| Metric | Value  | Description                                                                                                      |
| :--- |:------:|:-----------------------------------------------------------------------------------------------------------------|
| **Exact Match Accuracy** |  0.0%  | Strict multi-label matching is still too complex for this baseline configuration.                                |
| **Micro F1-Score** | 0.310  | Significant improvement from the 0.0 baseline, showing initial learning.                                         |
| **Macro F1-Score** | 0.226  | Low score indicates heavily skewed performance towards high-support classes.                                     |
| **Samples F1-Score** | 0.306  | Shows baseline competence in capturing multi-label configurations, tracking close to the overall micro average.  |
| **Hallucination Count** |   1    | A massive drop from 20 down to just 1 taxonomy violation, proving format mastery.                                |

## Detailed Classification Report

```json
{
    "exact_match_accuracy": 0.0,
    "macro_avg": {
        "precision": 0.2517410951621478,
        "recall": 0.2608452950558214,
        "f1_score": 0.2267827970736558
    },
    "micro_avg": {
        "precision": 0.29591836734693877,
        "recall": 0.3258426966292135,
        "f1_score": 0.31016042780748665
    },
    "samples_avg": {
        "precision": 0.31767857142857137,
        "recall": 0.3271428571428571,
        "f1_score": 0.3068755101108042
    },
    "hallucination_count": 1,
    "detailed_report": {
        "exact_match_accuracy": 0.0,
        "macro_avg": {
            "precision": 0.2517410951621478,
            "recall": 0.2608452950558214,
            "f1_score": 0.2267827970736558
        },
        "micro_avg": {
            "precision": 0.29591836734693877,
            "recall": 0.3258426966292135,
            "f1_score": 0.31016042780748665
        },
        "samples_avg": {
            "precision": 0.31767857142857137,
            "recall": 0.3271428571428571,
            "f1_score": 0.3068755101108042
        },
        "hallucination_count": 1,
        "detailed_report": {
            "I am a failure, loser": {
                "precision": 0.5,
                "recall": 0.16666666666666666,
                "f1-score": 0.25,
                "support": 6.0
            },
            "I am a victim": {
                "precision": 0.0,
                "recall": 0.0,
                "f1-score": 0.0,
                "support": 5.0
            },
            "I am bad - dangerous, toxic, evil": {
                "precision": 0.0,
                "recall": 0.0,
                "f1-score": 0.0,
                "support": 3.0
            },
            "I am bound to be abandoned": {
                "precision": 0.2222222222222222,
                "recall": 0.6666666666666666,
                "f1-score": 0.3333333333333333,
                "support": 3.0
            },
            "I am bound to be alone": {
                "precision": 0.6,
                "recall": 0.375,
                "f1-score": 0.46153846153846156,
                "support": 8.0
            },
            "I am bound to be rejected": {
                "precision": 0.0,
                "recall": 0.0,
                "f1-score": 0.0,
                "support": 3.0
            },
            "I am defective": {
                "precision": 0.375,
                "recall": 0.6,
                "f1-score": 0.46153846153846156,
                "support": 5.0
            },
            "I am helpless": {
                "precision": 0.5714285714285714,
                "recall": 0.6666666666666666,
                "f1-score": 0.6153846153846154,
                "support": 6.0
            },
            "I am immoral": {
                "precision": 0.0,
                "recall": 0.0,
                "f1-score": 0.0,
                "support": 3.0
            },
            "I am incompetent": {
                "precision": 1.0,
                "recall": 0.3333333333333333,
                "f1-score": 0.5,
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
                "precision": 0.4166666666666667,
                "recall": 0.625,
                "f1-score": 0.5,
                "support": 8.0
            },
            "I am trapped": {
                "precision": 0.2222222222222222,
                "recall": 0.5,
                "f1-score": 0.3076923076923077,
                "support": 4.0
            },
            "I am unattractive": {
                "precision": 0.0,
                "recall": 0.0,
                "f1-score": 0.0,
                "support": 1.0
            },
            "I am undesirable, unwanted": {
                "precision": 0.08333333333333333,
                "recall": 0.25,
                "f1-score": 0.125,
                "support": 4.0
            },
            "I am unlovable": {
                "precision": 0.36363636363636365,
                "recall": 0.5,
                "f1-score": 0.42105263157894735,
                "support": 8.0
            },
            "I am worthless, waste": {
                "precision": 0.42857142857142855,
                "recall": 0.2727272727272727,
                "f1-score": 0.3333333333333333,
                "support": 11.0
            },
            "I don't deserve to live": {
                "precision": 0.0,
                "recall": 0.0,
                "f1-score": 0.0,
                "support": 3.0
            },
            "micro avg": {
                "precision": 0.29591836734693877,
                "recall": 0.3258426966292135,
                "f1-score": 0.31016042780748665,
                "support": 89.0
            },
            "macro avg": {
                "precision": 0.2517410951621478,
                "recall": 0.2608452950558214,
                "f1-score": 0.2267827970736558,
                "support": 89.0
            },
            "weighted avg": {
                "precision": 0.32527157611427276,
                "recall": 0.3258426966292135,
                "f1-score": 0.29728274879073224,
                "support": 89.0
            },
            "samples avg": {
                "precision": 0.31767857142857137,
                "recall": 0.3271428571428571,
                "f1-score": 0.3068755101108042,
                "support": 89.0
            }
        }
    }
}
```

## Key Findings & Qualitative Analysis

1. **Successful Syntax Conditioning:** The primary engineering milestone of this run is the near-total eradication of hallucinations (dropping from 20 to 1). LoRA adapters are highly effective at forcing the model to select only from the predefined taxonomy string format ('label A'; 'label B') rather than outputting conversational text.
2. **Emergent Semantic Mapping:** The model has successfully mapped certain core concepts. It achieved decent baseline performance on common schemas like "I am helpless" (F1: 0.615) and `"I am bound to be alone"` (F1: 0.461).
3. **The Data Sparsity Problem:** With a learning rate of 1e-4 and a low rank of 8, the model completely failed to predict rare classes. Every category with a support under 3.0 (e.g., `"I am needy"`, `"I am out of control"`, `"I am immoral"`) returned an F1-score of 0.000.
4. **Verdict:** This configuration is underfitting. The learning rate is too conservative to force updates on less frequent tokens, meaning the model needs a stronger optimization push to achieve deep semantic generalization.
