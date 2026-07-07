# Experiment 01: Baseline Results

- **Model:** `meta-llama/Llama-3.1-8B-Instruct`
- **Context:** Zero-shot inference without any parameter updates or specialized prompt engineering.
- **Objective:** Establish a reference point for formatting adherence and taxonomic alignment on the CBT-Bench dataset.

## Summary Metrics

| Metric | Value  | Description                                                                                                         |
| :--- |:------:|:--------------------------------------------------------------------------------------------------------------------|
| **Exact Match Accuracy** |  0.0%  | No evaluation sample matched the target labels perfectly.                                                           |
| **Micro F1-Score** | 0.000  | Zero true positives recorded.                                                                                       |
| **Macro F1-Score** | 0.000  | Class-agnostic performance is zero.                                                                                 |
| **Samples F1-Score** | 0.000  | The untrained model could not reconstruct a single narrative's belief profile due to structural formatting errors.  |
| **Hallucination Count** |   20   | Out of 89 reference steps, 20 instances generated raw, out-of-taxonomy text.                                        |

## Detailed Classification Report
```json
{
    "exact_match_accuracy": 0.0,
    "macro_avg": { "precision": 0.0, "recall": 0.0, "f1_score": 0.0 },
    "micro_avg": { "precision": 0.0, "recall": 0.0, "f1_score": 0.0 },
    "samples_avg": { "precision": 0.0, "recall": 0.0, "f1_score": 0.0 },
    "hallucination_count": 20,
    "detailed_report": {
        "exact_match_accuracy": 0.0,
        "macro_avg": {
            "precision": 0.0,
            "recall": 0.0,
            "f1_score": 0.0
        },
        "micro_avg": {
            "precision": 0.0,
            "recall": 0.0,
            "f1_score": 0.0
        },
        "samples_avg": {
            "precision": 0.0,
            "recall": 0.0,
            "f1_score": 0.0
        },
        "hallucination_count": 20,
        "detailed_report": {
            "I am a failure, loser": {
                "precision": 0.0,
                "recall": 0.0,
                "f1-score": 0.0,
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
                "precision": 0.0,
                "recall": 0.0,
                "f1-score": 0.0,
                "support": 3.0
            },
            "I am bound to be alone": {
                "precision": 0.0,
                "recall": 0.0,
                "f1-score": 0.0,
                "support": 8.0
            },
            "I am bound to be rejected": {
                "precision": 0.0,
                "recall": 0.0,
                "f1-score": 0.0,
                "support": 3.0
            },
            "I am defective": {
                "precision": 0.0,
                "recall": 0.0,
                "f1-score": 0.0,
                "support": 5.0
            },
            "I am helpless": {
                "precision": 0.0,
                "recall": 0.0,
                "f1-score": 0.0,
                "support": 6.0
            },
            "I am immoral": {
                "precision": 0.0,
                "recall": 0.0,
                "f1-score": 0.0,
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
                "precision": 0.0,
                "recall": 0.0,
                "f1-score": 0.0,
                "support": 8.0
            },
            "I am trapped": {
                "precision": 0.0,
                "recall": 0.0,
                "f1-score": 0.0,
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
                "precision": 0.0,
                "recall": 0.0,
                "f1-score": 0.0,
                "support": 8.0
            },
            "I am worthless, waste": {
                "precision": 0.0,
                "recall": 0.0,
                "f1-score": 0.0,
                "support": 11.0
            },
            "I don't deserve to live": {
                "precision": 0.0,
                "recall": 0.0,
                "f1-score": 0.0,
                "support": 3.0
            },
            "micro avg": {
                "precision": 0.0,
                "recall": 0.0,
                "f1-score": 0.0,
                "support": 89.0
            },
            "macro avg": {
                "precision": 0.0,
                "recall": 0.0,
                "f1-score": 0.0,
                "support": 89.0
            },
            "weighted avg": {
                "precision": 0.0,
                "recall": 0.0,
                "f1-score": 0.0,
                "support": 89.0
            },
            "samples avg": {
                "precision": 0.0,
                "recall": 0.0,
                "f1-score": 0.0,
                "support": 89.0
            }
        }
    }
}
```

## Key Findings & Qualitative Analysis

1. **Complete Formatting Failure:** The base instruction-tuned model has no intrinsic understanding of the structured multi-label taxonomy (`belief A`; `belief B`). Instead of emitting clean tokens from the fixed psychological taxonomy, it engages in conversational reasoning or free-form text extraction.
2. **High Hallucination Rate:** The pipeline registered 20 distinct taxonomy violations. The model frequently tries to paraphrase the user's emotional state using generic terms rather than the precise labels required by the evaluation script.
3. **Justification for Fine-Tuning:** These results confirm that standard zero-shot prompt adherence is insufficient for structured psychological schema extraction. Parameter updates (via LoRA) are strictly necessary to ground the model's output distribution within the defined boundaries.

