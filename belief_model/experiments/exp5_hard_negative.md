# Experiment 5 — Hard-Negative Boundary Augmentation

## Goal

Experiment 5 tested whether hard-negative boundary examples could reduce false positives for broad “attractor” belief classes. The motivation was that the model was over-predicting several general labels, such as “I am undesirable, unwanted,” “I am worthless, waste,” “I am trapped,” “I am unlovable,” and “I am powerless, weak, vulnerable.”

The experiment also added positive anchor examples for under-predicted labels that the model often failed to predict.

## Data Strategy

* Base dataset: Experiment 1 dataset
* Added data: 52 Positive anchors and hard-negative boundary examples
* Generation method: Synthetic generation with explicit boundary constraints
* Purpose: Improve label boundaries and reduce over-prediction of broad attractor classes

The augmentation had two parts:

1. **Positive anchors** for under-predicted classes:

   * `I am incompetent`
   * `I am needy`
   * `I am bound to be rejected`
   * `I am bound to be alone`
   * `I am unattractive`

2. **Hard negatives** against over-predicted attractor classes:

   * `I am undesirable, unwanted`
   * `I am worthless, waste`
   * `I am trapped`
   * `I am unlovable`
   * `I am powerless, weak, vulnerable`

## Training Configuration

Only the strongest configuration from previous experiments was tested.

| Configuration | Macro F1 | Micro F1 | Samples F1 |
| ------------- | -------: | -------: | ---------: |
| r8_lr2e-4     | 0.403894 | 0.441989 |   0.444812 |

## Best Run

`r8_lr2e-4`

```text id="j3bleq"
Macro F1:   0.403894
Micro F1:   0.441989
Samples F1: 0.444812
```

## Interpretation

Experiment 5 kept Macro F1 close to the Experiment 1 champion, but reduced Micro F1 and Samples F1. This confirms that while hard-negative augmentation successfully targeted specific class-level boundaries, it heavily penalized overall multi-label belief-set recovery.

The confusion matrix reveals a dramatic overcorrection:

* The Attractor Class Suppression: The strategy succeeded in its main goal of lowering False Positives for prominent attractor labels. FPs for I am undesirable, unwanted dropped from 9 to 5, and FPs for I am powerless, weak, vulnerable plummeted from 6 to 2.
* The Collateral Damage on Recall: However, this aggressive curation caused the model to severely suppress these classes. True Positives (TPs) for I am worthless, waste fell from 8 to 3, while I am unlovable completely collapsed from 6 TPs to just 1.
* The Ghost Class Activation: On a positive note, the positive anchors successfully woke up dormant classes, generating the first True Positives for I am incompetent (0 to 1) and I am bound to be rejected (0 to 2).

This indicates that the hard-negative constraints were too aggressive. Instead of merely refining the decision boundaries, they introduced an intense negative bias that caused the model to panic and entirely avoid predicting core labels that frequently co-occur in multi-label CBT environments.

## Decision

Diagnostic only.

Experiment 5 was not selected as the final dataset because it reduced Micro F1 and Samples F1 compared with the Experiment 1 champion.

## Summary

Experiment 5 showed that hard-negative boundary examples can influence class-level behavior and reduce some under-prediction problems. However, the current version overcorrected the model and disturbed the original multi-label distribution. It is useful as an ablation study, but not as the final training strategy.
