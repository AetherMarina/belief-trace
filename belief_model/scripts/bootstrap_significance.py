import argparse
import json
import logging
import numpy as np
import pandas as pd
from pathlib import Path
from tqdm import tqdm
from sklearn.preprocessing import MultiLabelBinarizer
from sklearn.metrics import precision_recall_fscore_support

# Configure basic logging
logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
logger = logging.getLogger(__name__)


def load_predictions(file_path: Path):
    """Loads true and predicted labels from a saved predictions JSONL file."""
    y_true = []
    y_pred = []

    with open(file_path, "r", encoding="utf-8") as f:
        for line in f:
            if not line.strip():
                continue
            data = json.loads(line)
            y_true.append(data["true_labels"])
            y_pred.append(data["predicted_labels"])

    return y_true, y_pred


def calculate_f1_scores(y_true_bin, y_pred_bin):
    """Calculates Macro, Micro, and Samples F1 scores."""
    _, _, macro_f1, _ = precision_recall_fscore_support(y_true_bin, y_pred_bin, average='macro', zero_division=0)
    _, _, micro_f1, _ = precision_recall_fscore_support(y_true_bin, y_pred_bin, average='micro', zero_division=0)
    _, _, samples_f1, _ = precision_recall_fscore_support(y_true_bin, y_pred_bin, average='samples', zero_division=0)
    return macro_f1, micro_f1, samples_f1


def main():
    parser = argparse.ArgumentParser(description="Bootstrap Statistical Significance for Multi-Label F1")
    parser.add_argument("--predictions-model-a", type=str, required=True,
                        help="Path to predictions JSONL for Model A (Baseline)")
    parser.add_argument("--predictions-model-b", type=str, required=True,
                        help="Path to predictions JSONL for Model B (Variant)")
    parser.add_argument("--iterations", type=int, default=10000, help="Number of bootstrap resamples (default: 10000)")
    parser.add_argument("--seed", type=int, default=42, help="Random seed for reproducibility")
    args = parser.parse_args()

    path_a = Path(args.predictions_model_a)
    path_b = Path(args.predictions_model_b)

    if not path_a.exists() or not path_b.exists():
        logger.error("One or both prediction files do not exist.")
        return

    logger.info("Loading predictions...")
    y_true_a, y_pred_a = load_predictions(path_a)
    y_true_b, y_pred_b = load_predictions(path_b)

    # Sanity check: Ensure both files evaluated the exact same dataset
    if len(y_true_a) != len(y_true_b) or y_true_a != y_true_b:
        logger.error(
            "Mismatch in ground truth labels between Model A and Model B. Ensure both evaluated the exact same dataset.")
        return

    N = len(y_true_a)
    logger.info(f"Loaded {N} evaluation examples.")

    # Dynamically fit the MultiLabelBinarizer on all possible classes present in the files
    logger.info("Binarizing labels...")
    all_labels = set(label for sublist in (y_true_a + y_pred_a + y_pred_b) for label in sublist)
    mlb = MultiLabelBinarizer(classes=sorted(list(all_labels)))
    mlb.fit([sorted(list(all_labels))])

    y_true_bin = mlb.transform(y_true_a)
    y_pred_a_bin = mlb.transform(y_pred_a)
    y_pred_b_bin = mlb.transform(y_pred_b)

    # Initialize arrays to store differences (B - A)
    diff_macro = np.zeros(args.iterations)
    diff_micro = np.zeros(args.iterations)
    diff_samples = np.zeros(args.iterations)

    logger.info(f"Starting paired bootstrap resampling ({args.iterations} iterations)...")
    np.random.seed(args.seed)

    for i in tqdm(range(args.iterations), desc="Bootstrapping"):
        # 1. Resample indices with replacement (Paired assumption: same indices for both models)
        indices = np.random.choice(N, size=N, replace=True)

        # 2. Extract resampled arrays
        y_true_boot = y_true_bin[indices]
        y_pred_a_boot = y_pred_a_bin[indices]
        y_pred_b_boot = y_pred_b_bin[indices]

        # 3. Calculate F1s for Model A
        macro_a, micro_a, samples_a = calculate_f1_scores(y_true_boot, y_pred_a_boot)

        # 4. Calculate F1s for Model B
        macro_b, micro_b, samples_b = calculate_f1_scores(y_true_boot, y_pred_b_boot)

        # 5. Store the differences (B - A)
        diff_macro[i] = macro_b - macro_a
        diff_micro[i] = micro_b - micro_a
        diff_samples[i] = samples_b - samples_a

    logger.info("Computing 95% Confidence Intervals...")

    # Calculate 2.5th and 97.5th percentiles
    ci_macro = np.percentile(diff_macro, [2.5, 97.5])
    ci_micro = np.percentile(diff_micro, [2.5, 97.5])
    ci_samples = np.percentile(diff_samples, [2.5, 97.5])

    # Helper function to format and check significance
    def format_result(metric_name, ci_array):
        lower, upper = ci_array
        # If the interval does NOT cross 0, the difference is statistically significant
        is_significant = not (lower <= 0 <= upper)
        sig_str = "YES" if is_significant else "NO"
        return {
            "Metric": metric_name,
            "Lower 2.5%": f"{lower:.4f}",
            "Upper 97.5%": f"{upper:.4f}",
            "Significant (95% CI excludes 0)": sig_str
        }

    results = [
        format_result("Macro F1", ci_macro),
        format_result("Micro F1", ci_micro),
        format_result("Samples F1", ci_samples)
    ]

    df_results = pd.DataFrame(results)

    print("\n" + "=" * 65)
    print(" Bootstrap Significance Report (Model B - Model A) ")
    print("=" * 65)
    print(df_results.to_string(index=False))
    print("=" * 65)
    print("Interpretation: If 'Significant' is YES, the performance difference")
    print("is highly likely to be real, not just evaluation split noise.\n")


if __name__ == "__main__":
    main()
