import json
import pandas as pd
from pathlib import Path
import re


def generate_metrics_report(base_dir="."):
    """
    Scans the outputs/metrics directory to locate evaluation metrics across various
    experiments and hyperparameter configurations, compiling them into a unified report.

    Expected Folder/File Structure:
        ./outputs/metrics/evaluation_report_lora_cloud_exp1a_r8_lr1e-4.json

    Args:
        base_dir (str/Path): The root directory from which to start scanning. Defaults to ".".

    Returns:
        pd.DataFrame: A sorted DataFrame comparing Macro, Micro, and Samples F1 scores.
    """
    data = []

    # 1. Target the specific metrics directory
    metrics_dir = Path(base_dir) / "outputs" / "metrics"

    if not metrics_dir.exists():
        print(f"Metrics directory not found: {metrics_dir}")
        return pd.DataFrame()

    # Look for all JSON evaluation reports in that folder
    file_pattern = "evaluation_report_lora_*.json"

    for file_path in metrics_dir.glob(file_pattern):
        file_stem = file_path.stem

        # 2. Extract experiment tag and configuration from the filename
        # This regex strictly captures letters and numbers after 'exp', stopping at the underscore '_'
        # Example: 'evaluation_report_lora_cloud_exp1a_r8_lr1e-4' -> tag: '1a', config: 'r8_lr1e-4'
        match = re.search(r'exp([a-zA-Z0-9]+)_(.*)$', file_stem)

        if match:
            experiment = match.group(1)
            configuration = match.group(2)
        else:
            # Fallback just in case a file doesn't perfectly match the naming convention
            experiment = "unknown"
            configuration = file_stem.replace("evaluation_report_lora_", "")

        # 3. Read JSON and extract F1 scores
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                report = json.load(f)

            macro_f1 = report.get("macro_avg", {}).get("f1_score", None)
            micro_f1 = report.get("micro_avg", {}).get("f1_score", None)
            sample_f1 = report.get("samples_avg", {}).get("f1_score", None)

            data.append({
                "experiment": experiment,
                "configuration": configuration,
                "macro_f1": macro_f1,
                "micro_f1": micro_f1,
                "sample_f1": sample_f1
            })

        except Exception as e:
            print(f"Failed to process {file_path}: {e}")

    df = pd.DataFrame(data)

    # Sort the dataframe by configuration, then experiment for direct baseline/ablation comparison
    if not df.empty:
        # Extract leading numbers. If it's purely letters (like 'ss'), assign 999999 so it sorts at the bottom.
        df['exp_num'] = pd.to_numeric(df['experiment'].str.extract(r'^(\d+)')[0], errors='coerce').fillna(999999)

        # Sort by configuration, then numeric order, then alphabetical fallback
        df = df.sort_values(by=['configuration', 'exp_num', 'experiment']).drop(columns=['exp_num']).reset_index(
            drop=True)

    return df


if __name__ == "__main__":
    # Run from the repository root to automatically find the outputs/metrics/ logs
    df_metrics = generate_metrics_report()

    if not df_metrics.empty:
        print("--------- Experiment Evaluation Summary Matrix -----------")
        print(df_metrics.to_string(index=False))
        print("----------------------------------------------------------")
    else:
        print("No matching metric files found. Check your directory structure.")
