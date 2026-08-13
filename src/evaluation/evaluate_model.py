import argparse
import json
import logging
import re
import yaml
from pathlib import Path
from tqdm import tqdm
import torch
import pandas as pd
from sklearn.preprocessing import MultiLabelBinarizer
from sklearn.metrics import (
    classification_report,
    precision_recall_fscore_support,
    multilabel_confusion_matrix,
    accuracy_score
)
from unsloth import FastLanguageModel
from src.utils.logger import setup_logger
from src.utils.constants import BELIEF_LABELS, BASELINE_MODELS, MAX_SEQ_LENGTH
from src.utils.paths import resolve_dynamic_model_path


logger = logging.getLogger(__name__)

# --- Static Configuration ---
PROJECT_ROOT: Path = Path(__file__).resolve().parent.parent.parent
EVAL_FILE: Path = PROJECT_ROOT / "data" / "eval" / "eval.jsonl"


def parse_labels(value: str) -> list[str]:
    """Robustly parses and deduplicates semicolon-separated labels."""
    if not value:
        return []
    return sorted({
        label.strip()
        for label in re.split(r"\s*;\s*", value.strip())
        if label.strip()
    })


def load_eval_data(file_path: Path):
    """Loads exact prompts and ground truth labels from the JSONL."""
    examples = []
    with open(file_path, "r", encoding="utf-8") as f:
        for line in f:
            if not line.strip():
                continue
            messages = json.loads(line)["messages"]
            examples.append({
                "system": messages[0]["content"],
                "text": messages[1]["content"],
                "true_labels": parse_labels(messages[2]["content"]),
            })
    return examples


def main():
    parser = argparse.ArgumentParser(description="Evaluate Core Belief Models")
    parser.add_argument("--mode", type=str, choices=["baseline", "lora"], required=True)
    parser.add_argument("--run-name", type=str, required=True,
                        help="Used for config lookup and file naming (e.g. cloud_r8_lr2e-4)")
    parser.add_argument("--exp-tag", type=str, default="",
                        help="Optional experiment identifier (e.g., exp1, exp3)")
    args = parser.parse_args()

    # Dynamic Suffix for all outputs
    exp_prefix = f"_{args.exp_tag}" if args.exp_tag else ""
    suffix = f"{exp_prefix}_{args.run_name}"

    if args.mode == "baseline":
        try:
            model_path = BASELINE_MODELS[args.run_name]
        except KeyError:
            parser.error(
                "--run-name must be 'local' or 'cloud' when --mode baseline"
            )
    else:
        # Dynamically map the run-name back to the corresponding YAML config
        yaml_path = PROJECT_ROOT / "configs" / f"training_{args.run_name}.yaml"

        if not yaml_path.exists():
            raise FileNotFoundError(f"Configuration file not found: {yaml_path}")

        with open(yaml_path, "r", encoding="utf-8") as f:
            config = yaml.safe_load(f)

        # Extract the output path directly from the training section of the YAML
        # Make sure it resolves to an absolute path based on PROJECT_ROOT
        yaml_lora_save_path = resolve_dynamic_model_path(
            base_path=config["export"]["lora_save_path"],
            run_name=args.run_name,
            exp_tag=args.exp_tag
        )
        model_path = str(PROJECT_ROOT / yaml_lora_save_path)

        if not Path(model_path).exists():
            raise FileNotFoundError(f"Fine-tuned model adapter not found at: {model_path}")

    # Output file paths with dynamic suffix
    output_metrics_file = PROJECT_ROOT / "outputs" / "metrics" / f"evaluation_report_{args.mode}{suffix}.json"

    logger.info(f"Initializing Evaluation Engine in [{args.mode.upper()}] mode for run [{args.run_name}]...")
    output_metrics_file.parent.mkdir(parents=True, exist_ok=True)

    logger.info(f"Loading model from: {model_path}")
    model, tokenizer = FastLanguageModel.from_pretrained(
        model_name=model_path,
        max_seq_length=MAX_SEQ_LENGTH,
        dtype=None,
        load_in_4bit=True,
    )

    FastLanguageModel.for_inference(model)
    model.eval()  # Disable dropout layers for strict deterministic output

    logger.info("Loading evaluation dataset...")
    examples = load_eval_data(EVAL_FILE)
    logger.info(f"Loaded {len(examples)} evaluation examples.")

    pred_labels = []
    invalid_predictions = []
    valid_labels_set = set(BELIEF_LABELS)

    logger.info("Starting deterministic batch inference...")

    # Enable inference mode to disable autograd overhead
    with torch.inference_mode():
        for example in tqdm(examples, desc="Evaluating"):
            messages = [
                {"role": "system", "content": example["system"]},
                {"role": "user", "content": example["text"]}
            ]

            prompt = tokenizer.apply_chat_template(
                messages,
                tokenize=False,
                add_generation_prompt=True
            )

            inputs = tokenizer([prompt], return_tensors="pt").to("cuda")

            outputs = model.generate(
                **inputs,
                max_new_tokens=128,
                do_sample=False,  # Strictly greedy decoding
                use_cache=True,
                pad_token_id=tokenizer.eos_token_id
            )

            input_length = inputs.input_ids.shape[1]
            response = tokenizer.decode(outputs[0][input_length:], skip_special_tokens=True).strip()

            parsed_preds = parse_labels(response)

            # Separate valid taxonomy classes from hallucinations
            valid_preds = [label for label in parsed_preds if label in valid_labels_set]
            invalid_preds = [label for label in parsed_preds if label not in valid_labels_set]

            pred_labels.append(valid_preds)
            if invalid_preds:
                invalid_predictions.append({
                    "text": example["text"],
                    "hallucinated_labels": invalid_preds
                })

    logger.info("Computing Multi-Label Metrics...")
    mlb = MultiLabelBinarizer(classes=BELIEF_LABELS)
    # Fit ONLY on the predefined taxonomy
    mlb.fit([BELIEF_LABELS])

    true_labels_list = [ex["true_labels"] for ex in examples]
    y_true = mlb.transform(true_labels_list)
    y_pred = mlb.transform(pred_labels)

    # 1. Macro Metrics (Treats all classes equally)
    macro_p, macro_r, macro_f1, _ = precision_recall_fscore_support(y_true, y_pred, average='macro', zero_division=0)
    # 2. Micro Metrics (Global operational accuracy)
    micro_p, micro_r, micro_f1, _ = precision_recall_fscore_support(y_true, y_pred, average='micro', zero_division=0)
    # 3. Samples Metrics (Per-narrative accuracy)
    samples_p, samples_r, samples_f1, _ = precision_recall_fscore_support(y_true, y_pred, average='samples',
                                                                          zero_division=0)
    # 4. Exact-Match Accuracy (Strict subset accuracy)
    exact_match = accuracy_score(y_true, y_pred)

    detailed_report = classification_report(y_true, y_pred, target_names=mlb.classes_, zero_division=0,
                                            output_dict=True)
    string_report = classification_report(y_true, y_pred, target_names=mlb.classes_, zero_division=0)

    logger.info("\n--- Evaluation Results ---\n")
    logger.info(f"\n{string_report}")
    logger.info(f"Exact-Match Accuracy: {exact_match:.4f}\n")

    # Save Metrics
    results_dict = {
        "exact_match_accuracy": exact_match,
        "macro_avg": {"precision": macro_p, "recall": macro_r, "f1_score": macro_f1},
        "micro_avg": {"precision": micro_p, "recall": micro_r, "f1_score": micro_f1},
        "samples_avg": {"precision": samples_p, "recall": samples_r, "f1_score": samples_f1},
        "hallucination_count": len(invalid_predictions),
        "detailed_report": detailed_report
    }
    with open(output_metrics_file, "w", encoding="utf-8") as f:
        json.dump(results_dict, f, indent=4)

    # Save Predictions and Hallucinations dynamically
    predictions_file = PROJECT_ROOT / "outputs" / "predictions" / f"predictions_{args.mode}{suffix}.jsonl"
    predictions_file.parent.mkdir(parents=True, exist_ok=True)

    with open(predictions_file, "w", encoding="utf-8") as f:
        for ex, valid_pred in zip(examples, pred_labels):
            f.write(json.dumps({
                "text": ex["text"],
                "true_labels": ex["true_labels"],
                "predicted_labels": valid_pred
            }) + "\n")

    if invalid_predictions:
        hallucinations_file = PROJECT_ROOT / "outputs" / "predictions" / f"hallucinations_{args.mode}{suffix}.jsonl"
        with open(hallucinations_file, "w", encoding="utf-8") as f:
            for item in invalid_predictions:
                f.write(json.dumps(item) + "\n")

    # Save Confusion Matrices dynamically
    cm_dir = PROJECT_ROOT / "outputs" / "confusion_matrices"
    cm_dir.mkdir(parents=True, exist_ok=True)

    mcm = multilabel_confusion_matrix(y_true, y_pred)
    cm_data = []
    for i, class_name in enumerate(mlb.classes_):
        tn, fp, fn, tp = mcm[i].ravel()
        cm_data.append({
            "belief_class": class_name,
            "True_Positives": tp,
            "False_Positives": fp,
            "False_Negatives": fn,
            "True_Negatives": tn
        })

    cm_df = pd.DataFrame(cm_data)
    cm_df.to_csv(cm_dir / f"cm_summary_{args.mode}{suffix}.csv", index=False)
    logger.info(f"Evaluation artifacts saved successfully to '{output_metrics_file.parent}'.")


if __name__ == "__main__":
    # Example execution:
    # python -m src.evaluation.evaluate_model --mode lora --run-name cloud_r8_lr2e-4
    setup_logger(process_name="evaluate")
    main()
