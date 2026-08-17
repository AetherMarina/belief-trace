import json
import random
import logging
from pathlib import Path
from datasets import Dataset

from src.utils.logger import setup_logger

logger = logging.getLogger(__name__)

# Define paths securely relative to the project root
PROJECT_ROOT: Path = Path(__file__).resolve().parent.parent.parent
DATA_RAW_DIR: Path = PROJECT_ROOT / "data" / "raw"
DATA_AUG_DIR: Path = PROJECT_ROOT / "data" / "augmented"
DATA_TRAIN_DIR: Path = PROJECT_ROOT / "data" / "train"
DATA_EVAL_DIR: Path = PROJECT_ROOT / "data" / "eval"

# Create output directories if they do not exist
DATA_TRAIN_DIR.mkdir(parents=True, exist_ok=True)
DATA_EVAL_DIR.mkdir(parents=True, exist_ok=True)

# Input file names reflecting the semantic raw data layer
TRAIN_FILE = DATA_RAW_DIR / "cbtbench_core_beliefs_train.jsonl"
EVAL_FILE = DATA_RAW_DIR / "cbtbench_core_beliefs_eval.jsonl"


def load_jsonl(file_path: Path) -> list:
    """Helper function to safely load JSONL files."""
    data = []
    if not file_path.exists():
        logger.warning(f"File not found: {file_path}")
        return data

    with open(file_path, "r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                data.append(json.loads(line))
    return data


def save_jsonl(data: list, file_path: Path) -> None:
    """Helper function to save data to a human-readable JSONL format for observability."""
    with open(file_path, "w", encoding="utf-8") as f:
        for item in data:
            f.write(json.dumps(item, ensure_ascii=False) + "\n")


def convert_to_unsloth_messages(item: dict) -> dict | None:
    """
    Robustly converts raw semantic data into the standardized 'messages' format.
    Optimized for smaller LLMs using semicolon separation and concise prompts.
    """
    # Defensive mapping: Look for semantic 'text' first, fallback to operational 'input'
    user_text = item.get("text") or item.get("input", "")

    # Defensive mapping: Look for semantic 'core_beliefs', fallback to training variants
    core_beliefs = (item.get("core_beliefs") or item.get("output"))

    # Quality Control: Skip empty or invalid entries
    if not user_text or not core_beliefs:
        return None

        # Optimization 1: Use semicolon delimiter instead of JSON string for token efficiency
    assistant_response = "; ".join(core_beliefs)

    # Optimization 2: Short, laser-focused system prompt
    return {
        "messages": [
            {
                "role": "system",
                "content": "Identify the underlying core beliefs from the narrative text."
            },
            {
                "role": "user",
                "content": user_text
            },
            {
                "role": "assistant",
                "content": assistant_response
            }
        ]
    }


def main():
    logger.info("Starting dataset preparation with robust key mapping and optimizations...")

    # --- STEP 1: Training Data (Original Raw + Augmented) ---
    logger.info("Loading semantic training and synthetic data...")
    train_raw = load_jsonl(TRAIN_FILE)
    
    # Load all augmented files from the directory
    train_aug = []
    for file_path in DATA_AUG_DIR.glob("*.jsonl"):
        logger.info(f"Loading augmented data from {file_path}...")
        train_aug.extend(load_jsonl(file_path))

    combined_train_data = train_raw + train_aug

    # Shuffle data to ensure stable training dynamics
    random.seed(42)
    random.shuffle(combined_train_data)

    logger.info(f"Combined and shuffled {len(combined_train_data)} training samples.")

    train_messages = []
    for item in combined_train_data:
        formatted = convert_to_unsloth_messages(item)
        if formatted:
            train_messages.append(formatted)

    logger.info(f"Successfully formatted {len(train_messages)} valid training samples.")

    # Optimization 3: Save parallel JSONL for human observability
    human_readable_train = DATA_TRAIN_DIR / "train.jsonl"
    save_jsonl(train_messages, human_readable_train)
    logger.info(f"Human-readable training data saved to: {human_readable_train}")

    # Create HF Dataset object and save to disk
    train_dataset = Dataset.from_list(train_messages)
    train_dataset_path = DATA_TRAIN_DIR / "ready_dataset"
    train_dataset.save_to_disk(str(train_dataset_path))
    logger.info(f"Binary HF training dataset saved to: {train_dataset_path}")

    # --- STEP 2: Evaluation Data (Original Raw) ---
    logger.info("Loading semantic evaluation data...")
    eval_raw = load_jsonl(EVAL_FILE)

    eval_messages = []
    for item in eval_raw:
        formatted = convert_to_unsloth_messages(item)
        if formatted:
            eval_messages.append(formatted)

    # Save parallel JSONL for eval
    human_readable_eval = DATA_EVAL_DIR / "eval.jsonl"
    save_jsonl(eval_messages, human_readable_eval)
    logger.info(f"Human-readable evaluation data saved to: {human_readable_eval}")

    # Create HF Dataset
    eval_dataset = Dataset.from_list(eval_messages)
    eval_dataset_path = DATA_EVAL_DIR / "ready_dataset"
    eval_dataset.save_to_disk(str(eval_dataset_path))
    logger.info(f"Binary HF evaluation dataset saved to: {eval_dataset_path}")

    logger.info("Dataset preparation completed successfully! Ready for Unsloth.")


if __name__ == "__main__":
    setup_logger(process_name="prepare")
    main()
