import json
import logging
from pathlib import Path
from datasets import load_dataset

from src.utils.logger import setup_logger

logger = logging.getLogger(__name__)

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
RAW_DIR = PROJECT_ROOT / "data" / "raw"

# Create raw directory if it doesn't exist
RAW_DIR.mkdir(parents=True, exist_ok=True)


def export_subset(hf_file_name: str, output_name: str):
    """
    Downloads a specific JSON file from the HF repository.
    """
    logger.info(f"Downloading {hf_file_name} from Hugging Face...")

    try:
        # Use data_files to target the exact JSON file in the CBT-Bench repo
        dataset = load_dataset("Psychotherapy-LLM/CBT-Bench", data_files={"train": hf_file_name})
        rows = dataset["train"]

        output_file = RAW_DIR / f"{output_name}.jsonl"

        with open(output_file, "w", encoding="utf-8") as f:
            for item in rows:
                json_line = {
                    "id": item.get("id", ""),
                    # Map to "input" and "expected_output" to perfectly match prepare_dataset.py
                    "text": item.get("ori_text", ""),
                    "core_beliefs": item.get("core_belief_fine_grained", [])
                }
                f.write(json.dumps(json_line, ensure_ascii=False) + "\n")

        logger.info(f"Successfully exported {len(rows)} rows -> {output_file}")

    except Exception as e:
        logger.error(f"Failed to download or process {hf_file_name}: {e}")


if __name__ == "__main__":
    setup_logger(process_name="download")

    logger.info("Starting dataset download process...")

    # Reminder: We use the 112-row 'test' file for our training, and the 20-row 'seed' for eval/baseline
    export_subset("core_fine_test.json", "cbtbench_core_beliefs_train")
    export_subset("core_fine_seed.json", "cbtbench_core_beliefs_eval")

    logger.info("Download process completed.")
