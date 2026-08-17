import json
import re
from pathlib import Path


def extract_taxonomy():
    project_root = Path(__file__).resolve().parent.parent.parent

    # List of all dataset files to scan
    dataset_paths = [
        project_root / "data" / "train" / "train.jsonl",
        project_root / "data" / "eval" / "eval.jsonl"
    ]

    unique_beliefs = set()

    print("Starting dataset scan...\n")

    for dataset_path in dataset_paths:
        if not dataset_path.exists():
            print(f"Warning: {dataset_path.name} not found at {dataset_path}. Skipping...")
            continue

        print(f"Scanning {dataset_path.name}...")
        with open(dataset_path, "r", encoding="utf-8") as f:
            for line in f:
                if not line.strip():
                    continue

                item = json.loads(line)

                # The true labels are located in the final assistant message
                labels_str = item["messages"][2]["content"]

                # Split and clean the labels
                labels = [
                    label.strip()
                    for label in re.split(r"\s*;\s*", labels_str.strip())
                    if label.strip()
                ]
                unique_beliefs.update(labels)

    print("BELIEF_LABELS = [")
    for belief in sorted(unique_beliefs):
        print(f'    "{belief}",')
    print("]")

    print(f"\nTotal unique classes found: {len(unique_beliefs)}")


if __name__ == "__main__":
    extract_taxonomy()
