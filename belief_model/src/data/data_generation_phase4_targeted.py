import json
import random
import logging
from pathlib import Path
from typing import Dict, Set

from src.data.data_generation import (
    generate_example,
    DOMAINS,
    PERSONA_BY_DOMAIN,
    RAW_DIR,
    AUGMENTED_DIR
)
from src.utils.logger import setup_logger

logger = logging.getLogger(__name__)

HIGH_PRIORITY_BELIEFS: Set[str] = {
    "I am needy",
    "I am unattractive"
}

MEDIUM_PRIORITY_BELIEFS: Set[str] = {
    "I am a failure, loser",
    "I am a victim",
    "I am bound to be alone",
    "I am bound to be rejected",
    "I am incompetent",
    "I am undesirable, unwanted",
}

# Define Quotas
TARGET_QUOTAS = {
    **{belief: 15 for belief in HIGH_PRIORITY_BELIEFS},
    **{belief: 10 for belief in MEDIUM_PRIORITY_BELIEFS}
}


def get_current_counts(output_file: Path) -> Dict[str, int]:
    """Reads the output file to see how many of each target belief have already been generated."""
    counts = {b: 0 for b in TARGET_QUOTAS.keys()}
    if not output_file.exists():
        return counts

    with open(output_file, "r", encoding="utf-8") as f:
        for line in f:
            if not line.strip():
                continue
            try:
                item = json.loads(line)
                if item.get("augmentation_phase") == "targeted_from_base":
                    # Increment counts for all target beliefs present in this generated item
                    for b in item.get("core_beliefs", []):
                        if b in counts:
                            counts[b] += 1
            except json.JSONDecodeError:
                continue
    return counts


def generate_targeted_from_base(input_file: Path, output_file: Path) -> None:
    # Load the raw dataset into memory
    base_dataset = []
    with open(input_file, "r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                base_dataset.append(json.loads(line))

    logger.info(f"Loaded {len(base_dataset)} rows from base dataset.")

    # Check how many we still need
    current_counts = get_current_counts(output_file)
    remaining_targets = {b: max(0, TARGET_QUOTAS[b] - current_counts[b]) for b in TARGET_QUOTAS}

    total_remaining = sum(remaining_targets.values())
    logger.info(f"Total augmentations remaining to hit exact quotas: {total_remaining}")

    if total_remaining <= 0:
        logger.info("All quotas are already met! Exiting.")
        return

    # Generate until quotas are met
    with open(output_file, "a", encoding="utf-8") as f_out:

        # Loop through the dataset randomly to avoid using the same base rows sequentially
        while total_remaining > 0:
            random.shuffle(base_dataset)

            for row in base_dataset:
                beliefs = row["core_beliefs"]

                # Check if this row contains ANY belief we still need
                beliefs_we_need = [b for b in beliefs if remaining_targets.get(b, 0) > 0]

                if not beliefs_we_need:
                    continue  # Skip row if it doesn't help us hit our quotas

                domain = random.choice(DOMAINS)
                persona = random.choice(PERSONA_BY_DOMAIN[domain])

                logger.info(f"Generating based on ID {row.get('id')} to satisfy targets: {beliefs_we_need}")

                # Call the LLM using your existing imported function
                text, temperature = generate_example(beliefs=beliefs, domain=domain, persona=persona)

                if text:
                    synthetic_item = {
                        "text": text,
                        "core_beliefs": beliefs,  # Uses the EXACT belief set from the base dataset
                        "domain": domain,
                        "persona": persona,
                        "temperature": temperature,
                        "source_id": row.get("id"),
                        "augmentation_phase": "targeted_from_base"
                    }

                    f_out.write(json.dumps(synthetic_item, ensure_ascii=False) + "\n")
                    f_out.flush()

                    # Deduct the counts for ALL target beliefs present in this newly generated row
                    for b in beliefs_we_need:
                        remaining_targets[b] -= 1

                    total_remaining = sum(remaining_targets.values())

                    if total_remaining <= 0:
                        break


if __name__ == "__main__":
    setup_logger(process_name="generation_targeted_base")
    logger.info("Starting Dataset Augmentation based on Raw Dataset Quotas...")

    # Set a local seed for reproducibility of the shuffling
    random.seed(42)

    in_path = RAW_DIR / "cbtbench_core_beliefs_train.jsonl"
    out_path = AUGMENTED_DIR / "cbtbench_core_beliefs_augmented_phase4.jsonl"

    generate_targeted_from_base(input_file=in_path, output_file=out_path)

    logger.info("Targeted Dataset Augmentation Completed!")
