import logging
import random
import json
from pathlib import Path

from src.data.data_generation import (
    generate_example,
    DOMAINS,
    PERSONA_BY_DOMAIN,
    AUGMENTED_DIR,
)
from src.utils.logger import setup_logger

logger = logging.getLogger(__name__)

CONTRASTIVE_PAIRS = {
    "I am needy": [
        "I am bound to be rejected",
        "I am unlovable",
    ],
    "I am unattractive": [
        "I am undesirable, unwanted",
        "I am bound to be rejected",
    ],
    "I am defective": [
        "I am worthless, waste",
        "I am bad - dangerous, toxic, evil",
        "I am immoral",
    ],
    "I don’t deserve to live": [
        "I am worthless, waste",
        "I am helpless",
        "I am powerless, weak, vulnerable",
    ],
    "I am bound to be alone": [
        "I am bound to be rejected",
        "I am bound to be abandoned",
        "I am unlovable",
    ],
}


def load_existing_generation_ids(output_file: Path) -> set[str]:
    if not output_file.exists():
        return set()

    ids = set()

    with open(output_file, "r", encoding="utf-8") as f:
        for line in f:
            if not line.strip():
                continue

            try:
                item = json.loads(line)
                if item.get("generation_id"):
                    ids.add(item["generation_id"])
            except json.JSONDecodeError:
                continue

    return ids


def generate_phase3_contrastive(output_file: Path) -> None:
    """
    Generates contrastive examples to teach the model distinctions between
    commonly confused belief classes.

    For each pair:
    - generate target-positive examples
    - generate confuser-only examples
    """

    examples_per_pair = 3
    existing_ids = load_existing_generation_ids(output_file)

    with open(output_file, "a", encoding="utf-8") as f_out:
        for target_belief, confusers in CONTRASTIVE_PAIRS.items():
            for confuser in confusers:
                for i in range(examples_per_pair):
                    variants = [
                        {
                            "kind": "target_positive",
                            "beliefs": [target_belief],
                        },
                        {
                            "kind": "confuser_only",
                            "beliefs": [confuser],
                        },
                    ]

                    for variant in variants:
                        kind = variant["kind"]
                        active_beliefs = variant["beliefs"]

                        generation_id = (
                            f"phase3::{target_belief}::vs::{confuser}::{kind}::{i}"
                        )

                        if generation_id in existing_ids:
                            continue

                        domain = random.choice(DOMAINS)
                        persona = random.choice(PERSONA_BY_DOMAIN[domain])

                        logger.info(
                            "Phase 3 contrastive | target=%s | confuser=%s | kind=%s | example=%s/%s",
                            target_belief,
                            confuser,
                            kind,
                            i + 1,
                            examples_per_pair,
                        )

                        text, temperature = generate_example(
                            beliefs=active_beliefs,
                            domain=domain,
                            persona=persona,
                        )

                        if not text:
                            continue

                        synthetic_item = {
                            "text": text,
                            "core_beliefs": active_beliefs,
                            "domain": domain,
                            "persona": persona,
                            "temperature": temperature,
                            "target_belief": target_belief,
                            "confuser_belief": confuser,
                            "contrastive_kind": kind,
                            "augmentation_phase": "phase3_contrastive_disambiguation",
                            "generation_id": generation_id,
                        }

                        f_out.write(json.dumps(synthetic_item, ensure_ascii=False) + "\n")
                        f_out.flush()
                        existing_ids.add(generation_id)


if __name__ == "__main__":
    setup_logger(process_name="generation_phase3")
    logger.info("Starting Dataset Augmentation Process Phase 3...")

    out_path = AUGMENTED_DIR / "cbtbench_core_beliefs_augmented_phase3_contrastive.jsonl"
    generate_phase3_contrastive(output_file=out_path)

    logger.info("Dataset Augmentation Completed!")
