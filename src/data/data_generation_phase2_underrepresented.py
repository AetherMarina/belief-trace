import logging
import random
import json
from pathlib import Path
from typing import Set

from src.data.data_generation import (
    generate_example,
    DOMAINS,
    PERSONA_BY_DOMAIN,
    AUGMENTED_DIR,
)
from src.utils.logger import setup_logger

logger = logging.getLogger(__name__)

# Define the logical neighbors to prevent confounding with dominant classes
COHERENT_SETS = {
    "I am needy": ["I am bound to be rejected", "I am unlovable"],
    "I am unattractive": ["I am undesirable, unwanted", "I am defective"],
    "I am a failure, loser": ["I am incompetent"],
    "I am a victim": ["I am powerless, weak, vulnerable"],
    "I am bound to be alone": ["I am unlovable"],
    "I am bound to be rejected": ["I am undesirable, unwanted"],
    "I am incompetent": ["I am a failure, loser"],
    "I am undesirable, unwanted": ["I am defective"]
}

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


def generate_phase2(output_file: Path) -> None:
    # Use your defined targets
    targets = {
        **{belief: 50 for belief in HIGH_PRIORITY_BELIEFS},
        **{belief: 25 for belief in MEDIUM_PRIORITY_BELIEFS}
    }

    existing_ids = load_existing_generation_ids(output_file)

    with open(output_file, "a", encoding="utf-8") as f_out:
        for target_belief, needed in targets.items():
            for i in range(needed):
                generation_id = f"phase2::{target_belief}::{i}"

                if generation_id in existing_ids:
                    continue

                domain = random.choice(DOMAINS)
                persona = random.choice(PERSONA_BY_DOMAIN[domain])

                # Coherent Set Logic: 50% chance of being pure, 50% chance of grabbing 1 related belief
                active_beliefs = [target_belief]
                if random.random() > 0.5 and target_belief in COHERENT_SETS:
                    neighbor = random.choice(COHERENT_SETS[target_belief])
                    active_beliefs.append(neighbor)

                text, temperature = generate_example(beliefs=active_beliefs, domain=domain, persona=persona)

                if text:
                    synthetic_item = {
                        "text": text,
                        "core_beliefs": active_beliefs,
                        "domain": domain,
                        "persona": persona,
                        "temperature": temperature,
                        "target_belief": target_belief,
                        "augmentation_phase": "phase2_targeted_rare_balancing",
                        "generation_id": generation_id
                    }

                    f_out.write(json.dumps(synthetic_item, ensure_ascii=False) + "\n")
                    f_out.flush()
                    existing_ids.add(generation_id)


if __name__ == "__main__":
    setup_logger(process_name="generation_phase2")
    logger.info("Starting Dataset Augmentation Process Phase 2...")

    out_path = AUGMENTED_DIR / "cbtbench_core_beliefs_augmented_phase2.jsonl"
    generate_phase2(output_file=out_path)

    logger.info("Dataset Augmentation Completed!")
