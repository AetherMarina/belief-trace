import json
import random
import os
import time
import logging
from pathlib import Path
from typing import List, Tuple, Optional

from openai import OpenAI
from src.data.data_generation import (
    DOMAINS,
    PERSONA_BY_DOMAIN,
    AUGMENTED_DIR,
)
from src.utils.logger import setup_logger

logger = logging.getLogger(__name__)

# ---------------------------------------------------------
# EXPERIMENT 5 CONFIGURATION
# ---------------------------------------------------------
EXAMPLES_PER_SET = 2  # Generates 2 examples per specific label combination

# Type A: Positive Anchors for under-predicted minority classes
POSITIVE_ANCHORS = {
    "I am incompetent": [
        ["I am incompetent"],
        ["I am incompetent", "I am a failure, loser"],
    ],
    "I am needy": [
        ["I am needy"],
        ["I am needy", "I am bound to be rejected"],
        ["I am needy", "I am unlovable"],
    ],
    "I am bound to be rejected": [
        ["I am bound to be rejected"],
        ["I am bound to be rejected", "I am undesirable, unwanted"],
    ],
    "I am bound to be alone": [
        ["I am bound to be alone"],
        ["I am bound to be alone", "I am unlovable"],
    ],
    "I am unattractive": [
        ["I am unattractive"],
        ["I am unattractive", "I am undesirable, unwanted"],
    ],
}

# Type B: Hard Negatives to combat "Attractor Classes"
# Maps the OVER-PREDICTED class (high False Positive rate) to the label sets we want to train,
# along with the anti-prompt.
HARD_NEGATIVES = {
    "I am undesirable, unwanted": {
        "label_sets": [
            ["I am bound to be rejected"],
            ["I am bound to be alone"],
            ["I am unlovable"],
        ],
        "anti_prompt": "Crucially, this person knows they are highly attractive, desired, and wanted by others. They do not lack desirability. However, they believe external factors, timing, or their own internal barriers will always result in them being alone or rejected."
    },
    "I am worthless, waste": {
        "label_sets": [
            ["I am incompetent"],
            ["I am a failure, loser"],
            ["I am defective"],
        ],
        "anti_prompt": "Crucially, this person deeply values their own inherent worth and humanity. They know they have value as a person. However, they feel they are failing at specific tasks, roles, or have specific character flaws. Do NOT imply they feel like waste or inherently worthless."
    },
    "I am trapped": {
        "label_sets": [
            ["I am powerless, weak, vulnerable"],
            ["I am helpless"],
            ["I am out of control"],
        ],
        "anti_prompt": "Crucially, this person knows they are free to leave or escape their situation. They are absolutely not physically or circumstantially trapped. However, they feel too emotionally weak, overwhelmed, or helpless to execute those options."
    },
    "I am unlovable": {
        "label_sets": [
            ["I am bound to be rejected"],
            ["I am bound to be abandoned"],
            ["I am undesirable, unwanted"],
        ],
        "anti_prompt": "Crucially, this person knows they are fundamentally capable of being loved and have been deeply loved in the past. They do not believe there is a flaw preventing love. Instead, they believe bad luck, specific relational dynamics, or external forces will cause them to be abandoned or rejected."
    },
    "I am powerless, weak, vulnerable": {
        "label_sets": [
            ["I am a victim"],
            ["I am helpless"],
            ["I am trapped"],
        ],
        "anti_prompt": "Crucially, this person acknowledges they have immense personal strength, power, and capability in most areas of life. They do not view themselves as fundamentally weak. Instead, they feel victimized, trapped, or temporarily helpless in this one specific, overwhelming situation."
    },
}

# ---------------------------------------------------------
# INITIALIZATION
# ---------------------------------------------------------
random.seed(42)

client = OpenAI(
    base_url="https://generativelanguage.googleapis.com/v1beta/openai/",
    api_key=os.environ.get("GEMINI_API_KEY")
)


def generate_example_exp5(beliefs: List[str], domain: str, persona: str,
                          anti_prompt: str = "", max_retries: int = 3) -> Tuple[Optional[str], float]:
    """Generates synthetic examples, allowing for a negative constraint injection."""

    beliefs_text = "\n".join(f"- {b}" for b in beliefs)

    # Inject the contrastive learning constraint if it exists
    constraint_block = ""
    if anti_prompt:
        constraint_block = f"\nCRITICAL BOUNDARY CONSTRAINT (FOLLOW EXACTLY):\n{anti_prompt}\n"

    prompt = f"""
Generate a realistic first-person forum post.

Domain: {domain}
Persona: {persona}

The person should naturally express these underlying core beliefs:
{beliefs_text}
{constraint_block}
The generated story should strongly reflect at most 3-4 of the provided beliefs.
Not every belief must be equally visible.

Avoid direct self-judgments.

Do not write sentences such as:
- I am a failure
- I ruin everything
- I make everything worse
- nobody wants me

Express beliefs through events, decisions, relationships and consequences.

Return only the forum post body. Do not generate titles, subjects, headings, or markdown.

Requirements:
- Write as a real person seeking advice or sharing a problem.
- Use concrete life events rather than abstract self-analysis.
- Do NOT mention the beliefs explicitly.
- Do NOT list emotions.
- Make the beliefs inferable from the story.
- Length: 100-250 words.
- Return only the post text.
"""

    temperature = random.choice([0.6, 0.8, 1.0])

    for attempt in range(max_retries):
        try:
            response = client.chat.completions.create(
                model="gemini-3.1-pro-preview",
                messages=[
                    {"role": "system", "content": "You are an expert clinical psychologist and CBT researcher."},
                    {"role": "user", "content": prompt}
                ],
                temperature=temperature
            )
            return response.choices[0].message.content.strip(), temperature

        except Exception as e:
            logger.warning(f"API Error (Attempt {attempt + 1}/{max_retries}): {e}")
            if attempt < max_retries - 1:
                time.sleep(5)
            else:
                return None, temperature

    return None, temperature


def load_existing_generation_ids(output_file: Path) -> set[str]:
    """Loads existing generation IDs to allow safe script restarts."""
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


def generate_exp5_data(output_file: Path) -> None:
    existing_ids = load_existing_generation_ids(output_file)

    with open(output_file, "a", encoding="utf-8") as f_out:

        # 1. GENERATE POSITIVE ANCHORS (TYPE A)
        logger.info("Starting Phase 1: Positive Anchors...")
        for target_group, label_sets in POSITIVE_ANCHORS.items():
            for set_idx, label_set in enumerate(label_sets):
                for i in range(EXAMPLES_PER_SET):
                    gen_id = f"exp5::anchor::{target_group}::set{set_idx}::{i}"
                    if gen_id in existing_ids:
                        continue

                    domain = random.choice(DOMAINS)
                    persona = random.choice(PERSONA_BY_DOMAIN[domain])

                    logger.info(f"Generating Positive Anchor: {label_set}")
                    text, temp = generate_example_exp5(beliefs=label_set, domain=domain, persona=persona)

                    if text:
                        item = {
                            "text": text,
                            "core_beliefs": label_set,
                            "domain": domain, "persona": persona, "temperature": temp,
                            "augmentation_phase": "exp5_positive_anchor",
                            "generation_id": gen_id
                        }
                        f_out.write(json.dumps(item, ensure_ascii=False) + "\n")
                        f_out.flush()
                        existing_ids.add(gen_id)

        # 2. GENERATE HARD NEGATIVES (TYPE B)
        logger.info("Starting Phase 2: Hard-Negative Boundaries...")
        for attractor_label, config in HARD_NEGATIVES.items():
            anti_prompt = config["anti_prompt"]
            for set_idx, label_set in enumerate(config["label_sets"]):
                for i in range(EXAMPLES_PER_SET):
                    gen_id = f"exp5::hardneg::{attractor_label}::set{set_idx}::{i}"
                    if gen_id in existing_ids:
                        continue

                    domain = random.choice(DOMAINS)
                    persona = random.choice(PERSONA_BY_DOMAIN[domain])

                    logger.info(f"Generating Hard Negative against '{attractor_label}': {label_set}")
                    text, temp = generate_example_exp5(beliefs=label_set, domain=domain, persona=persona,
                                                       anti_prompt=anti_prompt)

                    if text:
                        item = {
                            "text": text,
                            "core_beliefs": label_set,
                            "domain": domain, "persona": persona, "temperature": temp,
                            "augmentation_phase": "exp5_hard_negative",
                            "generation_id": gen_id,
                            "anti_prompt_target": attractor_label  # Saves which label this was trained against
                        }
                        f_out.write(json.dumps(item, ensure_ascii=False) + "\n")
                        f_out.flush()
                        existing_ids.add(gen_id)


if __name__ == "__main__":
    setup_logger(process_name="generation_exp5")
    logger.info("Starting Experiment 5: Hard-Negative Boundary Augmentation...")

    out_path = AUGMENTED_DIR / "cbtbench_core_beliefs_augmented_phase5.jsonl"
    generate_exp5_data(output_file=out_path)

    logger.info("Experiment 5 Generation Completed!")
