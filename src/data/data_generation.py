import json
import random
import os
import time
import logging
from pathlib import Path
from typing import List, Tuple, Set, Optional

from openai import OpenAI

# 1. Configure Logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# 2. Project Constants
PROJECT_ROOT: Path = Path(__file__).resolve().parent.parent.parent
RAW_DIR: Path = PROJECT_ROOT / "data" / "raw"
AUGMENTED_DIR: Path = PROJECT_ROOT / "data" / "augmented"

AUGMENTED_DIR.mkdir(parents=True, exist_ok=True)

DOMAINS: List[str] = [
    "relationships", "career", "friendships", "family",
    "health", "education", "creativity", "financial stress"
]

PERSONA_BY_DOMAIN: dict[str, List[str]] = {
    "relationships": ["software engineer", "manager", "artist", "freelancer", "single parent",
                      "person recovering from breakup", "recent immigrant", "healthcare worker"],
    "career": ["software engineer", "manager", "artist", "freelancer", "recent immigrant", "college student",
               "healthcare worker", "small business owner"],
    "friendships": ["college student", "teenager", "recent immigrant", "software engineer", "artist", "freelancer",
                    "healthcare worker"],
    "family": ["single parent", "caregiver", "recent immigrant", "manager", "software engineer", "artist", "freelancer",
               "teenager", "healthcare worker"],
    "health": ["software engineer", "manager", "artist", "freelancer", "single parent",
               "person recovering from breakup", "college student", "recent immigrant", "caregiver",
               "healthcare worker"],
    "education": ["college student", "teenager", "recent immigrant"],
    "creativity": ["artist", "freelancer", "software engineer", "college student", "small business owner"],
    "financial stress": ["freelancer", "single parent", "recent immigrant", "artist", "college student", "manager",
                         "software engineer", "small business owner", "caregiver", "healthcare worker"]
}

RARE_BELIEFS: Set[str] = {
    "I am needy",
    "I am unattractive",
    "I don’t deserve to live",
    "I am immoral"
}

# 3. Initialization
random.seed(42)

client = OpenAI(
    base_url="https://generativelanguage.googleapis.com/v1beta/openai/",
    api_key=os.environ.get("GEMINI_API_KEY")
)


def generate_example(beliefs: List[str], domain: str, persona: str, max_retries: int = 3) -> Tuple[Optional[str], float]:
    """
    Calls the Gemini API to generate a realistic forum post implicitly reflecting core beliefs.

    Args:
        beliefs (List[str]): A list of fine-grained core beliefs to inject into the story.
        domain (str): The life domain/context for the story (e.g., 'career', 'health').
        persona (str): The persona of the author (e.g., 'software engineer').
        max_retries (int): Number of allowed API retry attempts in case of failure.

    Returns:
        Tuple[Optional[str], float]: The generated text and the temperature used.
                                     Returns (None, temperature) if generation fails.
    """
    beliefs_text = "\n".join(f"- {b}" for b in beliefs)

    prompt = f"""
Generate a realistic first-person forum post.

Domain: {domain}

Persona: {persona}

The person should naturally express these underlying core beliefs:

{beliefs_text}

The generated story should strongly reflect at most 3-4 of the provided beliefs.
Not every belief must be equally visible.

Avoid direct self-judgments.

Do not write sentences such as:
- I am a failure
- I ruin everything
- I make everything worse
- nobody wants me

Express beliefs through events, decisions, relationships and consequences.

Return only the forum post body.

Do not generate:
- titles
- subjects
- headings
- markdown formatting

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
                model="gemini-2.5-pro",
                messages=[
                    {"role": "system", "content": "You are an expert clinical psychologist and CBT researcher."},
                    {"role": "user", "content": prompt}
                ],
                temperature=temperature
            )
            generated_content = response.choices[0].message.content.strip()
            return generated_content, temperature

        except Exception as e:
            logger.warning(f"API Error (Attempt {attempt + 1}/{max_retries}): {e}")
            if attempt < max_retries - 1:
                time.sleep(5)  # Wait before retrying
            else:
                logger.error(f"Failed to generate example for domain '{domain}' after {max_retries} attempts.")
                return None, temperature


def augment_dataset(input_file: Path, output_file: Path) -> None:
    """
    Reads the raw dataset, dynamically generates synthetic augmented examples using an LLM,
    and appends the results to an output JSONL file. Supports stateful resuming.

    Args:
        input_file (Path): Path to the raw JSONL training data.
        output_file (Path): Path to save the augmented JSONL data.
    """
    # Load existing source_ids into a set to support resuming
    existing_ids: Set[str] = set()

    if output_file.exists():
        with open(output_file, "r", encoding="utf-8") as f_out:
            for line in f_out:
                if line.strip():
                    try:
                        existing_row = json.loads(line)
                        if "source_id" in existing_row:
                            existing_ids.add(existing_row["source_id"])
                    except json.JSONDecodeError:
                        continue

        logger.info(f"Found {len(existing_ids)} existing source_ids in the previously generated records.")

    # Process the input file and append new data
    with open(input_file, "r", encoding="utf-8") as f_in, \
            open(output_file, "a", encoding="utf-8") as f_out:

        for line in f_in:
            if not line.strip():
                continue

            row = json.loads(line)
            source_id = row.get("id")

            if source_id in existing_ids:
                logger.debug(f"Skipping row {source_id}, already created.")
                continue

            beliefs = row["core_beliefs"]

            # Determining the number of augmentations for sparse schemes
            n_augments = 3 if any(b in RARE_BELIEFS for b in beliefs) else 1

            for i in range(n_augments):
                domain = random.choice(DOMAINS)
                persona = random.choice(PERSONA_BY_DOMAIN[domain])

                logger.info(f"Generating for ID {source_id} (Augment {i + 1}/{n_augments}) | Domain: {domain}")

                text, temperature = generate_example(
                    beliefs=beliefs,
                    domain=domain,
                    persona=persona
                )

                if text is None:
                    logger.error(f"Skipping augment {i + 1} for ID {source_id} due to API failure.")
                    continue

                augmented_item = {
                    "text": text,
                    "core_beliefs": beliefs,
                    "domain": domain,
                    "persona": persona,
                    "temperature": temperature,
                    "source_id": source_id,
                    "augment_idx": i
                }

                json_line = json.dumps(augmented_item, ensure_ascii=False)
                f_out.write(json_line + "\n")
                f_out.flush()  # Force OS to dump buffer to disk

            logger.info(f"Successfully processed ID: {source_id}")


if __name__ == "__main__":
    logger.info("Starting Dataset Augmentation Process...")

    in_path = RAW_DIR / "cbtbench_core_beliefs_train.jsonl"
    out_path = AUGMENTED_DIR / "cbtbench_core_beliefs_augmented.jsonl"

    augment_dataset(input_file=in_path, output_file=out_path)

    logger.info("Dataset Augmentation Completed!")
