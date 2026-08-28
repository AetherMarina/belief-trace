from typing import List, Optional
from openai import OpenAI
import json
import logging
from pydantic import ValidationError

from .core import LLMProvider, ExtractedTriplet, TransitionType, Belief
from .config import DEFAULT_MODEL, OLLAMA_BASE_URL, VERIFIER_MODEL

logger = logging.getLogger(__name__)


class OllamaProvider(LLMProvider):
    """
    Implementation of the LLMProvider protocol using a local Ollama instance
    via the OpenAI-compatible SDK.
    """

    def __init__(self, model: str = DEFAULT_MODEL, base_url: str = OLLAMA_BASE_URL):
        self.model = model
        self.verifier_model = VERIFIER_MODEL
        self.client = OpenAI(base_url=base_url, api_key="ollama")

    def extract_beliefs(self, text: str, entity_id: str = "The main subject") -> List[ExtractedTriplet]:
        """
        Extract inferred surface beliefs expressed or implied by the narrative
        about Self, World, or Others and returns structured Pydantic DTOs.
        """
        system_prompt = f"""
            You are analyzing a narrative to extract surface beliefs held by a specific subject: {entity_id}.

            A surface belief is an explicit or strongly implied proposition about the Self, World, or Others
            that represents what {entity_id} currently believes, assumes, expects, or understands.
            
            Do NOT infer deep psychological schemas, latent core beliefs, diagnoses, or hidden motivations.
            Preserve the meaning of explicit belief statements as directly as possible. 
            Extract psychologically meaningful surface beliefs, not incidental factual details,
            unless those details directly express identity, capability, expectation, evaluation,
            or a belief about how Self, World, or Others are.
            
            Do not extract descriptive facts merely because they are mentioned.

            CRITICAL:
            - Extract ONLY beliefs held by {entity_id}.
            - Do not treat another character's statement as {entity_id}'s belief unless the narrative clearly shows that {entity_id} adopts or accepts it.
            - Attribute each belief to the correct subject.
            - Preserve explicit identity claims and explicit identity negations.

            Use ONLY these canonical subject categories:
            1. Self   - beliefs about {entity_id}'s identity, state, capabilities, or worth
            2. World  - beliefs about reality, the environment, or how the world works
            3. Others - beliefs about other people/entities, their nature, behavior, or intentions
            Never use a person's name or any other value as the subject.

            Format the output EXACTLY as a JSON object containing a "triplets" key, 
            which holds a list of objects representing the triplets.
            Each object in the list must have keys: "subject", "relation", "object".
            The relation must contain the predicate meaning. Do not place verb phrases inside the object.
            Return triplets in the order in which they occur or become apparent in the narrative.

            Example output:
            {{
                "triplets": [
                    {{"subject": "World", "relation": "REQUIRES", "object": "Strict Obedience"}},
                    {{"subject": "Self", "relation": "IS", "object": "Insignificant"}}
                ]
            }}
            
            Do not include any other text, markdown, or explanations. 
            """

        prompt = f"Analyze the following text and extract the core beliefs for {entity_id}:\n\nTEXT:\n{text}"

        try:
            response = self.client.chat.completions.create(
                model=self.model,
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": prompt}
                ],
                response_format={"type": "json_object"},
                temperature=0.0
            )

            raw_content = response.choices[0].message.content
            parsed_json = json.loads(raw_content)
            raw_triplets = parsed_json.get("triplets", [])

            # Map raw dictionary lists to Pydantic objects for type-safe validation
            valid_triplets = []
            for item in raw_triplets:
                try:
                    triplet = ExtractedTriplet(**item)
                    valid_triplets.append(triplet)
                except ValidationError as e:
                    logger.warning("Skipping invalid extracted triplet %s: %s", item, e)
            return valid_triplets

        except Exception as e:
            logger.error(f"[OllamaProvider Error] Extraction failed: {e}")
            return []

    def resolve_potential_contradiction(self, old_belief: Belief, new_triplet: ExtractedTriplet) \
            -> Optional[TransitionType]:
        """
        Evaluates whether a contradiction is a mere negation (SHATTERED)
        or an evolution into a new state (REFRAMED).
        """
        prompt = (
            f"Old belief: [{old_belief.subject}] "
            f"{old_belief.relation} [{old_belief.object}]\n"
            f"New belief: [{new_triplet.subject}] "
            f"{new_triplet.relation} [{new_triplet.object}]\n\n"

            "Determine whether the two beliefs genuinely conflict.\n"
            "If both beliefs can reasonably be true at the same time, "
            "return NOT_CONTRADICTION.\n\n"

            "Return exactly one label:\n"
            "NOT_CONTRADICTION - the beliefs can coexist without one invalidating the other\n"
            "SHATTERED - represents direct negation or explicit abandonment of an existing belief "
            "without introducing a substantive replacement\n"
            "REFRAMED - the new belief replaces the old belief with a substantive alternative\n"
        )

        try:
            response = self.client.chat.completions.create(
                model=self.verifier_model,
                messages=[{"role": "user", "content": prompt}],
                temperature=0.0
            )

            result = response.choices[0].message.content.strip().upper()

            if result == "NOT_CONTRADICTION":
                return None
            elif result == "SHATTERED":
                return TransitionType.SHATTERED
            else:
                return TransitionType.REFRAMED

        except Exception as e:
            logger.error(f"[OllamaProvider Error] Classification failed: {e}")
            return TransitionType.REFRAMED
