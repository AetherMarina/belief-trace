import json
from typing import List
from openai import OpenAI
from core import LLMProvider, ExtractedTriplet, ConflictReport, Belief
from config import DEFAULT_MODEL, OLLAMA_BASE_URL


class OllamaProvider(LLMProvider):
    """
    Implementation of the LLMProvider protocol using a local Ollama instance
    via the OpenAI-compatible SDK.
    """

    def __init__(self, model: str = DEFAULT_MODEL, base_url: str = OLLAMA_BASE_URL):
        self.model = model
        self.client = OpenAI(base_url=base_url, api_key="ollama")

    def extract_beliefs(self, text: str, entity_id: str = "The main subject") -> List[ExtractedTriplet]:
        """
        Analyzes narrative text and extracts core cognitive beliefs.
        Enforces [SELF, WORLD, OTHERS] ontology and returns structured Pydantic DTOs.
        """
        system_prompt = f"""
            You are an expert cognitive psychologist analyzing a narrative. 
            Your task is to extract the Core Beliefs (deep psychological schemas) of a specific subject: {entity_id}.

            CRITICAL: Extract ONLY the beliefs held by {entity_id}. You must strictly ignore claims, opinions, or statements made by other interlocutors in the text.

            Extract beliefs ONLY related to these three core ontological categories:
            1. [SELF] ({entity_id}'s identity, capabilities, worth)
            2. [WORLD] (How reality, rules, or the environment function)
            3. [OTHERS] (The inherent nature or intentions of other entities)

            Format the output EXACTLY as a JSON object containing a "triplets" key, 
            which holds a list of objects representing the triplets.
            Each object in the list must have keys: "subject", "relation", "object".
            Do not include any other text, markdown, or explanations. 

            Example output:
            {{
                "triplets": [
                    {{"subject": "World", "relation": "REQUIRES", "object": "Strict Obedience"}},
                    {{"subject": "Self", "relation": "IS", "object": "Insignificant"}}
                ]
            }}
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
            triplets_raw = parsed_json.get("triplets", [])

            # Map raw dictionary lists to Pydantic objects for type-safe validation
            return [ExtractedTriplet(**t) for t in triplets_raw]

        except Exception as e:
            print(f"[OllamaProvider Error] Extraction failed: {e}")
            return []

    def evaluate_transitions(self, current_beliefs: List[Belief], new_text: str) -> List[ConflictReport]:
        """
        Evaluates current active beliefs against new narrative context.
        Acts as the Cognitive Arbiter and returns a list of shattered beliefs with psychological reasons.
        """
        # Simplify active beliefs into a lightweight format for the LLM prompt context
        beliefs_context = [
            {
                "belief_id": b.belief_id,
                "subject": b.subject,
                "relation": b.relation,
                "object": b.object
            }
            for b in current_beliefs
        ]

        system_prompt = f"""
        You are a Cognitive Arbiter. You evaluate a subject's existing Core Beliefs against their new experiences.

        CURRENT BELIEFS:
        {json.dumps(beliefs_context, indent=2)}

        Task: Read the provided NEW TEXT. Identify if the new experience shatters, contradicts, or fundamentally alters any of the CURRENT BELIEFS.

        Return EXACTLY a JSON object with a "deprecated_beliefs" key.
        This key must contain a list of objects, where each object has:
        1. "deprecated_belief_id": The exact ID of the broken belief (e.g., "b_001").
        2. "reason": A brief psychological explanation of why this experience contradicts the belief.

        If none are broken, return an empty list for "deprecated_beliefs".
        Do not include any other text, markdown, or explanations.

        Example output:
        {{
            "deprecated_beliefs": [
                {{
                    "deprecated_belief_id": "b_001",
                    "reason": "The subject now expresses deep confusion about their identity, contradicting their prior absolute certainty."
                }}
            ]
        }}
        """
        prompt = f"NEW TEXT:\n{new_text}"

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

            parsed_json = json.loads(response.choices[0].message.content)
            deprecated_raw = parsed_json.get("deprecated_beliefs", [])

            # Convert raw LLM output into validated ConflictReport objects
            reports = []
            for item in deprecated_raw:
                reports.append(ConflictReport(
                    deprecated_belief_id=item.get("deprecated_belief_id"),
                    reason=item.get("reason", "Shattered by new narrative context.")
                ))
            return reports

        except Exception as e:
            print(f"[OllamaProvider Error] Conflict resolution failed: {e}")
            return []
