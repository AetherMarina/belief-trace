from enum import Enum
from typing import Optional, Dict, Any, List, Protocol
from pydantic import BaseModel, Field

# ========================================================================
# 1. ENUMS
# ========================================================================

class BeliefStatus(str, Enum):
    ACTIVE = "active"
    CHALLENGED = "challenged"
    DEPRECATED = "deprecated"

class TransitionType(str, Enum):
    SHATTERED = "shattered"
    # Future expansion: REFRAMED, REINFORCED, etc.

# ========================================================================
# 2. PROVENANCE SCHEMAS
# ========================================================================

class InferenceProvenance(BaseModel):
    """Explicitly tracks the origin of an LLM-inferred cognitive transition or belief."""
    model: str
    prompt_version: str
    temperature: Optional[float] = None
    extraction_run_id: Optional[str] = None
    # We keep a tiny metadata dict here ONLY for truly unpredictable extras
    # (like prompt tokens used, if the API provides it)
    extra_meta: Dict[str, Any] = Field(default_factory=dict)

# ========================================================================
# 3. CORE SCHEMAS
# ========================================================================

class Belief(BaseModel):
    """Represents a single, unique belief node in the longitudinal graph."""
    belief_id: str
    entity_id: str
    subject: str
    relation: str
    object: str
    first_seen_step: int
    last_seen_step: int
    status: BeliefStatus = BeliefStatus.ACTIVE
    source_id: str
    evidence_span: Optional[str] = None
    provenance: InferenceProvenance

class Transition(BaseModel):
    """Represents an edge explaining a cognitive state change between steps."""
    transition_id: str
    affected_belief_id: str
    resulting_belief_id: Optional[str] = None  # None if the belief was shattered without a direct replacement
    transition_type: TransitionType
    step: int
    reason: str
    provenance: InferenceProvenance  # Transitions are also LLM-inferred, so they need provenance too


# --- temporary LLM DATA TRANSFER OBJECTS (DTOs) ---

class ExtractedTriplet(BaseModel):
    """Raw cognitive triplet extracted by the LLM before it becomes a Belief node."""
    subject: str
    relation: str
    object: str


class ConflictReport(BaseModel):
    """Cognitive arbiter's report indicating which belief was shattered and why."""
    deprecated_belief_id: str
    reason: str


# ========================================================================
# 2. INTERFACES (Protocols)
# ========================================================================

class LLMProvider(Protocol):
    """
    Contract that any LLM client (Ollama, OpenAI) must fulfill to be
    integrated into the Belief-Graph framework.
    """

    def extract_beliefs(self, text: str, entity_id: str) -> List[ExtractedTriplet]:
        """Analyzes the input text and returns a list of cognitive triplets for the specified entity."""
        ...

    def evaluate_transitions(self, current_beliefs: List[Belief], new_text: str) -> List[ConflictReport]:
        """
        Acts as the Cognitive Arbiter. Compares active beliefs against new
        narrative experiences and returns a report of broken beliefs.
        """
        ...


# ========================================================================
# 3. THE LONGITUDINAL ENGINE (Tn Framework)
# ========================================================================

class LongitudinalEngine:
    """
    The core state machine that processes text sequentially, manages the
    belief lifecycle, and handles the JSONL persistence layer.
    """

    def __init__(self, llm_provider: LLMProvider):
        self.provider = llm_provider
        self.beliefs: Dict[str, Belief] = {}
        self.transitions: Dict[str, Transition] = {}
        self._belief_counter = 0
        self._transition_counter = 0

    def _generate_belief_id(self) -> str:
        self._belief_counter += 1
        return f"b_{self._belief_counter:03d}"

    def _generate_transition_id(self) -> str:
        self._transition_counter += 1
        return f"t_{self._transition_counter:03d}"

    def get_active_beliefs(self) -> List[Belief]:
        return [b for b in self.beliefs.values() if b.status == BeliefStatus.ACTIVE]

    def process_step(self, entity_id: str, step: int, text: str, source_id: str):
        """Processes a single longitudinal step (Tn), evaluating conflicts and extracting new beliefs."""
        print(f"\n--- Processing Step {step} | Source: {source_id} ---")

        # 0. Generate the provenance record for this specific execution
        # We safely try to grab the model name from the provider, defaulting to 'unknown' if not found
        current_provenance = InferenceProvenance(
            model=getattr(self.provider, 'model', 'unknown-llm'),
            prompt_version="v0.1",
            temperature=0.0
        )

        # Step 1: Extract new beliefs from the narrative text
        print(f"Extracting new beliefs for '{entity_id}' via LLMProvider...")
        extracted_triplets = self.provider.extract_beliefs(text=text, entity_id=entity_id)

        # Step 2: Evaluate conflicts against currently active beliefs
        active_beliefs = self.get_active_beliefs()
        conflict_reports = []
        if active_beliefs:
            print(f"Evaluating conflicts against {len(active_beliefs)} active beliefs...")
            conflict_reports = self.provider.evaluate_transitions(active_beliefs, text)

        # Step 3: Apply transitions (deprecate broken beliefs)
        for report in conflict_reports:
            belief_id = report.deprecated_belief_id
            if belief_id in self.beliefs:
                # Update status and lock the 'last_seen_step'
                self.beliefs[belief_id].status = BeliefStatus.DEPRECATED
                self.beliefs[belief_id].last_seen_step = step

                # Record the transition edge WITH provenance
                transition = Transition(
                    transition_id=self._generate_transition_id(),
                    affected_belief_id=belief_id,
                    resulting_belief_id=None,
                    transition_type=TransitionType.SHATTERED,
                    step=step,
                    reason=report.reason,
                    provenance=current_provenance
                )
                self.transitions[transition.transition_id] = transition
                print(f"[Conflict Detected] Belief '{belief_id}' shattered. Reason: {report.reason}")

        # Step 4: Register new beliefs WITH provenance
        for triplet in extracted_triplets:
            new_belief = Belief(
                belief_id=self._generate_belief_id(),
                entity_id=entity_id,
                subject=triplet.subject,
                relation=triplet.relation,
                object=triplet.object,
                first_seen_step=step,
                last_seen_step=step,
                status=BeliefStatus.ACTIVE,
                source_id=source_id,
                provenance=current_provenance
            )
            self.beliefs[new_belief.belief_id] = new_belief
            print(f"[New Belief] {new_belief.belief_id}: {triplet.subject} --({triplet.relation})--> {triplet.object}")

        # Step 5: Extend 'last_seen_step' for all surviving active beliefs ()
        for belief in self.get_active_beliefs():
            if belief.first_seen_step < step:
                belief.last_seen_step = step

    def export_to_jsonl(self, beliefs_path: str = "beliefs.jsonl",
                        transitions_path: str = "transitions.jsonl"):
        """Exports the current graph state to flat JSON Lines files."""
        with open(beliefs_path, "w", encoding="utf-8") as f:
            for belief in self.beliefs.values():
                f.write(belief.model_dump_json() + "\n")

        with open(transitions_path, "w", encoding="utf-8") as f:
            for transition in self.transitions.values():
                f.write(transition.model_dump_json() + "\n")

        print(f"\nState successfully exported to '{beliefs_path}' and '{transitions_path}'")