import uuid
from enum import Enum
from typing import Optional, Dict, Any, List, Protocol, Literal
from pydantic import BaseModel, Field, model_validator


# ========================================================================
# 1. ENUMS
# ========================================================================

class BeliefStatus(str, Enum):
    ACTIVE = "active"
    CHALLENGED = "challenged"
    DEPRECATED = "deprecated"

class TransitionType(str, Enum):
    SHATTERED = "shattered"
    REFRAMED = "reframed"
    # Future expansion: REINFORCED, etc.

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
    belief_id: str = Field(default_factory=lambda: str(uuid.uuid4()))
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

    def _belief_to_text(self) -> str:
        return f"Subject: {self.subject}; Relation: {self.relation}; Object: {self.object}"

class Transition(BaseModel):
    """Represents an edge explaining a cognitive state change between steps."""
    transition_id: str = Field(default_factory=lambda: str(uuid.uuid4()))
    affected_belief_id: str
    resulting_belief_id: Optional[str] = None  # None if the belief was shattered without a direct replacement
    triggering_observation_id: Optional[str] = None
    transition_type: TransitionType
    step: int
    reason: str
    provenance: InferenceProvenance  # Transitions are also LLM-inferred, so they need provenance too

    @model_validator(mode='after')
    def validate_transition_logic(self) -> 'Transition':
        """
        Ensures that architectural rules for cognitive transitions are strictly followed.
        """
        if self.transition_type == TransitionType.REFRAMED and not self.resulting_belief_id:
            raise ValueError(
                f"Transition error on {self.transition_id}: "
                f"A 'REFRAMED' transition requires a valid resulting_belief_id."
            )

        if self.transition_type == TransitionType.SHATTERED and self.resulting_belief_id:
            raise ValueError(
                f"Transition error on {self.transition_id}: "
                f"A 'SHATTERED' transition cannot have a resulting_belief_id. Use 'REFRAMED' instead."
            )

        return self

class BeliefObservation(BaseModel):
    """
    Records a specific instance where a belief was observed in the narrative.
    This provides an append-only history without mutating the core belief identity.
    """
    observation_id: str = Field(default_factory=lambda: str(uuid.uuid4()))
    belief_id: str
    step: int
    source_id: str
    evidence_span: Optional[str] = None
    provenance: InferenceProvenance


# --- temporary LLM DATA TRANSFER OBJECTS (DTOs) ---

class ExtractedTriplet(BaseModel):
    """Raw cognitive triplet extracted by the LLM before it becomes a Belief node."""
    subject: Literal["Self", "World", "Others"]
    relation: str
    object: str

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

    def resolve_potential_contradiction(self, old_belief, triplet):
        """
        Classify type of contradiction.
        """
        ...
