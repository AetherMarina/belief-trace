import uuid
from enum import Enum
from typing import Optional, Dict, Any, List, Protocol, Literal
from pydantic import BaseModel, Field, model_validator, computed_field


# ========================================================================
# 1. ENUMS
# ========================================================================

class BeliefStatus(str, Enum):
    ACTIVE = "active"
    CHALLENGED = "challenged"
    DEPRECATED = "deprecated"


class CoreBeliefDomain(str, Enum):
    SELF = "Self"
    WORLD = "World"
    OTHERS = "Others"


class CoreBeliefLabel(str, Enum):
    # --- SELF ---
    CAPABLE = "Capable"
    INCAPABLE = "Incapable"
    WORTHY = "Worthy"
    UNWORTHY = "Unworthy"
    AUTONOMOUS = "Autonomous"
    DEPENDENT = "Dependent"
    RESILIENT = "Resilient"
    VULNERABLE = "Vulnerable"

    # --- WORLD ---
    SAFE = "Safe"
    DANGEROUS = "Dangerous"
    PREDICTABLE = "Predictable"
    UNPREDICTABLE = "Unpredictable"
    CONTROLLABLE = "Controllable"
    UNCONTROLLABLE = "Uncontrollable"
    FAIR = "Fair"
    UNFAIR = "Unfair"

    # --- OTHERS ---
    TRUSTWORTHY = "Trustworthy"
    UNTRUSTWORTHY = "Untrustworthy"
    ACCEPTING = "Accepting"
    REJECTING = "Rejecting"
    SUPPORTIVE = "Supportive"
    UNSUPPORTIVE = "Unsupportive"
    RELIABLE = "Reliable"
    UNRELIABLE = "Unreliable"
    COLLABORATIVE = "Collaborative"
    EXPLOITATIVE = "Exploitative"
    THREATENING = "Threatening"
    HARMLESS = "Harmless"


ALLOWED_CORE_BELIEF_LABELS: dict[
    CoreBeliefDomain,
    set[CoreBeliefLabel],
] = {
    CoreBeliefDomain.SELF: {
        CoreBeliefLabel.CAPABLE,
        CoreBeliefLabel.INCAPABLE,
        CoreBeliefLabel.WORTHY,
        CoreBeliefLabel.UNWORTHY,
        CoreBeliefLabel.AUTONOMOUS,
        CoreBeliefLabel.DEPENDENT,
        CoreBeliefLabel.RESILIENT,
        CoreBeliefLabel.VULNERABLE,
    },

    CoreBeliefDomain.WORLD: {
        CoreBeliefLabel.SAFE,
        CoreBeliefLabel.DANGEROUS,
        CoreBeliefLabel.PREDICTABLE,
        CoreBeliefLabel.UNPREDICTABLE,
        CoreBeliefLabel.CONTROLLABLE,
        CoreBeliefLabel.UNCONTROLLABLE,
        CoreBeliefLabel.FAIR,
        CoreBeliefLabel.UNFAIR,
    },

    CoreBeliefDomain.OTHERS: {
        CoreBeliefLabel.TRUSTWORTHY,
        CoreBeliefLabel.UNTRUSTWORTHY,
        CoreBeliefLabel.ACCEPTING,
        CoreBeliefLabel.REJECTING,
        CoreBeliefLabel.SUPPORTIVE,
        CoreBeliefLabel.UNSUPPORTIVE,
        CoreBeliefLabel.COLLABORATIVE,
        CoreBeliefLabel.EXPLOITATIVE,
        CoreBeliefLabel.THREATENING,
        CoreBeliefLabel.HARMLESS,
        CoreBeliefLabel.RELIABLE,
        CoreBeliefLabel.UNRELIABLE,
    },
}


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


class CoreBelief(BaseModel):
    core_belief_id: str = Field(default_factory=lambda: str(uuid.uuid4()))
    entity_id: str
    domain: CoreBeliefDomain
    label: CoreBeliefLabel
    first_seen_step: int
    last_seen_step: int
    status: BeliefStatus = BeliefStatus.ACTIVE
    provenance: InferenceProvenance

    @model_validator(mode="after")
    def validate_domain_label(self):
        allowed = ALLOWED_CORE_BELIEF_LABELS[self.domain]

        if self.label not in allowed:
            raise ValueError(
                f"Label '{self.label}' is not valid "
                f"for domain '{self.domain.value}'"
            )

        return self

    @computed_field
    @property
    def active_duration(self) -> int:
        """Calculates the total step span this core schema has remained active."""
        return (self.last_seen_step - self.first_seen_step) + 1


class SurfaceToCoreMapping(BaseModel):
    mapping_id: str = Field(default_factory=lambda: str(uuid.uuid4()))
    surface_belief_id: str
    core_belief_id: str
    step: int
    confidence_score: Optional[float] = None
    provenance: InferenceProvenance


# --- temporary LLM DATA TRANSFER OBJECTS (DTOs) ---

class ExtractedTriplet(BaseModel):
    """Raw cognitive triplet extracted by the LLM before it becomes a Belief node."""
    subject: Literal["Self", "World", "Others"]
    relation: str
    object: str


class CoreBeliefMappingResult(BaseModel):
    """LLM's response when mapping surface belief to core belief."""
    is_core_belief: bool = Field(
        description="True if the surface belief reflects a deep psychological schema."
                    "False if it is a transient or situational thought."
    )
    domain: Optional[CoreBeliefDomain] = Field(
        default=None,
        description="The core domain. Must be null if is_core_belief is False."
    )
    label: Optional[CoreBeliefLabel] = Field(
        default=None,
        description="The taxonomy label. Must be null if is_core_belief is False."
    )
    confidence_score: float = Field(
        default=1.0, ge=0.0, le=1.0,
        description="Confidence in this mapping from 0.0 to 1.0"
    )

    @model_validator(mode="after")
    def validate_mapping(self):
        if self.is_core_belief:
            if self.domain is None or self.label is None:
                raise ValueError(
                    "domain and label are required when is_core_belief=True"
                )
            if self.label not in ALLOWED_CORE_BELIEF_LABELS[self.domain]:
                raise ValueError(
                    f"{self.label.value} is not valid for "
                    f"{self.domain.value}"
                )
        return self


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

    def map_to_core_belief(self, triplet: ExtractedTriplet) -> CoreBeliefMappingResult:
        """
        Evaluates a surface triplet and attempts to map it to a Core Belief taxonomy.
        """
        ...
