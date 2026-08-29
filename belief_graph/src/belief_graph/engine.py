import logging
from typing import List, Dict, Optional
from .core import (
    Belief, BeliefObservation, Transition, BeliefStatus,
    TransitionType, InferenceProvenance, ExtractedTriplet, LLMProvider, CoreBelief, SurfaceToCoreMapping
)
from .memory import BeliefMemory
from .matching import SemanticBeliefMatcher, MatchType
from .embedder import Embedder


logger = logging.getLogger(__name__)


class LongitudinalEngine:
    """
    The core orchestrator of the cognitive architecture.
    Processes narratives chronologically and maintains the belief graph state.
    """

    def __init__(
            self,
            llm_provider: LLMProvider,
            memory: BeliefMemory,
            embedder: Embedder,
            matcher: SemanticBeliefMatcher,
            provenance: InferenceProvenance,
            transition_provenance: InferenceProvenance
    ):
        self.llm_provider = llm_provider
        self.memory = memory
        self.embedder = embedder
        self.matcher = matcher
        self.provenance = provenance
        self.transition_provenance = transition_provenance

        # In-memory state of the graph (nodes and edges)
        self.beliefs: Dict[str, Belief] = {}
        self.transitions: List[Transition] = []
        self.observations: Dict[str, BeliefObservation] = {}
        self.core_beliefs: Dict[str, CoreBelief] = {}
        self.surface_to_core_mappings: Dict[str, SurfaceToCoreMapping] = {}

    # --- Analytical Helpers ---

    def get_active_beliefs(self) -> List[Belief]:
        """Returns all beliefs currently held by the entity."""
        return [b for b in self.beliefs.values() if b.status == BeliefStatus.ACTIVE]

    def get_deprecated_beliefs(self) -> List[Belief]:
        """Returns the 'graveyard' of shattered and replaced beliefs."""
        return [b for b in self.beliefs.values() if b.status == BeliefStatus.DEPRECATED]

    def get_belief_history(self, belief_id: str) -> List[BeliefObservation]:
        return sorted(
            (
                obs for obs in self.observations.values()
                if obs.belief_id == belief_id
            ),
            key=lambda obs: obs.step,
        )

    def get_transitions_for_belief(self, belief_id: str) -> List[Transition]:
        """Finds all transitions where this belief was shattered or created."""
        return [
            t for t in self.transitions
            if t.affected_belief_id == belief_id or t.resulting_belief_id == belief_id
        ]

        # --- Main Processing Loop ---

    def process_step(
            self,
            entity_id: str,
            step: int,
            text: str,
            source_id: str,
    ):
        logger.info("Processing step=%s source_id=%s entity_id=%s", step, source_id, entity_id)
        triplets = self.llm_provider.extract_beliefs(
            text=text,
            entity_id=entity_id,
        )
        logger.info("Processing step=%s extracted_triplets=%d", step, len(triplets))

        for triplet in triplets:
            self.process_triplet(
                entity_id=entity_id,
                triplet=triplet,
                step=step,
                source_id=source_id,
            )

    def process_triplet(
            self,
            entity_id: str,
            triplet: ExtractedTriplet,
            step: int,
            source_id: str,
            evidence_span: Optional[str] = None
    ) -> None:
        """
        The main processing loop for a newly extracted belief.
        """
        # 1. Generate Semantic Embedding
        triplet_text = f"Subject: {triplet.subject}; Relation: {triplet.relation}; Object: {triplet.object}"
        embedding = self.embedder.embed(triplet_text)

        # 2. Retrieve candidates from Memory (Filtered by Subject and ACTIVE status)
        candidates = self.memory.search_similar(
            embedding=embedding,
            entity_id=entity_id,
            subject=triplet.subject,
            status=BeliefStatus.ACTIVE,
            top_k=5
        )

        # 3. Semantic NLI Matching
        match_result = self.matcher.match(new_belief=triplet, candidates=candidates)
        logger.debug(
            "Triplet [%s] %s [%s] -> match=%s matched_id=%s similarity=%s",
            triplet.subject,
            triplet.relation,
            triplet.object,
            match_result.decision.value,
            match_result.matched_belief_id,
            match_result.similarity_score,
        )

        # 4. State Machine Routing
        if match_result.decision == MatchType.SAME:
            self._handle_same(match_result.matched_belief_id, step, source_id, evidence_span)

        elif match_result.decision == MatchType.DIFFERENT:
            self._handle_different(entity_id, triplet, embedding, step, source_id, evidence_span)

        elif match_result.decision == MatchType.CONTRADICTS:
            old_belief = self.beliefs[match_result.matched_belief_id]
            trans_type = self.llm_provider.resolve_potential_contradiction(old_belief, triplet)
            logger.info(
                "Potential contradiction: old=[%s] %s [%s] new=[%s] %s [%s] "
                "resolution=%s similarity=%.3f",
                old_belief.subject,
                old_belief.relation,
                old_belief.object,
                triplet.subject,
                triplet.relation,
                triplet.object,
                trans_type.value if trans_type else "NOT_CONTRADICTION",
                match_result.similarity_score,
            )

            if trans_type is None:
                self._handle_different(entity_id, triplet, embedding, step, source_id, evidence_span)
            elif trans_type == TransitionType.SHATTERED:
                reason = (
                    f"{match_result.reason} "
                    f"Classified as direct negation/abandonment. "
                    f"Triggering triplet: "
                    f"[{triplet.subject}] {triplet.relation} [{triplet.object}]"
                )
                self._handle_shattered(match_result.matched_belief_id, reason, step)
            else:
                self._handle_contradicts(
                    entity_id, triplet, embedding, match_result.matched_belief_id,
                    match_result.reason, step, source_id, evidence_span
                )

    def _handle_same(self, matched_id: str, step: int, source_id: str, evidence: str):
        """Reinforces an existing belief with a new observation."""
        belief = self.beliefs[matched_id]
        belief.last_seen_step = step

        observation = BeliefObservation(
            belief_id=belief.belief_id,
            step=step,
            source_id=source_id,
            evidence_span=evidence,
            provenance=self.provenance
        )
        self.observations[observation.observation_id] = observation

        # Only update the last_seen_step metadata in vector db
        self.memory.update_metadata(belief)

    def _handle_different(self, entity_id: str, triplet: ExtractedTriplet, embedding: List[float], step: int,
                          source_id: str,  evidence: str):
        """Creates a completely new cognitive entity in the graph."""
        new_belief = Belief(
            entity_id=entity_id,
            subject=triplet.subject,
            relation=triplet.relation,
            object=triplet.object,
            first_seen_step=step,
            last_seen_step=step,
            source_id=source_id,
            evidence_span=evidence,
            provenance=self.provenance
        )

        observation = BeliefObservation(
            belief_id=new_belief.belief_id,
            step=step,
            source_id=source_id,
            evidence_span=evidence,
            provenance=self.provenance
        )
        self.observations[observation.observation_id] = observation

        self.beliefs[new_belief.belief_id] = new_belief
        self.memory.add(new_belief, embedding)
        # Run Core Belief mapping
        self._process_core_mapping(entity_id, new_belief, triplet, step)

    def _handle_contradicts(self, entity_id: str, triplet: ExtractedTriplet, embedding: List[float], matched_id: str,
                            reason: str, step: int, source_id: str, evidence: str):
        """Executes a REFRAMED cognitive transition."""
        old_belief = self.beliefs[matched_id]

        # 1. Deprecate the old state
        old_belief.status = BeliefStatus.DEPRECATED
        self.memory.update_metadata(old_belief)

        # 2. Instantiate the new reframed state
        new_belief = Belief(
            entity_id=entity_id,
            subject=triplet.subject,
            relation=triplet.relation,
            object=triplet.object,
            first_seen_step=step,
            last_seen_step=step,
            source_id=source_id,
            evidence_span=evidence,
            provenance=self.provenance
        )

        observation = BeliefObservation(
            belief_id=new_belief.belief_id,
            step=step,
            source_id=source_id,
            evidence_span=evidence,
            provenance=self.provenance
        )
        self.observations[observation.observation_id] = observation

        self.beliefs[new_belief.belief_id] = new_belief
        self.memory.add(new_belief, embedding)

        # 3. Create the architectural transition log (Strict Pydantic rule applied)
        transition = Transition(
            transition_type=TransitionType.REFRAMED,
            step=step,
            affected_belief_id=old_belief.belief_id,
            resulting_belief_id=new_belief.belief_id,
            triggering_observation_id=observation.observation_id,
            reason=reason,
            provenance=self.transition_provenance
        )
        self.transitions.append(transition)

        # 4. Run Core Belief mapping
        self._process_core_mapping(entity_id, new_belief, triplet, step)

        logger.info("REFRAMED old=%s new=%s step=%s", old_belief.belief_id, new_belief.belief_id, step)

    def _handle_shattered(self, matched_id: str, reason: str, step: int):
        """Executes a SHATTERED cognitive transition (destruction without replacement)."""
        old_belief = self.beliefs[matched_id]

        old_belief.status = BeliefStatus.DEPRECATED
        self.memory.update_metadata(old_belief)

        transition = Transition(
            transition_type=TransitionType.SHATTERED,
            step=step,
            affected_belief_id=old_belief.belief_id,
            resulting_belief_id=None,
            triggering_observation_id=None,
            reason=reason,
            provenance=self.transition_provenance
        )
        self.transitions.append(transition)
        logger.info("SHATTERED belief=%s step=%s", old_belief.belief_id, step)

    def _process_core_mapping(self, entity_id: str, new_surface_belief: Belief, triplet: ExtractedTriplet, step: int):
        """Attempts to map a surface belief to a core schema, maintaining the hierarchical graph structure."""

        # 1. Evaluate the surface triplet via the LLM arbiter
        mapping_result = self.llm_provider.map_to_core_belief(triplet)

        if not mapping_result.is_core_belief or not mapping_result.domain or not mapping_result.label:
            logger.debug(f"Triplet evaluated as situational/transient. No core mapping generated for {entity_id}.")
            return

        # 2. Check if this core schema already exists for the given entity
        core_belief = None
        for cb in self.core_beliefs.values():
            if cb.entity_id == entity_id and cb.domain == mapping_result.domain and cb.label == mapping_result.label:
                core_belief = cb
                break

        # 3. Instantiate a new CoreBelief node if no match is found
        if not core_belief:
            core_belief = CoreBelief(
                entity_id=entity_id,
                domain=mapping_result.domain,
                label=mapping_result.label,
                first_seen_step=step,
                last_seen_step=step,
                provenance=self.provenance
            )
            self.core_beliefs[core_belief.core_belief_id] = core_belief
            logger.info(
                f"Generated new CoreBelief for {entity_id}: {core_belief.domain.value} - {core_belief.label}")
        else:
            # Extend the temporal boundary of the existing core belief
            core_belief.last_seen_step = step

        # 4. Generate the hierarchical edge connecting the surface belief to the core schema
        mapping_edge = SurfaceToCoreMapping(
            surface_belief_id=new_surface_belief.belief_id,
            core_belief_id=core_belief.core_belief_id,
            step=step,
            confidence_score=mapping_result.confidence_score,
            provenance=self.provenance
        )
        self.surface_to_core_mappings[mapping_edge.mapping_id] = mapping_edge

    # --- Persistence ---

    def export_to_jsonl(self, beliefs_path: str = "beliefs.jsonl", transitions_path: str = "transitions.jsonl",
                        observations_path: str = "observations.jsonl", core_beliefs_path: str = "core_beliefs.jsonl",
                        surface_to_core_mappings_path: str = "surface_to_core_mappings.jsonl"):
        """Exports the fully normalized graph state (Surface, Core, and Edges) to JSON Lines files."""
        with open(beliefs_path, "w", encoding="utf-8") as f:
            for belief in self.beliefs.values():
                f.write(belief.model_dump_json() + "\n")

        with open(transitions_path, "w", encoding="utf-8") as f:
            for transition in self.transitions:
                f.write(transition.model_dump_json() + "\n")

        with open(observations_path, "w", encoding="utf-8") as f:
            for observation in self.observations.values():
                f.write(observation.model_dump_json() + "\n")

        with open(core_beliefs_path, "w", encoding="utf-8") as f:
            for core_belief in self.core_beliefs.values():
                f.write(core_belief.model_dump_json() + "\n")

        with open(surface_to_core_mappings_path, "w", encoding="utf-8") as f:
            for mapping in self.surface_to_core_mappings.values():
                f.write(mapping.model_dump_json() + "\n")

        logger.info(f"State successfully exported to '{beliefs_path}', '{transitions_path}', '{observations_path}',"
                    f" '{core_beliefs_path}' and '{surface_to_core_mappings_path}'")
