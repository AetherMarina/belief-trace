import pytest

from belief_graph.core import (
    Belief,
    BeliefObservation,
    BeliefStatus,
    ExtractedTriplet,
    InferenceProvenance,
    TransitionType,
)
from belief_graph.engine import LongitudinalEngine
from belief_graph.matching import MatchResult, MatchType


class FakeProvider:
    def __init__(
        self,
        triplets=None,
        transition_type=TransitionType.REFRAMED,
    ):
        self.triplets = triplets or []
        self.transition_type = transition_type
        self.calls = []
        self.classify_calls = []

    def extract_beliefs(self, text: str, entity_id: str):
        self.calls.append({
            "text": text,
            "entity_id": entity_id,
        })
        return self.triplets

    def resolve_potential_contradiction(self, old_belief, new_triplet):
        self.classify_calls.append((old_belief, new_triplet))
        return self.transition_type


class FakeEmbedder:
    def __init__(self, embedding=None):
        self.embedding = embedding or [0.1, 0.2, 0.3]

    def embed(self, text: str):
        return self.embedding


class FakeMemory:
    def __init__(self):
        self.add_calls = []
        self.update_calls = []

    def search_similar(
        self,
        embedding,
        entity_id,
        subject,
        status=BeliefStatus.ACTIVE,
        top_k=5,
    ):
        return []

    def add(self, belief, embedding):
        self.add_calls.append((belief, embedding))

    def update_metadata(self, belief):
        self.update_calls.append(belief)


class FakeMatcher:
    def __init__(self, result: MatchResult):
        self.result = result

    def match(self, new_belief, candidates):
        return self.result


@pytest.fixture
def provenance():
    return InferenceProvenance(
        model="test-model",
        prompt_version="test-v1",
        temperature=0.0,
    )


def make_engine(provenance, match_result):
    llm_provider = FakeProvider([match_result])
    memory = FakeMemory()
    embedder = FakeEmbedder()
    matcher = FakeMatcher(match_result)

    engine = LongitudinalEngine(
        llm_provider=llm_provider,
        memory=memory,
        embedder=embedder,
        matcher=matcher,
        provenance=provenance,
        transition_provenance=provenance
    )

    return engine, memory, embedder


def make_existing_belief(provenance):
    return Belief(
        entity_id="alice",
        subject="Self",
        relation="IS",
        object="Vulnerable",
        first_seen_step=1,
        last_seen_step=1,
        status=BeliefStatus.ACTIVE,
        source_id="alice_t1",
        evidence_span="Alice felt small and exposed.",
        provenance=provenance,
    )


def test_same_adds_observation_without_creating_new_belief(provenance):
    existing = make_existing_belief(provenance)

    result = MatchResult(
        decision=MatchType.SAME,
        matched_belief_id=existing.belief_id,
        reason="Bidirectional entailment.",
        similarity_score=0.95,
    )

    engine, memory, _ = make_engine(provenance, result)
    engine.beliefs[existing.belief_id] = existing

    triplet = ExtractedTriplet(
        subject="Self",
        relation="IS",
        object="Vulnerable",
    )

    engine.process_triplet(
        entity_id="alice",
        triplet=triplet,
        step=2,
        source_id="alice_t2",
        evidence_span="Alice again feels vulnerable.",
    )

    assert len(engine.beliefs) == 1
    assert engine.beliefs[existing.belief_id].last_seen_step == 2

    assert len(engine.observations) == 1
    observation = next(iter(engine.observations.values()))
    assert isinstance(observation, BeliefObservation)
    assert observation.belief_id == existing.belief_id
    assert observation.step == 2
    assert observation.source_id == "alice_t2"

    assert len(engine.transitions) == 0
    assert len(memory.update_calls) == 1
    assert memory.add_calls == []


def test_different_creates_new_belief_and_initial_observation(provenance):
    result = MatchResult(
        decision=MatchType.DIFFERENT,
        matched_belief_id=None,
        reason="No equivalent or contradictory belief.",
    )

    engine, memory, embedder = make_engine(provenance, result)

    triplet = ExtractedTriplet(
        subject="World",
        relation="IS",
        object="Unpredictable",
    )

    engine.process_triplet(
        entity_id="alice",
        triplet=triplet,
        step=3,
        source_id="alice_t3",
        evidence_span="Nothing seems to follow stable rules.",
    )

    assert len(engine.beliefs) == 1
    new_belief = next(iter(engine.beliefs.values()))

    assert new_belief.entity_id == "alice"
    assert new_belief.subject == "World"
    assert new_belief.object == "Unpredictable"
    assert new_belief.status == BeliefStatus.ACTIVE
    assert new_belief.first_seen_step == 3
    assert new_belief.last_seen_step == 3
    assert new_belief.source_id == "alice_t3"

    assert len(engine.observations) == 1
    observation = next(iter(engine.observations.values()))
    assert observation.belief_id == new_belief.belief_id
    assert observation.step == 3

    assert len(engine.transitions) == 0
    assert memory.update_calls == []
    assert len(memory.add_calls) == 1
    assert memory.add_calls[0][0].belief_id == new_belief.belief_id
    assert memory.add_calls[0][1] == embedder.embedding


def test_contradicts_deprecates_old_creates_new_and_reframed_transition(provenance):
    existing = make_existing_belief(provenance)

    result = MatchResult(
        decision=MatchType.CONTRADICTS,
        matched_belief_id=existing.belief_id,
        reason="The new belief contradicts the previous belief.",
        similarity_score=0.92,
    )

    engine, memory, embedder = make_engine(provenance, result)
    engine.beliefs[existing.belief_id] = existing

    triplet = ExtractedTriplet(
        subject="Self",
        relation="IS",
        object="Capable",
    )

    engine.process_triplet(
        entity_id="alice",
        triplet=triplet,
        step=4,
        source_id="alice_t4",
        evidence_span="Alice realizes she can handle the challenge.",
    )

    assert len(engine.beliefs) == 2

    old_belief = engine.beliefs[existing.belief_id]
    assert old_belief.status == BeliefStatus.DEPRECATED
    assert old_belief.last_seen_step == 1

    new_belief = next(
        belief
        for belief_id, belief in engine.beliefs.items()
        if belief_id != existing.belief_id
    )

    assert new_belief.status == BeliefStatus.ACTIVE
    assert new_belief.first_seen_step == 4
    assert new_belief.last_seen_step == 4
    assert new_belief.object == "Capable"

    assert len(engine.observations) == 1
    observation = next(iter(engine.observations.values()))
    assert observation.belief_id == new_belief.belief_id

    assert len(engine.transitions) == 1
    transition = engine.transitions[0]

    assert transition.transition_type == TransitionType.REFRAMED
    assert transition.affected_belief_id == existing.belief_id
    assert transition.resulting_belief_id == new_belief.belief_id
    assert transition.triggering_observation_id == observation.observation_id
    assert transition.step == 4
    assert transition.reason == "The new belief contradicts the previous belief."

    assert len(memory.update_calls) == 1
    assert memory.update_calls[0].status == BeliefStatus.DEPRECATED

    assert len(memory.add_calls) == 1
    assert memory.add_calls[0][0].belief_id == new_belief.belief_id
    assert memory.add_calls[0][1] == embedder.embedding


def test_get_belief_history_returns_only_requested_belief_in_step_order(provenance):
    result = MatchResult(
        decision=MatchType.DIFFERENT,
        matched_belief_id=None,
        reason="unused",
    )
    engine, _, _ = make_engine(provenance, result)

    belief_a = make_existing_belief(provenance)

    belief_b = Belief(
        entity_id="alice",
        subject="World",
        relation="IS",
        object="Dangerous",
        first_seen_step=1,
        last_seen_step=1,
        status=BeliefStatus.ACTIVE,
        source_id="alice_t1",
        provenance=provenance,
    )

    engine.beliefs[belief_a.belief_id] = belief_a
    engine.beliefs[belief_b.belief_id] = belief_b

    obs_step_3 = BeliefObservation(
        belief_id=belief_a.belief_id,
        step=3,
        source_id="alice_t3",
        provenance=provenance,
    )

    obs_other_belief = BeliefObservation(
        belief_id=belief_b.belief_id,
        step=2,
        source_id="alice_t2",
        provenance=provenance,
    )

    obs_step_1 = BeliefObservation(
        belief_id=belief_a.belief_id,
        step=1,
        source_id="alice_t1",
        provenance=provenance,
    )

    # Intentionally insert observations out of chronological order.
    for observation in (
        obs_step_3,
        obs_other_belief,
        obs_step_1,
    ):
        engine.observations[
            observation.observation_id
        ] = observation

    history = engine.get_belief_history(
        belief_a.belief_id
    )

    assert [obs.step for obs in history] == [1, 3]
    assert all(
        obs.belief_id == belief_a.belief_id
        for obs in history
    )


def test_get_transitions_for_belief_returns_transition_for_old_and_new_belief(
    provenance,
):
    existing = make_existing_belief(provenance)

    result = MatchResult(
        decision=MatchType.CONTRADICTS,
        matched_belief_id=existing.belief_id,
        reason="The new belief contradicts the previous belief.",
        similarity_score=0.92,
    )

    engine, _, _ = make_engine(
        provenance,
        result,
    )
    engine.beliefs[existing.belief_id] = existing

    engine.process_triplet(
        entity_id="alice",
        triplet=ExtractedTriplet(
            subject="Self",
            relation="IS",
            object="Capable",
        ),
        step=4,
        source_id="alice_t4",
        evidence_span=(
            "Alice realizes she can handle the challenge."
        ),
    )

    new_belief = next(
        belief
        for belief_id, belief in engine.beliefs.items()
        if belief_id != existing.belief_id
    )

    old_transitions = engine.get_transitions_for_belief(
        existing.belief_id
    )
    new_transitions = engine.get_transitions_for_belief(
        new_belief.belief_id
    )

    assert len(old_transitions) == 1
    assert len(new_transitions) == 1

    assert (
        old_transitions[0].transition_id
        == new_transitions[0].transition_id
    )

    assert (
        old_transitions[0].affected_belief_id
        == existing.belief_id
    )

    assert (
        old_transitions[0].resulting_belief_id
        == new_belief.belief_id
    )


def test_process_step_extracts_and_processes_multiple_triplets(provenance):
    triplets = [
        ExtractedTriplet(
            subject="Self",
            relation="IS",
            object="Vulnerable",
        ),
        ExtractedTriplet(
            subject="World",
            relation="IS",
            object="Dangerous",
        ),
    ]

    result = MatchResult(
        decision=MatchType.DIFFERENT,
        matched_belief_id=None,
        reason="No matching belief.",
    )

    engine, memory, _ = make_engine(provenance, result)

    llm_provider = FakeProvider(triplets)
    engine.llm_provider = llm_provider

    engine.process_step(
        entity_id="alice",
        step=1,
        text="Alice feels exposed and the world seems dangerous.",
        source_id="alice_t1",
    )

    # Extractor called exactly once with the whole narrative step.
    assert len(llm_provider.calls) == 1
    assert llm_provider.calls[0]["entity_id"] == "alice"
    assert llm_provider.calls[0]["text"] == (
        "Alice feels exposed and the world seems dangerous."
    )

    # Both extracted triplets were processed.
    assert len(engine.beliefs) == 2
    assert len(engine.observations) == 2
    assert len(memory.add_calls) == 2

    extracted_objects = {
        belief.object
        for belief in engine.beliefs.values()
    }

    assert extracted_objects == {
        "Vulnerable",
        "Dangerous",
    }


def test_process_step_forwards_each_triplet_to_process_triplet(
    provenance,
    monkeypatch,
):
    triplets = [
        ExtractedTriplet(
            subject="Self",
            relation="IS",
            object="Vulnerable",
        ),
        ExtractedTriplet(
            subject="Others",
            relation="ARE",
            object="Untrustworthy",
        ),
    ]

    result = MatchResult(
        decision=MatchType.DIFFERENT,
        matched_belief_id=None,
        reason="unused",
    )

    engine, _, _ = make_engine(provenance, result)
    engine.llm_provider = FakeProvider(triplets)

    processed = []

    def fake_process_triplet(
        entity_id,
        triplet,
        step,
        source_id,
        evidence_span=None,
    ):
        processed.append({
            "entity_id": entity_id,
            "triplet": triplet,
            "step": step,
            "source_id": source_id,
            "evidence_span": evidence_span,
        })

    monkeypatch.setattr(
        engine,
        "process_triplet",
        fake_process_triplet,
    )

    engine.process_step(
        entity_id="alice",
        step=3,
        text="Some narrative text.",
        source_id="alice_t3",
    )

    assert len(processed) == 2

    assert processed[0]["triplet"] == triplets[0]
    assert processed[1]["triplet"] == triplets[1]

    assert all(
        item["entity_id"] == "alice"
        for item in processed
    )

    assert all(
        item["step"] == 3
        for item in processed
    )

    assert all(
        item["source_id"] == "alice_t3"
        for item in processed
    )


def test_contradiction_classified_as_shattered_deprecates_without_replacement(
    provenance,
):
    old_belief = Belief(
        entity_id="alice",
        subject="Self",
        relation="IS",
        object="Invincible",
        first_seen_step=1,
        last_seen_step=1,
        status=BeliefStatus.ACTIVE,
        source_id="alice_t1",
        provenance=provenance,
    )

    match_result = MatchResult(
        decision=MatchType.CONTRADICTS,
        matched_belief_id=old_belief.belief_id,
        reason="NLI detected semantic contradiction.",
    )

    memory = FakeMemory()
    embedder = FakeEmbedder()
    matcher = FakeMatcher(match_result)
    provider = FakeProvider(transition_type=TransitionType.SHATTERED)

    engine = LongitudinalEngine(
        llm_provider=provider,
        memory=memory,
        embedder=embedder,
        matcher=matcher,
        provenance=provenance,
        transition_provenance=provenance
    )

    engine.beliefs[old_belief.belief_id] = old_belief

    triplet = ExtractedTriplet(
        subject="Self",
        relation="IS NOT",
        object="Invincible",
    )

    engine.process_triplet(
        entity_id="alice",
        triplet=triplet,
        step=2,
        source_id="alice_t2",
    )

    assert old_belief.status == BeliefStatus.DEPRECATED

    # No replacement belief is created.
    assert len(engine.beliefs) == 1

    assert len(engine.transitions) == 1
    transition = engine.transitions[0]

    assert transition.transition_type == TransitionType.SHATTERED
    assert transition.affected_belief_id == old_belief.belief_id
    assert transition.resulting_belief_id is None
    assert transition.triggering_observation_id is None

    # Negated triplet is not stored as a new belief.
    assert all(
        belief.object != "Invincible"
        or belief.relation != "IS NOT"
        for belief in engine.beliefs.values()
    )

    assert len(provider.classify_calls) == 1


def test_contradiction_classified_as_reframed_creates_replacement_belief(
    provenance,
):
    old_belief = Belief(
        entity_id="alice",
        subject="Self",
        relation="IS",
        object="Invincible",
        first_seen_step=1,
        last_seen_step=1,
        status=BeliefStatus.ACTIVE,
        source_id="alice_t1",
        provenance=provenance,
    )

    match_result = MatchResult(
        decision=MatchType.CONTRADICTS,
        matched_belief_id=old_belief.belief_id,
        reason="NLI detected semantic contradiction.",
    )

    memory = FakeMemory()
    embedder = FakeEmbedder()
    matcher = FakeMatcher(match_result)
    provider = FakeProvider(transition_type=TransitionType.REFRAMED)

    engine = LongitudinalEngine(
        llm_provider=provider,
        memory=memory,
        embedder=embedder,
        matcher=matcher,
        provenance=provenance,
        transition_provenance=provenance
    )

    engine.beliefs[old_belief.belief_id] = old_belief

    triplet = ExtractedTriplet(
        subject="Self",
        relation="IS",
        object="Vulnerable",
    )

    engine.process_triplet(
        entity_id="alice",
        triplet=triplet,
        step=2,
        source_id="alice_t2",
    )

    assert old_belief.status == BeliefStatus.DEPRECATED

    assert len(engine.beliefs) == 2

    new_belief = next(
        belief
        for belief in engine.beliefs.values()
        if belief.belief_id != old_belief.belief_id
    )

    assert new_belief.subject == "Self"
    assert new_belief.relation == "IS"
    assert new_belief.object == "Vulnerable"
    assert new_belief.status == BeliefStatus.ACTIVE

    assert len(engine.transitions) == 1
    transition = engine.transitions[0]

    assert transition.transition_type == TransitionType.REFRAMED
    assert transition.affected_belief_id == old_belief.belief_id
    assert transition.resulting_belief_id == new_belief.belief_id

    assert len(provider.classify_calls) == 1
