import pytest

from belief_graph.core import (
    Belief,
    BeliefStatus,
    ExtractedTriplet,
    InferenceProvenance, TransitionType,
)
from belief_graph.embedder import OllamaEmbedder
from belief_graph.matching import MatchType, SemanticBeliefMatcher
from belief_graph.nli_matcher import NliMatcherProvider
from belief_graph.providers import OllamaProvider
from belief_graph.qdrant_memory import QdrantBeliefMemory


@pytest.fixture
def provenance():
    return InferenceProvenance(
        model="qwen3:8b",
        prompt_version="surface-v0.2",
        temperature=0.0,
    )


@pytest.fixture
def provider():
    return OllamaProvider(model="qwen3:8b")


@pytest.mark.integration
def test_distinct_from_mabel_vs_mabel_is_contradiction():
    provenance = InferenceProvenance(
        model="gemma3:4b",
        prompt_version="surface-v0.2",
        temperature=0.0,
    )

    old_belief = Belief(
        entity_id="Alice",
        subject="Self",
        relation="IS",
        object="Distinct from Mabel",
        first_seen_step=3,
        last_seen_step=3,
        status=BeliefStatus.ACTIVE,
        source_id="debug_t3",
        provenance=provenance,
    )

    new_triplet = ExtractedTriplet(
        subject="Self",
        relation="IS",
        object="Mabel",
    )

    embedder = OllamaEmbedder(model="mxbai-embed-large")
    memory = QdrantBeliefMemory()
    matcher = SemanticBeliefMatcher(
        provider=NliMatcherProvider()
    )

    old_text = (
        f"Subject: {old_belief.subject}; "
        f"Relation: {old_belief.relation}; "
        f"Object: {old_belief.object}"
    )

    new_text = (
        f"Subject: {new_triplet.subject}; "
        f"Relation: {new_triplet.relation}; "
        f"Object: {new_triplet.object}"
    )

    memory.add(
        old_belief,
        embedder.embed(old_text),
    )

    candidates = memory.search_similar(
        embedding=embedder.embed(new_text),
        entity_id="Alice",
        subject="Self",
        status=BeliefStatus.ACTIVE,
        top_k=5,
    )

    assert len(candidates) == 1

    print(f"Qdrant similarity: {candidates[0].score:.3f}")

    result = matcher.match(
        new_belief=new_triplet,
        candidates=candidates,
    )

    print(f"NLI decision: {result.decision}")
    print(f"NLI reason: {result.reason}")

    assert result.decision == MatchType.CONTRADICTS
    assert result.matched_belief_id == old_belief.belief_id


@pytest.mark.integration
def test_mabel_to_not_mabel_is_shattered(
    provider,
    provenance,
):
    old_belief = Belief(
        entity_id="Alice",
        subject="Self",
        relation="IS",
        object="Mabel",
        first_seen_step=3,
        last_seen_step=3,
        status=BeliefStatus.ACTIVE,
        source_id="test_t1",
        provenance=provenance,
    )

    new_triplet = ExtractedTriplet(
        subject="Self",
        relation="IS",
        object="Not Mabel",
    )

    result = provider.resolve_potential_contradiction(
        old_belief,
        new_triplet,
    )

    print(f"Verifier result: {result}")

    assert result == TransitionType.SHATTERED
