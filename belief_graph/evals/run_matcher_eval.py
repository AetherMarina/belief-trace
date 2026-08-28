from typing import List

from belief_graph.core import (
    Belief,
    BeliefStatus,
    ExtractedTriplet,
    InferenceProvenance,
)
from belief_graph.memory import BeliefMatch
from belief_graph.matching import SemanticBeliefMatcher
from belief_graph.nli_matcher import NliMatcherProvider
from matcher_cases import MatcherCase, DEV_CASES, HELD_OUT_CASES, CALIBRATION_CASES


def make_candidate(
    triplet: ExtractedTriplet,
    belief_id: str,
    score: float,
) -> BeliefMatch:

    belief = Belief(
        belief_id=belief_id,
        entity_id="test_entity",
        subject=triplet.subject,
        relation=triplet.relation,
        object=triplet.object,
        first_seen_step=1,
        last_seen_step=1,
        status=BeliefStatus.ACTIVE,
        source_id="matcher_eval",
        evidence_span=None,
        provenance=InferenceProvenance(
            model="test",
            prompt_version="matcher_eval",
            temperature=0.0,
        ),
    )

    return BeliefMatch(
        belief=belief,
        score=score,
    )


def run_eval(cases: List[MatcherCase], label: str):
    print("Controlled semantic evaluation of the belief matcher")
    provider = NliMatcherProvider()
    matcher = SemanticBeliefMatcher(provider)

    passed = 0

    for index, case in enumerate(cases, start=1):
        candidate = make_candidate(
            triplet=case.existing,
            belief_id=f"b_{index:03d}",
            score=case.similarity_score,
        )

        result = matcher.match(
            new_belief=case.new,
            candidates=[candidate],
        )

        success = result.decision == case.expected

        if success:
            passed += 1

        print("=" * 70)
        print(f"CASE:      {case.name}")
        print(
            f"EXISTING:  "
            f"{case.existing.subject} "
            f"{case.existing.relation} "
            f"{case.existing.object}"
        )
        print(
            f"NEW:       "
            f"{case.new.subject} "
            f"{case.new.relation} "
            f"{case.new.object}"
        )
        print(f"SIMILARITY: {case.similarity_score}")
        print(f"EXPECTED:   {case.expected.value}")
        print(f"PREDICTED:  {result.decision.value}")
        print(f"MATCHED ID: {result.matched_belief_id}")
        print(f"REASON:     {result.reason}")
        print(f"RESULT:     {'PASS' if success else 'FAIL'}")

        comparison = provider.compare(
            new_belief=case.new,
            candidate=candidate.belief,
        )

        ab = comparison.new_to_existing
        ba = comparison.existing_to_new

        same_score = comparison.same_score

        print(
            f"NLI new→existing: "
            f"{ab.label} | "
            f"E={ab.entailment:.3f} "
            f"C={ab.contradiction:.3f} "
            f"N={ab.neutral:.3f}"
        )

        print(
            f"NLI existing→new: "
            f"{ba.label} | "
            f"E={ba.entailment:.3f} "
            f"C={ba.contradiction:.3f} "
            f"N={ba.neutral:.3f}"
        )

        print(f"SAME SCORE: {same_score:.3f}")

    print()
    print(f"Passed {passed}/{len(cases)} {label} cases.")


if __name__ == "__main__":
    run_eval(DEV_CASES, "DEV SET")
    run_eval(HELD_OUT_CASES, "HELD-OUT SET")
    run_eval(CALIBRATION_CASES , "CALIBRATION_CASES SET")
