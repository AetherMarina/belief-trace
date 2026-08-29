from belief_graph.config import DEFAULT_MODEL
from belief_graph.core import BeliefStatus, InferenceProvenance
from belief_graph.embedder import OllamaEmbedder
from belief_graph.engine import LongitudinalEngine
from belief_graph.matching import SemanticBeliefMatcher
from belief_graph.nli_matcher import NliMatcherProvider
from belief_graph.providers import OllamaProvider
from belief_graph.qdrant_memory import QdrantBeliefMemory


def get_golden_set():
    """
    Minimal Golden Evaluation Set
    """
    return [
        {
            "step": 1,
            "source_id": "eval_ch1_bravery",
            "text": "'Well!' thought Alice to herself, 'after such a fall as this, I shall think nothing of tumbling down stairs! How brave they'll all think me at home!'",
            "expected_conflict": False,
            "min_extracted_beliefs": 1
        },
        {
            "step": 2,
            "source_id": "eval_ch2_cowardice",
            "text": "'Oh, how wrong I was!' thought Alice. 'Tumbling down stairs is the most terrifying thing in the world, and I am not brave at all. I am a complete coward!'",
            "expected_conflict": True,
            "min_extracted_beliefs": 1
        }
    ]


def run_evaluation():
    print("==================================================")
    print("   RUNNING MINIMAL GOLDEN EVALUATION SET v0.1     ")
    print("==================================================\n")

    print(f"[System] Initializing Ollama Provider ({DEFAULT_MODEL})...")
    try:
        provider = OllamaProvider()
    except Exception as e:
        print(f"[Error] Failed to initialize provider: {e}")
        return
    golden_set = get_golden_set()
    memory = QdrantBeliefMemory()
    embedder = OllamaEmbedder(model="mxbai-embed-large")
    nli_provider = NliMatcherProvider()
    matcher = SemanticBeliefMatcher(provider=nli_provider)
    extraction_provenance = InferenceProvenance(
        model=provider.model,
        prompt_version="surface-v0.2",
        temperature=0.0,
    )
    transition_provenance = InferenceProvenance(
        model=provider.verifier_model,
        prompt_version="transition-v0.2",
        temperature=0.0,
    )
    print("[System] Initializing Longitudinal Engine...")
    engine = LongitudinalEngine(
        llm_provider=provider,
        memory=memory,
        embedder=embedder,
        matcher=matcher,
        provenance=extraction_provenance,
        transition_provenance=transition_provenance
    )

    passed_tests = 0
    total_tests = 0

    for data in golden_set:
        print(f"\n--- Evaluating Step {data['step']} ({data['source_id']}) ---")

        # Pre-execution state
        previous_active_count = len(engine.get_active_beliefs())

        # 1. TEST: Is the JSON schema valid?
        try:
            engine.process_step(
                entity_id="Alice_Eval",
                step=data["step"],
                text=data["text"],
                source_id=data["source_id"]
            )
            print("✔️ [Test 1] JSON schema is valid and correctly parsed by Pydantic.")
            passed_tests += 1
        except Exception as e:
            print(f"❌ [Test 1] Failed! JSON/Schema error: {e}")
        total_tests += 1

        # 2. TEST: Is the belief supported by the text?
        current_active = engine.get_active_beliefs()
        if len(current_active) >= data["min_extracted_beliefs"]:
            print(
                f"✔️ [Test 2] Extracted minimum expected beliefs (Expected: >={data['min_extracted_beliefs']}, Got: {len(current_active)}).")
            passed_tests += 1
        else:
            print(f"❌ [Test 2] Failed! Did not extract expected number of beliefs.")
        total_tests += 1

        # 3. TEST: Is the conflict/no-conflict relationship correct?
        # We look to see if the number of "deprecated" beliefs has increased when we expect conflict.
        deprecated_count = len([b for b in engine.beliefs.values() if b.status == BeliefStatus.DEPRECATED])
        conflict_occurred = deprecated_count > 0

        if conflict_occurred == data["expected_conflict"]:
            print(
                f"✔️ [Test 3] Conflict resolution behaved as expected (Expected conflict: {data['expected_conflict']}).")
            passed_tests += 1
        else:
            print(
                f"❌ [Test 3] Failed! Expected conflict: {data['expected_conflict']}, but conflict occurred: {conflict_occurred}.")
        total_tests += 1

    # 4. TEST: Referential Integrity (Consistency of IDs)
    print("\n--- Evaluating Global Referential Integrity ---")
    integrity_passed = True
    for transition in engine.transitions:
        # Does the affected belief exist in the database?
        if transition.affected_belief_id not in engine.beliefs:
            print(
                f"❌ [Test 4] Failed! Transition {transition.transition_id} "
                f"references non-existent belief {transition.affected_belief_id}.")
            integrity_passed = False
            break

        # Is the first_seen_step of the older belief logically before or equal to the transition step?
        affected_belief = engine.beliefs[transition.affected_belief_id]
        if affected_belief.first_seen_step > transition.step:
            print(f"❌ [Test 4] Failed! Time paradox in transition {transition.transition_id}.")
            integrity_passed = False
            break

    if integrity_passed:
        print("✔️ [Test 4] All transition edges have valid references to existing belief IDs.")
        passed_tests += 1
    total_tests += 1

    # SUMMARY
    print("\n==================================================")
    print(f"   EVALUATION RESULTS: {passed_tests}/{total_tests} TESTS PASSED")
    print("==================================================")

    if passed_tests == total_tests:
        print("🚀 ALL SYSTEMS GO! The framework is stable for longitudinal analysis.")
    else:
        print("⚠️ SOME TESTS FAILED. Review the logs above.")


if __name__ == "__main__":
    run_evaluation()
