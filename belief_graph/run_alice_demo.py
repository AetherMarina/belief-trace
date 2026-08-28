import os
import logging

from belief_graph.core import InferenceProvenance
from belief_graph.embedder import OllamaEmbedder
from belief_graph.engine import LongitudinalEngine
from belief_graph.matching import SemanticBeliefMatcher
from belief_graph.nli_matcher import NliMatcherProvider
from belief_graph.providers import OllamaProvider
from belief_graph.config import DEFAULT_MODEL
from belief_graph.qdrant_memory import QdrantBeliefMemory


def configure_logging():
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
    )
    # Show detailed BeliefTrace diagnostics without making third-party libraries equally verbose.
    logging.getLogger("belief_graph").setLevel(logging.DEBUG)


def get_alice_dataset():
    """
    Returns a carefully curated list of 8 sequential passages from Alice's internal
    monologues and dialogues, designed to trigger cognitive dissonance and belief transitions.
    """
    return [
        {
            "step": 1,
            "source_id": "chapter_01_the_fall",
            "text": "'Well!' thought Alice to herself, 'after such a fall as this, I shall think nothing of tumbling down stairs! How brave they'll all think me at home! Why, I wouldn't say anything about it, even if I fell off the top of the house!'"
        },
        {
            "step": 2,
            "source_id": "chapter_02_the_puzzle",
            "text": "'I wonder if I've been changed in the night? Let me think: was I the same when I got up this morning? I almost think I can remember feeling a little different. But if I'm not the same, the next question is, Who in the world am I? Ah, that's the great puzzle!'"
        },
        {
            "step": 3,
            "source_id": "chapter_02_not_mabel",
            "text": "'I'm sure I'm not Ada,' she said, 'for her hair goes in such long ringlets, and mine doesn't go in ringlets at all; and I'm sure I can't be Mabel, for I know all sorts of things, and she, oh! she knows such a very little! Besides, she's she, and I'm I, and—oh dear, how puzzling it all is!'"
        },
        {
            "step": 4,
            "source_id": "chapter_02_mabel_fear",
            "text": "'I must be Mabel after all, and I shall have to go and live in that poky little house, and have next to no toys to play with, and oh! ever so many lessons to learn! No, I've made up my mind about it; if I'm Mabel, I'll stay down here!'"
        },
        {
            "step": 5,
            "source_id": "chapter_05_the_caterpillar",
            "text": "'I can't explain myself, I'm afraid, sir,' said Alice, 'because I'm not myself, you see.' 'I don't see,' said the Caterpillar. 'I'm afraid I can't put it more clearly,' Alice replied very politely, 'for I can't understand it myself to begin with; and being so many different sizes in a day is very confusing.'"        },
        {
            "step": 6,
            "source_id": "chapter_06_cheshire_cat",
            "text": "'But I don't want to go among mad people,' Alice remarked. 'Oh, you can't help that,' said the Cat: 'we're all mad here. I'm mad. You're mad.' 'How do you know I'm mad?' said Alice. 'You must be,' said the Cat, 'or you wouldn't have come here.'"
        },
        {
            "step": 7,
            "source_id": "chapter_07_tea_party",
            "text": "'At any rate I'll never go there again!' said Alice as she picked her way through the wood. 'It's the stupidest tea-party I ever was at in all my life!' Just as she said this, she noticed that one of the trees had a door leading right into it. 'That's very curious!' she thought. 'But everything's curious today.'"
        },
        {
            "step": 8,
            "source_id": "chapter_08_queen_identity",
            "text": "'My name is Alice, so please your Majesty,' said Alice very politely; but she added, to herself, 'Why, they're only a pack of cards, after all. I needn't be afraid of them! How should I know?' said Alice, surprised at her own courage. It's no business of mine.'"
        },
        {
            "step": 9,
            "source_id": "chapter_08_croquet_ground",
            "text": "Alice began to feel very uneasy: to be sure, she had not as yet had any dispute with the Queen, but she knew that it might happen any minute, 'and then,' thought she, 'what would become of me? They're dreadfully fond of beheading people here; the great wonder is, that there's any one left alive!'"
        },
        {
            "step": 10,
            "source_id": "chapter_12_the_evidence",
            "text": "'Who cares for you?' said Alice, (she had grown to her full size by this time.) 'You're nothing but a pack of cards!'"
        }
    ]


def main():
    print("==================================================")
    print("   LONGITUDINAL BELIEF GRAPH - ALICE DEMO v0.2    ")
    print("==================================================\n")

    # 1. Initialize the LLM Provider
    print(f"[System] Initializing Ollama Provider ({DEFAULT_MODEL})...")
    try:
        provider = OllamaProvider(model=DEFAULT_MODEL)
    except Exception as e:
        print(f"[Error] Failed to initialize provider: {e}")
        return

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

    # 2. Initialize the Core Engine
    print("[System] Initializing Longitudinal Engine...")
    engine = LongitudinalEngine(
        llm_provider=provider,
        memory=memory,
        embedder=embedder,
        matcher=matcher,
        provenance=extraction_provenance,
        transition_provenance=transition_provenance
    )

    # 3. Load the demo dataset
    dataset = get_alice_dataset()

    # 4. Execute the longitudinal pipeline
    for data in dataset:
        engine.process_step(
            entity_id="Alice",
            step=data["step"],
            text=data["text"],
            source_id=data["source_id"]
        )

    # 5. Export results
    print("\n==================================================")
    print("   PROCESSING COMPLETE. EXPORTING ARTIFACTS...    ")
    print("==================================================")

    # Create an output directory for clean project structure
    output_dir = "outputs"
    os.makedirs(output_dir, exist_ok=True)

    beliefs_path = os.path.join(output_dir, "beliefs.jsonl")
    transitions_path = os.path.join(output_dir, "transitions.jsonl")
    observations_path = os.path.join(output_dir, "observations.jsonl")

    engine.export_to_jsonl(beliefs_path=beliefs_path, transitions_path=transitions_path,
                           observations_path=observations_path)

    print(f"\n[Success] Demo finished. Check the '{output_dir}' folder for your generated JSONL graphs.")


if __name__ == "__main__":
    configure_logging()
    main()
