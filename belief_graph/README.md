# 🕸️ Longitudinal Belief Graph (v0.2)

**An experimental framework for tracking, modeling and visualizing inferred belief transitions across longitudinal narratives.**

> **Disclaimer:** This framework uses Large Language Models (LLMs) to infer cognitive structures from text. The extracted beliefs and their subsequent transitions are model-generated hypotheses and narrative tracking tools, not clinical psychological assessments.

## Overview

Traditional knowledge graphs struggle with temporal dynamics and psychological depth. 
The **Longitudinal Belief Graph** tracks how inferred surface beliefs about `[Self]`, `[World]` and `[Others]` map onto normalized core-belief categories, and how those beliefs recur and change across sequential narrative steps ($T_n$).

Instead of overwriting old data, this framework uses a **Cognitive Arbiter** to detect narrative conflicts, deprecate outdated beliefs, and map inferred state transitions (e.g., `SHATTERED`) while maintaining strict inference provenance.

## Key Features

*   **Temporal State Machine ($T_n$ Engine):** Processes narrative steps chronologically, evaluating each new narrative against only currently `ACTIVE` beliefs.
*   **Dual-Layer Cognitive Architecture:** Differentiates between transient surface thoughts (Triplets) and entrenched Core Belief schemas via dynamic `SurfaceToCoreMapping` edges.
*   **Reified Belief Nodes:** Every belief is a unique entity with a distinct lifecycle (`first_seen_step`, `last_seen_step`, `status`), avoiding the "parallel edge" problem of traditional multi-graphs.
*   **Inference Provenance:** Strict Pydantic schemas enforce that every inferred belief and transition records the exact LLM model, prompt version, and temperature associated with its extraction.
*   **Decoupled LLM Architecture:** A clean LLMProvider protocol separates model-dependent inference from graph logic. The local Ollama implementation supports independent models for surface/core inference and contradiction verification.
*   **Interactive Visualization:** Renders the longitudinal evolution of the mind into an interactive, physics-based HTML graph using PyVis and NetworkX.
*   **Longitudinal Metrics:** Computes `active_duration`, `transition_count`, `distinct_manifestations`, and `recurrence_count`, providing quantitative signals for studying persistence, recurrence, and cognitive change over time.

---

## Quick Start

### Prerequisites
- Python 3.10+
- [Ollama](https://ollama.ai/) installed locally.
- A local copy of a model, for example gemma3:
  ```bash
  ollama pull gemma3:4b
  ```

### 1. Installation

Clone the repository and set up virtual environment:
```bash
git clone https://github.com/AetherMarina/belief-trace.git
cd belief-trace/belief_graph

python -m venv venv
source venv/bin/activate

pip install -r requirements.txt
pip install -e .
pip install -e ".[dev]"
```

### 2. Run the Alice in Wonderland Demo

We provide a curated dataset of Alice's internal monologues from Alice's Adventures in Wonderland to demonstrate cognitive dissonance and identity shifts.
```bash
python run_alice_demo.py
```

## What happens?

1. The engine extracts Alice's initial absolute certainties (e.g., bravery).
2. It processes subsequent chapters where her physical reality changes.
3. The Cognitive Arbiter detects conflicts with active beliefs; the engine deprecates affected beliefs and creates SHATTERED and REFRAMED transitions. In v0.2, SHATTERED represents direct negation or explicit abandonment of an existing belief without introducing a substantive replacement, while REFRAMED represents contradictions that introduce an alternative belief.
4. The output is saved as flat, machine-readable artifacts in the `outputs/` directory: `beliefs.jsonl`, `transitions.jsonl`, `core_beliefs.jsonl`, `observations.jsonl` and `surface_to_core_mappings.jsonl`, which is git-ignored.
5. Frozen v0.1 and v0.2 examples are available under examples/alice/.

### 3. Visualize the Graph

Turn the JSONL artifacts into a beautiful, interactive web graph:
```bash
python visualize.py
```

Open `outputs/graph.html` in your web browser.
* `Surface Beliefs (Boxes)`: Green (Active), Red (Deprecated).
* `Core Schemas (Purple Ellipses)`: Node size scales dynamically based on how often the schema is triggered by surface thoughts (recurrence).
* `Edges`: Solid lines represent state transitions; dashed purple lines represent MAPS_TO relationships linking surface thoughts to core schemas.
* `Hover`: View the narrative source, inference provenance, and longitudinal metrics.

### Evaluation

Run the minimal golden evaluation set to verify the structural integrity, schema validation, and logical conflict resolution of the framework:
```bash
python evals/eval_golden.py
```
This runs a controlled 4-step pipeline test to ensure the LLM outputs match the strict Pydantic requirements and maintain referential integrity across graph edges.

### Project Structure
```text
belief_graph/
├── experiments/          # Sandbox scripts and prompt testing
│   └── data/             # Datasets for longitudinal analysis and graph visualization
├── evals/                # LLM evaluations and golden datasets
│   └──eval_golden.py    
├── examples/              
│   └── alice/            # Generated artifacts (beliefs.jsonl, transitions.jsonl, graph.html)
├── src/belief_graph/              
│   ├── config.py         # Global constants and default model configurations
│   ├── core.py           # Pydantic schemas, Protocols
│   ├── embedder.py       # Embedder protocol and implementation
│   ├── engine.py         # The core TnEngine
│   ├── matching.py       # Semantic belief matching and SAME/DIFFERENT/CONTRADICTS decisions 
│   ├── memory.py         # Belief memory interface and retrieval contracts
│   ├── nli_matcher.py    # NLI-based semantic entailment and contradiction classifier
│   ├── providers.py      # LLMProvider implementations (OllamaProvider)
│   └── qdrant_memory.py  # Qdrant-backed vector storage and semantic candidate retrieval
├── run_alice_demo.py     # End-to-end execution script with demo dataset
├── visualize.py          # NetworkX to PyVis HTML renderer
├── README.md             # Belief Graph documentation
└── requirements.txt      # Python dependencies for module
```

### What's New in v0.2

*   **Core Belief Taxonomies:** Implemented strict Pydantic models for domain-driven psychological schemas.
*   **Many-to-One Abstraction:** Multiple surface beliefs can map to the same normalized core-belief node through explicit `SurfaceToCoreMapping` edges, while the original surface beliefs remain preserved.
*   **New Transition Dynamics:** Introduced the `REFRAMED` transition type to model nuanced cognitive shifts and belief modifications (expanding upon the v0.1 limitation of only `SHATTERED` transitions).
*   **Terminal Analytics:** Added `print_core_metrics_report()` for immediate longitudinal core-belief analysis.

### Future Roadmap (v0.3+)

* Advanced Transition Types: Adding CHALLENGED, REINFORCED, and WEAKENED edges.
* Document Adapters: Automated POV extraction and speaker diarization for full-length books and therapy transcripts.
* Extended Cognitive Networks: Transitioning from broad psychological categories to explicit entity-resolution mapping. Future versions will track the protagonist's evolving beliefs regarding specific significant figures and instrumental objects, modeling the exact relational network that shapes their cognitive state.