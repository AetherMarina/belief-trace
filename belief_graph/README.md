# 🕸️ Longitudinal Belief Graph (v0.1)

**An experimental framework for tracking, modeling and visualizing inferred belief transitions across longitudinal narratives.**

> **Disclaimer:** This framework uses Large Language Models (LLMs) to infer cognitive structures from text. The extracted beliefs and their subsequent transitions are model-generated hypotheses and narrative tracking tools, not clinical psychological assessments.

## Overview

Traditional knowledge graphs struggle with temporal dynamics and conflicting facts. The **Longitudinal Belief Graph** is designed specifically to track how an entity's core beliefs (`[Self]`, `[World]`, `[Others]`) evolve, shatter, or mature over time across sequential narrative steps ($T_n$).

Instead of overwriting old data, this framework uses a **Cognitive Arbiter** to detect narrative conflicts, deprecate outdated beliefs, and map inferred state transitions (e.g., `SHATTERED`) while maintaining strict inference provenance.

## Key Features

*   **Temporal State Machine ($T_n$ Engine):** Processes narrative steps chronologically, evaluating each new narrative against only currently `ACTIVE` beliefs.
*   **Reified Belief Nodes:** Every belief is a unique entity with a distinct lifecycle (`first_seen_step`, `last_seen_step`, `status`), avoiding the "parallel edge" problem of traditional multi-graphs.
*   **Inference Provenance:** Strict Pydantic schemas enforce that every inferred belief and transition records the exact LLM model, prompt version, and temperature associated with its extraction.
*   **Decoupled LLM Architecture:** Powered by a clean `LLMProvider` protocol. Currently defaults to local Ollama (`gemma3:4b`) via the OpenAI SDK, but easily extensible to any API.
*   **Interactive Visualization:** Renders the longitudinal evolution of the mind into an interactive, physics-based HTML graph using PyVis and NetworkX.

---

## Quick Start

### Prerequisites
- Python 3.10+
- [Ollama](https://ollama.ai/) installed locally.
- - A local copy of a model, for example gemma3:
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
```

### 2. Run the Alice in Wonderland Demo

We provide a curated dataset of Alice's internal monologues from Alice's Adventures in Wonderland to demonstrate cognitive dissonance and identity shifts.
```bash
python run_alice_demo.py
```

## What happens?

1. The engine extracts Alice's initial absolute certainties (e.g., bravery).
2. It processes subsequent chapters where her physical reality changes.
3. The Cognitive Arbiter detects conflicts with active beliefs; the engine deprecates affected beliefs and creates SHATTERED transitions.
4. The output is saved as flat, machine-readable beliefs.jsonl and transitions.jsonl files in the outputs/ directory, which is git-ignored. 
5. A frozen v0.1 example is available under examples/alice/.

### 3. Visualize the Graph

Turn the JSONL artifacts into a beautiful, interactive web graph:
```bash
python visualize.py
```

Open `outputs/graph.html` in your web browser.
* Green Nodes: Active beliefs.
* Red Nodes: Deprecated/Shattered beliefs.
* Hover: View the narrative source, inference provenance, and model-generated rationale behind a shattered belief.

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
├── config.py             # Global constants and default model configurations
├── core.py               # Pydantic schemas, Protocols, and the core TnEngine
├── providers.py          # LLMProvider implementations (OllamaProvider)
├── run_alice_demo.py     # End-to-end execution script with demo dataset
├── visualize.py          # NetworkX to PyVis HTML renderer
├── README.md             # Belief Graph documentation
└── requirements.txt      # Python dependencies for module
```

### v0.1 Scope

v0.1 intentionally keeps the transition model minimal:
- Beliefs are extracted independently at each narrative step.
- Transition evaluation compares the current ACTIVE belief state against the raw new narrative evidence.
- SHATTERED is the only supported transition type.
- Belief recurrence/deduplication and direct replacement links are deferred to v0.2.

### Future Roadmap (v0.2+)

* Advanced Transition Types: Adding REFRAMED, REINFORCED, and WEAKENED edges.
* BeliefTrace Integration: Adding core belief extraction layer.
* Document Adapters: Automated POV extraction and speaker diarization for full-length books and therapy transcripts.