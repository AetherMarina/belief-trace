# 👣 BeliefTrace

BeliefTrace is an experimental research project for modeling how latent cognitive beliefs can be extracted from narrative language and traced over time.

The project currently consists of two complementary components: Belief Model and Longitudinal Belief Graph.

## Architecture

```text
Narrative text
      ↓
Belief Model
      ↓
Structured belief labels
      ↓
Longitudinal Belief Graph
      ↓
Belief relations, transitions, and longitudinal evolution
```

## Components

### 🧠 Belief Model

Fine-tuned LLM component for structured multi-label core-belief extraction from narrative text.

Research areas include:

- QLoRA / LoRA fine-tuning
- multi-label belief extraction
- synthetic data augmentation
- ablation studies
- model evaluation
- bootstrap significance testing

→ [Belief Model documentation](belief_model/README.md)


### 🕸️ Longitudinal Belief Graph

Longitudinal representation layer for tracking beliefs and how they evolve across narrative steps.

Planned / evolving capabilities include:

- belief nodes
- relations between beliefs
- longitudinal belief tracking
- belief transitions
- reframing and belief replacement
- graph visualization

→ [Longitudinal Belief Graph documentation](belief_graph/README.md)

## Research Motivation

BeliefTrace explores a broader question:

> Can AI systems move beyond isolated text classification toward representations of how cognitive patterns persist, interact, and change over time?

The **Belief Model** focuses on extracting structured belief representations from narrative text, while the **Longitudinal Belief Graph** focuses on their relationships, transitions, and longitudinal evolution.

Together, they provide an experimental foundation for belief-aware and human-centered AI systems.

## Project Structure

```
belief-trace/
├── belief_model/
│   ├── README.md
│   ├── requirements.txt
│   ├── constraints.txt
│   ├── configs/
│   ├── data/
│   ├── experiments/
│   ├── models/
│   ├── notebooks/
│   ├── outputs/
│   ├── scripts/
│   └── src/
│
├── belief_graph/
│   ├── README.md
│   ├── requirements.txt
│   ├── experiments/
│   ├── evals/
│   ├── examples/
│   ├── config.py
│   ├── core.py
│   ├── providers.py
│   ├── run_alice_demo.py
│   └── visualize.py
│
├── README.md
└── LICENSE
```

---

## Status

- ✅ Belief extraction and QLoRA experimentation
- ✅ Multi-label evaluation and ablation studies
- ✅ Statistical significance testing
- 🚧 Longitudinal Belief graph and belief-transition modeling

---

## 🛡️ Ethical and Data Considerations
BeliefTrace is an experimental research system for studying belief extraction and longitudinal belief representation from narrative text. Its outputs are probabilistic abstractions and must not be treated as factual representations of a person's internal mental state. The project is not intended for diagnosis, treatment, profiling, employee monitoring, risk assessment, or automated decision-making. Each component may introduce additional limitations documented in its respective README.

For model-specific limitations, including CBT-Bench taxonomy bias and data privacy considerations, see the [Belief Model documentation](belief_model/README.md#️-ethical-and-data-considerations).

## ⚖️ Authorship Disclaimer
**BeliefTrace** is an independent, personal project developed by Marina Kaličanin. 
It is not an official product of, nor is it endorsed by, any past or current employers. 
The architecture, logic, and implementation were created outside of professional work hours using personal resources.

## License
This project is licensed under the **Apache License 2.0**. See the [LICENSE](LICENSE) file for the full text.