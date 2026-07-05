# Belief-Trace

**Belief-Trace** is an experimental research project for extracting structured core beliefs from narrative text using instruction-tuned language models and parameter-efficient fine-tuning.

The project explores whether a model can map short free-form narratives to one or more predefined belief labels, such as beliefs related to inadequacy, rejection, safety, control, or worthiness.

> **Important:** Belief-Trace is a research prototype. It must not be used for diagnosis, treatment, risk assessment, or other clinical decision-making.

---

## Goals

Belief-Trace currently focuses on:

- Fine-tuning instruction models for multi-label core-belief extraction.
- Comparing a baseline model against LoRA/QLoRA-adapted models.
- Evaluating structured belief-label predictions with multilabel classification metrics.
- Supporting lightweight local experiments and larger cloud training runs.
- Creating a foundation for later work on belief relations, belief hierarchies, and longitudinal belief graphs.

---

## Current Workflow

```text
Narrative text
      ↓
Instruction-tuned language model
      ↓
Predicted core-belief labels
      ↓
Evaluation metrics, raw predictions, and per-label error analysis
```

The current task is framed as multilabel classification through language-model generation:

```text
Input:
"I always feel that I have to prove myself, otherwise people will see that I am not good enough."

Output:
"I am inadequate; I must prove my worth"
```

---

## Models

The project uses two environments.

| Environment | Model | Intended purpose |
|---|---|---|
| Local development | `meta-llama/Llama-3.2-1B-Instruct` | Dataset validation, pipeline debugging, local LoRA experiments |
| Cloud training | `meta-llama/Llama-3.1-8B-Instruct` | Larger-scale QLoRA experiments and final comparison runs |

The local workflow is designed for resource-constrained GPUs such as an NVIDIA RTX 3050 Laptop GPU with 4 GB VRAM.

---

## 📂 Project Structure

```
belief-trace/
├── configs/
│   ├── generation.yaml                   # Configuration for text generation
│   ├── training_cloud_r8_lr1e-4.yaml     # Configuration for cloud model training
│   ├── training_cloud_r8_lr2e-4.yaml     # Configuration for cloud model training
│   ├── training_cloud_r16_lr1e-4.yaml    # Configuration for cloud model training
│   ├── training_cloud_r16_lr2e-4.yaml    # Configuration for cloud model training
│   ├── training_local_r8.yaml            # Configuration for local model training
│   └── training_local_r16.yaml           # Configuration for local model training
├── data/
│   ├── augmented/              # Augmented datasets
│   ├── eval/                   # Evaluation datasets
│   ├── raw/                    # Raw, unprocessed data
│   └── train/                  # Training datasets
├── experiments/
│   ├── exp01_baseline.md       # Documentation for baseline experiment
│   ├── exp02_lora_r8.md        # Documentation for LoRA experiment with rank 8
│   └── exp03_lora_r16.md       # Documentation for LoRA experiment with rank 16
├── models/
│   ├── checkpoints/            # Intermediate Hugging Face Trainer checkpoints
│   ├── lora/                   # LoRA (Low-Rank Adaptation) models
│   └── merged/                 # Merged models (e.g., base model + LoRA)
├── notebooks/
│   ├── cbt_dataset_analysis.ipynb         # Jupyter notebook for dataset exploration and analysis
│   └── training_dataset_analysis.ipynb    # Jupyter notebook for analyzing training dataset
├── outputs/
│   ├── confusion_matrices/     # Confusion matrices from model evaluations
│   ├── metrics/                # Performance metrics from experiments
│   └── predictions/            # Model predictions
├── src/
│   ├── data/
│   │   ├── data_generation.py  # Script for generating synthetic data
│   │   ├── download_data.py    # Script for downloading raw data
│   │   └── prepare_dataset.py  # Script for preprocessing and preparing datasets
│   ├── evaluation/
│   │   └── evaluate_model.py   # Script for evaluating fine-tuned models
│   ├── training/
│   │   ├── callbacks.py        # Training metric logging callbacks
│   │   └── train_lora.py       # Script for training models with LoRA
│   └── utils/
│       ├── constants.py      # Master taxonomy (BELIEF_LABELS)
│       ├── logger.py         # Custom logging and HF Trainer bridging
│       └── extract_taxonomy.py # Utility to scrape unique classes from datasets
├── scripts/                
│   └── check_env.py            # Infrastructure check script
├── .gitignore                  # Specifies intentionally untracked files to ignore
├── README.md                   # Project README file
└── requirements.txt            # Python dependencies for the project
```

---

## Quick Start

### Prerequisites
* Python 3.11+
* [uv](https://docs.astral.sh/uv/getting-started/installation/)

### 1. Environment Setup
Clone the repository and set up virtual environment:
```bash
git clone https://github.com/AetherMarina/belief-trace.git
cd belief-trace

uv venv
source .venv/bin/activate

uv pip install -r requirements.txt -c constraints.txt --index-strategy unsafe-best-match
```

Verify that CUDA is available:

```bash
python scripts/check_env.py
```

### 2. Hugging Face Authentication

Some Llama model repositories require access approval. Make sure your account has access, then log in:
```bash
hf auth login
```

### 3. Run the Data Pipeline

Execute these scripts sequentially to ingest, augment, and align the dataset.

**Note on Synthetic Data Generation:** 
The augmentation script (`data_generation.py`) utilizes the standard `openai` Python client. While it is pre-configured to use the Gemini API endpoint, the architecture is provider-agnostic. You can easily swap the `base_url` and `api_key` in the script to use any compatible LLM provider (e.g., standard OpenAI, Groq, Together AI, or a local vLLM server).

*Recommendation:* We strongly advise using a large, frontier-class LLM (e.g., Gemini 1.5 Pro, GPT-4o, Llama 3 70B+) for data generation. Inferring latent cognitive beliefs from text requires high-capacity reasoning that smaller models struggle to maintain consistently.

```bash
python -m src.data.download_data     # 1. Ingest raw data into data/raw

# Set the API key for your chosen provider
export GEMINI_API_KEY="your_api_key_here"

python -m src.data.data_generation   # 2. Synthetic augmentation into data/augmented
python -m src.data.prepare_dataset   # 3. Preprocessing and alignment into data/train
```

### 4. Run Training
For local testing (RTX 3050 - 4GB): Uses Llama-3.2-1B-Instruct.
```bash
python -m src.training.train_lora --config configs/training_local.yaml
```
For cloud training (RTX 3090/4090 - 24GB): Uses Llama-3.1-8B-Instruct.
```bash
python -m src.training.train_lora --config configs/training_cloud16.yaml
```

### 5. Run Evaluation
Evaluate the baseline (untrained) model:
```bash
python -m src.evaluation.evaluate_model --mode baseline --run-name local
```
*Note: Use --run-name cloud or local for baseline evaluation.*

Evaluate a LoRA-adapted model:
```bash
python -m src.evaluation.evaluate_model --mode lora --run-name cloud_r16_lr2e-4
```
*Note: The --run-name should match the suffix of your training config file (e.g., cloud_r16_lr2e-4).*

---

## Reproducing Experiments
If you want to run the full suite of baseline evaluations and LoRA hyperparameter sweeps sequentially, we provide automated bash scripts for both local and cloud environments.

**Prerequisites:** 
1. Ensure your virtual environment is active (`source .venv/bin/activate`).
2. Ensure you have already executed the **Data Pipeline** (Step 3) so that the processed datasets are available in `data/train`.

For Cloud Environments:
```bash
chmod +x run_cloud_experiments.sh
./run_cloud_experiments.sh
```

For Local Environments:
```bash
chmod +x run_local_experiments.sh
./run_local_experiments.sh
```

---

## System Architecture & Explanations
### 1. Dataset Format

Training and evaluation data use the chat-style JSONL format expected by instruction-tuned models.

JSONL Schema:

```json
{
  "messages": [
    {
      "role": "system",
      "content": "Identify the underlying core beliefs from the narrative text."
    },
    {
      "role": "user",
      "content": "I feel like I always need to prove myself or people will realize I am not good enough."
    },
    {
      "role": "assistant",
      "content": "I am inadequate; I must prove my worth"
    }
  ]
}
```
- Label Rules: Each assistant response contains one or more belief labels separated by semicolons (`belief A`; `belief B`). For reliable evaluation, label wording remains strictly consistent across training and evaluation data.

### 2. QLoRA Configuration & Cloud Experiments

The project begins with conservative settings and compares a small number of meaningful configurations rather than performing a large hyperparameter sweep.

**Recommended QLoRA Targets:**
Target modules include `q_proj`, `k_proj`, `v_proj`, `o_proj`, `gate_proj`, `up_proj`, `down_proj`. With `r = 16` and `alpha = 16`, the effective LoRA scaling is `alpha / r = 1`.

**Cloud Experiment Matrix:**
Each experiment uses its own output and adapter path so results are never overwritten.

| Run name | LoRA rank (r) | LoRA alpha | Learning rate |
|---|--------------:|---:|---:|
| `cloud_r8_lr2e-4` |             8 | 8 | `2e-4` |
| `cloud_r16_lr2e-4` |            16 | 16 | `2e-4` |
| `cloud_r8_lr1e-4` |             8 | 8 | `1e-4` |
| `cloud_r16_lr1e-4` |            16 | 16 | `1e-4` |

### 3. Evaluation Methodology

The evaluation pipeline (`src/evaluation/evaluate_model.py`) performs the following steps:
- Loads the evaluation dataset, preserving the original system prompt.
- Parses and deduplicates semicolon-separated ground-truth labels and model outputs.
- Generates deterministic predictions using greedy decoding, evaluation mode, and torch.inference_mode().
- Validates predictions against the fixed BELIEF_LABELS taxonomy, isolating hallucinations.
- Converts valid labels into multilabel vectors for metric calculation.

**Metrics Explained:**

* Macro F1: Averages performance across all belief classes equally, highlighting performance on rare, imbalanced labels.
* Micro F1: Aggregates true positives, false positives, and false negatives across all labels before computing F1; it reflects overall label-level performance.
* Samples F1: Computes F1 for each narrative’s predicted label set, then averages across narratives. (useful for assessing per-example label-set alignment)
* Exact-Match Accuracy: The percentage of evaluations where the predicted set of beliefs matched the ground truth perfectly with zero deviations. 
* Hallucination Count: Number of evaluation examples for which the model produced at least one label outside the predefined taxonomy.

### 4. Output Artifacts

Each experiment generates a standardized set of artifacts for benchmarking and error analysis:

```text
outputs/
├── runs/
│   └── <run_name>/
│       ├── training.log
│       └── checkpoints/
│
├── metrics/
│   └── evaluation_report_<run_name>.json
│
├── predictions/
│   └── predictions_<run_name>.jsonl
│
└── confusion_matrices/
    └── cm_summary_<run_name>.csv
```

### 5. Logging Standards

Belief-Trace uses standard Python logging for project-level logs.
* Runnable Scripts: Configure logging once at the entry point (setup_logger(process_name="...")).
* Reusable Modules: Only instantiate a logger (logger = logging.getLogger(__name__)).
* Training Metrics: Hugging Face Trainer logs can be piped into the same file via a custom TrainerCallback.

---

## ⚠️ Ethical and Data Considerations

Belief-Trace is an experimental utility that fine-tunes and interfaces with probabilistic Large Language Models (LLMs). Belief-related language is highly context-dependent, subjective, and can be ambiguous. 

Users must be aware of the following inherent limitations:

**Dataset Negativity Bias:** The CBT-Bench dataset is rooted in Cognitive Behavioral Therapy, which inherently focuses on identifying and restructuring maladaptive or limiting schemas. Consequently, the predefined taxonomy consists entirely of negative core beliefs (e.g., *I am inadequate*, *I am unlovable*). Models fine-tuned exclusively on this data will develop a "negativity bias" and lack the representational capacity to map positive, resourceful, or adaptive beliefs. When presented with empowering narratives, the model may hallucinate or forcefully fit the text into a negative class. This limitation must be factored into any downstream evaluation or application.

**Belief-Trace should strictly be treated as:**
- An experimental NLP research system.
- A structured text-analysis prototype.
- A tool for studying model behavior, latent inference, and annotation design.

**It must NOT be used as:**
- A psychological diagnosis tool or replacement for a clinician/therapist.
- A system that can reliably infer factual hidden mental states from text.
- A tool for profiling individuals, employee monitoring, or automated decision-making.

**Non-Deterministic Outputs:** LLMs are probabilistic engines. Any generated inference regarding cognitive schemas or latent beliefs is an abstraction, subject to inaccuracies, biases, or hallucinations. 

**Data Privacy:** Use only appropriately licensed public datasets for training and evaluation. Do not upload private, sensitive, or personally identifiable narratives to third-party APIs without explicit consent and rigorous privacy reviews. Local, zero-persistence execution is strongly recommended for handling sensitive text.

**Use at Your Own Risk:** This open-source repository is provided "as is," without warranty of any kind. Developers are responsible for implementing their own safety fallbacks, data protection compliance, and ethical reviews.

---

## ⚖️ Authorship Disclaimer
**BeliefTrace** is an independent, personal project developed by Marina Kaličanin. 
It is not an official product of, nor is it endorsed by, any past or current employers. 
The architecture, logic, and implementation were created outside of professional work hours using personal resources.

## License
This project is licensed under the **Apache License 2.0**. See the [LICENSE](LICENSE) file for the full text.