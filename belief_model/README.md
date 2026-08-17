# 🧠 Belief Model

**Belief Model is the model component of BeliefTrace**. It is an experimental research module for extracting structured core beliefs from narrative text using instruction-tuned language models and parameter-efficient fine-tuning.

The modul explores whether a model can map short free-form narratives to one or more predefined belief labels, such as beliefs related to inadequacy, rejection, safety, control, or worthiness.

> **Important:** Belief Model is a research prototype. It must not be used for diagnosis, treatment, risk assessment, or other clinical decision-making.

---

## Goals

Belief Model currently focuses on:

- Fine-tuning instruction models for multi-label core-belief extraction.
- Comparing a baseline model against LoRA/QLoRA-adapted models.
- Evaluating structured belief-label predictions with multilabel classification metrics.
- Supporting lightweight local experiments and larger cloud training runs.

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
belief_model/
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
│   ├── dataset_manifests/       
│   │   ├── phase*_manifest.py  # Augmentation experiments
│   ├── exp*.md                 # Per-run configurations, metrics, and observations
│   └── README.md               # Index of experiment records
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
│   │   ├── data_generation.py             # Script for generating synthetic data
│   │   ├── data_generation_*.py           # Scripts for generating additional synthetic data based on different criteria
│   │   ├── download_data.py               # Script for downloading raw data
│   │   └── prepare_dataset.py             # Script for preprocessing and preparing datasets
│   ├── evaluation/
│   │   └── evaluate_model.py   # Script for evaluating base and fine-tuned models
│   ├── training/
│   │   ├── callbacks.py        # Training metric logging callbacks
│   │   └── train_lora.py       # Script for training models with LoRA
│   └── utils/
│       ├── constants.py        # Master taxonomy (BELIEF_LABELS)
│       ├── logger.py           # Custom logging and HF Trainer bridging
│       └── extract_taxonomy.py # Utility to scrape unique classes from datasets
├── scripts/                
│   ├── check_env.py                      # Infrastructure check script
│   ├── compare_different_experiments.py  # Aggregates and prints the ablation summary table
│   ├── run_cloud_experiments.sh
│   └── run_local_experiments.sh
├── .gitignore                  # Specifies intentionally untracked files to ignore
├── constraints.txt             
├── README.md                   # Belief Model documentation
└── requirements.txt            # Python dependencies for module
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
cd belief-trace/belief_model

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
python -m src.data.download_data               # 1. Ingest raw data into data/raw

# Set the API key for your chosen provider
export GEMINI_API_KEY="your_api_key_here"

python -m src.data.data_generation             # 2.a Synthetic augmentation into data/augmented
python -m src.data.prepare_dataset             # 3. Preprocessing and alignment into data/train
```
**Important Data Pipeline Rule**:
To experiment with specialized synthetic runs (Experiments 2–5), substitute step 2.a with python -m src.data.data_generation_*.
When testing a variation, always include the base augmentation (`python -m src.data.data_generation`).

### 4. Run Training
For local testing (RTX 3050 - 4GB): Uses Llama-3.2-1B-Instruct.
```bash
python -m src.training.train_lora --run-name local_r8 --exp-tag exp1
```
For cloud training (RTX 3090/4090 - 24GB): Uses Llama-3.1-8B-Instruct.
```bash
python -m src.training.train_lora --run-name cloud_r16_lr2e --exp-tag exp1
```

### 5. Run Evaluation
Evaluate the baseline (untrained) model:
```bash
python -m src.evaluation.evaluate_model --mode baseline --run-name local --exp-tag exp1
```
*Note: Use --run-name cloud or local for baseline evaluation.*

Evaluate a LoRA-adapted model:
```bash
python -m src.evaluation.evaluate_model --mode lora --run-name cloud_r16_lr2e-4 --exp-tag exp1
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
chmod +x scripts/run_cloud_experiments.sh
./scripts/run_cloud_experiments.sh
```

For Local Environments:
```bash
chmod +x scripts/run_local_experiments.sh
./scripts/run_local_experiments.sh
```

## Comparing Experiments (Ablation Analysis)

To generate a comparative ablation matrix across your fine-tuning configuration runs, use the `compare_different_experiments.py` tracking script. 

### Prerequisites
The script expects all evaluation JSON outputs to be stored inside the outputs/metrics/ directory. It dynamically parses both the alphanumeric experiment tag and the training configuration directly from the filename using the following naming convention:
`outputs/metrics/evaluation_report_lora_cloud_exp<tag>_<config>.json`

*Example*: `evaluation_report_lora_cloud_exp1a_r8_lr1e-4.json` will be parsed as Experiment: **1a**, Configuration: **r8_lr1e-4**.

### Execution
Run the following commands from the belief_model/ directory:

```bash
python scripts/compare_different_experiments.py
```
Expected Output:
```text
-------- Experiment Evaluation Summary Matrix ---------
experiment configuration  macro_f1  micro_f1  sample_f1
         1    r8_lr2e-4  0.405712  0.489362   0.458508
         2    r8_lr2e-4  0.424379  0.422222   0.427735
         3    r8_lr2e-4  0.403894  0.441989   0.444812
-------------------------------------------------------
```

## 📊 Statistical Significance Testing

To estimate whether performance differences between fine-tuned experiments are likely to be real rather than artifacts of a small evaluation split, this repository uses **paired bootstrap resampling**.

<details>
<summary><b>Why Bootstrap over a standard T-Test? (Click to expand)</b></summary>

Multi-label F1 scores such as Macro F1 and Micro F1 are corpus-level metrics computed from aggregated prediction counts. They are not simple normally distributed per-example measurements, so standard parametric tests such as a basic t-test are not ideal.

**How the script works:**
1. **Paired Indexing:** Draws N examples with replacement and evaluates both models on the exact same sampled indices.
2. **Corpus Recalculation:** Recomputes Macro F1, Micro F1, and Samples F1 from scratch for each bootstrap sample.
3. **Difference Mapping**: Computes the metric difference as Model B - Model A.
4. **95% Confidence Interval:** After 10,000 iterations, the script reports the 2.5th and 97.5th percentiles of the bootstrap differences. If the interval excludes 0.0, the observed performance shift is treated as statistically meaningful at approximately the 95% confidence level.
</details>

### Execution
Run the script from the belief_model/ directory, passing your "Control" model and "Variant" model predictions:

```bash
python scripts/bootstrap_significance.py \
  --predictions-model-a outputs/predictions/predictions_lora_exp1_cloud_r8_lr2e-4.jsonl \
  --predictions-model-b outputs/predictions/predictions_lora_exp3_cloud_r8_lr2e-4.jsonl
```
Available Arguments:
* `--predictions-model-a`: (Required) Path to predictions .jsonl for Model A (usually the Experiment 1 baseline champion).
* `--predictions-model-b`: (Required) Path to predictions .jsonl for Model B.
* `--iterations`: (Optional) Number of bootstrap iterations. Defaults to `10000`.
* `--seed`: (Optional) Random seed for exact reproducibility. Defaults to `42`.


<details><summary><b>Example Output (Click to expand)</b></summary>
```text
============================================================
 Bootstrap Significance Report (Model B - Model A) 
============================================================
    Metric Lower 2.5% Upper 97.5% Significant (95% CI excludes 0)
  Macro F1    -0.1615      0.0127                              NO
  Micro F1    -0.1781     -0.0040                             YES
Samples F1    -0.1590      0.0115                              NO
============================================================
Interpretation: If 'Significant' is YES, the performance difference
is highly likely to be real, not just evaluation split noise.
```</details>

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
Target modules include `q_proj`, `k_proj`, `v_proj`, `o_proj`, `gate_proj`, `up_proj`, `down_proj`. In the tested configurations, LoRA alpha was matched to rank (`alpha = r`), giving an effective LoRA scaling of `alpha / r = 1`.

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

Belief Model uses standard Python logging for project-level logs.
* Runnable Scripts: Configure logging once at the entry point (setup_logger(process_name="...")).
* Reusable Modules: Only instantiate a logger (logger = logging.getLogger(__name__)).
* Training Metrics: Hugging Face Trainer logs can be piped into the same file via a custom TrainerCallback.

---

## 🛡️ Ethical and Data Considerations

Belief Model is an experimental utility that fine-tunes and interfaces with probabilistic Large Language Models (LLMs). Belief-related language is highly context-dependent, subjective, and can be ambiguous. 

Users must be aware of the following inherent limitations:

**Dataset Negativity Bias:** The CBT-Bench dataset is rooted in Cognitive Behavioral Therapy, which inherently focuses on identifying and restructuring maladaptive or limiting schemas. Consequently, the predefined taxonomy consists entirely of negative core beliefs (e.g., *I am inadequate*, *I am unlovable*). Models fine-tuned exclusively on this data will develop a "negativity bias" and lack the representational capacity to map positive, resourceful, or adaptive beliefs. When presented with empowering narratives, the model may hallucinate or forcefully fit the text into a negative class. This limitation must be factored into any downstream evaluation or application.

**Belief Model should strictly be treated as:**
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
