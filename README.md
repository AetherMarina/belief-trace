# BeliefTrace
Tracing latent cognitive beliefs through narrative language.

## 📂 Project Structure

```
belief-trace/
├── configs/
│   ├── generation.yaml         # Configuration for text generation
│   └── training.yaml           # Configuration for model training
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
│   ├── lora/                   # LoRA (Low-Rank Adaptation) models
│   └── merged/                 # Merged models (e.g., base model + LoRA)
├── notebooks/
│   ├── dataset_analysis.ipynb  # Jupyter notebook for dataset exploration and analysis
│   └── error_analysis.ipynb    # Jupyter notebook for analyzing model errors
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
│   │   ├── evaluate_baseline.py# Script for evaluating baseline models
│   │   └── evaluate_model.py   # Script for evaluating fine-tuned models
│   └── training/
│       └── train_lora.py       # Script for training models with LoRA
├── .gitignore                  # Specifies intentionally untracked files to ignore
├── README.md                   # Project README file
└── requirements.txt            # Python dependencies for the project
```
