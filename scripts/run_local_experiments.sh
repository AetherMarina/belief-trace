#!/usr/bin/env bash
set -euo pipefail

# Baseline evaluation
echo "Running: python -m src.evaluation.evaluate_model --mode baseline --run-name local"
python -m src.evaluation.evaluate_model --mode baseline --run-name local

for run in \
  local_r8 \
  local_r16
do
  echo "Running: python -m src.training.train_lora --config configs/training_${run}.yaml"
  python -m src.training.train_lora --config "configs/training_${run}.yaml"
  echo "Running: python -m src.evaluation.evaluate_model --mode lora --run-name ${run}"
  python -m src.evaluation.evaluate_model --mode lora --run-name "$run"
done