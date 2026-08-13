#!/usr/bin/env bash
set -euo pipefail

exp_tag="exp1"
# Baseline evaluation
echo "Running: python -m src.evaluation.evaluate_model --mode baseline --run-name local --exp-tag ${exp_tag}"
python -m src.evaluation.evaluate_model --mode baseline --run-name local --exp-tag ${exp_tag}

for run in \
  local_r8 \
  local_r16
do
  echo "Running: python -m src.training.train_lora --run-name "${run}" --exp-tag ${exp_tag}"
  python -m src.training.train_lora --run-name "${run}" --exp-tag ${exp_tag}
  echo "Running: python -m src.evaluation.evaluate_model --mode lora --run-name "$run" --exp-tag ${exp_tag}"
  python -m src.evaluation.evaluate_model --mode lora --run-name "$run" --exp-tag ${exp_tag}
done