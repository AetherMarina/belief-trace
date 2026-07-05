#!/usr/bin/env bash
set -euo pipefail

echo "Running: python -m src.evaluation.evaluate_model --mode baseline --run-name cloud"
python -m src.evaluation.evaluate_model --mode baseline --run-name cloud

for run in \
  cloud_r16_lr2e-4 \
  cloud_r16_lr1e-4 \
  cloud_r8_lr2e-4 \
  cloud_r8_lr1e-4
do
  echo "Running: python -m src.training.train_lora --config configs/training_${run}.yaml"
  python -m src.training.train_lora --config "configs/training_${run}.yaml"

  echo "Running: python -m src.evaluation.evaluate_model --mode lora --run-name ${run}"
  python -m src.evaluation.evaluate_model --mode lora --run-name "$run"
done