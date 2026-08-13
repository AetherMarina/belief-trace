#!/usr/bin/env bash
set -euo pipefail

exp_tag="exp1"
echo "Running: python -m src.evaluation.evaluate_model --mode baseline --run-name cloud --exp-tag ${exp_tag}"
python -m src.evaluation.evaluate_model --mode baseline --run-name cloud --exp-tag ${exp_tag}

for run in \
  cloud_r16_lr2e-4 \
  cloud_r16_lr1e-4 \
  cloud_r8_lr2e-4 \
  cloud_r8_lr1e-4
do
  echo "Running: python -m src.training.train_lora --run-name "${run}" --exp-tag ${exp_tag}"
  python -m src.training.train_lora --run-name "${run}" --exp-tag ${exp_tag}

  echo "Running: python -m src.evaluation.evaluate_model --mode lora --run-name "$run" --exp-tag ${exp_tag}"
  python -m src.evaluation.evaluate_model --mode lora --run-name "$run" --exp-tag ${exp_tag}
done