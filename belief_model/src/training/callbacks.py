import json
import logging
from transformers import TrainerCallback


class FileLoggingCallback(TrainerCallback):
    """Write Hugging Face Trainer metrics into the project logger."""

    def __init__(self, logger: logging.Logger):
        self.logger = logger

    def on_log(self, args, state, control, logs=None, **kwargs):
        if not logs:
            return

        metrics = {
            key: value
            for key, value in logs.items()
            if isinstance(value, (int, float, str, bool))
        }

        self.logger.info(
            "trainer_metrics step=%s epoch=%.3f %s",
            state.global_step,
            state.epoch or 0.0,
            json.dumps(metrics, sort_keys=True),
        )
