import argparse
import yaml
import logging
from pathlib import Path
# Unsloth MUST be imported before TRL and Datasets
from unsloth import FastLanguageModel, is_bfloat16_supported
from unsloth.chat_templates import get_chat_template
from datasets import load_from_disk
from trl import SFTTrainer, SFTConfig

from src.training.callbacks import FileLoggingCallback
from src.utils.logger import setup_logger


logger = logging.getLogger(__name__)


def load_config(config_path: str) -> dict:
    """Loads the YAML configuration file."""
    path = Path(config_path)
    if not path.exists():
        raise FileNotFoundError(f"Configuration file not found: {config_path}")

    with open(path, "r", encoding="utf-8") as file:
        return yaml.safe_load(file)


def main():
    # Parse Command Line Arguments
    parser = argparse.ArgumentParser(description="Belief-Trace LoRA Fine-Tuning Pipeline")
    parser.add_argument(
        "--config",
        type=str,
        default="configs/training_local_r8.yaml",
        help="Path to the training configuration YAML file."
    )
    args = parser.parse_args()

    logger.info(f"Loading configuration from {args.config}...")
    config = load_config(args.config)

    # Extract Parameters from Config
    model_name = config["model_name"]
    max_seq_length = config["max_seq_length"]
    dataset_dir = "data/train/ready_dataset"

    logger.info(f"Initializing Unsloth FastLanguageModel for: {model_name}")

    # Load Model and Tokenizer (Auto 4-bit quantization for VRAM efficiency)
    model, tokenizer = FastLanguageModel.from_pretrained(
        model_name=model_name,
        max_seq_length=max_seq_length,
        dtype=None,
        load_in_4bit=True,
    )

    tokenizer = get_chat_template(
        tokenizer,
        chat_template="llama-3",
    )

    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    logger.info(f"Using native chat template. EOS token is: {tokenizer.eos_token}")

    # Add LoRA Adapters
    logger.info("Injecting LoRA adapters based on configuration...")
    model = FastLanguageModel.get_peft_model(
        model,
        r=config["lora"]["r"],
        target_modules=config["lora"]["target_modules"],
        lora_alpha=config["lora"]["alpha"],
        lora_dropout=config["lora"]["dropout"],
        bias="none",
        use_gradient_checkpointing="unsloth",
        random_state=config["training"]["seed"],
        use_rslora=False,
        loftq_config=None,
    )

    # Load and Format Dataset
    logger.info("Loading prepared training dataset...")
    dataset = load_from_disk(dataset_dir)

    def formatting_prompts_func(examples):
        convos = examples["messages"]
        texts = [tokenizer.apply_chat_template(convo, tokenize=False, add_generation_prompt=False) for convo in convos]
        return {"text": texts}

    dataset = dataset.map(formatting_prompts_func, batched=True)

    # Configure SFTTrainer (Modern TRL API)
    logger.info("Setting up SFTTrainer with SFTConfig...")
    trainer = SFTTrainer(
        model=model,
        processing_class=tokenizer,
        train_dataset=dataset,
        args=SFTConfig(
            dataset_text_field="text",
            max_length=max_seq_length,
            dataset_num_proc=2,
            packing=False,
            per_device_train_batch_size=config["training"]["per_device_train_batch_size"],
            gradient_accumulation_steps=config["training"]["gradient_accumulation_steps"],
            warmup_steps=config["training"]["warmup_steps"],
            num_train_epochs=config["training"]["num_train_epochs"],
            learning_rate=float(config["training"]["learning_rate"]),
            fp16=not is_bfloat16_supported(),
            bf16=is_bfloat16_supported(),
            logging_steps=config["training"]["logging_steps"],
            optim=config["training"]["optim"],
            weight_decay=config["training"]["weight_decay"],
            lr_scheduler_type="linear",
            seed=config["training"]["seed"],
            output_dir=config["training"]["output_dir"],
            report_to="none"
        ),
        callbacks=[FileLoggingCallback(logger)],
    )

    # Execute Training
    logger.info("Starting LoRA fine-tuning process...")
    trainer_stats = trainer.train()
    logger.info(f"Training completed successfully. Runtime info: {trainer_stats}")

    # Handle Exporting based on YAML config
    if config["export"]["save_lora"]:
        lora_path = config["export"]["lora_save_path"]
        logger.info(f"Saving trained LoRA adapters to {lora_path}...")
        model.save_pretrained(lora_path)
        tokenizer.save_pretrained(lora_path)

    if config["export"]["export_gguf"]:
        gguf_path = config["export"]["gguf_save_path"]
        quant_method = config["export"]["quantization_method"]
        logger.info(f"Exporting to GGUF ({quant_method}) for local edge deployment...")
        try:
            model.save_pretrained_gguf(gguf_path, tokenizer, quantization_method=quant_method)
            logger.info("GGUF export successful!")
        except Exception as e:
            logger.error(f"GGUF export encountered an issue: {e}")
    else:
        logger.info("GGUF export skipped as per configuration.")


if __name__ == "__main__":
    setup_logger(process_name="train")
    main()
