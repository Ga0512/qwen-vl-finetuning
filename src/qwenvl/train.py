import argparse
import os

os.environ["TORCH_DYNAMO_DISABLE"] = "1"

import unsloth
import torch
from trl import SFTTrainer, SFTConfig

from qwenvl.config import TrainConfig
from qwenvl.data import prepare_dataset
from qwenvl.model import load_model_for_train, merge_and_save


def main():
    parser = argparse.ArgumentParser(description="Qwen3VL training CLI")
    parser.add_argument(
        "--config",
        type=str,
        required=True,
        help="Path to YAML training config",
    )
    args = parser.parse_args()

    cfg = TrainConfig.from_yaml(args.config)

    print(f"Loading dataset: {cfg.data.dataset_name}")
    dataset = prepare_dataset(
        dataset_name=cfg.data.dataset_name,
        split=cfg.data.dataset_split,
        num_samples=cfg.data.num_samples,
        instruction=cfg.data.instruction,
    )

    print(f"Loading model: {cfg.model.base_model_name}")
    model, tokenizer = load_model_for_train(cfg.model)

    print("Starting training...")
    trainer = SFTTrainer(
        model=model,
        processing_class=tokenizer,
        train_dataset=dataset,
        formatting_func=lambda x: x["messages"],
        args=SFTConfig(
            output_dir=cfg.training.output_dir,
            per_device_train_batch_size=cfg.training.per_device_train_batch_size,
            gradient_accumulation_steps=cfg.training.gradient_accumulation_steps,
            learning_rate=float(cfg.training.learning_rate),
            num_train_epochs=cfg.training.num_train_epochs,
            fp16=cfg.training.fp16,
            bf16=cfg.training.bf16,
            logging_steps=cfg.training.logging_steps,
            remove_unused_columns=cfg.training.remove_unused_columns,
        ),
    )

    trainer.train()

    if cfg.merge.enabled:
        print(f"Merging LoRA weights and saving to {cfg.merge.output_dir} ...")
        merge_and_save(
            model=model,
            tokenizer=tokenizer,
            output_dir=cfg.merge.output_dir,
            save_method=cfg.merge.save_method,
        )
        print("Merge complete!")

    print("Done!")


if __name__ == "__main__":
    main()
