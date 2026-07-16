from dataclasses import dataclass, field
from typing import Optional
import yaml


@dataclass
class LoraConfig:
    r: int = 16
    lora_alpha: int = 16
    lora_dropout: float = 0.0
    bias: str = "none"
    finetune_vision_layers: bool = True
    finetune_language_layers: bool = True
    finetune_attention_modules: bool = True
    finetune_mlp_modules: bool = True


@dataclass
class ModelConfig:
    base_model_name: str = "unsloth/Qwen3-VL-4B-Instruct-bnb-4bit"
    load_in_4bit: bool = True
    lora: LoraConfig = field(default_factory=LoraConfig)


@dataclass
class DataConfig:
    dataset_name: str = "unsloth/LaTeX_OCR"
    dataset_split: str = "train"
    num_samples: Optional[int] = 100
    instruction: str = "Write the LaTeX representation for this image."


@dataclass
class TrainingConfig:
    output_dir: str = "qwenvl-lora-checkpoints"
    per_device_train_batch_size: int = 1
    gradient_accumulation_steps: int = 8
    learning_rate: float = 2e-4
    num_train_epochs: int = 10
    fp16: bool = True
    bf16: bool = False
    logging_steps: int = 1
    remove_unused_columns: bool = False


@dataclass
class MergeConfig:
    enabled: bool = True
    output_dir: str = "qwenvl-model-completo"
    save_method: str = "merged_16bit"


@dataclass
class TrainConfig:
    model: ModelConfig = field(default_factory=ModelConfig)
    data: DataConfig = field(default_factory=DataConfig)
    training: TrainingConfig = field(default_factory=TrainingConfig)
    merge: MergeConfig = field(default_factory=MergeConfig)

    @classmethod
    def from_yaml(cls, path: str) -> "TrainConfig":
        with open(path) as f:
            raw = yaml.safe_load(f)
        return cls._from_dict(raw)

    @classmethod
    def _from_dict(cls, raw: dict) -> "TrainConfig":
        model = ModelConfig(**raw.get("model", {}))
        if "lora" in raw.get("model", {}):
            model.lora = LoraConfig(**raw["model"]["lora"])

        train_kw = dict(raw.get("training", {}))
        if "learning_rate" in train_kw:
            train_kw["learning_rate"] = float(train_kw["learning_rate"])
        if "num_train_epochs" in train_kw:
            train_kw["num_train_epochs"] = int(train_kw["num_train_epochs"])
        if "per_device_train_batch_size" in train_kw:
            train_kw["per_device_train_batch_size"] = int(train_kw["per_device_train_batch_size"])
        if "gradient_accumulation_steps" in train_kw:
            train_kw["gradient_accumulation_steps"] = int(train_kw["gradient_accumulation_steps"])
        if "logging_steps" in train_kw:
            train_kw["logging_steps"] = int(train_kw["logging_steps"])

        return cls(
            model=model,
            data=DataConfig(**raw.get("data", {})),
            training=TrainingConfig(**train_kw),
            merge=MergeConfig(**raw.get("merge", {})),
        )
