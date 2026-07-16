from unsloth import FastVisionModel
from qwenvl.config import ModelConfig, MergeConfig


def load_model_for_train(cfg: ModelConfig):
    model, tokenizer = FastVisionModel.from_pretrained(
        cfg.base_model_name,
        load_in_4bit=cfg.load_in_4bit,
    )

    model = FastVisionModel.get_peft_model(
        model,
        finetune_vision_layers=cfg.lora.finetune_vision_layers,
        finetune_language_layers=cfg.lora.finetune_language_layers,
        finetune_attention_modules=cfg.lora.finetune_attention_modules,
        finetune_mlp_modules=cfg.lora.finetune_mlp_modules,
        r=cfg.lora.r,
        lora_alpha=cfg.lora.lora_alpha,
        lora_dropout=cfg.lora.lora_dropout,
        bias=cfg.lora.bias,
    )

    return model, tokenizer


def load_model_for_infer(model_path: str, load_in_4bit: bool = True):
    model, tokenizer = FastVisionModel.from_pretrained(
        model_name=model_path,
        load_in_4bit=load_in_4bit,
    )
    return model, tokenizer


def merge_and_save(
    model,
    tokenizer,
    output_dir: str,
    save_method: str = "merged_16bit",
):
    model.save_pretrained_merged(output_dir, tokenizer, save_method=save_method)
