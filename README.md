# qwenvl

CLI for fine-tuning and inference of **Qwen3-VL** with LoRA/QLoRA using [unsloth](https://github.com/unslothai/unsloth).

Powers ⚡
- **Flash Attention 2** for fast training and inference
- **4-bit quantization via BitsAndBytes** for low VRAM usage

## Installation

```bash
pip install -e .
```

## Training

```bash
qwenvl-train --config configs/train.yaml
```

The YAML controls everything: base model, dataset, hyperparameters, LoRA, and post-training merge.

### Configuration (`configs/train.yaml`)

| Section | Field | Description |
|---------|-------|-------------|
| `model` | `base_model_name` | Base model (e.g. `unsloth/Qwen3-VL-4B-Instruct-bnb-4bit`) |
| `model` | `load_in_4bit` | Load in 4-bit |
| `model.lora` | `r`, `lora_alpha`, `lora_dropout` | LoRA params |
| `data` | `dataset_name` | HuggingFace dataset |
| `data` | `num_samples` | `null` to use all samples |
| `data` | `instruction` | Instruction prompt |
| `training` | `output_dir` | Checkpoint directory |
| `training` | `learning_rate`, `num_train_epochs` | Hyperparameters |
| `merge` | `enabled` | Whether to merge LoRA into full model |
| `merge` | `output_dir` | Merged model output dir |

## Inference

```bash
qwenvl-infer \
    --model qwenvl-model-completo \
    --image path/to/image.jpg \
    --prompt "Describe this image." \
    --max-tokens 512
```

Flags:
- `--model` — trained model directory (merged or not)
- `--image` — path to input image
- `--prompt` — instruction text (default: `"Describe this image in detail."`)
- `--max-tokens` — max generated tokens
- `--no-4bit` — disable 4-bit loading

## Project Structure

```
qwenvl/
├── README.md
├── requirements.txt
├── pyproject.toml
├── configs/
│   └── train.yaml
└── src/qwenvl/
    ├── __init__.py
    ├── config.py      # YAML config dataclasses
    ├── data.py        # Dataset loading & preprocessing
    ├── model.py       # Model loading, LoRA, merge
    ├── train.py       # Training CLI
    └── infer.py       # Inference CLI
```
