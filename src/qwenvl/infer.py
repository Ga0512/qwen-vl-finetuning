import argparse
import os

os.environ["TORCH_DYNAMO_DISABLE"] = "1"

import torch
from PIL import Image
from transformers import Qwen3VLForConditionalGeneration, AutoProcessor
from qwen_vl_utils import process_vision_info


def main():
    parser = argparse.ArgumentParser(description="Qwen3VL inference CLI")
    parser.add_argument(
        "--model",
        type=str,
        required=True,
        help="Path to the merged model directory",
    )
    parser.add_argument(
        "--image",
        type=str,
        required=True,
        help="Path to input image",
    )
    parser.add_argument(
        "--prompt",
        type=str,
        default="Describe this image in detail.",
        help="Text prompt for the model",
    )
    parser.add_argument(
        "--4bit",
        action="store_true",
        default=True,
        dest="load_4bit",
        help="Load model in 4-bit quantization (default: True)",
    )
    parser.add_argument(
        "--max-tokens",
        type=int,
        default=512,
        help="Maximum new tokens to generate",
    )
    parser.add_argument(
        "--no-4bit",
        action="store_false",
        dest="load_4bit",
        help="Disable 4-bit loading",
    )
    args = parser.parse_args()

    print("Loading model...")

    model = Qwen3VLForConditionalGeneration.from_pretrained(
        args.model,
        device_map="auto",
        quantization_config=(
            {"load_in_4bit": True, "bnb_4bit_compute_dtype": torch.float16}
            if args.load_4bit
            else None
        ),
    )
    processor = AutoProcessor.from_pretrained(args.model)

    print("Processing image...")
    image = Image.open(args.image).convert("RGB")

    messages = [
        {
            "role": "user",
            "content": [
                {"type": "image", "image": image},
                {"type": "text", "text": args.prompt},
            ],
        }
    ]

    text_prompt = processor.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=True
    )
    image_inputs, video_inputs = process_vision_info(messages)

    inputs = processor(
        text=[text_prompt],
        images=image_inputs,
        videos=video_inputs,
        padding=True,
        return_tensors="pt",
    ).to("cuda")

    print("Generating...")
    with torch.inference_mode():
        generated_ids = model.generate(
            **inputs,
            max_new_tokens=args.max_tokens,
            do_sample=False,
        )

    trimmed = generated_ids[0][inputs.input_ids.shape[1] :]
    result = processor.decode(
        trimmed,
        skip_special_tokens=True,
        clean_up_tokenization_spaces=False,
    )

    print("\n" + "=" * 40)
    print("              OUTPUT")
    print("=" * 40)
    print(result)
    print("=" * 40)


if __name__ == "__main__":
    main()
