from datasets import load_dataset, Dataset


def prepare_dataset(
    dataset_name: str,
    split: str = "train",
    num_samples: int | None = None,
    instruction: str = "",
    image_column: str = "image",
    text_column: str = "text",
) -> Dataset:
    dataset = load_dataset(dataset_name, split=split)

    if num_samples is not None:
        dataset = dataset.select(range(min(num_samples, len(dataset))))

    def convert_to_conversation(sample):
        return {
            "messages": [
                {
                    "role": "user",
                    "content": [
                        {"type": "text", "text": instruction},
                        {"type": "image"},
                    ],
                },
                {
                    "role": "assistant",
                    "content": [
                        {"type": "text", "text": sample[text_column]},
                    ],
                },
            ],
            "images": [sample[image_column]],
        }

    dataset = dataset.map(convert_to_conversation, remove_columns=dataset.column_names)
    return dataset
