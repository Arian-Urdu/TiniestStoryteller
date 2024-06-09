import datasets
from datasets import Dataset, DatasetDict
from preprocessing.remove_forbidden_letters import filter_dataset




def is_not_empty(story):
    return bool(story["text"].strip())


def fix_encoding(text):
    encodings = {
        "â€˜": "'",
        "â€š": ",",
        "â€?": '"',
        "â€™": "'",
        "â€œ": '"',
        "â€“": "-",
        "â€”": "-",
        "â€¦": "...",
        "Ã©": "é",
        "Â´": "'",
        "Â’": "'",
        "Â·": "'",
        "Â£": "£",
        "Ã‰": "É",
        "â€": '"',
        "â€": '"' # not redundant as unicode characters are actually different
    }
    for encoding, character in encodings.items():
        text = text.replace(encoding, character)
    return text


def preprocess_split(split):
    # remove empty stories
    split = split.filter(is_not_empty)

    # dedupe and fix encodings
    unique_samples = set()
    for sample in split["text"]:
        story = fix_encoding(sample)
        unique_samples.add(story)
    split = Dataset.from_dict({"text": list(unique_samples)})

    # remove forbidden characters
    split = filter_dataset(split)

    return split

def preprocess_dataset(dataset):
    preprocessed_splits = {}
    for split in dataset.keys():
        split_data = dataset[split]
        print(f"Preprocessing split: {split}")
        preprocessed_split = preprocess_split(split_data)
        preprocessed_splits[split] = preprocessed_split
    preprocessed_dataset = DatasetDict(preprocessed_splits)
    return preprocessed_dataset

# Load dataset
dataset = datasets.load_dataset("roneneldan/TinyStories")
preprocessed_dataset = preprocess_dataset(dataset)


preprocessed_dataset.save_to_disk("preprocessed_dataset")
print("Process complete - preprocessed dataset saved to disk")