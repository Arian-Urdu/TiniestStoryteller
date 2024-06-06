from preprocessing.used_letters import forbidden_letters
import datasets


def filter_dataset(dataset):
    filtered_dataset = dataset.filter(lambda story: not any(char in story['text'].lower() for char in forbidden_letters))
    return filtered_dataset
