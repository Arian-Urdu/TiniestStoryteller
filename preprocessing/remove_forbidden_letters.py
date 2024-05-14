from used_letters import forbidden_letters
from used_letters import process_string
import datasets

dataset = datasets.load_dataset("roneneldan/TinyStories", split="train")

# list all indices of stories which contain a forbidden letter
forbidden_indices = []

for i in range(0, len(dataset['text'])):
    processed_story = process_string(dataset['text'][i])
    if any(letter in processed_story for letter in forbidden_letters):
        forbidden_indices.append(i)
