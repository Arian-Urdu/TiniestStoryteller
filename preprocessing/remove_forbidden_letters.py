from preprocessing.used_letters import forbidden_letters
import datasets
import time

dataset = datasets.load_dataset("roneneldan/TinyStories", split="train")



start_time = time.time()

filtered_dataset = dataset.filter(lambda story: not any(char in story['text'].lower() for char in forbidden_letters))

end_time = time.time()

print(f"Elapsed time: {end_time - start_time}s")

print(len(filtered_dataset))
print(filtered_dataset)
print(len(dataset['text']))
