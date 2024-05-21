# from used_letters import forbidden_letters
# from used_letters import process_string
import datasets
import re
import time

forbidden_letters = {'ê', '_', 'ò', '²', 'š', 'ž', 'ª', 'á', 'ÿ', 'ï', '½', 'ä', 'æ', 'ˆ', 'ã', 'ñ', 'º', '³', '¾', '¼', 'é', 'â', 'œ', 'µ', 'î', 'è', 'å', 'ç', '¹', 'ð'}

def process_string(text):
    text = text.lower()
    text = re.sub(r"n't", " not", text)
    text = re.sub(r"'(m|re|s|d|ll|ve)", r" '\1", text)

    # Remove punctuation marks except apostrophes
    text = re.sub(r'[^\w\s\']', '', text)

    return text


dataset = datasets.load_dataset("roneneldan/TinyStories", split="train")
# dataset = dataset.select(range(10000))

# list all indices of stories which contain a forbidden letter
forbidden_indices = []

start_time = time.time()

for i in range(0, len(dataset['text'])):
    if i == 100:
        end_time = time.time()
    processed_story = process_string(dataset['text'][i])
    if any(letter in processed_story for letter in forbidden_letters):
        forbidden_indices.append(i)

print(f"Elapsed time: {end_time - start_time}s")

print(len(forbidden_indices))
print(forbidden_indices)

forbidden_indices_as_string = str(forbidden_indices)

f = open("forbidden_indices.txt", "w")
f.write(forbidden_indices_as_string)
f.close()
