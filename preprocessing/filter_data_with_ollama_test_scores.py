import datasets
from ollama import chat
import numpy as np
import time
from datasets import Dataset
from preprocessing.used_letters import forbidden_letters
from preprocessing.remove_forbidden_letters import filter_dataset

# Load dataset
dataset = datasets.load_dataset("roneneldan/TinyStories", split = "train")
print(dataset)
print(len(dataset))

def is_not_empty(story):
    return bool(story["text"].strip())

dataset = dataset.filter(is_not_empty)

print(len(dataset))

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
        "â€": '"'
    }
    for encoding, character in encodings.items():
        text = text.replace(encoding, character)
    return text

unique_samples = set()
for sample in dataset["text"]:
    story = fix_encoding(sample)
    unique_samples.add(story)


dataset = Dataset.from_dict({"text": list(unique_samples)})
print(len(dataset))
# dataset = dataset.select(range(40))

# forbidden_words = set()
# for story in dataset["text"]:
#     # story = story.lower()
#     words = story.split()
#     for i in range(0, len(words)):
#         for char in forbidden_letters:
#             if char in words[i]:
#                 try:
#                     print(words[i - 1], words[i], words[i + 1])
#                 except:
#                     print(words[i])
#                 break

dataset = filter_dataset(dataset)

# Use a subset for testing
# dataset = dataset.select(range(100))

# Prompt template
base_prompt = (
    "Please evaluate the following stories based on the following five categories: Grammar, Creativity, "
    "Consistency, Originality, and Vocabulary. Each category should be scored with an integer "
    "between 1 and 10, where 1 is the lowest score and 10 is the highest. The output for each story should be in the "
    "format of five numbers separated by colons, for example, '5:8:3:10:4'. Concatenate the scores for each story with "
    "a '---', so for example for two stories output 9:4:6:7:1---10:5:7:2:4. Output the scores only in the following "
    "format: <Grammar score>:<Creativity score>:<Consistency score>:<Originality score>:<Vocabulary score>."
    "Output the scores as numbers and nothing but that, absolutely no additional text or explanation. It is very "
    "important that you output NUMBERS ONLY and in the desired format. DO NOT OUTPUT TEXT! "
    "I'm giving you {batch_size} stories in this text and the prompt will repeat for each. "
    "Please print {batch_size} scores (NOT ONE).\n\n{story}\n------------------------------------"
)

def parse_ollama_output(output):
    score_strings = output.split(":")
    scores = [int(score) for score in score_strings]
    return scores

# def test():
#     prompt = "Hello, how are you?"
#     msgs = {"role": "system", "content": prompt}
#     output = chat(model = "llama3", messages = msgs)
#     print(output)
#
# test()

def ask_llm_batch(stories):
    # base_prompt.format(batch_size = len(stories))
    # batch_prompt = base_prompt + "\n".join([f"### Story {i + 1}\n{story}" for i, story in enumerate(stories)])
    batch_prompt = "\n\n".join([base_prompt.format(batch_size = len(stories), story = f"### Story {i + 1}\n" + story) for i, story in enumerate(stories)])
    # print(f"Batch prompt:\n{batch_prompt}\n")  # Debug statement to check batch prompt
    msgs = [
        {"role": "system", "content": batch_prompt}
    ]
    output = chat(model="llama3", messages=msgs)
    print(f"LLM response:\n{output['message']['content']}\n")  # Debug statement to check LLM response
    return output['message']['content'].split('---')

def score_stories_batch(stories):
    ollama_outputs = ask_llm_batch(stories)
    scores = [parse_ollama_output(output) for output in ollama_outputs if output.strip()]
    # print(f"Scores: {scores}")
    return scores

def filter_stories_batch(dataset, min_score, batch_size):
    # filtered_stories = []
    # all_scores = []
    indices = []
    num_samples = len(dataset)
    for i in range(0, num_samples, batch_size):
        start_time = time.time()
        batch = dataset.select(range(i, min(i + batch_size, num_samples)))
        batch_texts = [story['text'] for story in batch]
        scores = score_stories_batch(batch_texts)
        if len(scores) != batch_size:
            print("Error in ollama output")
            return
        for j, (story, score) in enumerate(zip(batch, scores)):
            # all_scores.append((story, score))  # Store both story and score
            if np.average(score) > min_score:
                # filtered_stories.append((story, score))
                # print(i + j)
                indices.append(i + j)
        end_time = time.time()
        print(f"Time required for batch {i}: {round(end_time - start_time, 3)}s")
    return indices

min_score = 3
batch_size = 2  # Adjust batch size based on API limits and performance

_start_time = time.time()

# hq_stories, all_scores = filter_stories_batch(dataset, min_score, batch_size)
indices = filter_stories_batch(dataset, min_score, batch_size)

_end_time = time.time()

# print(f"Elapsed time: {_end_time - _start_time}")

with open("forbidden_indices.txt", 'w') as f:
    f.write(','.join(map(str, indices)))
    f.write(f"Elapsed time: {_end_time - _start_time}")

# # Print high-quality stories with their scores
# print("High Quality Stories:")
# for story, score in hq_stories:
#     print(f"Story: {story['text']}\nScore: {score}\n")
#
# # Print scores for each story for verification
# print("Scores for each story in high-quality stories:")
# for story, score in hq_stories:
#     print(f"Story: {story['text']}\nScore: {score}\n")
#
# # Verify the number of stories processed
# print(f"Total number of stories processed: {len(all_scores)}")
# print(f"Expected number of stories: {len(dataset)}")

