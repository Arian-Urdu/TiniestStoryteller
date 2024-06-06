import datasets
from ollama import chat
import numpy as np
import time

dataset = datasets.load_dataset("roneneldan/TinyStories", split="train")

# used for testing
dataset = dataset.select(range(20))

prompt_template = "Please evaluate the following story based on the following five categories: Grammar, Creativity, Consistency, Originality, and Vocabulary. Each category should be scored with an integer between 1 and 10, where 1 is the lowest score and 10 is the highest. The output should be in the format of six numbers separated by colons, for example, '5:8:3:10:4'. Output the scores only in the following format: <Grammar score>:<Creativity score>:<Consistency score>:<Originality score>:<Vocabulary score>:<Overall Enjoyment score>. Output the numbers and nothing but the numbers, absolutely no additional text or explanation. Story: {story}"

def parse_ollama_output(output):
    score_strings = output.split(":")
    print(f"Score strings: {score_strings}")
    scores = [int(score) for score in score_strings]
    return scores

def ask_llm_batch(stories):
    batch_prompt = "\n\n".join([prompt_template.format(story=story) for story in stories])
    msgs = [
        {"role": "system", "content": batch_prompt}
    ]
    output = chat(model="llama3", messages=msgs)
    return output['message']['content']#.split('\n')

def score_stories_batch(stories, i):
    ollama_outputs = ask_llm_batch(stories)
    print(i, ollama_outputs)
    scores = [parse_ollama_output(output) for output in ollama_outputs]
    averages = [np.average(score) for score in scores]
    return averages

def filter_stories_batch(dataset, min_score, batch_size):
    filtered_stories = []
    num_samples = len(dataset)
    for i in range(0, num_samples, batch_size):
        batch = dataset.select(range(i, min(i + batch_size, num_samples)))
        batch_texts = [story['text'] for story in batch]
        # print(i, batch_texts)
        scores = score_stories_batch(batch_texts, i)
        # print(i, scores)
        for story, score in zip(batch, scores):
            if score > min_score:
                filtered_stories.append(story)
    return filtered_stories

min_score = 2
batch_size = 10 # Adjust batch size based on API limits and performance

_start_time = time.time()

hq_stories = filter_stories_batch(dataset, min_score, batch_size)

_end_time = time.time()

# print(f"Elapsed time: {_end_time - _start_time}")

# print(len(dataset))
# print(len(hq_stories))
