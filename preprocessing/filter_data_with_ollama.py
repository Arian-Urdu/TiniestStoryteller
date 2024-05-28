import datasets
from ollama import chat
import numpy as np
import time

dataset = datasets.load_dataset("roneneldan/TinyStories", split="train")

# used for testing
dataset = dataset.select(range(2))


prompt = "Please evaluate the following story based on the following six categories: Grammar, Creativity, Consistency, Originality, Vocabulary, and Overall Enjoyment. Each category should be scored with an integer between 1 and 10, where 1 is the lowest score and 10 is the highest. The output should be in the format of six numbers separated by colons, for example, '5:8:3:9:10:4'. Output the scores only in the following format: <Grammar score>:<Creativity score>:<Consistency score>:<Originality score>:<Vocabulary score>:<Overall Enjoyment score>. Output nothing but the scores, no additional text or explanation."


def parse_ollama_output(output):
    score_strings = output.split(":")
    scores = [int(score) for score in score_strings]
    return scores

def ask_llm(system_prompt, datapoint):
    msgs = [
        {"role": "system", "content": system_prompt},
        { "role": "user", "content": datapoint },
    ]
    output = chat(model="llama3", messages=msgs)
    return(output['message']['content'])

def score_story(story):
    ollama_output = ask_llm(prompt, story)
    scores = parse_ollama_output(ollama_output)
    return np.average(scores)

# for i in range(0, len(dataset['text'])):
#     ollama_output = ask_llm(prompt, dataset['text'][i])
#     scores = parse_ollama_output(ollama_output)
#     print(scores)



min_score = 3

_start_time = time.time()

hq_stories = dataset.filter(lambda story: score_story(story['text']) > min_score)

_end_time = time.time()

print(f"Elapsed time: {_end_time - _start_time}")

print(hq_stories)