import datasets
from ollama import chat
import numpy as np
import time

dataset = datasets.load_dataset("roneneldan/TinyStories", split="train")

# used for testing
dataset = dataset.select(range(100))

prompt_template = "Please evaluate the following story based on the following six categories: Grammar, Creativity, Consistency, Originality, Vocabulary, and Overall Enjoyment. Each category should be scored with an integer between 1 and 10, where 1 is the lowest score and 10 is the highest. The output should be in the format of six numbers separated by colons, for example, '5:8:3:9:10:4'. Output the scores only in the following format: <Grammar score>:<Creativity score>:<Consistency score>:<Originality score>:<Vocabulary score>:<Overall Enjoyment score>. Output nothing but the scores, no additional text or explanation. Story: {story}"
prompt = """Please evaluate the following stories based on the following five categories:
Grammar, Creativity, Consistency, Originality, and Vocabulary.
Each category should be scored with an integer beween 1 and 10,
where 1 is the lowest score and 10 is the highest.
The first story begins after the first ###, then each new story is introduced with another ###

The output for each story should be in the format of five numbers separated by colons,
for example, '5:8:3:9:10'. Concatenate the scores for each story with a '---', so for example 
for two stories output 9:4:6:7:1---10:5:7:2:4.
Output the scores only in the following format:
<Grammar score>:<Creativity score>:<Consistency score>:<Originality score>:<Vocabularity score>.

Output nothing but the scores, no additional text or explanation. 

{input}
"""

def parse_ollama_output(output):
    score_strings = output.split(":")
    scores = [int(score) for score in score_strings]
    return scores


def ask_llm_batch(stories):
    batch_prompt = prompt.format(input = "###\n" + "\n###\n".join(stories) + "\n###")

    # print(batch_prompt)
    msgs = [
        {"role": "system", "content": batch_prompt}
    ]
    output = chat(model = "llama3", messages = msgs)
    return output

def ask_llm_batch_(stories):
    batch_prompt = "\n\n".join([prompt_template.format(story=story) for story in stories])
    print(batch_prompt)
    msgs = [
        {"role": "system", "content": batch_prompt}
    ]
    output = chat(model="llama3", messages=msgs)
    return output['message']['content'].split('\n')

def score_stories_batch(stories):
    ollama_output = ask_llm_batch(stories)
    individual_responses = ollama_output["message"]["content"].split("---")
    scores = 0

def score_stories_batch_(stories):
    ollama_outputs = ask_llm_batch(stories)
    scores = [parse_ollama_output(output) for output in ollama_outputs]
    averages = [np.average(score) for score in scores]
    return averages

def filter_stories_batch(dataset, min_score, batch_size):
    filtered_stories = []
    num_samples = len(dataset)
    for i in range(0, num_samples, batch_size):
        print(f"i: {i}")
        batch = dataset.select(range(i, min(i + batch_size, num_samples)))
        batch_texts = [story['text'] for story in batch]
        output = ask_llm_batch(batch_texts)
        print(output["message"]["content"])
        print("-----------------------------------------------------")
    #     scores = score_stories_batch(batch_texts)
    #     for story, score in zip(batch, scores):
    #         if score > min_score:
    #             filtered_stories.append(story)
    # return filtered_stories

min_score = 3
batch_size = 40 # Adjust batch size based on API limits and performance

_start_time = time.time()

#hq_stories = filter_stories_batch(dataset, min_score, batch_size)
filter_stories_batch(dataset, min_score, batch_size)

_end_time = time.time()

print(f"Elapsed time: {_end_time - _start_time}")

# print(hq_stories)
