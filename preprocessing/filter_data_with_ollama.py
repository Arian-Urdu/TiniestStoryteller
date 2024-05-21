from ask_llm import ask_llm
import datasets
import ollama
import numpy as np

dataset = datasets.load_dataset("roneneldan/TinyStories", split="train")

# used for testing
dataset = dataset.select(range(10))


prompt = "Please evaluate the following story based on the following six categories: Grammar, Creativity, Consistency, Originality, Vocabulary, and Overall Enjoyment. Each category should be scored with an integer between 1 and 10, where 1 is the lowest score and 10 is the highest. The output should be in the format of six numbers separated by colons, for example, '5:8:3:9:10:4'. Output the scores only in the following format: <Grammar score>:<Creativity score>:<Consistency score>:<Originality score>:<Vocabulary score>:<Overall Enjoyment score>"


def parse_ollama_output(output):
    score_strings = output.split(":")
    scores = [int(score) for score in score_strings]
    return scores

def ask_llm(system_prompt, datapoint):
    msgs = [
        {"role": "system", "content": system_prompt},
        { "role": "user", "content": datapoint },
    ]
    output = ollama.chat(model="llama3", messages=msgs )

    return(output['message']['content'])

for i in range(0, len(dataset['text'])):
    ollama_output = ask_llm(prompt, dataset['text'][i])
    scores = parse_ollama_output(ollama_output)
    print(scores)


# for later:
# min_score = 3
# low_scoring_indices = []
#     if np.average(scores) < min_score:
#         low_scoring_indices.append(i)
#
