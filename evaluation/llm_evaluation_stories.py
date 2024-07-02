import torch
from openai import OpenAI
import random
import os
try:
    from config import tokenizer, device
except ImportError:
    import sys
    sys.path.append(sys.path[0] + '/..')
    from config import tokenizer, device
from load_checkpoint import model
import numpy as np

client = OpenAI(
    base_url="http://localhost:11434/v1",
    api_key = "ollama",
)

system_prompt = ("Please evaluate the following story based on the following three categories: "
                 "Grammar, Spelling, and Consistency. "
                 "Each category should be scored with an integer between 1 and 3, 1 being the worst and 3 the best. "
                 "When rating a category, disregard all other errors except for those relating to that category. "
                 "The output should be in the format of three numbers separated by colons, for example, '2:1:3'. "
                 "Output the scores only in the following format: "
                 "<Grammar score>:<Spelling score>:<Consistency score>. "
                 "Output nothing but the scores, no additional text or explanation.")

number_prompts = 10

scores = []

indices = random.sample(range(1, number_prompts + 1), number_prompts)

prompts_path = os.path.join(os.path.dirname(os.path.realpath(__file__)), "prompts_for_story_eval.txt")
with open(prompts_path, 'r', encoding='utf-8') as f:
    prompts = f.read().split('\n')

def cut_after_last_full_stop(input_string):
    last_full_stop_index = input_string.rfind('.')

    if last_full_stop_index == -1:
        return input_string

    result_string = input_string[:last_full_stop_index + 1]
    return result_string

def parse_ollama_output(output):
    score_strings = output.split(":")
    scores = [float(score) for score in score_strings]
    return scores

for i in indices:
    # get random prompt from file
    prompt = prompts[i]

    # let model generate
    input_ids = tokenizer.encode(prompt, return_tensors='pt').to(device)
    with torch.no_grad():
        model_response = model.generate(input_ids, max_new_tokens=10)
    model_response = tokenizer.decode(model_response[0].tolist())

    # remove prompt from response
    model_response = model_response[len(prompt) + 1:]
    # cut off after first full stop
    model_response = cut_after_last_full_stop(model_response)

    # prepare prompt
    # print(f"Prompt: {system_prompt}")
    msgs = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": model_response},
    ]

    # let llm score output
    llm_response = client.chat.completions.create(model="llama3", messages=msgs).choices[0].message.content
    score = np.mean(parse_ollama_output(llm_response))
    if float(score) < 1.0 or float(score) > 10:
        raise ValueError(f"Score must be between 1.0 and 10.0. Got {score}")
    scores.append(float(score))
    print("\n-------------------------------\n")

print(f"Final score: {np.mean(scores)}")
