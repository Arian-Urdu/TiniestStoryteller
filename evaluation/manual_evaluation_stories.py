import random
import os

import torch

try:
    from config import tokenizer, device
except ImportError:
    import sys
    sys.path.append(sys.path[0] + '/..')
    from config import tokenizer, device

from evaluation.load_checkpoint import load_model
import numpy as np

number_prompts = 10

print("Starting evaluation\n"
      "-------------------------------\n"
      "See the prompt and give the generated story a score from 1 to 3.\n\n")

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

def evaluate_model_stories_manual(model = None, modelpath = ""):
    if model == None:
        if modelpath == "":
            raise Exception("Invalid arguments")
        model = load_model(modelpath)

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

        # let operator score output
        score = input(f"Prompt:\n{prompt}...\n\n"
                      f"Model response:\n...{model_response}\n\n"
                      f"Your score: ")
        if float(score) < 1.0 or float(score) > 3:
            raise ValueError(f"Score must be between 1 and 3. Got {score}")
        scores.append(float(score))
        print("\n-------------------------------\n")

    return np.mean(scores)


modelpath = "model_checkpoint_2024-07-07-09:25.pth"
evaluate_model_stories_manual(modelpath)