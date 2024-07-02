import os
import numpy as np
import torch

from evaluation.load_checkpoint import load_model
from config import tokenizer, device
import random

number_prompts = 10

print("Starting evaluation\n"
      "-------------------------------\n"
      "See the prompt and give the generated text a score from 1 to 3.\n"
      "An example response is provided but needn't be adhered to.\n\n")

scores = []


indices = random.sample(range(1, number_prompts + 1), number_prompts)

prompts_path = os.path.join(os.path.dirname(os.path.realpath(__file__)), "prompts_for_logic_eval.txt")
with open(prompts_path, 'r', encoding='utf-8') as f:
    prompts = f.read().split('\n')

def evaluate_model_logic_manual(model = None, modelpath = ""):
    if model == None:
        if modelpath == "":
            raise Exception("Invalid arguments")
        model = load_model(modelpath)

    for i in indices:
        # get random prompt from file
        prompt, example_response = prompts[i].split("---")

        # let model generate
        input_ids = tokenizer.encode(prompt, return_tensors='pt').to(device)
        with torch.no_grad():
            model_response = model.generate(input_ids, max_new_tokens=10)
        model_response = tokenizer.decode(model_response[0].tolist())

        # remove prompt from response
        model_response = model_response[len(prompt) + 1:]
        # cut off after first full stop
        model_response = model_response.split(".")[0] + "."

        # let operator score output
        score = input(f"Prompt:\n{prompt}\n\n"
                      f"Model response:\n{model_response}\n\n"
                      f"Example response:\n{example_response}\n\n"
                      f"Your score: ")
        if float(score) < 1.0 or float(score) > 10:
            raise ValueError(f"Score must be between 1.0 and 10.0. Got {score}")
        scores.append(float(score))
        print("\n-------------------------------\n")

    return np.mean(scores)
