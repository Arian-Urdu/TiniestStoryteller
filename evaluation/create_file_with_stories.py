import torch
import os
from load_checkpoint import model
from config import tokenizer, device, dataset
import random

eos_token_id = tokenizer.eos_token_id
pad_token_id = tokenizer.pad_token_id

file_path = os.path.join(os.path.dirname(os.path.realpath(__file__)), "stories_for_system_prompt_selection.txt")

random_indices = random.sample(range(len(dataset["validation"]["text"])), 50)

stories = [dataset["validation"]["text"][i] for i in random_indices]

def truncate_story(story):
    num_words = random.randint(1, 3)
    words = story.split()
    truncated_story = " ".join(words[:num_words])
    return truncated_story

prompts = [truncate_story(story) for story in stories]

for prompt in prompts:
    # let model generate
    input_ids = tokenizer.encode(prompt, return_tensors='pt').to(device)
    with torch.no_grad():
        model_response = model.generate(input_ids, max_new_tokens=300)
    model_response_ids = model_response[0].tolist()

    if eos_token_id in model_response_ids:
        eos_index = model_response_ids.index(eos_token_id)
        # truncate at end of sequence token
        model_response_ids = model_response_ids[:eos_index]

    if pad_token_id in model_response_ids:
        pad_index = model_response_ids.index(pad_token_id)
        # truncate at pad token
        model_response_ids = model_response_ids[:pad_index]

    model_response_decoded = tokenizer.decode(model_response_ids)

    # remove prompt from response
    model_response_decoded = model_response_decoded[len(prompt) + 1:]

    with open(file_path, 'a') as file:
        file.write(f"{prompt} --- {model_response_decoded}\n")
        file.write("---------------------------------------------------\n")