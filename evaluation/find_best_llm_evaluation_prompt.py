import os
from openai import OpenAI
import numpy as np

system_prompts_path = os.path.join(os.path.dirname(os.path.realpath(__file__)), "system_prompts.txt")
stories_path = os.path.join(os.path.dirname(os.path.realpath(__file__)), "stories_for_system_prompt_selection.txt")
manual_ratings_path = os.path.join(os.path.dirname(os.path.realpath(__file__)), "manual_ratings.txt")

client = OpenAI(
    base_url="http://localhost:11434/v1",
    api_key = "ollama",
)

def parse_output(output):
    score_strings = output.split(":")
    scores = [float(score) for score in score_strings]
    return scores

def squared_error(y_true, y_pred):
    error = 0
    for t, p in zip(y_true, y_pred):
        error += (t - p) ** 2
    return error ** 0.5

with open(system_prompts_path, 'r', encoding='utf-8') as f:
    system_prompts = f.read().split('\n\n')

with open(stories_path, 'r', encoding='utf-8') as f:
    stories = f.read().split('---------------------------------------------------\n')
    stories = [story.replace(" --- ", " ") for story in stories]

with open(manual_ratings_path, 'r', encoding='utf-8') as f:
    manual_ratings = f.read().split('\n')

errors = []
i = 0
for system_prompt in system_prompts:
    print(i)
    errors_per_prompt = []
    for story_index in range(len(stories)):
        print(f"   {story_index}")
        story = stories[story_index]
        msgs = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": story},
        ]

        # let llm score output
        # try:
        llm_response = client.chat.completions.create(model="llama3", messages=msgs).choices[0].message.content
        llm_score = parse_output(llm_response)
        # print(f"LLM score: {llm_score}")
        manual_score = parse_output(manual_ratings[story_index])
        # print(f"Manual score: {manual_score}")
        errors_per_prompt.append(squared_error(manual_score, llm_score))
        # except:
        #     continue
    # print(np.mean(errors_per_prompt))
    errors.append(np.mean(errors_per_prompt))
    i += 1


print(errors)