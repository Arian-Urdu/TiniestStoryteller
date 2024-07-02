from openai import OpenAI
import numpy as np
import os
try:
    from config import train_data
except ImportError:
    import sys
    sys.path.append(sys.path[0] + '/..')
    from config import train_data

client = OpenAI(
    base_url="http://localhost:11434/v1",
    api_key = "ollama",
)


file_path = os.path.join(os.path.dirname(os.path.realpath(__file__)), "curriculum.txt")

system_prompt = ("Please evaluate the grammar, syntax, and consistency of the given story. "
                 "Output separate scores for each category, all being between 1 and 5 where 1 is bad and 5 is good. "
                 "Focus on each category in part and disregard all mistakes that donot relate to it when evaluating "
                 "each particular score. The final scores should be output in the format of three integers separated "
                 "by colons, for example '2:4:3' in the order <grammar score>:<syntax score>:<consistency score>. "
                 "Output nothing but the scores, no additional text or explanation.")


ratings = {}

def parse_ollama_output(output):
    score_strings = output.split(":")
    scores = [float(score) for score in score_strings]
    return scores


for i, story in enumerate(train_data["text"]):
    msgs = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": story},
        ]
    llm_response = client.chat.completions.create(model="llama3", messages=msgs).choices[0].message.content
    score = np.mean(parse_ollama_output(llm_response))
    ratings[i] = score
    if i % 1000 == 0:
        print(i)