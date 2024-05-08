import yaml
from tinystories_model import use_tinyStories_model
from gpt_eval import gpt_eval

# Load the YAML file
with open("benchmarks/Evalutation_prompts.yaml", "r") as file:
    data = yaml.safe_load(file)

"""
All Models:
-----------
TinyStories-1M
TinyStories-3M
TinyStories-8M
TinyStories-1Layer-21M
TinyStories-28M
TinyStories-2Layers-33M
TinyStories-33M
"""

def benchmark(model):

    # Iterate over the list of strings
    for prompt in data:
        model_output = use_tinyStories_model(model, prompt)

        def add_separators(string_a, string_b):
            separator = "***"
            index = string_b.find(string_a) + len(string_a)
            return string_b[:index] + separator + string_b[index:]

        story_with_separators = add_separators(prompt,model_output)

        eval = gpt_eval(story_with_separators)
        print(eval)

        f = open(model + " benchmark.txt", "a")
        f.write(eval + '\n')  # Add a newline after each write
        
benchmark("TinyStories-3M")