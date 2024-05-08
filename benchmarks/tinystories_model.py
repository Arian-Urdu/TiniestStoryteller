from transformers import AutoModelForCausalLM, AutoTokenizer, GenerationConfig

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

def use_tinyStories_model(model_name, prompt):

    model = AutoModelForCausalLM.from_pretrained('roneneldan/'+ model_name)
    tokenizer = AutoTokenizer.from_pretrained("EleutherAI/gpt-neo-125M")

    # prompt = "Once upon a time there was"

    input_ids = tokenizer.encode(prompt, return_tensors="pt")

    output = model.generate(input_ids, max_length = 1000, num_beams=1)
    output_text = tokenizer.decode(output[0], skip_special_tokens=True)

    return output_text