import torch
try:
    from evaluation.load_checkpoint import load_model
    from config import tokenizer, device
    from evaluation.llm_evaluation_stories import cut_after_last_full_stop
except ImportError:
    import sys
    sys.path.append(sys.path[0] + '/..')
    from evaluation.load_checkpoint import load_model
    from config import tokenizer, device
    from evaluation.llm_evaluation_stories import cut_after_last_full_stop


model = load_model("output\\model_checkpoint_2024-07-07-09-25.pth")

prompt = ""

while True:
    prompt = input("Prompt: ")
    if prompt == "\exit":
        break

    input_ids = tokenizer.encode(prompt, return_tensors='pt').to(device)
    with torch.no_grad():
        model_response = model.generate(input_ids, max_new_tokens = 100, temperature = 0.7)
    model_response = tokenizer.decode(model_response[0].tolist())

    # remove prompt from response
    model_response = model_response[len(prompt) + 1:]
    # remove leading whitespace
    if model_response[0] == " ":
        model_response = model_response[1:]
    # cut off after first full stop
    model_response = cut_after_last_full_stop(model_response)

    print(model_response)
