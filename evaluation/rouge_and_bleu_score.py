import random
import torch
from config import tokenizer, device
import evaluate

rouge = evaluate.load("rouge")
bleu = evaluate.load("bleu")

def get_preds_and_refs(model, data):
    all_predictions = []
    all_references = []
    for _ in range(100):
        ref_idx = random.randint(0, len(data) - 1)
        references = tokenizer.decode(data[ref_idx])
        prompt = ' '.join(references.split()[0:3])
        input_ids = tokenizer.encode(prompt, return_tensors = 'pt').to(device)
        with torch.no_grad():
            predicted_ids = model.generate(input_ids, max_new_tokens = len(references.split()) - 3)
        predictions = tokenizer.decode(predicted_ids[0], skip_special_tokens = True)
        all_predictions.append(predictions)
        all_references.append(references)
    return all_predictions, all_references

def score_rouge(model, data):
    predictions, references = get_preds_and_refs(model, data)
    score = rouge.compute(predictions = predictions, references = references)["rougeL"]
    return score

def score_bleu(model, data):
    predictions, references = get_preds_and_refs(model, data)
    score = bleu.compute(predictions = predictions, references = references)["bleu"]
    return score


