# hyperparameters
import os
import time

import datasets
import torch
from transformers import PreTrainedTokenizerFast

batch_size = 32  # how many independent sequences will we process in parallel?
block_size = 256  # what is the maximum context length for predictions?
max_iters = 2000000
num_epochs = 20
eval_interval = 500
eval_iters = 50 # was: 200
learning_rate = 3e-4 # was: 3e-4
device = 'cuda' if torch.cuda.is_available() else 'cpu'
n_embed = 64 # has to be divisible(without rem) by n_head, given head_size definition further below
n_head = 8
n_layer = 7
dropout = 0.3
max_grad_norm = 1.0
# wandb logging
wandb_log = True # disabled by default
wandb_project = 'TiniestStoryteller'
wandb_run_name = 'run' + str(time.time())
gradient_accumulation_steps = 2
# config
config_keys = [k for k,v in globals().items() if not k.startswith('_') and isinstance(v, (int, float, bool, str))]
config = {k: globals()[k] for k in config_keys} # for logging

current_dir = os.path.dirname(os.path.realpath(__file__))

dataset_path = os.path.join(current_dir, 'preprocessing', 'preprocessed_dataset')
dataset = datasets.load_from_disk(dataset_path)
train_data = dataset['train']
val_data = dataset['validation']
print("Loaded dataset from disk")

# Smaller Dataset for testing
# train_data = train_data.select(range(200000))

tokenizer_path = os.path.join(current_dir, 'tokenizers', 'bpe_tokenizer_4096.json')
tokenizer = PreTrainedTokenizerFast(
    tokenizer_file = tokenizer_path,
    bos_token = "<|endoftext|>",
    eos_token = "<|endoftext|>"
)

tokenizer.add_special_tokens({'pad_token': '[PAD]'})
                              
vocab_size = len(tokenizer)
print(f"Loaded Tokenizer: {tokenizer}")

