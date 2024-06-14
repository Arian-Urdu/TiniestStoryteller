import os
import numpy as np

import torch
import datasets
from transformers import PreTrainedTokenizerFast
from torch.optim.lr_scheduler import ReduceLROnPlateau
from torch.utils.data import DataLoader

import wandb

from config import batch_size, block_size, max_iters, eval_interval, eval_iters, learning_rate, device, wandb_log, wandb_project, wandb_run_name, config
from transformer_model import LanguageModel

print(f'Using device : {device}')
# to use gpu/device, data and model params has to be moved to the device


dataset_path = os.path.join('preprocessing', 'preprocessed_dataset')
dataset = datasets.load_from_disk(dataset_path)
train_data = dataset['train']
val_data = dataset['validation']
print("Loaded dataset from disk")   

# Smaller Dataset for testing
train_data = train_data.select(range(1000))

tokenizer_path = os.path.join('tokenizers', 'bpe_tokenizer.json')
tokenizer = PreTrainedTokenizerFast(
    tokenizer_file = tokenizer_path,
    bos_token = "<|endoftext|>",
    eos_token = "<|endoftext|>"
)
tokenizer.add_special_tokens({'pad_token': '[PAD]'})

vocab_size = tokenizer.vocab_size
print(f"Loaded Tokenizer: {tokenizer} \n with size: {vocab_size}")

def encode_batch(data):
    encoded = tokenizer(data["text"], padding=False, truncation=True, max_length=block_size)#, return_tensors="pt")
    return encoded

train_data = train_data.map(encode_batch, batched = True)
val_data = val_data.map(encode_batch, batched = True)


# train_data = train_data.remove_columns("text")

# train_dataloader = DataLoader(train_data, shuffle=True, batch_size=batch_size)
# for batch in train_dataloader:
#    break
# print({k: np.shape(v) for k, v in batch.items()})
   
# Using Truncation so ignore token_type_ids and padding attention_mask
train_data = train_data["input_ids"]
val_data = val_data["input_ids"]



# flatten lists (wahrscheinlich unsauber)
train_data = [encoding for encoded_story in train_data for encoding in encoded_story]
val_data = [encoding for encoded_story in val_data for encoding in encoded_story]


train_data = torch.tensor(train_data, dtype = torch.long)
val_data = torch.tensor(val_data, dtype = torch.long)



def get_batch1(split):
    # generate a small batch of data of inputs x and targets y
    data = train_data if split == 'train' else val_data
    #ix = torch.randint(len(data) - block_size, (batch_size,))
    ix = torch.tensor((0,1))
    print(ix)
    x = torch.stack([data[i:i + block_size] for i in ix])
    y = torch.stack([data[i + 1:i + block_size + 1] for i in ix])
    x, y = x.to(device), y.to(device)
    return x, y
x, y = get_batch1("train")
print(x)
print(x.shape)
print(x.get_device())