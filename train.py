import os
from datetime import datetime
import numpy as np

import torch
import datasets
from transformers import PreTrainedTokenizerFast
from torch.optim.lr_scheduler import ReduceLROnPlateau
from torch.optim.lr_scheduler import LinearLR
from torch.utils.data import DataLoader
import torch.nn.utils as nn_utils
from accelerate import Accelerator
from evaluation.llm_evaluation_logic import evaluate_model_logic
from evaluation.llm_evaluation_stories import evaluate_model_stories
# from curriculum_learning.split_curriculum import split_curriculum

from tqdm.auto import tqdm
import wandb

from config import batch_size, block_size, max_iters, eval_interval, eval_iters, learning_rate, device, wandb_log, \
    wandb_project, wandb_run_name, config, train_data, val_data, tokenizer, vocab_size, num_epochs, gradient_accumulation_steps, \
    max_grad_norm
from transformer_model import LanguageModel

# to use gpu/device, data and model params has to be moved to the device
print(f'Using device : {device}')


# Encode Data with Tokenizer
def encode_batch(data):
    encoded = tokenizer(data["text"], padding=True, truncation=True, max_length=block_size)  # , return_tensors="pt")
    return encoded


print(f"Loaded Tokenizer with size: {vocab_size}")

# for testing
# train_data = train_data.select(range(2000))
# val_data = val_data.select(range(1000))

# train_data = train_data.map(encode_batch, batched=True)
val_data = val_data.map(encode_batch, batched=True)

# print(train_data[0])

# partition_size = len(train_data) // 3

dataset_path = os.path.join(os.path.dirname(os.path.realpath(__file__)), 'curriculum_learning', 'split_dataset')
dataset_curriculum = datasets.load_from_disk(dataset_path)

high_train_data = dataset_curriculum["high_data"]
high_train_data = high_train_data.map(encode_batch, batched = True)
mid_train_data = dataset_curriculum["mid_data"]
mid_train_data = mid_train_data.map(encode_batch, batched = True)
low_train_data = dataset_curriculum["low_data"]
low_train_data = low_train_data.map(encode_batch, batched = True)
# print(f"Data split: {type(high_train_data)}, {len(high_train_data)}")

# high_train_data = train_data[:partition_size]
# mid_train_data = train_data[partition_size:2*partition_size]
# low_train_data = train_data[2*partition_size:len(train_data)]



# mask_data = train_data["attention_mask"]
high_mask_data = high_train_data["attention_mask"]
mid_mask_data = mid_train_data["attention_mask"]
low_mask_data = low_train_data["attention_mask"]
val_mask_data = val_data["attention_mask"]
# train_data = train_data["input_ids"]
high_train_data = high_train_data["input_ids"]
mid_train_data = mid_train_data["input_ids"]
low_train_data = low_train_data["input_ids"]
val_data = val_data["input_ids"]




# Create Pytorch Dataset
class TinyDataset_Preprocessed(torch.utils.data.Dataset):
    def __init__(self, data, tokenizer, mask):
        self.data = data
        self.tokenizer = tokenizer
        self.mask = mask

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        row = self.data[idx]
        mask_row = self.mask[idx]
        input = row
        label = row[1:] + [self.tokenizer.pad_token_id]
        return {'input': torch.tensor(input), 'mask': torch.tensor(mask_row), 'label': torch.tensor(label)}

    def collate_fn(self, batch):
        input_pad = torch.nn.utils.rnn.pad_sequence([item['input'] for item in batch], batch_first=True,
                                                    padding_value=0)
        label_pad = torch.nn.utils.rnn.pad_sequence([item['label'] for item in batch], batch_first=True,
                                                    padding_value=0)
        return {'input': input_pad, 'label': label_pad}


# train_dataset = TinyDataset_Preprocessed(train_data, tokenizer, mask_data)
high_train_dataset = TinyDataset_Preprocessed(high_train_data, tokenizer, high_mask_data)
mid_train_dataset = TinyDataset_Preprocessed(mid_train_data, tokenizer, mid_mask_data)
low_train_dataset = TinyDataset_Preprocessed(low_train_data, tokenizer, low_mask_data)
print('Loaded Pytorch train_dataset with length:', len(high_train_dataset))
# print('Check first input-label pair:', train_dataset[0])

val_dataset = TinyDataset_Preprocessed(val_data, tokenizer, val_mask_data)
print('Loaded Pytorch val_dataset with length:', len(val_dataset))
# print('Check first input-label pair:', val_dataset[0])

# Create DataLoaders
high_train_loader = DataLoader(high_train_dataset, batch_size=batch_size, shuffle=False)
mid_train_loader = DataLoader(mid_train_dataset, batch_size = batch_size, shuffle = False)
low_train_loader = DataLoader(low_train_dataset, batch_size = batch_size, shuffle = False)
loader_curriculum = [high_train_loader, mid_train_loader, low_train_loader]
val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)

""" 
Traing Loop
----------------------------------------------------------------
"""

# average out loss over multiple batches
# because every single batch individually will be more or less lucky
# iterate eval_iter times and average out the loss
# for both train and val, this will be lot less noisy
@torch.no_grad()
def estimate_loss(partition):
    out = {}
    model.eval()

    for split in ['train', 'val']:
        losses = torch.zeros(eval_iters)
        if split == 'train':
            loader = loader_curriculum[partition]
        else:
             loader = val_loader
        for k, batch in enumerate(loader):
            if k > (eval_iters - 1):
                break
            X = batch['input']
            Y = batch['label']
            X, Y = X.to(device), Y.to(device)
            attention_mask = (X != 0).unsqueeze(1).expand(-1, X.size(1), -1)  # Shape (B, T, T)
            attention_mask = attention_mask.to(device)
            logits, loss = model(X, attention_mask, Y)
            losses[k] = loss.item()
        out[split] = losses.mean()
    model.train()
    return out


# Initialize model and move to device
model = LanguageModel()
m = model.to(device)

# create a PyTorch optimizer with LR scheduler
optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate)
# scheduler = ReduceLROnPlateau(optimizer, 'min', patience=5, cooldown=5, verbose = True)
scheduler = LinearLR(optimizer, start_factor=1.0, end_factor=0.5, total_iters=30)

accelerator = Accelerator(gradient_accumulation_steps)

model, optimizer, high_train_loader, mid_train_loader, low_train_loader = (
    accelerator.prepare(model, optimizer, high_train_loader, mid_train_loader, low_train_loader))

# create wandb run
if wandb_log:
    wandb.init(project=wandb_project, name=wandb_run_name, config=config)

try:
    partition = 0
    for epoch in range(num_epochs):
        if epoch == 5:
            partition = 1
            print(f"Switching to partition {partition}")
        if epoch == 10:
            partition = 2
            print(f"Switching to partition {partition}")
        print("Currently on epoch number", epoch, "out of", num_epochs)
        # progress bar via tqdm
        num_training_steps = len(loader_curriculum[partition])
        # progress_bar = tqdm(range(num_training_steps))
        # for iteration, batch in enumerate(train_loader):
        for iteration, batch in enumerate(loader_curriculum[partition]):
            with accelerator.accumulate(model):
                # every once in a while evaluate the loss on train and val sets
                # interesting that we're not printing loss every iter
                # instead we're estimating the non noisy loss every eval_intervar
                # only for printing purposes


                # sample a batch of data
                xb = batch["input"]
                yb = batch["label"]
                xb, yb = xb.to(device), yb.to(device)

                    # Create the attention mask
                attention_mask = (xb != 0).unsqueeze(1).expand(-1, xb.size(1), -1)  # Shape (B, T, T)
                attention_mask = attention_mask.to(device)

                # evaluate the loss
                logits, loss = model(xb, attention_mask, yb)

                accelerator.backward(loss)

                nn_utils.clip_grad_norm_(model.parameters(), max_grad_norm)

                if (iteration % eval_interval == 0) or (iteration == max_iters - 1):
                    losses = estimate_loss(partition)
                    print(f"step {iteration}: train loss {losses['train']:.4f}, val loss {losses['val']:.4f}")

                    total_norm = 0.0
                    sum_of_squares = 0.0
                    max_norm = 0.0
                    num_params = 0
                    for name, param in model.named_parameters():
                        if param.requires_grad:
                            if param.grad is not None:
                                norm = param.grad.norm().item()
                                total_norm += norm
                                sum_of_squares += norm ** 2
                                if max_norm < norm:
                                    max_norm = norm
                                num_params += 1
                    if num_params > 0:
                        average_norm = total_norm / num_params
                        rms_norm = sum_of_squares / num_params
                    else:
                        average_norm = 0
                        rms_norm = 0

                    scheduler.step(losses['train'])

                    # logging
                    if wandb_log:
                        wandb.log({
                            "iter": iteration,
                            "train/loss": losses['train'],
                            "val/loss": losses['val'],
                            "average_norm": average_norm,
                            "rms_norm": rms_norm,
                            "max_norm": max_norm,
                            "lr": scheduler.state_dict()['_last_lr'][0],
                            "dataset": len(loader_curriculum[partition])
                        })

                    # print(f"Learning rate: {scheduler.state_dict()['_last_lr'][0]}")
                    # print(f"Scheduler state dict: {scheduler.state_dict().items()}")
                    # print(f"Learning rate: {(scheduler._last_lr())[0]}")

                optimizer.step()
                # scheduler.step()
                optimizer.zero_grad(set_to_none=True)
                # progress_bar.update(1)

except KeyboardInterrupt:
    pass

""" 
----------------------------------------------------------------
"""

# Training done
print("Please wait, genereating sample output with model... (this might take a while)")

# current time
today = datetime.now().strftime('%Y-%m-%d-%H:%M')

# Save the model and optimizer state dictionaries
torch.save({
    'model_state_dict': model.state_dict(),
    'optimizer_state_dict': optimizer.state_dict(),
    'epoch': iteration,
    'loss': loss.item(),
},
    (os.path.join(os.path.dirname(os.path.realpath(__file__)), 'output', 'model_checkpoint_' + today + '.pth')))

# Generate from the model
context = torch.zeros((1, 1), dtype=torch.long, device=device)
output = tokenizer.decode(m.generate(context, max_new_tokens=500)[0].tolist())
print(output)

# Write output to file
disclaimer = """
OUTPUT FROM MODEL:
"""
with open((os.path.join(os.path.dirname(os.path.realpath(__file__)), 'output', 'Tinystories_' + today + '.txt')), 'w',
          encoding='utf-8') as f:
    f.write(disclaimer)
    f.write("\n")
    f.write(output)

if wandb_log:
    stories_score = evaluate_model_stories(model = model)
    logic_score = evaluate_model_logic(model = model)
    wandb.log({"stories_eval": stories_score,
               "logic_eval": logic_score})