import torch

from torch.optim.lr_scheduler import ReduceLROnPlateau

import wandb

from config import batch_size, block_size, max_iters, eval_interval, eval_iters, learning_rate, device, wandb_log, wandb_project, wandb_run_name, config, train_data, val_data, tokenizer, vocab_size
from transformer_model import LanguageModel

print(f'Using device : {device}')
# to use gpu/device, data and model params has to be moved to the device

print("Using Tokenizer with size:", vocab_size)

torch.manual_seed(1337)

def encode_batch(batch):
    encoded = tokenizer(batch["text"], padding=False, truncation=True, max_length=block_size)#, return_tensors="pt")
    return encoded

train_data = train_data.map(encode_batch, batched = True)
val_data = val_data.map(encode_batch, batched = True)


train_data = train_data["input_ids"]
val_data = val_data["input_ids"]

# flatten lists (wahrscheinlich unsauber)
train_data = [encoding for encoded_story in train_data for encoding in encoded_story]
val_data = [encoding for encoded_story in val_data for encoding in encoded_story]




# data loading
def get_batch(split):
    # generate a small batch of data of inputs x and targets y
    data = train_data if split == 'train' else val_data
    ix = torch.randint(len(data) - block_size, (batch_size,))
    x = torch.stack([data[i:i + block_size] for i in ix])
    y = torch.stack([data[i + 1:i + block_size + 1] for i in ix])
    x, y = x.to(device), y.to(device)
    return x, y

# average out loss over multiple batches
# because every single batch individually will be more or less lucky
# iterate eval_iter times and average out the loss
# for both train and val, this will be lot less noisy
@torch.no_grad()
def estimate_loss():
    out = {}
    model.eval()
    for split in ['train', 'val']:
        losses = torch.zeros(eval_iters)
        for k in range(eval_iters):
            X, Y = get_batch(split)
            logits, loss = model(X, Y)
            losses[k] = loss.item()
        out[split] = losses.mean()
    model.train()
    return out


model = LanguageModel()
m = model.to(device)

# create a PyTorch optimizer
optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate)

scheduler = ReduceLROnPlateau(optimizer, 'min', patience=5, cooldown=5)

# create wandb run
if wandb_log:
    wandb.init(project=wandb_project, name=wandb_run_name, config=config)

try:
    for iteration in range(max_iters):

        if (iteration % 50 == 0):
            print(iteration)
        # every once in a while evaluate the loss on train and val sets
        # interesting that we're not printing loss every iter
        # instead we're estimating the non noisy loss every eval_intervar
        # only for printing purposes
        if (iteration % eval_interval == 0) or (iteration == max_iters - 1):

            losses = estimate_loss()
            print(f"step {iteration}: train loss {losses['train']:.4f}, val loss {losses['val']:.4f}")
            # logging
            if wandb_log:
                wandb.log({
                    "iter": iteration,
                    "train/loss": losses['train'],
                    "val/loss": losses['val'],
                    #"lr": learning_rate,
                    })

            scheduler.step(losses['train'])

            print(f"Learning rate: {(scheduler.get_last_lr())[0]}")

        # sample a batch of data
        xb, yb = get_batch('train')


        # evaluate the loss
        logits, loss = model(xb, yb)
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        optimizer.step()
except:
    print("Please wait, genereating sample output with model... (this might take a while)")

# Save the model and optimizer state dictionaries
torch.save({
    'model_state_dict': model.state_dict(),
    'optimizer_state_dict': optimizer.state_dict(),
    'epoch': iteration,
    'loss': loss.item(),
}, 'model_checkpoint.pth')

# generate from the model
context = torch.zeros((1, 1), dtype=torch.long, device=device)
output = tokenizer.decode(m.generate(context, max_new_tokens=500)[0].tolist())
print(output)

# write longer output to file
disclaimer = """
THIS FILE CONTAINS GPT GENERATED TEXT.
"""
# output_long = tokenizer.decode(m.generate(context, max_new_tokens=1000)[0].tolist())
with open('./Tinystories.txt', 'w', encoding='utf-8') as f:
    f.write(disclaimer)
    f.write(output)