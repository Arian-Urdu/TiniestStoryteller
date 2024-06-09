import os
import time

import datasets
import torch
import torch.nn as nn
from torch.nn import functional as F
from transformers import PreTrainedTokenizerFast

import wandb

# hyperparameters
batch_size = 64  # how many independent sequences will we process in parallel?
block_size = 128  # what is the maximum context length for predictions?
max_iters = 2000000
eval_interval = 500
eval_iters = 200 # was: 200
learning_rate = 3e-4
device = 'cuda' if torch.cuda.is_available() else 'cpu'
n_embed = 64  # has to be divisible(without rem) by n_head, given head_size definition further below
n_head = 8
n_layer = 7
dropout = 0.3
# wandb logging
wandb_log = False # disabled by default
wandb_project = 'TiniestStoryteller'
wandb_run_name = 'run' + str(time.time())
# config
config_keys = [k for k,v in globals().items() if not k.startswith('_') and isinstance(v, (int, float, bool, str))]
config = {k: globals()[k] for k in config_keys} # for logging
# ------------

print(f'Using device : {device}')
# to use gpu/device, data and model params has to be moved to the device

torch.manual_seed(1337)




dataset_path = os.path.join('preprocessing', 'preprocessed_dataset')
dataset = datasets.load_from_disk(dataset_path)
train_data = dataset['train']
val_data = dataset['validation']
print("Loaded dataset from disk")

# Smaller Dataset for testing
#train_data = train_data.select(range(10000))

tokenizer_path = os.path.join('tokenizers', 'bpe_tokenizer.json')
tokenizer = PreTrainedTokenizerFast(
    tokenizer_file = tokenizer_path,
    bos_token = "<|endoftext|>",
    eos_token = "<|endoftext|>"
)
vocab_size = tokenizer.vocab_size
print(f"Loaded Tokenizer: {tokenizer}")

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


train_data = torch.tensor(train_data, dtype = torch.long)
val_data = torch.tensor(val_data, dtype = torch.long)


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


# single head of self attention
class Head(nn.Module):
    """ one head of self attention """

    def __init__(self, head_size):
        super().__init__()
        self.key = nn.Linear(n_embed, head_size, bias=False)
        self.query = nn.Linear(n_embed, head_size, bias=False)
        self.value = nn.Linear(n_embed, head_size, bias=False)
        # in Pytorch convention a variable that's not a parameter of the model is called a buffer
        self.register_buffer('tril', torch.tril(torch.ones(block_size, block_size)))
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        B, T, C = x.shape
        # emit keys and queries for x
        k = self.key(x)  # (B, T, hs)
        q = self.query(x)  # (B, T, hs)
        # compute attention
        wei = q @ k.transpose(-2, -1) * k.shape[-1] ** -0.5  # (B, T, hs) @ (B, hs, T) -> (B, T, T)
        wei = wei.masked_fill(self.tril[:T, :T] == 0, float('-inf'))  # (B, T, T)
        wei = F.softmax(wei, dim=-1)  # (B, T, T)
        wei = self.dropout(wei)  # dropout some of the affinities
        # perform the weighted aggregation of the values
        v = self.value(x)  # (B, T, hs)
        out = wei @ v  # (B, T, T) @ (B, T, hs) -> (B, T, hs)
        return out


class MultiHeadAttention(nn.Module):

    def __init__(self, num_heads, head_size):
        super().__init__()
        self.heads = nn.ModuleList([Head(head_size) for _ in range(num_heads)])
        self.proj = nn.Linear(n_embed, n_embed)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        out = torch.cat([h(x) for h in self.heads], dim=-1)
        out = self.proj(out)  # outcome of the linear layer to project back into the residual pathway
        out = self.dropout(out)  # final dropout
        return out


class FeedForward(nn.Module):
    " simple linear layer followed by non linearity "

    def __init__(self, n_embed):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(n_embed, 4 * n_embed),  # as mentioned in the paper
            nn.ReLU(),
            nn.Linear(4 * n_embed, n_embed),  # projection layer : the final projection back into the residual pathway
            nn.Dropout(dropout),  # dropout before final projection
        )

    def forward(self, x):
        return self.net(x)


class Block(nn.Module):
    """ a transformer block : communication then computation """

    def __init__(self, n_embed, n_head):
        super().__init__()
        head_size = n_embed // n_head
        self.sa = MultiHeadAttention(n_head, head_size)
        self.ffwd = FeedForward(n_embed)
        self.ln1 = nn.LayerNorm(n_embed)
        self.ln2 = nn.LayerNorm(n_embed)

    def forward(self, x):
        x = x + self.sa(self.ln1(x))
        x = x + self.ffwd(self.ln2(x))
        return x


class LanguageModel(nn.Module):

    def __init__(self):
        super().__init__()
        # each token directly reads off the logits for the next token from a lookup table
        self.token_embedding_table = nn.Embedding(vocab_size, n_embed)
        self.position_embedding_table = nn.Embedding(block_size, n_embed)
        self.blocks = nn.Sequential(*[Block(n_embed, n_head=n_head) for _ in range(n_layer)])
        self.ln_f = nn.LayerNorm(n_embed)
        self.lm_head = nn.Linear(n_embed, vocab_size)

    def forward(self, idx, targets=None):
        B, T = idx.shape

        # idx and targets are both (B,T) tensor of integers
        token_emb = self.token_embedding_table(idx)  # (B,T,C)
        pos_emb = self.position_embedding_table(torch.arange(T, device=device))  # (T, C)
        x = token_emb + pos_emb  # (B, T, C)
        x = self.blocks(x)
        x = self.ln_f(x)
        logits = self.lm_head(x)  # (B,T,vocab_size)

        if targets is None:
            loss = None
        else:
            B, T, C = logits.shape
            logits = logits.view(B * T, C)
            targets = targets.view(B * T)
            loss = F.cross_entropy(logits, targets)

        return logits, loss

    def generate(self, idx, max_new_tokens):
        # idx is (B, T) array of indices in the current context
        for _ in range(max_new_tokens):
            # crop idx to the last block size token
            idx_cond = idx[:, -block_size:]
            # get the predictions
            logits, loss = self(idx_cond)
            # focus only on the last time step
            logits = logits[:, -1, :]  # becomes (B, C)
            # apply softmax to get probabilities
            probs = F.softmax(logits, dim=-1)  # (B, C)
            # sample from the distribution
            idx_next = torch.multinomial(probs, num_samples=1)  # (B, 1)
            # append sampled index to the running sequence
            idx = torch.cat((idx, idx_next), dim=1)  # (B, T+1)
        return idx


model = LanguageModel()
m = model.to(device)

# create a PyTorch optimizer
optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate)

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
        # sample a batch of data
        xb, yb = get_batch('train')
        # evaluate the loss
        logits, loss = model(xb, yb)
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        optimizer.step()
        # print("iteration complete")
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