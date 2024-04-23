import datasets

import torch
from torch import nn
from torch.utils.data import Dataset, DataLoader
from torch.optim import Adam

import transformers
from transformers import GPT2TokenizerFast


dataset = datasets.load_dataset("roneneldan/TinyStories", split="train")
print(dataset[0])

# Set device to cuda if avaliable else cpu
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Define the tokenizer
tokenizer = GPT2TokenizerFast.from_pretrained("gpt2")
if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

# Tokenize the dataset
def tokenize_function(examples):
    return tokenizer(examples['text'], truncation=True, padding='max_length', max_length=1024)

dataset = dataset.map(tokenize_function, batched=True)
print(len(tokenizer))

dataset.set_format(type="torch", columns=['text', 'input_ids', 'attention_mask'])
print(dataset.format)


class TransformerModel(nn.Module):
    def __init__(self, vocab_size, embed_size, hidden_size, num_layers, num_heads, dropout):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, embed_size)
        self.transformer = nn.Transformer(
            d_model=embed_size,
            nhead=num_heads,
            num_encoder_layers=num_layers,
            num_decoder_layers=num_layers,
            dim_feedforward=hidden_size,
            dropout=dropout,
        )
        self.fc = nn.Linear(embed_size, vocab_size)

    def forward(self, x):
        embedded = self.embedding(x)
        output = self.transformer(embedded, embedded)
        output = self.fc(output)
        return output
    
# Initialize the model and the optimizer
model = TransformerModel(len(tokenizer), embed_size=128, hidden_size=256, num_layers=2, num_heads=8, dropout=0.1).to(device)

# Use ADAM
optimizer = Adam(model.parameters(), lr=0.001)


# Define the dataset
class TransformerDataset(Dataset):
    def __init__(self, text, sequence_length):
        self.text = text
        self.sequence_length = sequence_length

    def __len__(self):
        return len(self.text) - self.sequence_length

    def __getitem__(self, idx):
        return (
            torch.tensor(self.text[idx:idx+self.sequence_length]),
            torch.tensor(self.text[idx+1:idx+self.sequence_length+1]),
        )

# Create the dataset and dataloader
sequence_length = 30
dataset = TransformerDataset(dataset["input_ids"], sequence_length)
dataloader = DataLoader(dataset, batch_size=128)
print(dataloader)

# Train the model
for epoch in range(2):
    for batch in dataloader:
        print(batch)
        x, y = batch
        x = x.to(device)
        y = y.to(device)
        optimizer.zero_grad()
        y_pred = model(x)
        loss = nn.functional.cross_entropy(y_pred.view(-1, len(tokenizer)), y.view(-1))
        loss.backward()
        optimizer.step()
    print(f'Epoch {epoch}, Loss {loss.item()}')
    if float(loss.item()) < 0.06:
        break