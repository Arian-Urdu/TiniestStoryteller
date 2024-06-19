import torch
from transformer_model import LanguageModel
from config import device

model = LanguageModel()

checkpoint = torch.load('../model_checkpoint.pth')

# instantiate model and optimizer
model = LanguageModel().to(device)
optimizer = torch.optim.AdamW(model.parameters(), lr = checkpoint['learning_rate'])

# load model and optimizer
model.load_state_dict(checkpoint['model_state_dict'])
optimizer.load_state_dict(checkpoint['optimizer_state_dict'])

