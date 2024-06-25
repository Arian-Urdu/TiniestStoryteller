import torch
from transformer_model import LanguageModel
from config import device
import os

model = LanguageModel()

model_path = os.path.join(os.path.dirname(os.path.dirname(os.path.realpath(__file__))), 'output\\model_checkpoint_2024-06-19-08-56.pth')
checkpoint = torch.load(model_path)

# instantiate model and optimizer
model = LanguageModel().to(device)
optimizer = torch.optim.AdamW(model.parameters())#, lr = checkpoint['learning_rate'])

# load model and optimizer
model.load_state_dict(checkpoint['model_state_dict'])
model.eval() # set model to evaluation
optimizer.load_state_dict(checkpoint['optimizer_state_dict'])

