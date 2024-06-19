import torch
from config import batch_size, block_size, max_iters, eval_interval, eval_iters, learning_rate, device, wandb_log, \
    wandb_project, wandb_run_name, config, train_data, val_data, tokenizer, vocab_size, num_epochs
from transformer_model import LanguageModel

# Load the checkpoint
checkpoint = torch.load('output/model_checkpoint_2024-06-19-08:56.pth')

# Define your model architecture
model = LanguageModel()  # Replace MyModel with your model class

# Load the state_dict for model and optimizer
model.load_state_dict(checkpoint['model_state_dict'])

# Set the model to evaluation mode
model.eval()
model.to(device)


num_gen = 3

for i in range(num_gen):
    
    # Generate from the model
    context = torch.zeros((1, 1), dtype=torch.long, device=device)
    intro = "Once upon a time "
    src = torch.tensor([tokenizer.encode(intro)]).to(device)
    with torch.no_grad():
        output_list = model.generate(src, max_new_tokens=500)[0].tolist()


    # Truncate the output at the first occurrence of the [PAD] token
    truncated_output = []
    for token in output_list:
        if token == tokenizer.pad_token_id:
            break
        truncated_output.append(token)


    # Process the output
    output = tokenizer.decode(truncated_output)
    print(output)

