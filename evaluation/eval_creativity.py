import torch
import random
from sklearn.metrics.pairwise import cosine_similarity
from config import batch_size, block_size, max_iters, eval_interval, eval_iters, learning_rate, device, wandb_log, \
    wandb_project, wandb_run_name, config, train_data, val_data, tokenizer, vocab_size, num_epochs
from transformer_model import LanguageModel

# Load the checkpoint
checkpoint = torch.load('output/model_checkpoint_2024-07-03-14:14.pth')

# Define model architecture
model = LanguageModel()  

# Load the state_dict for model and optimizer
model.load_state_dict(checkpoint['model_state_dict'])

# Set the model to evaluation mode
model.eval()
model.to(device)

# Function to generate text from the model
def generate_text(input_text, model, tokenizer, max_new_tokens=500):
    src = torch.tensor([tokenizer.encode(input_text)]).to(device)
    with torch.no_grad():
        output_list = model.generate(src, max_new_tokens=max_new_tokens)[0].tolist()
    
    # Truncate the output at the first occurrence of the [PAD] token
    truncated_output = []
    for token in output_list:
        if token == tokenizer.pad_token_id:
            break
        truncated_output.append(token)
    
    return tokenizer.decode(truncated_output)

# Function to calculate cosine similarity
def calculate_cosine_similarity(vec1, vec2):
    # Pad the shorter vector with zeros to match the length of the longer vector
    max_length = max(vec1.shape[1], vec2.shape[1])
    if vec1.shape[1] < max_length:
        vec1 = torch.nn.functional.pad(vec1, (0, max_length - vec1.shape[1]), "constant", 0)
    if vec2.shape[1] < max_length:
        vec2 = torch.nn.functional.pad(vec2, (0, max_length - vec2.shape[1]), "constant", 0)
    
    # Move tensors to CPU and convert to numpy arrays
    vec1 = vec1.cpu().numpy().reshape(1, -1)
    vec2 = vec2.cpu().numpy().reshape(1, -1)
    return cosine_similarity(vec1, vec2)[0][0]

# Calculate cosine similarity for 10 random examples from the training set
num_examples = 3
cosine_sims = []

for _ in range(num_examples):
    example_index = random.randint(0, len(train_data) - 1)
    example = train_data[example_index]['text']  
    input_text = ' '.join(example.split()[:10])
    actual_continuation = ' '.join(example.split()[10:])
    
    # Generate text using the model
    generated_text = generate_text(input_text, model, tokenizer)
    
    # Encode both the generated text and the actual continuation
    generated_encoding = tokenizer.encode(generated_text, return_tensors='pt').to(device)
    actual_encoding = tokenizer.encode(actual_continuation, return_tensors='pt').to(device)
    
    # Calculate cosine similarity
    cosine_sim = calculate_cosine_similarity(generated_encoding, actual_encoding)
    cosine_sims.append(cosine_sim)
    
    print(f"Example {example_index}:")
    print(f"Input text: {input_text}")
    print(f"Generated text: {generated_text}")
    print("." * 40)
    print("\n")
    print(f"Actual continuation: {input_text} {actual_continuation}")
    print(f"Cosine Similarity: {cosine_sim}")
    print('-' * 80)

average_cosine_sim = sum(cosine_sims) / len(cosine_sims)
print(f"Average Cosine Similarity over {num_examples} examples: {average_cosine_sim}")
