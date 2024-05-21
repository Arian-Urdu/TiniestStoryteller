from transformers import GPT2TokenizerFast
from datasets import load_dataset

def load_and_prepare_data(filename, dataset_name="roneneldan/TinyStories", num_samples=1000, max_length=128):
    tokenizer = GPT2TokenizerFast.from_pretrained("gpt2")
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    dataset = load_dataset(dataset_name, data_files={'train': filename})
    print(f"Dataset loaded with {len(dataset['train'])} entries. Processing {num_samples} entries.")


    small_dataset = dataset['train'].select(range(min(num_samples, len(dataset['train']))))

 
    def tokenize_function(examples):
        return tokenizer(examples['text'], truncation=True, padding='max_length', max_length=max_length)
    
 
    tokenized_dataset = small_dataset.map(tokenize_function, batched=True)
    print("Tokenization complete.")

    return tokenized_dataset

if __name__ == "__main__":
    tokenized_data = load_and_prepare_data('TinyStoriesV2-GPT4-train.txt', num_samples=1000, max_length=128)