from tokenizers import (decoders, models, normalizers, pre_tokenizers, processors, trainers, Tokenizer, )
try:
    from config import train_data as dataset
except ImportError:
    import sys
    sys.path.append(sys.path[0] + '/..')
    from config import train_data as dataset
from transformers import PreTrainedTokenizerFast


# dataset = dataset.select(range(10000))

def get_training_corpus():
    for i in range(0, len(dataset), 1000):
        yield dataset[i : i + 1000]["text"]

bpe_tokenizer = Tokenizer(models.BPE())



bpe_tokenizer.pre_tokenizer = pre_tokenizers.ByteLevel(add_prefix_space = False)


trainer = trainers.BpeTrainer(vocab_size = 2047, special_tokens = ["<|endoftext|>"], min_frequency = 3)

bpe_tokenizer.train_from_iterator(get_training_corpus(), trainer = trainer)


bpe_tokenizer.post_processor = processors.ByteLevel(trim_offsets = False)




bpe_tokenizer.decoder = decoders.ByteLevel()

bpe_tokenizer.save("bpe_tokenizer_2048.json")

# wrapped_tokenizer = PreTrainedTokenizerFast(
#     tokenizer_file = "bpe_tokenizer.json",
#     bos_token = "<|endoftext|>",
#     eos_token = "<|endoftext|>"
# )