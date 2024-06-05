from tokenizers import (decoders, models, normalizers, pre_tokenizers, processors, trainers, Tokenizer, )
import datasets
from transformers import PreTrainedTokenizerFast

dataset = datasets.load_dataset("roneneldan/TinyStories", split="train")

# dataset = dataset.select(range(10000))

def get_training_corpus():
    for i in range(0, len(dataset), 1000):
        yield dataset[i : i + 1000]["text"]

wp_tokenizer = Tokenizer(models.WordPiece(unk_token = "[UNK]"))

wp_tokenizer.normalizer = normalizers.Sequence([normalizers.NFD(), normalizers.Lowercase(), normalizers.StripAccents()])

wp_tokenizer.pre_tokenizer = pre_tokenizers.Whitespace()

special_tokens = ["[UNK]", "[PAD]", "[BOS]", "[EOS]"]
trainer = trainers.WordPieceTrainer(vocab_size = 25000, special_tokens = special_tokens, min_frequency = 3)

wp_tokenizer.train_from_iterator(get_training_corpus(), trainer)

bos_token_id = wp_tokenizer.token_to_id("[BOS]")
eos_token_id = wp_tokenizer.token_to_id("[EOS]")

wp_tokenizer.post_processor = processors.TemplateProcessing(
    single = f"[BOS]:0 $A:0 [EOS]:0",
    special_tokens = [("[BOS]", bos_token_id), ("[EOS]", eos_token_id)],
)

wp_tokenizer.decoder = decoders.WordPiece(prefix = "##")

wp_tokenizer.save("wp_tokenizer.json")
#
# wrapped_tokenizer = PreTrainedTokenizerFast(
#     wp_tokenizer_file = "wp_tokenizer.json",
#     unk_token = "[UNK]",
#     pad_token = "[PAD]",
#     bos_token = "[BOS]",
#     eos_token = "[EOS]",
# )

'''
questions for chatgpt:
1. do we need any other special tokens? leave out [CLS] and [MASK], add [BOS] and [EOS]
2. does it make sense to train the tokenizer on the same dataset as we'll use for the model we want to
train or could that lead to anomalies? yes!
3. what would you suggest is a good vocab size for the tinystories dataset? 20000-50000
4. min_frequency? yes, see word_frequency.py
'''