from tokenizers import (decoders, models, normalizers, pre_tokenizers, processors, trainers, Tokenizer, Regex, )
import datasets
from transformers import PreTrainedTokenizerFast

dataset = datasets.load_dataset("roneneldan/TinyStories", split="train")

# dataset = dataset.select(range(10000))

def get_training_corpus():
    for i in range(0, len(dataset), 1000):
        yield dataset[i : i + 1000]["text"]

xln_tokenizer = Tokenizer(models.Unigram())

xln_tokenizer.normalizer = normalizers.Sequence([normalizers.Replace("``", '"'), normalizers.Replace("''", '"'), normalizers.NFKD(), normalizers.StripAccents(), normalizers.Replace(Regex(" {2,}"), " "),])

xln_tokenizer.pre_tokenizer = pre_tokenizers.Metaspace()

special_tokens = ["<unk>", "<pad>", "<s>", "</s>"]
trainer = trainers.UnigramTrainer(vocab_size = 25000, special_tokens = special_tokens, unk_token = "<unk>")

xln_tokenizer.train_from_iterator(get_training_corpus(), trainer = trainer)

bos_token_id = xln_tokenizer.token_to_id("<s>")
eos_token_id = xln_tokenizer.token_to_id("</s>")

xln_tokenizer.post_processor = processors.TemplateProcessing(
    single = "$A:0 </s>:0 <s>:2",
    special_tokens = [("</s>", eos_token_id), ("<s>", bos_token_id)],
)

xln_tokenizer.decoder = decoders.Metaspace()

xln_tokenizer.save("xln_tokenizer.json")

wrapped_tokenizer = PreTrainedTokenizerFast(
    tokenizer_file = "xln_tokenizer.json",
    bos_token = "<s>",
    eos_token = "</s>",
    unk_token = "<unk>",
    pad_token = "<pad>",
    padding_side = "left",
)