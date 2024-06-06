import torchtext; torchtext.disable_torchtext_deprecation_warning()
from torchtext.data.utils import get_tokenizer
from collections import Counter
from torchtext.vocab import vocab 
import io
import os
import torch


# Path to entire dataset in one txt
tinystories_all = 'data/tiny_stories_all.txt'
# tinystories_small = 'data/TinyStories-valid.txt'

# Get tokenizer using torchtext
tokenizer = get_tokenizer('basic_english')
print(tokenizer("Using basic_english Tokenizer"))


def build_vocab(filepath, tokenizer):
    if os.path.exists('out/vocab_obj.pth'):
        vocab_obj = torch.load('out/vocab_obj.pth')
        print('Found saved vocab, using: out/vocab_obj.pth')
        return vocab_obj
    else:
        print('No saved vocab found, creating new this might take a couple minutes...')
        counter = Counter()
        with io.open(filepath, encoding="utf8") as f: 
            for string_ in f:
                counter.update(tokenizer(string_))
        vocab_obj = vocab(counter, specials=['<unk>', '<pad>', '<bos>', '<eos>'])
        torch.save(vocab_obj, 'out/vocab_obj.pth')
        return vocab_obj

class BasicEnglishTokenizer:
    def __init__(self, prefix='basic_english') -> None:
        # get the tokenizer from torchtext
        self.vocab = build_vocab(tinystories_all, tokenizer)
        self.vocab.set_default_index(0)
        self.tok =  get_tokenizer('basic_english')
        self.prefix = prefix

    def encode(self, txt):
        tokenized_txt = self.tok(txt)
        return self.vocab.lookup_indices(tokenized_txt)
    
    def decode(self, ids):
        return self.vocab.lookup_tokens(ids)

    def vocab_size(self):
        return len(self.vocab)

if __name__ == '__main__':
  tknz = BasicEnglishTokenizer()

  print("tknz.vocab_size()", tknz.vocab_size())
  print('tknz.sp.unk_id()', tknz.vocab['<unk>'])

  ids_foo = tknz.encode('hello my name is Arian')
  ids_bar = tknz.encode('ciao il mio nome è Arian')
  ids_zoo = tknz.encode('emma')
  print('ids_foo', ids_foo)
  print('ids_bar', ids_bar)
  print('ids_zoo', ids_zoo)
  txt_foo = tknz.decode(ids_foo)
  txt_bar = tknz.decode(ids_bar)
  txt_zoo = tknz.decode(ids_zoo)
  print('txt_foo', txt_foo)
  print('txt_bar', txt_bar)
  print('txt_zoo', txt_zoo)