
import torch
import tokenizer
import datasets


class TinyDataset(torch.utils.data.Dataset):
  def __init__(self):
    self.ds = datasets.load_dataset("roneneldan/TinyStories")
    self.tk = tokenizer.BasicEnglishTokenizer()

  def __len__(self):
    return len(self.ds['train'])

  def __getitem__(self, idx):
    row = self.ds['train'][idx]['text']
    input = [self.tk.vocab['<bos>']] + self.tk.encode(row)
    label = (self.tk.encode(row)) + [self.tk.vocab['<eos>']]
    return { 'input': torch.tensor(input), 'label': torch.tensor(label) }

  def collate_fn(self, batch):
    input_pad = torch.nn.utils.rnn.pad_sequence([item['input'] for item in batch], batch_first=True, padding_value=0)
    label_pad = torch.nn.utils.rnn.pad_sequence([item['label'] for item in batch], batch_first=True, padding_value=0)
    return { 'input': input_pad, 'label': label_pad }


if __name__ == '__main__':
  dataset = TinyDataset()
  tokenizer1 = tokenizer.BasicEnglishTokenizer()
  print('ds.ds', dataset.ds)
  print('len(ds)', len(dataset))
  print('ds[362]', dataset[362])
  entry_362 = dataset[362]['label']
  print(tokenizer1.decode(entry_362.tolist()))
