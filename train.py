import torch
import gpt
import process_dataset
import tokenizer
import random

is_mps = torch.backends.mps.is_available() 
device = "mps" if is_mps else "cpu"

is_cuda = torch.cuda.is_available()
device = "cuda:0" if is_cuda else "cpu"

print('using device :', device)

random.seed(42)
torch.manual_seed(42)
model = gpt.GPT().to(device)
model.num_params()


tk = tokenizer.BasicEnglishTokenizer()
ds = process_dataset.TinyDataset()
dl = torch.utils.data.DataLoader(ds, batch_size=4, shuffle=True, collate_fn=ds.collate_fn)
opt = torch.optim.Adam(model.parameters(), lr=0.0001)


for epoch in range(5):
  org = "Once upon a time "
  src = torch.tensor([tk.encode(org)]).to(device)
  trs = model.generate(src)
  print(f"{org} - {tk.decode(trs.tolist()[0])}")

  for idx, batch in enumerate(dl):
    if idx % 10 == 0: print(idx)
    x = batch['input'].to(device)
    y = batch['label'].to(device)
    p = model(x)

    p = p.view(-1, p.size(-1))
    y = y.view(-1)
    l = torch.nn.functional.cross_entropy(p, y, ignore_index=0)
    if idx % 1000 == 0: print(f"Loss: {l.item():.4f}")
    if idx % 5000 == 0: torch.save(model.state_dict(), f"weights_{epoch}_{idx}.pt")
    l.backward()
    opt.step()
    opt.zero_grad()