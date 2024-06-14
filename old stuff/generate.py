import gpt
import torch
import tokenizer

is_mps = torch.backends.mps.is_available() 
device = "mps" if is_mps else "cpu"

is_cuda = torch.cuda.is_available()
device = "cuda:0" if is_cuda else "cpu"

model = gpt.GPT().to(device)
path = 'out/weights_0_5000.pt'
model.load_state_dict(torch.load(path))

tk = tokenizer.BasicEnglishTokenizer()
    
org = "Once upon a time "
src = torch.tensor([tk.encode(org)]).to(device)
trs = model.generate(src)
print(tk.decode(trs.tolist()[0]))
