import torch
from torch.utils.data import DataLoader
from datakit import MindvaultDataset, Preprocess
from modelkit import MindVaultGPT
import tiktoken
from torch.optim import AdamW
from tqdm import tqdm

with open('data/input.txt', 'r') as file:
    data = file.read()
tok= tiktoken.get_encoding('r50k_base')

# parameters 
seg = 5
block_size = 64
device = 'cuda' if torch.cuda.is_available() else 'cpu'
epochs = 5
# --------

# Preprocess data
preprocess = Preprocess(data, seg, block_size, tokenizer=tok)
data = preprocess.process()

# Create DataLoader
train_size = int(0.8 * len(data))
d_train = data[:train_size]
d_val = data[train_size:]
t_dataset = MindvaultDataset(d_train, seg)
v_dataset = MindvaultDataset(d_val, seg)
t_loader = DataLoader(t_dataset, batch_size=32, shuffle=True)
v_loader = DataLoader(v_dataset, batch_size=32, shuffle=True)

# Create model
m = MindVaultGPT(vocab_size = tok.n_vocab, block_size = block_size)
m.to(device)

# Create optimizer
optim = AdamW(m.parameters(), lr=1e-4)

# Training loop
for e in range(epochs):
    t_losses = []
    t_loader = tqdm(t_loader, leave=False)
    t_loader.set_description(f'Epoch {e}/{epochs}')
    for x, y in t_loader:
        for xi, yi in zip(x, y):
            xi, yi = xi.to(device), yi.to(device)
            logits, t_loss = m(xi, yi)
            optim.zero_grad()
            t_loss.backward()
            optim.step()
            t_losses.append(t_loss.item())
            t_loader.set_postfix(loss=t_loss.item())
    t_loss_avg = torch.tensor(t_losses).mean().item()
    v_losses = []
    v_loader = tqdm(v_loader, leave=False)
    for x, y in v_loader:
        for xi, yi in zip(x, y):
            xi, yi = xi.to(device), yi.to(device)
            logits, v_loss = m(xi, yi)
            v_losses.append(v_loss.item())
            v_loader.set_postfix(loss=v_loss.item())
    v_loss_avg = torch.tensor(v_losses).mean().item()
    print(f'Epoch {e}, Train Loss: {t_loss}, Val Loss: {v_loss}')