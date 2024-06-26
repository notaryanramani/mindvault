from .architecture  import Decoder
import torch.nn as nn
import torch
import torch.nn.functional as F


class MindVaultGPT(nn.Module):
    def __init__(self, vocab_size, n_embd = 256, n_heads = 4, batch_size = 32, block_size = 128, seg = 5, top_k = 10, dropout = 0.2, n_layers = 4) -> None:
        super().__init__()
        self.token_emb = nn.Embedding(vocab_size, n_embd)
        self.pos_emb = nn.Embedding(block_size, n_embd)
        
        self.decoder = nn.ModuleList([Decoder(n_embd, n_heads, batch_size, block_size, seg, top_k, dropout) for _ in range(n_layers)])
        self.ln = nn.LayerNorm(n_embd)
        self.fc = nn.Linear(n_embd, vocab_size)

    def forward(self, x, targets = None, ki=None, vi=None):
        B, T = x.size()

        x = self.token_emb(x)
        pos = torch.arange(T).to(x.device)
        x = x + self.pos_emb(pos)
        k, v = ki, vi

        for layer in self.decoder:
            x, k, v, ak, av = layer(x, k, v)
        
        x = self.ln(x)
        logits = self.fc(x)

        if targets is not None:
            logits = logits.view(B * T, -1)
            targets = targets.view(-1)
            loss = F.cross_entropy(logits, targets)
        
        else: 
            loss = None
        
        if ki is not None:
            k = torch.cat([ki, ak], dim=-2)
        if vi is not None:
            v = torch.cat([vi, av], dim=-2)

        return logits, k, v, loss

    def predict(self, x):
        logits, _ = self(x)
        logits = logits[:, -1, :]
        return logits
