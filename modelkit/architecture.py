import torch
import torch.nn as nn


class Head(nn.Module):
    def __init__(self, n_embd, head_size) -> None:
        super().__init__()
        self.q = nn.Linear(n_embd, head_size)
        self.k = nn.Linear(n_embd, head_size)
        self.v = nn.Linear(n_embd, head_size)

        self.register_buffer("mask", torch.tril(torch.ones(n_embd, n_embd)))

    def forward(self, x):
        _, _, C = x.shape

        q = self.q(x)
        k = self.k(x)
        
        att = (q @ k.transpose(-2, -1)) / C ** 0.5
        att = att.masked_fill(self.mask[:x.size(1), :x.size(1)] == 0, float("-inf"))
        att = torch.softmax(att, dim=-1)

        v = self.v(x)
        out = att @ v
        return out
    

class MultiHeadAttention(nn.Module):
    def __init__(self, n_embd, n_heads) -> None:
        super().__init__()
        head_size = n_embd // n_heads
        self.heads = nn.ModuleList([Head(n_embd, head_size) for _ in range(n_heads)])

    def forward(self, x):
        return torch.cat([head(x) for head in self.heads], dim=-1)


class FeedForward(nn.Module):
    def __init__(self, n_embd, dropout = 0.1) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(n_embd, n_embd * 4),
            nn.ReLU(),
            nn.Linear(n_embd * 4, n_embd)
        )
        self.ln = nn.LayerNorm(n_embd)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        x = self.ln(x)
        x = self.net(x) + x
        out = self.dropout(x)
        return out


class Decoder(nn.Module):
    def __init__(self, n_embd, n_heads, dropout = 0.2) -> None:
        super().__init__()
        self.attn = MultiHeadAttention(n_embd, n_heads)
        self.ff = FeedForward(n_embd, dropout = dropout)
        self.ln1 = nn.LayerNorm(n_embd)
        self.ln2 = nn.LayerNorm(n_embd)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        x = self.ln1(x)
        x = x + self.attn(x)
        x = self.ln2(x)
        x = x + self.ff(x)
        out = self.dropout(x)
        return out