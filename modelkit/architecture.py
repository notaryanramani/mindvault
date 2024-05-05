import torch
import torch.nn as nn
from datakit import KNN


class Head(nn.Module):
    def __init__(self, n_embd, head_size, knn, top_k, dropout) -> None:
        super().__init__()

        # attention layers
        self.q = nn.Linear(n_embd, head_size)
        self.k = nn.Linear(n_embd, head_size)
        self.v = nn.Linear(n_embd, head_size)

        # memory bank
        self.knn = knn
        self.top_k = top_k
        self.gate = nn.Parameter(torch.randn(1))

        self.dropout = nn.Dropout(dropout)
        self.register_buffer("mask", torch.tril(torch.ones(n_embd, n_embd)))

    def forward(self, x):
        B, T, C = x.shape

        q = self.q(x)
        k = self.k(x)
        v = self.v(x)

        # scaled dot product attention
        att = (q @ k.transpose(-2, -1)) / C ** 0.5
        att = att.masked_fill(self.mask[:x.size(1), :x.size(1)] == 0, float("-inf"))
        att = torch.softmax(att, dim=-1)
        out = att @ v

        # memory attention 
        self.knn.add(k, v)
        mem_kvs = self.knn.search(q, self.top_k) #  return shape: (B, T, 3, 2, HS)
        mem_k, mem_v = mem_kvs[:, :, :, 0, :], mem_kvs[:, :, :, 1, :]
        mem_att = (q.unsqueeze(-2) @ mem_k.transpose(-2, -1)) / C ** 0.5 
        mem_att = torch.softmax(mem_att, dim=-1)
        mem_out = mem_att @ mem_v
        mem_out = mem_out.view(B, T, -1)

        # gating mechanism
        md_out = out * self.gate + (mem_out * (1 - self.gate)) 
        md_out = self.dropout(out)

        return md_out
    

class MultiHeadAttention(nn.Module):
    def __init__(self, n_embd, n_heads, batch_size, block_size, seg, top_k, dropout) -> None:
        super().__init__()
        head_size = n_embd // n_heads
        self.knn = KNN(head_size, batch_size * block_size * seg)
        self.heads = nn.ModuleList([Head(n_embd, head_size, self.knn, top_k, dropout) for _ in range(n_heads)])

    def forward(self, x):
        out = torch.cat([head(x) for head in self.heads], dim=-1)
        self.knn.clear()
        return out


class FeedForward(nn.Module):
    def __init__(self, n_embd, dropout) -> None:
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
    def __init__(self, n_embd, n_heads, batch_size, block_size, seg, top_k, dropout = 0.1) -> None:
        super().__init__()
        self.attn = MultiHeadAttention(n_embd, n_heads, batch_size, block_size, seg, top_k, dropout)
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