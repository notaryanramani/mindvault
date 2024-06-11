import torch
import torch.nn as nn
from .knn import KNN
import torch.nn.functional as F


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
        self.head_size = head_size

    def forward(self, x, ki=None, vi=None, idx=None):
        Bq, Tq, Cq = x.shape
        
        q = self.q(x)
        k = self.k(x)
        v = self.v(x)
        self.knn.add(k, v)

        start_idx = idx * self.head_size 
        end_idx = start_idx + self.head_size

        if ki is not None:
            k = torch.cat([ki[:, :, start_idx:end_idx], k], dim=-2)
            kv_length = ki.shape[1]
        if vi is not None:
            v = torch.cat([vi[:, :, start_idx:end_idx], v], dim=-2)

        Bk, Tk, Ck = k.shape

        # scaled dot product attention
        mask = torch.tril(torch.ones(Tq, Tk), diagonal=Tk - Tq).to(q.device)
        out = F.scaled_dot_product_attention(q, k, v, attn_mask=mask)

        # memory attention 
        mem_kvs = self.knn.search(q, self.top_k) #  return shape: (B, T, 3, 2, HS)
        mem_k, mem_v = mem_kvs[:, :, :, 0, :], mem_kvs[:, :, :, 1, :]
        mem_att = (q.unsqueeze(-2) @ mem_k.transpose(-2, -1)) / Cq ** 0.5 
        mem_att = torch.softmax(mem_att, dim=-1)
        mem_out = mem_att @ mem_v
        mem_out = mem_out.view(Bq, Tq, -1)

        # gating mechanism
        md_out = out * self.gate + (mem_out * (1 - self.gate)) 
        md_out = self.dropout(out)

        if ki is not None:
            k = k[:, :kv_length, :]
            ak = k[:, kv_length:, :]
        if vi is not None:
            v = v[:, :kv_length, :]
            av = v[:, kv_length:, :]

        return md_out, k, v, ak, av
    

class MultiHeadAttention(nn.Module):
    def __init__(self, n_embd, n_heads, batch_size, block_size, seg, top_k, dropout) -> None:
        super().__init__()
        head_size = n_embd // n_heads
        self.knn = KNN(head_size, batch_size * block_size * seg)
        self.heads = nn.ModuleList([Head(n_embd, head_size, self.knn, top_k, dropout) for _ in range(n_heads)])

    def forward(self, q, k=None, v=None):
        out, keys, values = [], [], []
        akeys, avalues = [], []
        for idx, head in enumerate(self.heads):
            oi, k, v, ak, av = head(q, k, v, idx)
            out.append(oi)
            akeys.append(ak)
            avalues.append(av)
            keys.append(k)
            values.append(v)

        out = torch.cat(out, dim=-1)
        akeys = torch.cat(akeys, dim=-1)
        avalues = torch.cat(avalues, dim=-1)
        keys = torch.cat(keys, dim=-1)
        values = torch.cat(values, dim=-1)
        
        self.knn.clear()
        return out, keys, values, akeys, avalues


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

    def forward(self, x, k=None, v=None):
        x = self.ln1(x)
        q, k, v, ak, av = self.attn(x, k, v)
        x = x + q
        x = self.ln2(x)
        x = x + self.ff(x)
        out = self.dropout(x)
        return out, k, v, ak, av
