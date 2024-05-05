import faiss
import numpy as np
import torch
import os
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE" # to fix the error: OMP: Error #15: Initializing libiomp5md.dll, but found libiomp5md.dll already initialized.


class KNN:
    def __init__(self, n_embd, max_mem):
        self.n_embd = n_embd
        self.index = faiss.IndexFlatL2(n_embd)
        os.makedirs('data', exist_ok=True)
        self.path = os.path.join('data', 'mydb.npy')
        self.db = np.zeros((max_mem, 2, n_embd), dtype=np.float32)
        self.max_mem = max_mem
        self.db_offset = 0
        

    def add(self, k, v):
        self.device = k.device

        k = k.view(-1, self.n_embd)
        v = v.view(-1, self.n_embd)

        k = k.detach().cpu().numpy()
        v = v.detach().cpu().numpy()

        self.add_to_db(k, v)
        k = np.ascontiguousarray(k)

        self.index.add(k)


    def add_to_db(self, k, v):
        kv = np.concatenate([np.expand_dims(k, axis=-2), np.expand_dims(v, axis=-2)], axis=-2)  
        kv_len = kv.shape[0]
        ids = np.arange(kv_len) + self.db_offset
        self.db_offset += kv_len
        self.db[ids] = kv


    def search(self, q, top_k):
        batch_size = q.shape[0]

        q = q.view(-1, self.n_embd)
        qs = np.ascontiguousarray(q.detach().cpu().numpy())
        _, i = self.index.search(qs, top_k)
        out = self.db[i]

        out = torch.tensor(out).to(self.device)
        out = out.view(batch_size, -1, top_k, 2, self.n_embd) 
        return  out 
    

    def clear(self):
        self.index.reset()
        self.db_offset = 0
        self.db[:] = 0
