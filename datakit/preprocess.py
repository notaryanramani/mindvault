import torch

class Preprocess:
    def __init__(self, text, seg, block_size, tokenizer):
        self.text = text
        self.seg = seg
        self.block_size = block_size
        self.chunk_size = seg * block_size
        self.tokenizer = tokenizer


    def process(self):
        tokens = self.tokenizer.encode(self.text)
        chunks = [tokens[i:i+self.chunk_size] for i in range(0, len(tokens), self.chunk_size) if len(tokens[i:i+self.chunk_size]) >= self.chunk_size]
        chunked = torch.tensor(chunks)
        return chunked

