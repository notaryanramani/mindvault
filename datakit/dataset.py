from torch.utils.data import Dataset

class MindvaultDataset(Dataset):
    def __init__(self, data, seg) -> None:
        self.data = data
        self.seg = seg

    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        x = self.data[idx, :-1]
        y = self.data[idx, 1:]

        x = x.chunk(self.seg, dim=-1)
        y = y.chunk(self.seg, dim=-1)

        return x, y

    