import torch
import torch.distributed as dist


class MyDataset(torch.utils.data.Dataset):
    """
    Dummy dataset for testing
    """
    def __init__(self, n = 1024):
        self.data = torch.randn(n, 32)
        self.targets = (self.data * 1.3) - 0.65
        # Synchronize data across all ranks
        with torch.no_grad():
            dist.broadcast(self.data, src = 0)
            dist.broadcast(self.targets, src = 0)
        
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        return self.data[idx], self.targets[idx]