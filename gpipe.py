"""
GPipe schedule:
--------------------------------------
Rank 0 | F F F F             B B B B |
Rank 1 |   F F F F         B B B B   |
Rank 2 |     F F F F     B B B B     |
Rank 3 |       F F F F B B B B       |
--------------------------------------
"""

import torch
import torch.distributed as dist
import torch.nn as nn
from src.data import MyDataset
from src.utils import sequential_forward, sequential_backward

if __name__ == "__main__":
    dist.init_process_group(backend="gloo")
    rank = dist.get_rank()
    world_size = dist.get_world_size()

    # This is the full model
    model = nn.Sequential(
        nn.Linear(32, 32),
        nn.ReLU(),
        nn.Linear(32, 32),
        nn.ReLU(),
        nn.Linear(32, 32),
        nn.ReLU(),
        nn.Linear(32, 32),
        nn.Identity() # an even number of layers is easier to split
    )

    # Each rank gets a part of the model
    layers_per_rank = len(model) // world_size
    local_model = model[rank * layers_per_rank : (rank + 1) * layers_per_rank]
    print(f"Rank {rank} model: {local_model}")
    
    # dataset = MyDataset()
    # inputs, targets = dataset.in
    
    inputs = torch.randn(256, 32) # inputs to the full model
    targets = torch.randn(256, 32) # targets
    
    inputs, outputs = sequential_forward(local_model, inputs)
    
    sequential_backward(inputs, outputs, targets, nn.MSELoss())