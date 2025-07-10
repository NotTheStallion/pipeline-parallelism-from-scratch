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



def pipelined_iteration_gpipe(model_part, inputs, targets, loss_fn):
    """
    Implement one iteration of pipelined training using GPipe
    - Split the inputs and targets into microbatches
    - Perform forward passes for all microbatches (use sequential_forward)
    - Perform backward passes for all microbatches (use sequential_backward)
    """
    rank = dist.get_rank()
    world_size = dist.get_world_size()
    microbatches = torch.chunk(inputs, world_size)
    microtargets = torch.chunk(targets, world_size)
    
    total_loss = 0
    forward_inputs = []
    forward_outputs = []

    # Forward pass for all microbatches
    for i, microbatch in enumerate(microbatches):
        inputs, outputs = sequential_forward(model_part, microbatch)
        forward_inputs.append(inputs)
        forward_outputs.append(outputs)

    # Backward pass for all microbatches
    for i, (microbatch, microtarget) in enumerate(zip(microbatches, microtargets)):
        loss = sequential_backward(forward_inputs[i], forward_outputs[i], microtarget, loss_fn)
        if rank == world_size - 1:
            total_loss += loss.item()

    return total_loss




def pipelined_training_gpipe(model_part):
    """
    Perform pipelined training on a full dataset
    For each batch:
    - Perform pipelined iteration (use pipelined_iteration)
    - Update the model parameters
    """
    rank = dist.get_rank()
    world_size = dist.get_world_size()

    dataset = MyDataset()
    loss_fn = nn.MSELoss()
    optimizer = torch.optim.Adam(model_part.parameters())
    batch_size = 8
    data_loader = torch.utils.data.DataLoader(dataset, batch_size=batch_size, shuffle=True)

    for epoch in range(10):
        epoch_loss = 0
        for inputs, targets in data_loader:
            optimizer.zero_grad()
            loss = pipelined_iteration_gpipe(model_part, inputs, targets, loss_fn)
            optimizer.step()
            if rank == world_size - 1:
                epoch_loss += loss

        if rank == world_size - 1:
            print(f"[Rank {rank}] Epoch {epoch} loss: {epoch_loss / len(data_loader)}")





if __name__ == "__main__":
    from torch.profiler import profile, ProfilerActivity
    
    
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
    
    inputs = torch.randn(256, 32) # inputs to the full model
    targets = torch.randn(256, 32) # targets
    
    inputs, outputs = sequential_forward(local_model, inputs)
    
    sequential_backward(inputs, outputs, targets, nn.MSELoss())
    
    pipelined_iteration_gpipe(local_model, inputs, targets, nn.MSELoss())
    
    with profile(
        activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA],  # Auto-detects CUDA
        profile_memory=True,
        record_shapes=True,
        with_stack=True,
        schedule=torch.profiler.schedule(wait=0, warmup=1, active=3)
    ) as prof:
        pipelined_training_gpipe(local_model)
    
    prof.export_chrome_trace(f"pipeline_trace_rank{rank}.json")
    
    dist.destroy_process_group()