import torch
import torch.distributed as dist
import torch.nn as nn
from src.data import MyDataset
from src.utils import sequential_forward, sequential_backward, fb_forward






def pipelined_iteration_1f1b(model_part, inputs, targets, loss_fn, chunck_num=2):
    """
    Implement one iteration of pipelined training using GPipe
    - Split the inputs and targets into microbatches
    - Perform forward passes for all microbatches (use sequential_forward)
    - Perform backward passes for all microbatches (use sequential_backward)
    """
    rank = dist.get_rank()
    world_size = dist.get_world_size()
    microbatches = torch.chunk(inputs, world_size * chunck_num)
    microtargets = torch.chunk(targets, world_size * chunck_num)
    
    total_loss = 0
    global_inputs = []
    global_outputs = []
    global_grads = []
    
    # Forward & Backward pass for all microbatches
    # if rank == 0:
    #     print(chunck_num, world_size)
    #     print(len(microbatches), len(microtargets))
    total_loss = fb_forward(model_part, microbatches, microtargets, loss_fn, chunck_num)

    return total_loss




def pipelined_training_1f1b(model_part, chunck_num=2):
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
    batch_size = world_size * chunck_num
    data_loader = torch.utils.data.DataLoader(dataset, batch_size=batch_size, shuffle=True)

    for epoch in range(10):
        epoch_loss = 0
        for inputs, targets in data_loader:
            optimizer.zero_grad()
            loss = pipelined_iteration_1f1b(model_part, inputs, targets, loss_fn, chunck_num)
            optimizer.step()
            if rank == world_size - 1:
                epoch_loss += loss

        if rank == world_size - 1:
            print(f"[Rank {rank}] Epoch {epoch} loss: {epoch_loss / len(data_loader)}")





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
    
    chunck_num = 4 # @param 
    inputs = torch.randn(256, 32) # inputs to the full model
    targets = torch.randn(256, 32) # targets
    
    # inputs, outputs = sequential_forward(local_model, inputs)
    
    # sequential_backward(inputs, outputs, targets, nn.MSELoss())
    
    # pipelined_iteration_1f1b(local_model, inputs, targets, nn.MSELoss(), chunck_num)
    
    pipelined_training_1f1b(local_model, chunck_num)
    
    dist.destroy_process_group()