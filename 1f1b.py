import torch
import torch.distributed as dist
import torch.nn as nn
from src.data import MyDataset
from src.utils import sequential_forward, sequential_backward, fb_forward


def _forward(x, model_part):
    rank = dist.get_rank()
    world_size = dist.get_world_size()
    
    if not x.requires_grad:
        x.requires_grad_(True)
    
    return model_part(x)
    
def _backward(microoutput, microtarget, grad_output, loss_fn):
    rank = dist.get_rank()
    world_size = dist.get_world_size()
    
    if rank == world_size - 1:
        loss = loss_fn(microoutput, microtarget)
        loss.backward()
        return loss.item()
    else:
        microoutput.backward(grad_output)
        return None
        

def schedule_1f1b(model_part, microbatches, microtargets, loss_fn):
    rank = dist.get_rank()
    world_size = dist.get_world_size()
    
    microoutputs = [None] * len(microbatches)

    # Warm-up phase
    for step in range(world_size):
        # microbatch index forward is i = step - rank if i > 0

        i = step - rank
        
        if i >= 0:
            microbatches[i] = microbatches[i].requires_grad_(True)
            
            # Forward pass
            microoutputs[i] = _forward(microbatches[i], model_part)
            print(f"\033[94m+ F{i} [{rank}]\033[0m")
            
            if rank != 0:
                # print(f"\033[94mo{i} [{rank} <- {rank - 1}]\033[0m")
                dist.recv(microoutputs[i], src=rank - 1)
                
            if rank != world_size - 1:
                # print(f"\033[94mi{i} [{rank} -> {rank + 1}]\033[0m")
                dist.send(microoutputs[i], dst=rank + 1)
        
        # print(f"Rank {rank}: Warm-up forward microbatch {i}")
    
    print("\033[91mEND WARMUP\033[0m")
    exit(0)
    



def pipelined_iteration_1f1b(model_part, inputs, targets, loss_fn, chunck_num=2):
    """
    Implement one iteration of pipelined training using GPipe
    - Split the inputs and targets into microbatches
    - Perform forward passes for all microbatches (use sequential_forward)
    - Perform backward passes for all microbatches (use sequential_backward)
    """
    rank = dist.get_rank()
    world_size = dist.get_world_size()
    microbatches = list(torch.chunk(inputs, world_size * chunck_num))
    microtargets = list(torch.chunk(targets, world_size * chunck_num))
    
    total_loss = 0
    global_inputs = []
    global_outputs = []
    global_grads = []
    
    # Forward & Backward pass for all microbatches
    # if rank == 0:
    #     print(chunck_num, world_size)
    #     print(len(microbatches), len(microtargets))
    # total_loss = fb_forward(model_part, microbatches, microtargets, loss_fn, chunck_num)
    total_loss = schedule_1f1b(model_part, microbatches, microtargets, loss_fn)

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