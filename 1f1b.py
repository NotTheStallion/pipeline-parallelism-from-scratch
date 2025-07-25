import torch
import torch.distributed as dist
import torch.nn as nn
from src.data import MyDataset
from src.utils import sequential_forward, sequential_backward, fb_forward
import nvtx
import time



def _forward(microinputs, index, model_part):
    
    nvtx.push_range(message=f"F{index}", color="blue", domain="1f1b", 
                            category="forward", payload=rank)
    
    microinput = microinputs[index]
    
    # time.sleep(0.3)  # Simulate some processing time
    
    # if not microinput.requires_grad:
    microinput.requires_grad_(True).retain_grad()
    
    result = model_part(microinput)
    
    nvtx.pop_range(domain="1f1b")
    return result
    
def _backward(microoutputs, microtargets, grad_outputs, index, loss_fn, rank, world_size):
    nvtx.push_range(message=f"B{index}", color="red", domain="1f1b", 
                            category="backward", payload=rank)
    # time.sleep(0.6)  # Simulate some processing time
    
    if microoutputs:
        microoutput = microoutputs[index]
    if microtargets:
        microtarget = microtargets[index]
    if grad_outputs:
        grad_output = grad_outputs[index]
    
    if rank == world_size - 1:
        loss = loss_fn(microoutput, microtarget)
        loss.backward()
        
        nvtx.pop_range(domain="1f1b")
        return loss.item()
    else:
        microoutput.backward(grad_output)
        
        nvtx.pop_range(domain="1f1b")
        return None
        

def schedule_1f1b(model_part, microbatches, microtargets, loss_fn, ops=False, comms=False):
    rank = dist.get_rank()
    world_size = dist.get_world_size()
    
    total_loss = 0
    
    microoutputs = [None] * len(microbatches)
    micrograds = [None] * len(microbatches)

    # Warm-up phase
    for step in range(world_size):
        # microbatch index forward is i = step - rank if i > 0

        i = step - rank
        
        if i >= 0:
            if rank != 0:
                nvtx.push_range(message=f"m_i{i} from {rank-1}", color="green", domain="1f1b", 
                            category="comm", payload=rank)
                # time.sleep(0.1)
                
                if comms : print(f"\033[94mi{i} [{rank} <- {rank - 1}]\033[0m")
                dist.recv(microbatches[i], src=rank - 1)
                
                nvtx.pop_range(domain="1f1b")
            
            # * No grad is requires in the first rank for inputs
            microbatches[i].requires_grad_(True)
            microbatches[i].retain_grad()
            
            # Forward pass
            if ops : print(f"\033[94m+ F{i} [{rank}]\033[0m")
            
            microoutputs[i] = _forward(microbatches, i, model_part)
                
            if rank != world_size - 1 and step != world_size - 1:
                nvtx.push_range(message=f"m_o{i} to {rank+1}", color="green", domain="1f1b", 
                            category="comm", payload=rank)
                # time.sleep(0.1)
                
                if comms : print(f"\033[94mo{i} [{rank} -> {rank + 1}]\033[0m")
                dist.send(microoutputs[i], dst=rank + 1)
                
                nvtx.pop_range(domain="1f1b")

    
    if rank == world_size - 1 and (ops or comms):
        print("\033[91mEND WARMUP\033[0m")
    
    # 1F1B steady phase
    start_idx = world_size - rank
    i = world_size - rank
    while i < len(microbatches) + start_idx:
        
        # Backward pass (for microbatch i - (world_size - 1 - rank))
        b_idx = i - (world_size - rank)
        if b_idx >= 0:
              
            if rank == world_size - 1:
                # Last rank computes loss
                
                if ops : print(f"\033[91m+ B{b_idx} [{rank}]\033[0m")
                loss = _backward(microoutputs, microtargets, None, b_idx, loss_fn, rank, world_size)
                total_loss += loss
            
            
            if rank != world_size - 1:
                # Receive gradient from next rank
                micrograds[b_idx] = torch.empty_like(microbatches[b_idx])
                
                nvtx.push_range(message=f"m_g{b_idx} from {rank+1}", color="green", domain="1f1b", 
                            category="comm", payload=rank)
                # time.sleep(0.1)
                
                if comms : print(f"G{b_idx} [{rank} <- {rank + 1}]")
                dist.recv(micrograds[b_idx], src=rank+1)
                
                nvtx.pop_range(domain="1f1b")
            
            
            # Send from previous rank
            if rank != world_size - 1 and i-1 < len(microbatches):
                nvtx.push_range(message=f"m_O{i-1} to {rank+1}", color="green", domain="1f1b", 
                            category="comm", payload=rank)
                # time.sleep(0.1)
                
                if comms : print(f"= O{i-1} [{rank} -> {rank + 1}]")
                dist.send(microoutputs[i-1], dst=rank+1)
                
                nvtx.pop_range(domain="1f1b")
            
            
            if rank != world_size - 1:
                if ops : print(f"\033[91m+ B{b_idx} [{rank}]\033[0m")
                
                # if b_idx == 0 and rank == 0:
                #     print("inside o1:", microoutputs[0])
                #     print("inside g1:", micrograds[0])
                
                # if b_idx == 0 and rank == 1:
                #     print("inside o2:", microoutputs[0])
                #     print("inside g2:", micrograds[0])
                _backward(microoutputs, None, micrograds, b_idx, loss_fn, rank, world_size)
                # if b_idx == 0 and rank == 1:
                #     print("inside o1:", microbatches[0])
                #     print("inside o1.grad:", microbatches[0].grad)
                
                # if b_idx == 0 and rank == 0:
                #     print("inside input:", microbatches[0])
                #     print("inside input.grad:", microbatches[0].grad)
        
        
            # Send grads to previosu rank
            if rank != 0:
                nvtx.push_range(message=f"m_G{b_idx} to {rank-1}", color="green", domain="1f1b", 
                            category="comm", payload=rank)
                # time.sleep(0.1)
                
                if comms : print(f"G{b_idx} [{rank} -> {rank - 1}]")
                dist.send(microbatches[b_idx].grad, dst=rank - 1)
                
                nvtx.pop_range(domain="1f1b")
        
        
        
        # Forward pass
        if rank == 0:
            
            # First rank processes new microbatch
            # * skip forward if non existent microbatch
            if i >= len(microbatches):
                if ops : print(f"\033[93m- F{i} [{rank}]\033[0m")
            else:
                # * No grad is requires in the first rank for inputs
                microbatches[i].requires_grad_(True)
                microbatches[i].retain_grad()
                
                if ops : print(f"\033[94m+ F{i} [{rank}]\033[0m")
                microoutputs[i] = _forward(microbatches, i, model_part)

        
        if rank != 0:
            # * skip forward if non existent microbatch
            if i >= len(microbatches):
                if ops : print(f"\033[93m- F{i} [{rank}]\033[0m")
            else:
                # Receive from previous rank
                microbatches[i] = torch.empty_like(microbatches[i])
                
                nvtx.push_range(message=f"m_I{i} from; {rank-1}", color="green", domain="1f1b", 
                            category="comm", payload=rank)
                # time.sleep(0.1)
                
                if comms : print(f"I{i} [{rank} <- {rank - 1}]")
                dist.recv(microbatches[i], src=rank-1)
                
                nvtx.pop_range(domain="1f1b")
                
                
                microbatches[i].requires_grad_(True)
                microbatches[i].retain_grad()
            
                if ops : print(f"\033[94m+ F{i} [{rank}]\033[0m")
                microoutputs[i] = _forward(microbatches, i, model_part)
        
        
        i += 1
        # print(f"Rank {rank}: Forward microbatch {i}")
        
    
    if rank == world_size - 1 and (ops or comms):
        print("\033[91mEND STEADY\033[0m")

    return microbatches, micrograds, microoutputs, total_loss



def pipelined_iteration_1f1b(model_part, inputs, targets, loss_fn, chunck_num=2, model=None, check=False):
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
    
    global_outputs = []
    global_grads = []
    
    # Forward & Backward pass for all microbatches
    # if rank == 0:
    #     print(chunck_num, world_size)
    #     print(len(microbatches), len(microtargets))
    # total_loss = fb_forward(model_part, microbatches, microtargets, loss_fn, chunck_num)

    microbatches, micrograds, microoutputs, total_loss = schedule_1f1b(model_part, microbatches, microtargets, loss_fn)
    
    
    if check == True:
        # Send microoutputs from rank world_size - 1 to rank 0
        if rank == world_size - 1:
            for microoutput in microoutputs:
                dist.send(microoutput, dst=0)
        elif rank == 0:
            last_rank_outputs = []
            for _ in range(len(microoutputs)):
                tensor = torch.empty_like(microoutputs[0])
                dist.recv(tensor, src=world_size - 1)
                last_rank_outputs.append(tensor)

    
    if rank == 0 and check == True:
        import copy
        full_inputs = copy.deepcopy(microbatches)
        full_targets = copy.deepcopy(microtargets)
        
        chunck_model = [model[r * chunck_num : (r + 1) * chunck_num] for r in range(world_size)]

        for i, input in enumerate(full_inputs):
            input.requires_grad_(True)
            input.retain_grad()
            
            # Forward pass through the chunked model
            o1 = chunck_model[0](input)
            o1.retain_grad()
            o2 = chunck_model[1](o1)
            o2.retain_grad()
            o3 = chunck_model[2](o2)
            o3.retain_grad()
            output = chunck_model[3](o3)
            
            
            loss = loss_fn(output, full_targets[i])
            loss.backward()
            
            # print("o2 :", o2)
            # print("o2.grad :", o2.grad)
            
            
            # print(chunck_model[0])
            
            # print("o1 :", o1)
            # print("o1.grad :", o1.grad)
            # print("input :", input)
            # print("input.grad :", input.grad)
            # print("traget :", full_targets[i])
            
            global_outputs.append(output)
            global_grads.append(input.grad)

            

        
        print(f"Checking inputs...")
        for i, input in enumerate(full_inputs):
            assert torch.allclose(input, microbatches[i]), f"Mismatch in microbatch {i} input"
        
        print(f"Checking targets...")
        for i, target in enumerate(full_targets):
            assert torch.allclose(target, microtargets[i]), f"Mismatch in microbatch {i} target"

        print(f"Checking outputs...")
        for i, output in enumerate(global_outputs):
            assert torch.allclose(output, last_rank_outputs[i]), f"Mismatch in microbatch {i} output"
        
        print("CHecking W gradients...")
        for param, chunk_param in zip(local_model.parameters(), chunck_model[0].parameters()):
            assert torch.allclose(param.grad, chunk_param.grad), f"Gradient for {param} doesn't match chunked model parameter {chunk_param}. Difference: {torch.norm(param.grad - chunk_param.grad)}"

    
    return total_loss




def pipelined_training_1f1b(model_part, chunck_num=2, model=None):
    """
    Perform pipelined training on a full dataset
    For each batch:
    - Perform pipelined iteration (use pipelined_iteration)
    - Update the model parameters
    """
    rank = dist.get_rank()
    world_size = dist.get_world_size()

    check = True
    dataset = MyDataset()
    loss_fn = nn.MSELoss()
    optimizer = torch.optim.Adam(model_part.parameters())
    batch_size = world_size * chunck_num
    
    # * Ensure the data is shuffled in the same way across all devices
    generator = torch.Generator()
    generator.manual_seed(42)
    data_loader = torch.utils.data.DataLoader(dataset, batch_size=batch_size, shuffle=True, generator=generator)

    for epoch in range(10):
        epoch_loss = 0
        for inputs, targets in data_loader:
            optimizer.zero_grad()
            loss = pipelined_iteration_1f1b(model_part, inputs, targets, loss_fn, chunck_num, model, check=check)
            check=False
            optimizer.step()
            if rank == world_size - 1:
                epoch_loss += loss

        if rank == world_size - 1:
            print(f"[Rank {rank}] Epoch {epoch} loss: {epoch_loss / len(data_loader)}")





if __name__ == "__main__":
    dist.init_process_group(backend="gloo")
    rank = dist.get_rank()
    world_size = dist.get_world_size()

    torch.manual_seed(42)

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

    for param in model.parameters():
        dist.broadcast(param.data, src=0) 

    # Each rank gets a part of the model
    layers_per_rank = len(model) // world_size
    local_model = model[rank * layers_per_rank : (rank + 1) * layers_per_rank]
    print(f"Rank {rank} model: {local_model}")
    
    chunck_num = 2 # @param 
    inputs = torch.randn(256, 32) # inputs to the full model
    targets = torch.randn(256, 32) # targets
    
    # Broadcast inputs and targets to all ranks
    dist.broadcast(inputs, src=0)
    dist.broadcast(targets, src=0)
    
    # inputs, outputs = sequential_forward(local_model, inputs)
    
    # sequential_backward(inputs, outputs, targets, nn.MSELoss())
    
    # pipelined_iteration_1f1b(local_model, inputs, targets, nn.MSELoss(), chunck_num)
    
    pipelined_training_1f1b(local_model, chunck_num, model)
    
    dist.destroy_process_group()