import torch
import torch.distributed as dist

def sequential_forward(model_part, inputs):
    """
    Handles the forward pass in a distributed pipeline
    
    - For all ranks except the first (rank 0), receives inputs from the previous rank
    - Processes the inputs through the local model segment
    - For all ranks except the last, sends the outputs to the next rank
    """
    rank = dist.get_rank()
    world_size = dist.get_world_size()
    
    if rank != 0:
        # Receive inputs from the previous rank
        inputs = torch.zeros_like(inputs, requires_grad=True)
        inputs.retain_grad()
        dist.recv(inputs, src=rank - 1)
    
    # print(f"Rank {rank} inputs grad: {inputs.requires_grad}")
    
    outputs = model_part(inputs)

    if rank != world_size - 1:
        # Send outputs to the next rank
        dist.send(outputs, dst=rank + 1)

    return inputs, outputs


def sequential_backward(inputs, outputs, targets, loss_fn):
    """
    Executes a backward pass in a pipeline-parallel distributed setup
    
    - Last rank computes the loss and backwards from there
    - Other ranks receive gradients from the next rank and perform backward on outputs with received gradients
    - All ranks except first send gradients to the previous rank

    hint: tensor.backward() can take a gradient tensor as an argument
    
    Returns the loss on the last rank
    """
    rank = dist.get_rank()
    world_size = dist.get_world_size()
    
    
    if rank == world_size - 1:
        # Compute loss and backward
        loss = loss_fn(outputs, targets)
        loss.backward()
    else:
        # Receive gradients from the next rank and backward
        grad_outputs = torch.zeros_like(outputs, requires_grad=True)
        dist.recv(grad_outputs, src=rank + 1)
        outputs.backward(grad_outputs)

    if rank != 0:
        # Send gradients to the previous rank
        dist.send(inputs.grad, dst=rank - 1)


    if rank == world_size - 1:
        return loss



def pipelined_iteration(model, inputs, targets, loss_fn):
    """
    Executes a pipelined forward and backward pass through the model
    
    - For each rank, performs a forward pass
    - For the last rank, computes the loss and performs backward pass
    - For other ranks, receives gradients from the next rank and performs backward on outputs with received gradients
    
    Returns the total loss on the last rank
    """
    inputs, outputs = sequential_forward(model, inputs)
    
    return sequential_backward(inputs, outputs, targets, loss_fn)


def fb_forward(model_part, microbatches, microtargets, loss_fn):
    
    rank = dist.get_rank()
    world_size = dist.get_world_size()
    
    micrograds = [0] * world_size * 2
    microoutputs = [0] * world_size * 2
    microbatches = microbatches * 2
    microtargets = microtargets * 2
    
    for i in range(world_size):
        if i >= world_size:
            break
        
        if rank != 0:
            # Receive inputs from the previous rank
            microbatches[i].requires_grad_(True)
            microbatches[i].retain_grad()
            # print(f"\033[94mReceive inputs {i} rank {rank} <- {rank - 1}\033[0m")
            dist.recv(microbatches[i], src=rank - 1)
        
        if rank != world_size - 1:
            outputs = model_part(microbatches[i])
            microoutputs[i] = outputs

        if rank != world_size - 1:
            # Send outputs to the next rank
            # print(f"\033[94mSend outputs {i} rank {rank} -> {rank + 1}\033[0m")
            dist.send(outputs, dst=rank + 1)
        
        if rank == world_size - 1:
            # print(f"F/Backward pass rank {rank} batch {i}")
            
            # Compute loss and backward
            outputs = model_part(microbatches[i])
            microoutputs[i] = outputs
            
            loss = loss_fn(outputs, microtargets[i])
            loss.backward()
            
            if i != world_size - 1:
                # Send gradients to the previous rank
                # print(f"\033[91mSend inputs.grad {i} rank {rank} -> {rank - 1}\033[0m")
                dist.send(microbatches[i].grad, dst=rank - 1)
            print(f"Rank {rank} loss: {loss.item()}")
        
        if rank == world_size - 2 and i != world_size - 1:
            # Receive gradients from the next rank and backward
            grad_outputs = torch.zeros_like(microoutputs[i], requires_grad=True)
            micrograds[i] = grad_outputs
            
            # print(f"\033[94;1mReceive grad_outputs {i} rank {rank} <- {rank + 1}\033[0m")
            dist.recv(grad_outputs, src=rank + 1)
            
            microoutputs[i].backward(grad_outputs)
    
    
    if rank == 0:
        # import pdb; pdb.set_trace()
        print("=========== Yellow triangle")
    
    for i, inputs in enumerate(microbatches):
        if i >= world_size:
            break
        
        if i != world_size - 2:
            if i <= rank < world_size - 2:
                # Receive gradients from the next rank and backward
                grad_outputs = torch.zeros_like(microoutputs[i], requires_grad=True)
                micrograds[i] = grad_outputs
                
                # print(f"\033[93m+ Recv {i} rank {rank} <- {rank + 1}\033[0m")
                dist.recv(grad_outputs, src=rank + 1)
                microoutputs[i].backward(grad_outputs)

            if i < rank < world_size - 1:
                # Send gradients to the previous rank
                # print(f"\033[93m+ Send {i} rank {rank} -> {rank - 1}\033[0m")
                dist.send(inputs.grad, dst=rank - 1)
    
    if rank == 0:
        print("========== Trapezoid")
    
    # Interleaving forward and backward passes
    for i in range(world_size, world_size * 2):
        
        if rank != 0:
            # Receive inputs from the previous rank
            microbatches[i].requires_grad_(True)
            microbatches[i].retain_grad()
            print(f"\033[93mReceive inputs {i} rank {rank} <- {rank - 1}\033[0m")
            dist.recv(microbatches[i], src=rank - 1)
        
        
        if rank != world_size - 1 :
            tmp_outputs = model_part(microbatches[i])
            microoutputs[i] = tmp_outputs
            
            # Send outputs to the next rank
            print(f"\033[95mSend outputs {i} rank {rank} -> {rank + 1}\033[0m")
            dist.send(tmp_outputs, dst=rank + 1)
        
            k = i - ( world_size - rank ) + 1

            grad_outputs = torch.zeros_like(microoutputs[k], requires_grad=True)
            micrograds[k] = grad_outputs
            print(f"\033[92mReceive grad_outputs {k} rank {rank} <- {rank + 1}\033[0m")
            dist.recv(grad_outputs, src=rank + 1)
            microoutputs[k].backward(grad_outputs)
        
        if rank != 0:
            # Send gradients to the previous rank
            k = i - ( world_size - rank )
            print(f"\033[94mSend inputs.grad {k} rank {rank} -> {rank - 1}\033[0m")
            dist.send(microbatches[k].grad, dst=rank - 1)
        
        
        if rank == world_size - 1 :
            # Compute loss and backward
            outputs = model_part(microbatches[i])
            microoutputs[i] = outputs
            
            loss = loss_fn(outputs, microtargets[i])
            loss.backward()
            
            print(f"Rank {rank} loss: {loss.item()}")
    
    if rank == 0:
        print("Black triangle")
    
    for i, _ in enumerate(microbatches):
        if 0 <= rank < world_size - 1 - i:
            # Receive gradients from the next rank and backward
            grad_outputs = torch.zeros_like(outputs, requires_grad=True)
            dist.recv(grad_outputs, src=rank + 1)
            outputs.backward(grad_outputs)

        if 0 < rank <= world_size - 1 - i:
            # Send gradients to the previous rank
            dist.send(inputs.grad, dst=rank - 1)
    
    
    
    return inputs, targets, loss_fn
    