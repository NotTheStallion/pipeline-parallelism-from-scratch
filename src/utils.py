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
    
    for inputs, targets in zip(microbatches, microtargets):
        if rank != 0:
            # Receive inputs from the previous rank
            inputs = torch.zeros_like(inputs, requires_grad=True)
            inputs.retain_grad()
            dist.recv(inputs, src=rank - 1)
        
        if rank != world_size - 1:
            outputs = model_part(inputs)

        if rank != world_size - 1:
            # Send outputs to the next rank
            dist.send(outputs, dst=rank + 1)
        
        if rank == world_size - 1:
            # Compute loss and backward
            outputs = model_part(inputs)
            
            loss = loss_fn(outputs, targets)
            loss.backward()
            
            dist.send(inputs.grad, dst=rank - 1)
            
            print(f"Rank {rank} loss: {loss.item()}")
    
    
    for i, _ in enumerate(microbatches):
        if i <= rank < world_size - 1:
            # Receive gradients from the next rank and backward
            grad_outputs = torch.zeros_like(outputs, requires_grad=True)
            dist.recv(grad_outputs, src=rank + 1)
            outputs.backward(grad_outputs)

        if i < rank < world_size - 1:
            # Send gradients to the previous rank
            dist.send(inputs.grad, dst=rank - 1)
    
    
    
    # Interleaving forward and backward passes
    for i, (inputs, targets) in enumerate(zip(microbatches, microtargets)):
        if rank != 0:
            # Receive inputs from the previous rank
            inputs = torch.zeros_like(inputs, requires_grad=True)
            inputs.retain_grad()
            dist.recv(inputs, src=rank - 1)
        
        if rank != world_size - 1 :
            grad_outputs = torch.zeros_like(outputs, requires_grad=True)
            dist.recv(grad_outputs, src=rank + 1)
            outputs.backward(grad_outputs)
            
            
            outputs = model_part(inputs)
            # Send outputs to the next rank
            dist.send(outputs, dst=rank + 1)
        
        if rank != 0:
            if i != 0 & rank != world_size - 1:
                # Send gradients to the previous rank
                dist.send(inputs.grad, dst=rank - 1)
        
        if rank == world_size - 1 :
            # Compute loss and backward
            outputs = model_part(inputs)
            
            loss = loss_fn(outputs, targets)
            loss.backward()
            
            dist.send(inputs.grad, dst=rank - 1)
            
            print(f"Rank {rank} loss: {loss.item()}")
    
    
    return inputs, targets, loss_fn
    