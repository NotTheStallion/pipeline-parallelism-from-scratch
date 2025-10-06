import torch
import torch.distributed as dist
from torch import nn
import matplotlib.pyplot as plt
import matplotlib.patches as patches


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



def plot_memory_and_schedule(schedule, T, delta_mem, p, m, filename_prefix="predef_zbts_p-1_hand"):        
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(14, 8), sharex=True,
                                gridspec_kw={'height_ratios': [2, 1]})

    op_colors = {'F_S': 'royalblue', 'F_T': 'orange', 'B': 'crimson', 'W': 'forestgreen'}

    for stage in range(1, p+1):
        for s, e, mb, op in sorted(schedule[stage]):
            ax1.barh(stage, e - s, left=s, height=0.6,
                    color=op_colors[op], edgecolor='black')
            ax1.text(s + (e - s)/2, stage, f"{op}{mb}", va='center', ha='center',
                    fontsize=12, color='white')
            
    ax1.set_ylabel("GPU", fontsize=12)
    ax1.set_yticks(range(1, p+1))
    ax1.set_ylim(0.5, p + 0.5)
    ax1.set_title("GPU Operation Schedule (F=Forward, B=Backward, W=Weight Update)", fontsize=14)
    ax1.grid(True, linestyle='--', alpha=0.4)
    handles = [patches.Patch(color=op_colors[c]) for c in op_colors]
    labels = list(op_colors.keys())
    ax1.legend(handles, labels, title='Operations', loc='upper right', fontsize=10, title_fontsize=12)


    time_points = range(15)
    for stage in range(1, p+1):
        events = sorted((s, op) for s, e, mb, op in schedule[stage])
        mem_timeline = []
        cur_mem = 0
        idx = 0
        for t in time_points:
            while idx < len(events) and events[idx][0] < t:
                cur_mem += delta_mem[events[idx][1]]
                idx += 1
            mem_timeline.append(cur_mem)
        ax2.plot(time_points, mem_timeline, label=f"GPU{stage}")

    ax2.set_xlabel("Time", fontsize=12)
    ax2.set_ylabel("Memory (GB)", fontsize=12)
    ax2.set_title("Per-GPU Memory Usage Over Time", fontsize=14)
    ax2.grid(True, linestyle='--', alpha=0.4)
    ax2.legend(fontsize=13)
    plt.tight_layout()
    plt.savefig("" + filename_prefix + ".png")



def precedes(a, b, y):
    # Check if task a precedes task b
    
    if a == b:
        return 1
    
    if (a, b) in y:
        return y[(a, b)]
    elif (b, a) in y:
        return 1 - y[(b, a)]
    else:
        raise ValueError(f"No precedence relation defined for {a} and {b}")
        