import torch
import torch.distributed as dist
from torch import nn


# Deeper neural network class to be used as teacher:
class DeepNN(nn.Module):
    def __init__(self, num_classes=10):
        super(DeepNN, self).__init__()
        self.features = nn.Sequential(
            nn.Conv2d(3, 128, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv2d(128, 64, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Conv2d(64, 64, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv2d(64, 32, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),
        )
        self.classifier = nn.Sequential(
            nn.Linear(2048, 512),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(512, num_classes)
        )

    def forward(self, x):
        x = self.features(x)
        x = torch.flatten(x, 1)
        x = self.classifier(x)
        return x



# Lightweight neural network class to be used as student:
class LightNN(nn.Module):
    def __init__(self, num_classes=10):
        super(LightNN, self).__init__()
        self.features = nn.Sequential(
            nn.Conv2d(3, 16, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Conv2d(16, 16, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),
        )
        self.classifier = nn.Sequential(
            nn.Linear(1024, 256),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(256, num_classes)
        )

    def forward(self, x):
        x = self.features(x)
        x = torch.flatten(x, 1)
        x = self.classifier(x)
        return x



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

