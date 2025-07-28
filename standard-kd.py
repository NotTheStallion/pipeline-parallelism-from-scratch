import torch
import torch.nn as nn
import torch.optim as optim
import torch.distributed as dist
from src.utils import sequential_forward, sequential_backward
from torch.utils.data import DataLoader, TensorDataset
from src.data import MyDataset





if __name__== "__main__":
    dist.init_process_group(backend="gloo")
    rank = dist.get_rank()
    world_size = dist.get_world_size()

    torch.manual_seed(42)

    nn_deep = nn.Sequential(
        nn.Linear(32, 64),
        nn.ReLU(),
        nn.Linear(64, 128),
        nn.ReLU(),
        nn.Linear(128, 64),
        nn.ReLU(),
        nn.Linear(64, 32),
        nn.Identity() # an even number of layers is easier to split
    )
    
    for param in nn_deep.parameters():
        dist.broadcast(param.data, src=0) 

    
    nn_light = nn.Sequential(
        nn.Linear(32, 16),
        nn.ReLU(),
        nn.Linear(16, 16),
        nn.ReLU(),
        nn.Linear(16, 16),
        nn.ReLU(),
        nn.Linear(16, 32),
        nn.Identity() # an even number of layers is easier to split
    )
    
    for param in nn_light.parameters():
        dist.broadcast(param.data, src=0) 

    total_params_deep = "{:,}".format(sum(p.numel() for p in nn_deep.parameters()))
    if rank == world_size - 1:
        print(f"DeepNN parameters: {total_params_deep}")
    total_params_light = "{:,}".format(sum(p.numel() for p in nn_light.parameters()))
    if rank == world_size - 1:
        print(f"LightNN parameters: {total_params_light}")

    
    dataset = MyDataset()
    loss_fn = nn.MSELoss()
    optimizer = torch.optim.Adam(nn_deep.parameters())
    batch_size = 16 # @param
    
    # * Ensure the data is shuffled in the same way across all devices
    generator = torch.Generator()
    generator.manual_seed(42)
    data_loader = torch.utils.data.DataLoader(dataset, batch_size=batch_size, shuffle=True, generator=generator)
    
    if rank == world_size - 1:
        print("Training DeepNN CE")

    for epoch in range(10):
        epoch_loss = 0
        for inputs, targets in data_loader:
            optimizer.zero_grad()
            outputs = nn_deep(inputs)
            loss = loss_fn(outputs, targets)
            loss.backward()
            optimizer.step()
            epoch_loss += loss

        if rank == world_size - 1:
            print(f"Epoch {epoch} loss: {epoch_loss / len(data_loader)}")
    
    
    # if rank == world_size - 1:
    #     print("Training LightNN CE")
    
    # optimizer = torch.optim.Adam(nn_light.parameters())
    
    # for epoch in range(10):
    #     epoch_loss = 0
    #     for inputs, targets in data_loader:
    #         optimizer.zero_grad()
    #         outputs = nn_light(inputs)
    #         loss = loss_fn(outputs, targets)
    #         loss.backward()
    #         optimizer.step()
    #         epoch_loss += loss

    #     if rank == world_size - 1:
    #         print(f"Epoch {epoch} loss: {epoch_loss / len(data_loader)}")
    
    
    
    if rank == world_size - 1:
        print("Training LightNN CE + KD")
        
    optimizer = torch.optim.Adam(nn_light.parameters())
    
    T = 2.0  # Temperature for softening the outputs
    soft_target_loss_weight = 0.5  # Weight for the soft target loss
    ce_loss_weight = 0.5  # Weight for the cross-entropy loss   
    
    for epoch in range(10):
        epoch_loss = 0
        for inputs, targets in data_loader:
            optimizer.zero_grad()
            
            with torch.no_grad():
                teacher_outputs = nn_deep(inputs)
            student_outputs = nn_light(inputs)
            
            # # Pytorch version of KD
            # soft_targets = nn.functional.softmax(teacher_outputs / T, dim=-1)
            # soft_prob = nn.functional.log_softmax(student_outputs / T, dim=-1)
            
            # soft_targets_loss = torch.sum(soft_targets * (soft_targets.log() - soft_prob)) / soft_prob.size()[0] * (T**2)
            # ce_loss = loss_fn(student_outputs, targets)
            # loss = soft_target_loss_weight * soft_targets_loss + ce_loss_weight * ce_loss
            
            # Custom version of KD
            y_hat = nn.functional.softmax(teacher_outputs / T, dim=-1)
            y = nn.functional.softmax(student_outputs / T, dim=-1)
            soft_targets_loss = torch.sum(y_hat * y.log()) * (T**2)
            ce_loss = loss_fn(student_outputs, targets)
            loss = - soft_targets_loss + ce_loss
            
            loss.backward()
            optimizer.step()
            
            epoch_loss += loss

        if rank == world_size - 1:
            print(f"Epoch {epoch} loss: {epoch_loss / len(data_loader)}")
    
    
    
    
    
    dist.destroy_process_group()