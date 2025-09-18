from zb_utils import LinearDX, LayerDW, replace_linear_with_linear_dw
import torch
import torch.nn as nn








import torch
import torch.nn as nn
import torch.optim as optim
import torch.distributed as dist
from src.utils import sequential_forward, sequential_backward
from torch.utils.data import DataLoader, TensorDataset
from src.data import MyDataset
import nvtx
import time
import copy


def _forward(microinputs, index, model_part, teacher=False):
    color = "orange" if teacher else "blue"
    nvtx.push_range(message=f"F{index}", color=color, domain="tspipe", 
                    category="forward", payload=rank)
    
    microinput = microinputs[index]
    
    # time.sleep(0.3)  # Simulate some processing time
    
    if teacher:
        with torch.no_grad():
            result = model_part(microinput)
    else:
        microinput.requires_grad_(True).retain_grad()
        result = model_part(microinput)
    
    nvtx.pop_range(domain="tspipe")
    return result
    
def _backward(student_outputs, microtargets, teacher_outputs, grad_outputs, index, loss_fn, rank, world_size, retain_graph=False):
    nvtx.push_range(message=f"B{index}", color="red", domain="tspipe", 
                            category="backward", payload=rank)
    # time.sleep(0.3)  # Simulate some processing time
    
    if student_outputs:
        student_output = student_outputs[index]
    if teacher_outputs:
        teacher_output = teacher_outputs[index]
    if microtargets:
        microtarget = microtargets[index]
    if grad_outputs:
        grad_output = grad_outputs[index]
    
    if rank == world_size - 1:
        
        y_hat = nn.functional.softmax(teacher_output / T, dim=-1)
        y = nn.functional.softmax(student_output / T, dim=-1)
        soft_targets_loss = torch.sum(y_hat * y.log()) * (T**2)
        ce_loss = loss_fn(student_output, microtarget)
        loss = - soft_targets_loss + ce_loss
        
        print(f"Loss for microbatch {index}: {loss.item()}")

        loss.backward(retain_graph=retain_graph)
        # del microoutput
        
        nvtx.pop_range(domain="tspipe")
        return loss.item()
    else:
        student_output.backward(grad_output, retain_graph=retain_graph)
        # del grad_output
        # del microoutput
        
        nvtx.pop_range(domain="tspipe")
        return None



def train_zbts(nn_deep_part, nn_light_part, inputs, targets, loss_fn, T, soft_target_loss_weight, ce_loss_weight, check, epochs, batches_per_epoch, optimizer):
    return None, None, None, None, None, None


def zbts(nn_deep_part, nn_light_part, inputs, targets, loss_fn, T, soft_target_loss_weight, ce_loss_weight):
    rank = dist.get_rank()
    world_size = dist.get_world_size()

    check = True
    loss_fn = nn.MSELoss(reduction='sum')
    optimizer = torch.optim.Adam(nn_light_part.parameters())
    epochs = 2 # @param

    if rank == world_size - 1:
        print(f"Training with {len(data_loader)} batches per epoch in {epochs} epochs and {world_size - 1} chunks")

    global_inputs = copy.deepcopy(inputs)
    global_targets = copy.deepcopy(targets)
    
    loss, global_teacher_inputs, global_teacher_outputs, _, global_student_outputs, global_grads = train_zbts(nn_deep_part, nn_light_part, global_inputs, global_targets, loss_fn, T, soft_target_loss_weight, ce_loss_weight, check, epochs, len(data_loader), optimizer)
    
    return global_inputs, global_targets, global_teacher_outputs, global_teacher_inputs, global_student_outputs, global_grads
    


if __name__== "__main__":
    dist.init_process_group(backend="gloo")
    rank = dist.get_rank()
    world_size = dist.get_world_size()
    
    print(f"Rank {rank} of {world_size} started")

    torch.manual_seed(42)

    nn_deep = nn.Sequential(
        nn.Linear(32, 32),
        nn.ReLU(),
        nn.Linear(32, 32),
        nn.ReLU(),
        nn.Linear(32, 32),
        nn.ReLU(),
        nn.Linear(32, 32),
        nn.ReLU(),
        nn.Linear(32, 32),
        nn.ReLU(),
        nn.Linear(32, 32),
        nn.ReLU(),
        nn.Linear(32, 32),
        nn.ReLU(),
        nn.Linear(32, 32),
        nn.ReLU(),
        nn.Linear(32, 32),
        nn.ReLU(),
        nn.Linear(32, 32),
        nn.ReLU(),
        nn.Linear(32, 32),
        nn.ReLU(),
        nn.Linear(32, 32),
        nn.Identity() # an even number of layers is easier to split
    )
    
    for param in nn_deep.parameters():
        dist.broadcast(param.data, src=0) 

    
    nn_light = nn.Sequential(
        nn.Linear(32, 32),
        nn.ReLU(),
        nn.Linear(32, 32),
        nn.ReLU(),
        nn.Linear(32, 32),
        nn.ReLU(),
        nn.Linear(32, 32),
        nn.ReLU(),
        nn.Linear(32, 32),
        nn.ReLU(),
        nn.Linear(32, 32),
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

    
    dataset = MyDataset(n=42, seed=42)
    loss_fn = nn.MSELoss()
    batch_size = 21 # @param
    epochs = 2 # @param
    
    # * Ensure the data is shuffled in the same way across all devices
    generator = torch.Generator()
    generator.manual_seed(42)
    data_loader = torch.utils.data.DataLoader(dataset, batch_size=batch_size, shuffle=True, generator=generator)
    
    if rank == world_size - 1:
        print("Training DeepNN CE")
    
    optimizer = torch.optim.Adam(nn_deep.parameters())

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
    
    # for epoch in range(epochs):
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
    
    # Saving weights of untrained LightNN
    dist_nn_light = nn_light.state_dict()
    
    optimizer = torch.optim.Adam(nn_light.parameters())
    
    T = 2.0  # Temperature for softening the outputs
    soft_target_loss_weight = 0.5  # Weight for the soft target loss
    ce_loss_weight = 0.5  # Weight for the cross-entropy loss   
    
    # for epoch in range(epochs):
    #     epoch_loss = 0
    #     for inputs, targets in data_loader:
    #         optimizer.zero_grad()
            
    #         with torch.no_grad():
    #             teacher_outputs = nn_deep(inputs)
    #         student_outputs = nn_light(inputs)
            
    #         # # Pytorch version of KD
    #         # soft_targets = nn.functional.softmax(teacher_outputs / T, dim=-1)
    #         # soft_prob = nn.functional.log_softmax(student_outputs / T, dim=-1)
            
    #         # soft_targets_loss = torch.sum(soft_targets * (soft_targets.log() - soft_prob)) / soft_prob.size()[0] * (T**2)
    #         # ce_loss = loss_fn(student_outputs, targets)
    #         # loss = soft_target_loss_weight * soft_targets_loss + ce_loss_weight * ce_loss
            
    #         # Custom version of KD
    #         y_hat = nn.functional.softmax(teacher_outputs / T, dim=-1)
    #         y = nn.functional.softmax(student_outputs / T, dim=-1)
    #         soft_targets_loss = torch.sum(y_hat * y.log()) * (T**2)
    #         ce_loss = loss_fn(student_outputs, targets)
    #         loss = - soft_targets_loss + ce_loss
            
    #         loss.backward()
    #         optimizer.step()
            
    #         epoch_loss += loss

    #     if rank == world_size - 1:
    #         print(f"Epoch {epoch} loss: {epoch_loss / len(data_loader)}")
    
    
    loss_fn = nn.MSELoss(reduction='sum')
    
    # ! TSPipe single GPU
    
    layers_per_rank = len(nn_light) // world_size
    nn_light_part = nn_light[rank * layers_per_rank : (rank + 1) * layers_per_rank]
    nn_light_list = [ nn_light[r * layers_per_rank : (r + 1) * layers_per_rank] for r in range(world_size)]
    print(f"Rank {rank} LightNN model: {nn_light_part}")

    layers_per_rank = len(nn_deep) // world_size
    nn_deep_part = nn_deep[rank * layers_per_rank : (rank + 1) * layers_per_rank]
    nn_deep_list = [ nn_deep[r * layers_per_rank : (r + 1) * layers_per_rank] for r in range(world_size)]
    print(f"Rank {rank} DeepNN model: {nn_deep_part}")
    
    
    
    
    
    # length : epochs * len(data_loader) * (world_size - 1)
    _inputs = []
    _targets = []
    for epoch in range(epochs):
        for ins, tas in data_loader:
            _inputs.extend(list(torch.chunk(ins, world_size - 1)))
            _targets.extend(list(torch.chunk(tas, world_size - 1)))
    
    
    if rank == world_size - 1:
        print(f"batches per epoch: {len(data_loader)}")
    
    
    # !critical : model part doesn't do full piepline. 
    dist_inputs, dist_targets, dist_teacher_outputs, dist_teacher_inputs, dist_student_outputs, dist_global_grads = zbts(nn_deep_part, nn_light_part, _inputs, _targets, loss_fn, T, soft_target_loss_weight, ce_loss_weight)
    
    dist.destroy_process_group()