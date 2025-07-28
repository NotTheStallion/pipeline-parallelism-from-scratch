import torch
import torch.nn as nn
import torch.optim as optim
import torch.distributed as dist
from src.utils import sequential_forward, sequential_backward
from torch.utils.data import DataLoader, TensorDataset
from src.data import MyDataset
import nvtx
import time


def _forward(microinputs, index, model_part, teacher=False):
    color = "orange" if teacher else "blue"
    nvtx.push_range(message=f"F{index}", color=color, domain="tspipe", 
                    category="forward", payload=rank)
    
    microinput = microinputs[index]
    
    time.sleep(0.3)  # Simulate some processing time
    
    if teacher:
        with torch.no_grad():
            result = model_part(microinput)
    else:
        microinput.requires_grad_(True).retain_grad()
        result = model_part(microinput)
    
    nvtx.pop_range(domain="tspipe")
    return result
    
def _backward(microoutputs, microtargets, grad_outputs, index, loss_fn, rank, world_size):
    nvtx.push_range(message=f"B{index}", color="red", domain="tspipe", 
                            category="backward", payload=rank)
    time.sleep(0.6)  # Simulate some processing time
    
    if microoutputs:
        microoutput = microoutputs[index]
    if microtargets:
        microtarget = microtargets[index]
    if grad_outputs:
        grad_output = grad_outputs[index]
    
    if rank == world_size - 1:
        loss = loss_fn(microoutput, microtarget)
        loss.backward()
        del microoutput
        
        nvtx.pop_range(domain="tspipe")
        return loss.item()
    else:
        microoutput.backward(grad_output)
        del grad_output
        del microoutput
        
        nvtx.pop_range(domain="tspipe")
        return None


def trapezoid(global_inputs, global_teacher_outputs, global_student_outputs, global_grads, i_f, s_f, t_f, nn_deep_part, nn_light_part, loss_fn, T, soft_target_loss_weight, ce_loss_weight, rank, world_size):
    # first student forward index : s_f
    # first teacher forward index : t_f
    # num of single box diagonal : i_f
    rank = dist.get_rank()
    world_size = dist.get_world_size()
    
    for i in range(world_size - 1):
        if rank != 0:
            # Receive inputs from the previous rank
            nvtx.push_range(message=f"m_s_i{s_f+i} from {rank-1}", color="green", domain="tspipe",
                        category="comm", payload=rank)
            time.sleep(0.1)
            
            dist.recv(global_inputs[s_f+i], src=rank - 1)
            
            nvtx.pop_range(domain="tspipe")
        
        global_student_outputs[s_f+i] = _forward(global_inputs, s_f+i, nn_light_part)

        if rank != world_size - 1:
            # Send outputs to the next rank
            nvtx.push_range(message=f"m_s_o{s_f+i} to {rank+1}", color="green", domain="tspipe",
                        category="comm", payload=rank)
            time.sleep(0.1)
            
            dist.send(global_student_outputs[s_f+i], dst=rank + 1)
            
            nvtx.pop_range(domain="tspipe")
    
    
    
    # red and orange trapezoid
    for i in range(2 * (world_size - 1)):
        if rank != 0:
            # Receive inputs from the previous rank
            nvtx.push_range(message=f"m_s_i{t_f+i} from {rank-1}", color="green", domain="tspipe",
                        category="comm", payload=rank)
            time.sleep(0.1)
            
            dist.recv(global_inputs[t_f+i], src=rank - 1)
            
            nvtx.pop_range(domain="tspipe")
        
        global_teacher_outputs[t_f+i] = _forward(global_inputs, t_f+i, nn_light_part, teacher=True)

        if rank != world_size - 1:
            # Send outputs to the next rank
            nvtx.push_range(message=f"m_s_o{t_f+i} to {rank+1}", color="green", domain="tspipe",
                        category="comm", payload=rank)
            time.sleep(0.1)
            
            dist.send(global_teacher_outputs[t_f+i], dst=rank + 1)
            
            nvtx.pop_range(domain="tspipe")
    
    i_f+= world_size - 1
    s_f += world_size - 1
    return i_f, s_f, t_f



def train_tspipe(nn_deep_part, nn_light_part, global_inputs, global_targets, loss_fn, T, soft_target_loss_weight, ce_loss_weight, check, epochs, data_loader_len, optimizer):
    rank = dist.get_rank()
    world_size = dist.get_world_size()
    loss = 0
    
    global_teacher_outputs = [None] * len(global_inputs)
    global_student_outputs = [None] * len(global_inputs)
    global_grads = [None] * len(global_inputs) 
    
    i_f=0
    s_f=0
    t_f=0
    while True:
        # Warmup phase
        if i_f < world_size - 1:
            # First forwards
            if rank != 0:
                # Receive inputs from the previous rank
                nvtx.push_range(message=f"m_t_i{i_f} from {rank-1}", color="green", domain="tspipe", 
                            category="comm", payload=rank)
                time.sleep(0.1)
                
                dist.recv(global_inputs[i_f], src=rank - 1)
                
                nvtx.pop_range(domain="tspipe")
            
            global_teacher_outputs[i_f] = _forward(global_inputs, i_f, nn_deep_part, teacher=True)

            if rank != world_size - 1:
                # Send outputs to the next rank
                nvtx.push_range(message=f"m_t_o{i_f} to {rank+1}", color="green", domain="tspipe", 
                            category="comm", payload=rank)
                time.sleep(0.1)
                
                dist.send(global_teacher_outputs[i_f], dst=rank + 1)
                
                nvtx.pop_range(domain="tspipe")
            
            t_f += 1
            
        else:
            # trapezoidal phase
            if t_f >= len(global_inputs):
                break
            
            if s_f >= len(global_inputs):
                break
            
            i_f, s_f, t_f = trapezoid(global_inputs, global_teacher_outputs, global_student_outputs, global_grads, i_f, s_f, t_f, nn_deep_part, nn_light_part, loss_fn, T, soft_target_loss_weight, ce_loss_weight, rank, world_size)
            
            if s_f >= 3 & t_f >= 9:
                break
            
            
            
        
        i_f += 1
    
    return loss


def tspipe(nn_deep_part, nn_light_part, data_loader, loss_fn, T, soft_target_loss_weight, ce_loss_weight):
    rank = dist.get_rank()
    world_size = dist.get_world_size()

    check = True
    dataset = MyDataset()
    loss_fn = nn.MSELoss()
    optimizer = torch.optim.Adam(nn_light_part.parameters())
    batch_size = 12 # @param
    epochs = 10 # @param
    
    # * Ensure the data is shuffled in the same way across all devices
    generator = torch.Generator()
    generator.manual_seed(42)
    data_loader = torch.utils.data.DataLoader(dataset, batch_size=batch_size, shuffle=True, generator=generator)

    global_inputs = []
    global_targets = []
    for epoch in range(epochs):
        for inputs, targets in data_loader:
            global_inputs.extend(list(torch.chunk(inputs, world_size - 1)))
            global_targets.extend(list(torch.chunk(targets, world_size - 1)))
    
    train_tspipe(nn_deep_part, nn_light_part, global_inputs, global_targets, loss_fn, T, soft_target_loss_weight, ce_loss_weight, check, epochs, len(data_loader), optimizer)
            
    


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
    batch_size = 12 # @param
    
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
    
    # Saving weights of untrained LightNN
    dist_nn_light = nn_light.state_dict()
    
    optimizer = torch.optim.Adam(nn_light.parameters())
    
    T = 2.0  # Temperature for softening the outputs
    soft_target_loss_weight = 0.5  # Weight for the soft target loss
    ce_loss_weight = 0.5  # Weight for the cross-entropy loss   
    
    # for epoch in range(10):
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
    
    
    
    tspipe(nn_deep, nn_light, data_loader, loss_fn, T, soft_target_loss_weight, ce_loss_weight)
    
    
    dist.destroy_process_group()