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


def trapezoid(global_inputs, global_targets, global_teacher_outputs, global_teacher_inputs, global_student_outputs, global_grads, i_f, s_f, t_f, nn_deep_part, nn_light_part, loss_fn, T, soft_target_loss_weight, ce_loss_weight, rank, world_size):
    # first student forward index : s_f
    # first teacher forward index : t_f
    # num of single box diagonal : i_f
    rank = dist.get_rank()
    world_size = dist.get_world_size()
    
    loss = 0
    
    for i in range(world_size - 1):
        if s_f+i >= len(global_inputs):
            break
        
        if rank != 0:
            # Receive inputs from the previous rank
            nvtx.push_range(message=f"m_s_i{s_f+i} from {rank-1}", color="green", domain="tspipe",
                        category="comm", payload=rank)
            # time.sleep(0.1)
            
            dist.recv(global_inputs[s_f+i], src=rank - 1)
            
            nvtx.pop_range(domain="tspipe")
        
        global_student_outputs[s_f+i] = _forward(global_inputs, s_f+i, nn_light_part)

        if rank != world_size - 1:
            # Send outputs to the next rank
            nvtx.push_range(message=f"m_s_o{s_f+i} to {rank+1}", color="green", domain="tspipe",
                        category="comm", payload=rank)
            # time.sleep(0.1)
            
            dist.send(global_student_outputs[s_f+i], dst=rank + 1)
            
            nvtx.pop_range(domain="tspipe")
    
    
    
    # red and orange trapezoid
    num_f_before_backward = 2 * ( world_size - 1 - rank)
    num_f = 0
    num_b = 0
    
    for i in range(4 * (world_size - 1)):
        # if t_f+i >= len(global_inputs):
        #     break
        
        
        if num_f < num_f_before_backward and num_b < 2 * (world_size - 1):
            # Teacher forwards before student backward
            
            if rank != 0 and t_f+num_f < len(global_inputs):
                # Receive inputs from the previous rank
                nvtx.push_range(message=f"m_t_i{t_f+num_f} from {rank-1}", color="green", domain="tspipe",
                            category="comm", payload=rank)
                # time.sleep(0.1)
                
                dist.recv(global_teacher_inputs[t_f+num_f], src=rank - 1)
                
                nvtx.pop_range(domain="tspipe")
            
            if t_f+num_f < len(global_inputs):
                global_teacher_outputs[t_f+num_f] = _forward(global_teacher_inputs, t_f+num_f, nn_deep_part, teacher=True)
            
            pos = s_f + num_b//2
            
            if rank != world_size - 1 and num_f+1 == num_f_before_backward and pos < len(global_inputs):
                # Receive gradients from the next rank
                global_grads[pos] = torch.empty_like(global_inputs[pos])
                
                nvtx.push_range(message=f"m_s_G{pos},{num_b%2} from {rank+1}", color="green", domain="tspipe",
                            category="comm", payload=rank)
                # time.sleep(0.1)
                
                dist.recv(global_grads[pos], src=rank + 1)
                
                nvtx.pop_range(domain="tspipe")
            
            
            if rank != world_size - 1 and t_f+num_f < len(global_inputs):
                # Send outputs to the next rank
                nvtx.push_range(message=f"m_t_o{t_f+num_f} to {rank+1}", color="green", domain="tspipe",
                            category="comm", payload=rank)
                # time.sleep(0.1)
                
                dist.send(global_teacher_outputs[t_f+num_f], dst=rank + 1)
                
                nvtx.pop_range(domain="tspipe")
            
            num_f += 1
        
        elif num_f >= num_f_before_backward and num_b < 2 * (world_size - 1):
            # Student backward
            pos = s_f + num_b//2
            
            if pos >= len(global_inputs):
                num_b += 1
                continue
            
            
            if rank != world_size - 1 and num_b > 0 :
                # Receive gradients from the next rank
                global_grads[pos] = torch.empty_like(global_inputs[pos])
                
                nvtx.push_range(message=f"m_s_g{pos},{num_b%2} from {rank+1}", color="green", domain="tspipe",
                            category="comm", payload=rank)
                # time.sleep(0.1)
                
                dist.recv(global_grads[pos], src=rank + 1)
                
                nvtx.pop_range(domain="tspipe")
            
            
            if rank != 0 and num_b <= 1 and t_f+num_f+num_b%2 < len(global_inputs):
                # Receive inputs from the previous rank
                nvtx.push_range(message=f"M_t_I{t_f+num_f+num_b%2} from {rank-1}", color="green", domain="tspipe",
                            category="comm", payload=rank)
                # time.sleep(0.1)
                
                dist.recv(global_inputs[t_f+num_f+num_b%2], src=rank - 1)
                
                nvtx.pop_range(domain="tspipe")
            
            # print(f"[rank : {rank}] s_f {s_f} num_b {num_b}")
            # print(f"Backwarding first, list pos : {pos} / microGbatch {s_f + num_b}")
            if num_b % 2 == 0:
                __loss = _backward(global_student_outputs, global_targets, global_teacher_outputs, global_grads, pos, loss_fn, rank, world_size, retain_graph=True)
                _loss = 0
            else:
                _loss = _backward(global_student_outputs, global_targets, global_teacher_outputs, global_grads, pos, loss_fn, rank, world_size)
                
            
            
            if rank == world_size - 1:
                loss += _loss
            # print("="*20)
            # print(global_inputs[pos].grad.shape)
            
            
            if rank != 0:
                # Send gradients to the previous rank
                nvtx.push_range(message=f"m_s_g{pos},{num_b%2} to {rank-1}", color="green", domain="tspipe",
                            category="comm", payload=rank)
                # time.sleep(0.1)
                
                dist.send(global_inputs[pos].grad, dst=rank - 1)
                
                nvtx.pop_range(domain="tspipe")
            
            num_b += 1
        else:
            # Resume teacher forwards
            if num_f > 2 * (world_size - 1):
                break
            
            if t_f + num_f >= len(global_inputs):
                num_f += 1
                continue
            
            
            if rank != 0 and num_f > num_f_before_backward + 1:
                # Receive inputs from the previous rank
                nvtx.push_range(message=f"M_t_i{t_f+num_f} from {rank-1}", color="green", domain="tspipe",
                            category="comm", payload=rank)
                # time.sleep(0.1)
                
                dist.recv(global_teacher_inputs[t_f+num_f], src=rank - 1)
                
                nvtx.pop_range(domain="tspipe")
            
            global_teacher_outputs[t_f+num_f] = _forward(global_teacher_inputs, t_f+num_f, nn_deep_part, teacher=True)

            if rank != world_size - 1:
                # Send outputs to the next rank
                nvtx.push_range(message=f"M_t_o{t_f+num_f} to {rank+1}", color="green", domain="tspipe",
                            category="comm", payload=rank)
                # time.sleep(0.1)
                
                dist.send(global_teacher_outputs[t_f+num_f], dst=rank + 1)
                
                nvtx.pop_range(domain="tspipe")
            
            num_f += 1
            
            
    
    i_f += world_size - 1
    s_f += world_size - 1
    t_f += 2 * (world_size - 1)
    
    if rank == world_size - 1:
        return i_f, s_f, t_f, loss
    return i_f, s_f, t_f, None



def train_tspipe(nn_deep_part, nn_light_part, global_inputs, global_targets, loss_fn, T, soft_target_loss_weight, ce_loss_weight, check, epochs, data_loader_len, optimizer):
    rank = dist.get_rank()
    world_size = dist.get_world_size()
    loss = 0
    
    global_teacher_outputs = [None] * len(global_inputs)
    global_teacher_inputs = copy.deepcopy(global_inputs)
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
                # time.sleep(0.1)
                
                dist.recv(global_teacher_inputs[i_f], src=rank - 1)
                
                nvtx.pop_range(domain="tspipe")
            
            global_teacher_outputs[i_f] = _forward(global_teacher_inputs, i_f, nn_deep_part, teacher=True)

            if rank != world_size - 1:
                # Send outputs to the next rank
                nvtx.push_range(message=f"m_t_o{i_f} to {rank+1}", color="green", domain="tspipe", 
                            category="comm", payload=rank)
                # time.sleep(0.1)
                # print(f"\033[91mRank {rank} sending teacher output {global_teacher_outputs[i_f].shape} to rank {rank + 1}\033[0m")
                dist.send(global_teacher_outputs[i_f], dst=rank + 1)
                
                nvtx.pop_range(domain="tspipe")
            
            t_f += 1
            
        else:
            # trapezoidal phase
            if rank == world_size - 1:
                print(f"Epoch {(s_f // (world_size - 1))/data_loader_len}/{epochs}")
            optimizer.zero_grad()
            
            i_f, s_f, t_f, loss = trapezoid(global_inputs, global_targets, global_teacher_outputs, global_teacher_inputs, global_student_outputs, global_grads, i_f, s_f, t_f, nn_deep_part, nn_light_part, loss_fn, T, soft_target_loss_weight, ce_loss_weight, rank, world_size)
            
            if rank == world_size - 1:
                print(loss)
            
            nvtx.push_range(message=f"OP", color="purple", domain="tspipe",
                            category="comm", payload=rank)
            # time.sleep(0.1)
            
            optimizer.step()
            
            nvtx.pop_range(domain="tspipe")
            
            
            if t_f >= len(global_inputs) & s_f >= len(global_inputs):
                break

        i_f += 1
    
    return loss, global_teacher_inputs, global_teacher_outputs, global_inputs, global_student_outputs, global_grads


def tspipe(nn_deep_part, nn_light_part, inputs, targets, loss_fn, T, soft_target_loss_weight, ce_loss_weight):
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
    
    # if rank == 0:
    #     print(f"input 0: {global_inputs[0]}")
    #     print(f"target 0: {global_targets[0]}")
    
    if rank == world_size - 1:
        print(f"Global inputs: {len(global_inputs)}")
        print(f"Global targets: {len(global_targets)}")
    
    loss, global_teacher_inputs, global_teacher_outputs, _, global_student_outputs, global_grads = train_tspipe(nn_deep_part, nn_light_part, global_inputs, global_targets, loss_fn, T, soft_target_loss_weight, ce_loss_weight, check, epochs, len(data_loader), optimizer)
    
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
    print(f"Rank {rank} LightNN model: {nn_light_part}")

    layers_per_rank = len(nn_deep) // world_size
    nn_deep_part = nn_deep[rank * layers_per_rank : (rank + 1) * layers_per_rank]
    print(f"Rank {rank} DeepNN model: {nn_deep_part}")
    
    
    
    nn_light_list = [ nn_light[r * layers_per_rank : (r + 1) * layers_per_rank] for r in range(world_size)]
    nn_deep_list = [ nn_deep[r * layers_per_rank : (r + 1) * layers_per_rank] for r in range(world_size)]
    
    
    # length : epochs * len(data_loader) * (world_size - 1)
    _inputs = []
    _targets = []
    for epoch in range(epochs):
        for ins, tas in data_loader:
            _inputs.extend(list(torch.chunk(ins, world_size - 1)))
            _targets.extend(list(torch.chunk(tas, world_size - 1)))
    
    
    if rank == world_size - 1:
        print(f"batches per epoch: {len(data_loader)}")
    
    _epoch = 0
    _batch = 0
    _microbatch = 0
    
    for i, (_input, targets) in enumerate(zip(_inputs, _targets)):
        
        epoch = ((i // (world_size - 1)) // len(data_loader)) % epochs
        batch = (i // (world_size - 1)) % len(data_loader)
        microbatch = i
        
        if rank == world_size - 1:
            print(f"Epoch {epoch}, Batch {batch}, Microbatch {microbatch}")
        
        # if i % len(data_loader) == 0:
        #     optimizer.zero_grad()
        #     loss = 0
        
        _input.requires_grad_(True).retain_grad()
        
        with torch.no_grad():
            # teacher_outputs = nn_deep(_input)
            _input1 = nn_deep_list[0](_input)
            _input2 = nn_deep_list[1](_input1)
            _input3 = nn_deep_list[2](_input2)
            teacher_outputs = nn_deep_list[3](_input3)
        
        # student_outputs = nn_light(_input)
        s_input1 = nn_light_list[0](_input)
        s_input1.retain_grad()
        s_input2 = nn_light_list[1](s_input1)
        s_input2.retain_grad()
        s_input3 = nn_light_list[2](s_input2)
        s_input3.retain_grad()
        student_outputs = nn_light_list[3](s_input3)
        
        y_hat = nn.functional.softmax(teacher_outputs / T, dim=-1)
        y = nn.functional.softmax(student_outputs / T, dim=-1)
        soft_targets_loss = torch.sum(y_hat * y.log()) * (T**2)
        ce_loss = loss_fn(student_outputs, targets)
        _loss = - soft_targets_loss + ce_loss
        
        # if rank == world_size - 1:
        #     print(f"Microbatch : {i} , Loss: {_loss.item()}")
        
        _loss.backward()
        
        loss += _loss.item()
        
        if i % (world_size - 1) == world_size - 2:
            optimizer.step()
        
        # epoch_loss += loss

        if rank == world_size - 1 and epoch != _epoch:
            # print(f"epoch loss: {loss/(len(data_loader))}")
            loss = 0
            optimizer.zero_grad()
        
        _epoch = epoch
        _batch = batch
        _microbatch = microbatch
        
        break
    
    # if rank == world_size - 1:
    #     print(f"Final loss: {loss/(len(data_loader))}")
    
    
    # if rank == world_size - 1:
    #     print(f"Inputs: {_inputs[0]}")
    #     print(f"Targets: {_targets[0]}")
        
        # print(f"First rank input grad: {_inputs[0].grad}")
        
        # print(f"student outputs: {student_outputs}")
        # print(f"teacher outputs: {teacher_outputs}")
        
        # print(f"Last rank student input: {_input3}")
        # print(f"Last rank student input grad: {s_input3.grad}")
    
    # !critical : model part doesn't do full piepline. 
    dist_inputs, dist_targets, dist_teacher_outputs, dist_teacher_inputs, dist_student_outputs, dist_global_grads = tspipe(nn_deep_part, nn_light_part, _inputs, _targets, loss_fn, T, soft_target_loss_weight, ce_loss_weight)
    
    
    if rank == 0:
        print("Comparing inputs and targets")
        for inp, dist_inp in zip(_inputs, dist_inputs):
            assert torch.allclose(inp, dist_inp), "Mismatch between _inputs and dist_inputs"
        
        for tar, dist_tar in zip(_targets, dist_targets):
            assert torch.allclose(tar, dist_tar), "Mismatch between _targets and dist_targets"
    
    if rank == world_size - 1:
        print("Comparing teacher outputs")
        assert torch.allclose(teacher_outputs, dist_teacher_outputs[0]), "Mismatch between global_teacher_outputs and dist_teacher_outputs"
        # for t_out, dist_t_out in zip(_teacher_outputs, dist_teacher_outputs):
        #     assert torch.allclose(t_out, dist_t_out), "Mismatch between global_teacher_outputs and dist_teacher_outputs"
        
        print("Comparing student outputs")
        assert torch.allclose(student_outputs, dist_student_outputs[0]), "Mismatch between global_teacher_outputs and dist_teacher_outputs"
        # for s_out, dist_s_out in zip(_student_outputs, dist_student_outputs):
        #     assert torch.allclose(s_out, dist_s_out), "Mismatch between global_student_outputs and dist_student_outputs"
    
    # checking last layer output
    # if rank == 0 :
    #     print(f"First rank inputs: {dist_inputs[0]}")
    #     print(f"First rank targets: {dist_targets[0]}")
    
    # if rank == world_size - 1:  
        # print(f"Last rank student output: {dist_student_outputs[0]}")
        # print(f"last rank teacher output: {dist_teacher_outputs[0]}")
        # print(f"Last rank student input: {dist_inputs[0]}")
        # print(f"Last rank student input grad: {dist_inputs[0].grad}")
        
    
    # checking gradients in first layer
    # if rank == 0:
    #     print(_inputs[0].grad.shape)
    #     print(dist_inputs[0].grad.shape)
        
        # print(f"Inputs grad: {_inputs[0].grad}")
        # print(f"Distributed Inputs grad: {dist_inputs[0].grad}")
        
        # torch.allclose(_inputs[0].grad, dist_inputs[0].grad)
    
    dist.destroy_process_group()