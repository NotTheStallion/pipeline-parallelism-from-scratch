import torch
import torch.nn as nn
import torch.optim as optim
import torchvision.transforms as transforms
import torchvision.datasets as datasets
from src.utils import DeepNN, LightNN
import torch.distributed as dist
from src.utils import sequential_forward, sequential_backward


def train(model, train_loader, epochs, learning_rate, device):
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)

    model.train()

    for epoch in range(epochs):
        running_loss = 0.0
        for inputs, labels in train_loader:
            # inputs: A collection of batch_size images
            # labels: A vector of dimensionality batch_size with integers denoting class of each image
            inputs, labels = inputs.to(device), labels.to(device)

            optimizer.zero_grad()
            outputs = model(inputs)

            # outputs: Output of the network for the collection of images. A tensor of dimensionality batch_size x num_classes
            # labels: The actual labels of the images. Vector of dimensionality batch_size
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            running_loss += loss.item()

        print(f"Epoch {epoch+1}/{epochs}, Loss: {running_loss / len(train_loader)}")

def test(model, test_loader, device):
    model.to(device)
    model.eval()

    correct = 0
    total = 0

    with torch.no_grad():
        for inputs, labels in test_loader:
            inputs, labels = inputs.to(device), labels.to(device)

            outputs = model(inputs)
            _, predicted = torch.max(outputs.data, 1)

            total += labels.size(0)
            correct += (predicted == labels).sum().item()

    accuracy = 100 * correct / total
    print(f"Test Accuracy: {accuracy:.2f}%")
    return accuracy


def train_knowledge_distillation(teacher, student, train_loader, epochs, learning_rate, T, soft_target_loss_weight, ce_loss_weight, device):
    ce_loss = nn.CrossEntropyLoss()
    optimizer = optim.Adam(student.parameters(), lr=learning_rate)

    teacher.eval()  # Teacher set to evaluation mode
    student.train() # Student to train mode

    for epoch in range(epochs):
        running_loss = 0.0
        for inputs, labels in train_loader:
            inputs, labels = inputs.to(device), labels.to(device)

            optimizer.zero_grad()

            # Forward pass with the teacher model - do not save gradients here as we do not change the teacher's weights
            with torch.no_grad():
                teacher_logits = teacher(inputs)

            # Forward pass with the student model
            student_logits = student(inputs)

            #Soften the student logits by applying softmax first and log() second
            soft_targets = nn.functional.softmax(teacher_logits / T, dim=-1)
            soft_prob = nn.functional.log_softmax(student_logits / T, dim=-1)

            # Calculate the soft targets loss. Scaled by T**2 as suggested by the authors of the paper "Distilling the knowledge in a neural network"
            soft_targets_loss = torch.sum(soft_targets * (soft_targets.log() - soft_prob)) / soft_prob.size()[0] * (T**2)

            # Calculate the true label loss
            label_loss = ce_loss(student_logits, labels)

            # Weighted sum of the two losses
            loss = soft_target_loss_weight * soft_targets_loss + ce_loss_weight * label_loss

            loss.backward()
            optimizer.step()

            running_loss += loss.item()

        print(f"Epoch {epoch+1}/{epochs}, Loss: {running_loss / len(train_loader)}")



def custom_train_knowledge_distillation(teacher, student, train_loader, epochs, learning_rate, T, soft_target_loss_weight, ce_loss_weight, device):
    ce_loss = nn.CrossEntropyLoss()
    optimizer = optim.Adam(student.parameters(), lr=learning_rate)

    teacher.eval()  # Teacher set to evaluation mode
    student.train() # Student to train mode

    for epoch in range(epochs):
        running_loss = 0.0
        for inputs, labels in train_loader:
            inputs, labels = inputs.to(device), labels.to(device)

            optimizer.zero_grad()

            with torch.no_grad():
                teacher_logits = teacher(inputs)
            student_logits = student(inputs)


            y_hat = nn.functional.softmax(teacher_logits / T, dim=-1)
            y = nn.functional.softmax(student_logits / T, dim=-1)

            soft_targets_loss = torch.sum(y_hat * y.log()) * (T**2)
            label_loss = ce_loss(student_logits, labels)
            loss = - soft_targets_loss + label_loss

            loss.backward()
            optimizer.step()

            running_loss += loss.item()

        print(f"Epoch {epoch+1}/{epochs}, Loss: {running_loss / len(train_loader)}")


def tspipe(inputs, labels, teacher, student, ce_loss, optimizer, T, soft_target_loss_weight, ce_loss_weight):
    rank = dist.get_rank()
    world_size = dist.get_world_size()
    loss = 0.0
    
    # Warm up phase
    
    

    return loss


def dist_train_knowledge_distillation(teacher, student, train_loader, epochs, learning_rate, T, soft_target_loss_weight, ce_loss_weight, device):
    dist.init_process_group(backend="gloo")
    rank = dist.get_rank()
    world_size = dist.get_world_size()
    
    # Each rank gets a part of the teacher model
    teacher_layers = list(teacher.children())
    layers_per_rank = len(teacher_layers) // world_size
    teacher_part = teacher_layers[rank * layers_per_rank : (rank + 1) * layers_per_rank]
    print(f"Rank {rank} teacher model part: {teacher_part}")
    
    # Each rank gets a part of the student model
    student_layers = list(student.children())
    layers_per_rank = len(student_layers) // world_size
    student_part = student_layers[rank * layers_per_rank : (rank + 1) * layers_per_rank]
    print(f"Rank {rank} student model part: {student_part}")
    
    
    ce_loss = nn.CrossEntropyLoss()
    optimizer = optim.Adam(student.parameters(), lr=learning_rate)

    teacher.eval()  # Teacher set to evaluation mode
    student.train() # Student to train mode
    
    global_inputs = []
    global_labels = []
    for epoch in range(epochs):
        for inputs, labels in train_loader:
            global_inputs.append(inputs)
            global_labels.append(labels)
            
    tspipe(global_inputs, global_labels, teacher_part, student_part, ce_loss, optimizer, T, soft_target_loss_weight, ce_loss_weight)
    
    dist.destroy_process_group()



if __name__== "__main__":
    # Check if the current `accelerator <https://pytorch.org/docs/stable/torch.html#accelerators>`__
    # is available, and if not, use the CPU
    device = torch.accelerator.current_accelerator().type if torch.accelerator.is_available() else "cpu"
    print(f"Using {device} device")

    # Below we are preprocessing data for CIFAR-10. We use an arbitrary batch size of 128.
    transforms_cifar = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])

    # Loading the CIFAR-10 dataset:
    train_dataset = datasets.CIFAR10(root='/beegfs/mkherraz', train=True, download=True, transform=transforms_cifar)
    test_dataset = datasets.CIFAR10(root='/beegfs/mkherraz', train=False, download=True, transform=transforms_cifar)


    #Dataloaders
    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=128, shuffle=True)
    test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=128, shuffle=False)

    torch.manual_seed(42)
    nn_deep = DeepNN(num_classes=10).to(device)
    # train(nn_deep, train_loader, epochs=10, learning_rate=0.001, device=device)
    test_accuracy_deep = test(nn_deep, test_loader, device)

    # Instantiate the lightweight network:
    torch.manual_seed(42)
    nn_light = LightNN(num_classes=10).to(device)
    
    
    torch.manual_seed(42)
    new_nn_light = LightNN(num_classes=10).to(device)
    
    # Print the norm of the first layer of the initial lightweight model
    print("Norm of 1st layer of nn_light:", torch.norm(nn_light.features[0].weight).item())
    # Print the norm of the first layer of the new lightweight model
    print("Norm of 1st layer of new_nn_light:", torch.norm(new_nn_light.features[0].weight).item())
    
    
    total_params_deep = "{:,}".format(sum(p.numel() for p in nn_deep.parameters()))
    print(f"DeepNN parameters: {total_params_deep}")
    total_params_light = "{:,}".format(sum(p.numel() for p in nn_light.parameters()))
    print(f"LightNN parameters: {total_params_light}")
    
    # train(nn_light, train_loader, epochs=10, learning_rate=0.001, device=device)
    test_accuracy_light_ce = test(nn_light, test_loader, device)
    
    
    # Apply ``train_knowledge_distillation`` with a temperature of 2. Arbitrarily set the weights to 0.75 for CE and 0.25 for distillation loss.
    dist_train_knowledge_distillation(teacher=nn_deep, student=new_nn_light, train_loader=train_loader, epochs=10, learning_rate=0.001, T=2, soft_target_loss_weight=0.25, ce_loss_weight=0.75, device=device)
    test_accuracy_light_ce_and_kd = test(new_nn_light, test_loader, device)

    # Compare the student test accuracy with and without the teacher, after distillation
    print(f"Teacher accuracy: {test_accuracy_deep:.2f}%")
    print(f"Student accuracy without teacher: {test_accuracy_light_ce:.2f}%")
    print(f"Student accuracy with CE + KD: {test_accuracy_light_ce_and_kd:.2f}%")