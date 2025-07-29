import torch
import torch.nn as nn
from src.data import MyDataset


if __name__ == "__main__":
    torch.manual_seed(42)

    
    model = nn.Sequential(
        nn.Linear(32, 16),
        nn.ReLU(),
        nn.Linear(16, 16),
        nn.ReLU(),
        nn.Linear(16, 16),
        nn.ReLU(),
        nn.Linear(16, 32),
        nn.Identity() # an even number of layers is easier to split
    )

    
    dataset = MyDataset(dist=False)
    loss_fn = nn.MSELoss()
    batch_size = 12 # @param
    
    generator = torch.Generator()
    generator.manual_seed(42)
    data_loader = torch.utils.data.DataLoader(dataset, batch_size=batch_size, shuffle=True, generator=generator)


    optimizer = torch.optim.Adam(model.parameters())


    for epoch in range(10):
        epoch_loss = 0
        for inputs, targets in data_loader:
            optimizer.zero_grad()
            inputs.requires_grad = True
            inputs.retain_grad()
            
            print(f"Inputs shape: {inputs.shape}, targets shape: {targets.shape}")
            
            outputs = model(inputs)
            
            print(inputs.grad)
            
            half_idx = outputs.shape[0] // 2
            
            first_half_outputs = outputs[:half_idx]
            first_half_targets = targets[:half_idx]
            print(f"first half outputs shape: {first_half_outputs.shape}, targets shape: {first_half_targets.shape}")
            
            
            first_half_loss = loss_fn(first_half_outputs, first_half_targets)
            first_half_loss.backward(retain_graph=True)
            
            print(inputs.grad.shape)
            # print(inputs.grad)
            
            second_half_outputs = outputs[half_idx:]
            second_half_targets = targets[half_idx:]
            print(f"second half outputs shape: {second_half_outputs.shape}, targets shape: {second_half_targets.shape}")
            
            second_half_loss = loss_fn(second_half_outputs, second_half_targets)
            second_half_loss.backward()
            
            print(inputs.grad.shape)
            # print(inputs.grad)
            # print(f"Inputs shape: {inputs.shape}, output shape : {outputs.shape}, targets shape: {targets.shape}")

            optimizer.step()
            epoch_loss += first_half_loss.item() + second_half_loss.item()

        print(f"Epoch {epoch} loss: {epoch_loss / len(data_loader)}")