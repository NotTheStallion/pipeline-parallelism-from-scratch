import torch
import torch.nn as nn
from zb_utils import LayerDW, replace_linear_with_linear_dw, LinearDX


model = nn.Sequential(
    nn.Flatten(),
    nn.Linear(10, 5)
)

replace_linear_with_linear_dw(model, device='cpu')



# First check
linear_dw_layer = model[1]
print(f"Using layer: {linear_dw_layer}")

# Toy input
inputs = [torch.randn(3, 10, requires_grad=True),  # Microbatch 1
          torch.randn(3, 10, requires_grad=True)]   # Microbatch 2

# forward passes for both microbatches
print("\n--- FORWARD PASS ---")

outputs = []
for mb_id, x in enumerate(inputs):
    print(f"Forwarding microbatch {mb_id}")
    out = model(x)
    outputs.append(out)
    
    # ! move the saved tensors from the layer's .last_* into internal queues
    linear_dw_layer.move_last_computed("input", mb_id)



print("\n--- BACKWARD PASS (dX) ---")
# Compute MSE loss
fake_targets = [torch.zeros_like(o) for o in outputs]
mse_loss = nn.MSELoss()

fake_grads_from_top = []
for out, target in zip(outputs, fake_targets):
    loss = mse_loss(out, target)
    fake_grads_from_top.append(torch.autograd.grad(loss, out, retain_graph=True)[0])


for mb_id, (out, fake_grad) in enumerate(zip(outputs, fake_grads_from_top)):
    print(f"Computing dX for microbatch {mb_id}")
    # This backward() call propagates the gradient backwards.
    # It will trigger LinearDX.backward, which computes dX and
    # saves fake_grad as linear_dw_layer.last_grad_output.
    # We then immediately move it to the queue (see step 3).
    out.backward(fake_grad, retain_graph=True) 
    linear_dw_layer.move_last_computed("grad_output", mb_id)
    # After this, input[mb_id].grad will be populated.

print(f"Gradient of input 0: {inputs[0].grad is not None}") # True
print(f"Gradient of input 1: {inputs[1].grad is not None}") # True
# @Note: linear_dw_layer.weight.grad is still None
print(f"Gradient of weights: {linear_dw_layer.weight.grad is not None}") # False
print(f"Gradient of bias: {linear_dw_layer.bias.grad is not None}")     # False


print("\n--- WEIGHT GRADIENT COMPUTATION (dW) ---")
for mb_id in range(len(inputs)):
    print(f"Computing dW from microbatch {mb_id}")
    # This is the key function that uses the saved .ctx["input"] and .ctx["grad_output"]
    # to compute dW and dB and accumulate them into linear_dw_layer.weight.grad and .bias.grad.
    linear_dw_layer.backward(mb_id)

print(f"Gradient of weights: {linear_dw_layer.weight.grad is not None}") # True
print(f"Gradient of bias: {linear_dw_layer.bias.grad is not None}")     # True
print(f"Shape of dW: {linear_dw_layer.weight.grad.shape}") # Should be (5, 10)