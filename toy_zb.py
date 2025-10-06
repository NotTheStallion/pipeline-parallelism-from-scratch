import torch
import torch.nn as nn
from src.zb_utils import LayerDW, replace_linear_with_linear_dw, LinearDX


model = nn.Sequential(
    nn.Flatten(),
    nn.Linear(10, 5)
)

replace_linear_with_linear_dw(model, device='cpu')



linear_dw_layer = model[1]
print(f"Using layer: {linear_dw_layer}")


inputs = [torch.randn(3, 10, requires_grad=True),
          torch.randn(3, 10, requires_grad=True)]



outputs = []
for mb_id, x in enumerate(inputs):
    out = model(x)
    outputs.append(out)
    
    # ! move the saved tensors from the layer's .last_* into internal queues
    linear_dw_layer.move_last_computed("input", mb_id)




fake_targets = [torch.zeros_like(o) for o in outputs]
mse_loss = nn.MSELoss()

fake_grads_from_top = []
for out, target in zip(outputs, fake_targets):
    loss = mse_loss(out, target)
    fake_grads_from_top.append(torch.autograd.grad(loss, out, retain_graph=True)[0])


for mb_id, (out, fake_grad) in enumerate(zip(outputs, fake_grads_from_top)):
    # This backward() call propagates the gradient backwards.
    # It will trigger LinearDX.backward, which computes dX and
    # saves fake_grad as linear_dw_layer.last_grad_output.
    # We then immediately move it to the queue (see step 3).
    out.backward(fake_grad, retain_graph=True) 
    linear_dw_layer.move_last_computed("grad_output", mb_id)
    # After this, input[mb_id].grad will be populated.

assert inputs[0].grad is not None, "Gradient of input 0 should not be None"
assert inputs[1].grad is not None, "Gradient of input 1 should not be None"
# @Note: linear_dw_layer.weight.grad is still None
assert linear_dw_layer.weight.grad is None, "Gradient of weights should be None"
assert linear_dw_layer.bias.grad is None, "Gradient of bias should be None"


for mb_id in range(len(inputs)):
    # This is the key function that uses the saved .ctx["input"] and .ctx["grad_output"]
    # to compute dW and dB and accumulate them into linear_dw_layer.weight.grad and .bias.grad.
    linear_dw_layer.backward(mb_id)

assert linear_dw_layer.weight.grad is not None, "Gradient of weights should not be None"
assert linear_dw_layer.bias.grad is not None, "Gradient of bias should not be None"
assert linear_dw_layer.weight.grad.shape == (5, 10), f"Shape of dW should be (5, 10), got {linear_dw_layer.weight.grad.shape}"