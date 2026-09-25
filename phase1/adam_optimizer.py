import torch

def loss_function(params):
    x, y = params[0], params[1]
    return 0.1 * x**2 + 5 * y**2

params = torch.tensor([-8.0, 3.0], requires_grad=True)
optimizer = torch.optim.Adam(params, lr=0.3)

for step in range(30):
    optimizer.zero_grad()
    loss = loss_function(params)
    loss.backward()
    optimizer.step()
    print(f"Step {step}: x = {params[0].item():.4f}, y = {params[1].item():.4f}, loss = {loss.item():.4f}")

print("Adam final point:", params.detach().numpy())