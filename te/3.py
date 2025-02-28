import torch
from torch.distributions.normal import Normal

# Define a normal distribution
mean, std = torch.tensor(0.0), torch.tensor(1.0)
distribution = Normal(mean, std)

# Sample using sample()
sample = distribution.sample()
print("Sample:", sample)

# Sample using rsample()
rsample = distribution.rsample()
print("Rsample:", rsample)

# Trying to compute gradients
sample.requires_grad_()  # This will have no effect
rsample.requires_grad_()
gradient = torch.autograd.grad(rsample, [mean, std], grad_outputs=torch.tensor(1.0))
print("Gradient with rsample:", gradient)