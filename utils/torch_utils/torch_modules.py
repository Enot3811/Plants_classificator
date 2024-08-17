import torch


class Flatten(torch.nn.Module):
    def forward(self, x: torch.Tensor):
        return torch.flatten(x, 1)
