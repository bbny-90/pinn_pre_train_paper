import torch

class SiLU(torch.nn.Module):
    @staticmethod
    def forward(x):
        return x * torch.sigmoid(x)