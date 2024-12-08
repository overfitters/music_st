import torch
import torch.nn as nn

device = torch.device("cuda" if torch.cuda.is_available else "cpu")

N_FFT = 512
N_CHAN = round(1 + N_FFT / 2)
OUT_CHAN = 32


class RandomCNN(nn.Module):
    def __init__(self):
        super().__init__()

        # 2-D CNN
        self.conv1 = nn.Conv2d(1, OUT_CHAN, kernel_size=(3, 1), stride=1, padding=0)
        self.LeakyReLU = nn.LeakyReLU(0.2)

        # Set the random parameters to be constant.
        weight = torch.randn(self.conv1.weight.data.shape)
        self.conv1.weight = torch.nn.Parameter(weight, requires_grad=False)
        bias = (
            torch.zeros(self.conv1.bias.data.shape)
            if self.conv1.bias is not None
            else None
        )
        self.conv1.bias = torch.nn.Parameter(bias, requires_grad=False)

    def forward(self, x_delta):
        out = self.LeakyReLU(self.conv1(x_delta))
        return out
