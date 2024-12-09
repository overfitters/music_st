import torch
import torch.nn as nn

device = torch.device("cuda" if torch.cuda.is_available else "cpu")

N_FFT = 2048 #512
N_CHAN = round(1 + N_FFT / 2)
OUT_CHAN = 32


# ulyanov
class Random2DCNN(nn.Module):
    def __init__(self):
        super().__init__()

        self.conv1 = nn.Conv2d(1, OUT_CHAN, kernel_size=(3, 1), stride=1, padding=0)
        self.LeakyReLU = nn.LeakyReLU(0.2)

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


# alishdipani
class CNNModel(nn.Module):
    def __init__(self):
        super(CNNModel, self).__init__()
        self.cnn1 = nn.Conv1d(
            in_channels=1025, out_channels=4096, kernel_size=3, stride=1, padding=1
        )

    def forward(self, x):
        out = self.cnn1(x)
        out = out.view(out.size(0), -1)
        return out


class GramMatrix(nn.Module):
    def forward(self, input):
        a, b, c = input.size()  # a=batch size(=1)
        # b=number of feature maps
        # (c,d)=dimensions of a f. map (N=c*d)
        features = input.view(a * b, c)  # resise F_XL into \hat F_XL
        G = torch.mm(features, features.t())  # compute the gram product
        # we 'normalize' the values of the gram matrix
        # by dividing by the number of element in each feature maps.
        return G.div(a * b * c)


class StyleLoss(nn.Module):
    def __init__(self, target, weight):
        super(StyleLoss, self).__init__()
        self.target = target.detach() * weight
        self.weight = weight
        self.gram = GramMatrix()
        self.criterion = nn.MSELoss()

    def forward(self, inp):
        self.output = inp.clone()
        self.G = self.gram(inp)
        self.G.mul_(self.weight)
        self.loss = self.criterion(self.G, self.target)
        return self.output

    def backward(self, retain_graph=True):
        self.loss.backward(retain_graph=retain_graph)
        return self.loss
