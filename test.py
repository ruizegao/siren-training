import torch
from torch.utils.data import DataLoader
from torch.optim import Adam
import dataio
from torch import nn
from siren_pytorch import SirenNet, Siren

net = SirenNet(
    dim_in = 2,                        # input dimension, ex. 2d coor
    dim_hidden = 256,                  # hidden dimension
    dim_out = 3,                       # output dimension, ex. rgb value
    num_layers = 5,                    # number of layers
    final_activation = nn.Sigmoid(),   # activation of final layer (nn.Identity() for direct output)
    w0_initial = 30.                   # different signals may require different omega_0 in the first layer - this is a hyperparameter
)

coor = torch.randn(1, 2)
print(net(coor)) # (1, 3) <- rgb value
