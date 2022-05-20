from cmath import tanh
import torch
import torch.nn as nn

class NeuralNetwork(nn.Module):
    def __init__(self, input_size, hid_neurons, w1, w2, hid_layers=1, output_size=1):
        super().__init__()
        self.L1 = nn.Linear(input_size, hid_neurons, bias=False)
        # Do not train first layer weights
        for param in self.L1.parameters():
            param.requires_grad = False
        self.L2 = nn.Linear(hid_neurons, output_size, bias=False)
        self.tanh = nn.Tanh()
        self.L1.weight.data = torch.as_tensor(w1, dtype=torch.float32)
        self.L2.weight.data = torch.as_tensor(w2, dtype=torch.float32)

    def forward(self,x):

        x = self.L1(x) # w1*xx_k_n
        x = self.tanh(x) # tanh(w1*xx_k_n)
        x = torch.square(x) # tanh(w1*xx_k_n).^2
        out = self.L2(x) # w2*tanh(w1*xx_k_n).^2
        return out
