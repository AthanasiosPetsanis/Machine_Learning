import torch
import torch.nn as nn
import torch.nn.functional as F


class CreateNet(nn.Module):
    def __init__(self, arch, input_size, hid_neurons, hid_layers,output_size):
        super().__init__()
        if arch=='conv':
            self.model= ConvNetwork(input_size, hid_neurons, hid_layers,output_size)
        else: self.model= FCNetwork(input_size, hid_neurons, hid_layers,output_size)
        
    def forward(self,x):
        out=self.model(x)
        return out

class FCNetwork(nn.Module):
    def __init__(self, input_size, hid_neurons, hid_layers,output_size):
        super().__init__()
        self.layers = nn.ModuleList()
        self.hid_layers=hid_layers
        self.layers.append(nn.Linear(input_size,hid_neurons))
        for i in range(hid_layers-1):
            self.layers.append(nn.Linear(hid_neurons,hid_neurons))
        self.out=nn.Linear(hid_neurons,output_size)
        
    def forward(self,x):
        for idx in range(self.hid_layers):
            x=self.layers[idx](x)
            x=torch.square(torch.tanh(x)) # tanh^2 
        #x=torch.flatten(x,1)
        out=self.out(x)
        # out=torch.sigmoid(out) # none
        return out

class ConvNetwork(nn.Module):
    def __init__(self, input_size, filters, hid_layers,output_size):
        super().__init__()
        self.layers = nn.ModuleList()
        self.hid_layers=hid_layers
        self.layers.append(nn.Conv1d(in_channels=1, out_channels=filters, kernel_size=3,padding=1))
        for i in range(hid_layers-1):
            self.layers.append(nn.Conv1d(in_channels=filters, out_channels=filters, kernel_size=3,padding=1))
        self.out=nn.Linear(filters*input_size,output_size)
        
    def forward(self,x):
        x=x.unsqueeze(1)
        for idx in range(self.hid_layers):
            x=self.layers[idx](x)
            x=F.relu(x)
        #out=torch.flatten(x,1)
        out=self.out(out)
        #out=torch.flatten(out,1)
        return out