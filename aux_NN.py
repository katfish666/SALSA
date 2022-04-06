import torch
import torch.nn as nn
import torch.nn.functional as F

class NeuralNetwork(nn.Module):
    def __init__(self,input_dim,output_dim,
                 layer_dims=[128]):
        super().__init__()
        
        self.dims = [input_dim] + layer_dims + [output_dim] 
        self.linears = nn.ModuleList([nn.Linear(self.dims[i],
                                                self.dims[i+1])\
                                      for i in range(len(self.dims)-1)])

    def forward(self, x):
        for layer in self.linears[:-1]:
            x = layer(x)
            x = nn.ReLU()(x)
        x = self.linears[-1](x)
        
        return x