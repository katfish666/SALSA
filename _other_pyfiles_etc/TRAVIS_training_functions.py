#< ---- 20 char ---->< ---- 20 char ---->< ---- 20 char ---->< --- 18 char --->

import numpy as np
import torch
import torch.nn as nn
from torch.autograd import Variable
import torch.nn.functional as F

import math
import random

from tqdm.auto import trange, tqdm

# from smiles_enumerator import SmilesEnumerator

def padded_cross_entropy_loss(tgt, model_out, pad_mask,out_mask):
#     pad_mask = torch.where(seq == pad_idx, torch.zeros_like(seq),
#                            torch.ones_like(seq))
    
    pad_mask = torch.ones_like(pad_mask.float()) - pad_mask.float()
    pad_mask = torch.flatten(pad_mask[:,1:]).float()
    weights = torch.ones_like(out_mask[0,0]) - out_mask[0,0]
    criterion = nn.CrossEntropyLoss(reduction = 'none',weight = weights)
    
    unred_loss = criterion(torch.flatten(model_out[:,:-1],
                                         end_dim = -2),
                           torch.flatten(tgt[:,1:]))
    
    return torch.matmul(unred_loss,pad_mask)/torch.sum(pad_mask)


def train_step(model, sample, optimizer,
               variational = False, use_out_mask = False,
               prop_pred_loss = False, recon_alpha = 1.0):
    
    optimizer.zero_grad()
    
    if not use_out_mask:
        sample['out_mask'] = torch.zeros_like(sample['out_mask'])
        sample['pad_mask'] = torch.zeros_like(sample['pad_mask'])
        sample['avg_mask'] = torch.ones_like(sample['avg_mask'])
        
    property_loss = 0
    if prop_pred_loss:
        mean,var,model_out,prop_pred = model(sample['seq'],
                                             sample['pad_mask'],
                                             sample['avg_mask'],
                                             sample['out_mask'],
                                             variational)
        property_loss = nn.MSELoss()(prop_pred,sample['props'])
        
    else:
        mean,model_out = model(sample['seq'],
                                   sample['pad_mask'],
                                   sample['avg_mask'],
                                   sample['out_mask'],
                                   variational)
    
    reproduction_loss = padded_cross_entropy_loss(sample['seq'],
                                                  model_out,
                                                  sample['pad_mask'],
                                                  sample['out_mask'])
        
    loss = (recon_alpha*reproduction_loss + property_loss)

    loss.backward()
    optimizer.step()
    
    return loss.item(), reproduction_loss.item()

class sigmoidal_annealing:
    
    def __init__(self,slope,mid,warm_steps = 0,constant_param = None):
        self.slope = slope
        self.mid = mid
        self.warm_steps = warm_steps
        self.constant_param = constant_param
        
    def weight(self,steps):
        if self.constant_param is not None:
            return self.constant_param
        else:
            return torch.nn.Sigmoid()(torch.tensor(self.slope*(self.warm_steps
                                                               + steps 
                                                               -self.mid)))
        
class cyclical_annealing:
    
    def __init__(self, num_cycles, tot_steps, prop_to_max):
        self.num_cycles = num_cycles
        self.tot_steps = tot_steps
        self.prop_to_max = prop_to_max
        
    def weight(self,steps):
        tau = (steps%(-(-self.tot_steps//self.num_cycles)))\
            /(self.tot_steps/self.num_cycles)/(self.prop_to_max)
        if tau < 1.0:
            return tau
        else:
            return 1.0

def fit(model, dataloader, optimizer, scheduler,
        n_epochs=1, n_steps = -1, use_cuda = False, quiet = False,
        variational = False, use_out_mask = True, prop_pred_loss = False,
        recon_alpha = 1.0):
    
    all_losses = []
    
    if use_cuda is None:
        use_cuda = torch.cuda.is_available()
        
    device =  torch.device("cuda" if use_cuda else "cpu")
    data_type = torch.int64
    
    model = model.to(device)
    model = model.train()
    
    loss_avg = 0
    if quiet:
        epoch_iter = range(n_epochs)
    else:
        epoch_iter = trange(n_epochs, total = n_epochs,
                                  desc = 'Training in progress...')

    for epoch in epoch_iter:
        if quiet:
            data_iter = enumerate(dataloader)
        else:
            data_iter = tqdm(enumerate(dataloader),leave = True,total=len(dataloader))
            
        for i, sample in data_iter:
            print(len(sample))
            if (n_steps != -1) & (i > n_steps-1):
                break
                
            for key, value in sample.items():
                sample[key] = sample[key].to(device)
            
            loss = train_step(model, sample, optimizer,
                              use_out_mask = use_out_mask,
                              prop_pred_loss = prop_pred_loss,
                              recon_alpha = recon_alpha)
                              
            all_losses.append(loss)
            
        if scheduler:
            scheduler.step()
            
    return np.array(all_losses)