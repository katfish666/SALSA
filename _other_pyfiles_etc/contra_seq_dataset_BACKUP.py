import torch
from torch.utils.data import Dataset

import random
import numpy as np
import pandas as pd

from data_utils import tokenize

import torch
from torch.utils.data import Dataset

import random
import numpy as np
import pandas as pd
import os
import csv

from data_utils import tokenize

class ContraSeqDataset(Dataset):
    
    def __init__(self, anc_path, aug_path,
             start_token='<', 
             end_token='>', 
             pad_token = 'X',
             max_len = 120,
             use_cuda=None):
        
        super().__init__() 
        
        self.anc_smi = np.transpose(pd.read_csv(anc_path)['smiles'].values)
        self.aug_smi = np.asarray(pd.read_csv(aug_path))
        
        augs_arr = np.asarray([x[1] for x in self.aug_smi])
        
        tokens,_ = tokenize( np.hstack((self.anc_smi,augs_arr)) )  
        all_tokens = list(set(tokens + start_token + end_token + pad_token))
        self.tokens = ''.join(list(np.sort(all_tokens)))  
        self.n_tokens = len(self.tokens)
               
        self.s_token = start_token
        self.e_token = end_token
        self.p_token = pad_token

        self.max_sm_len = max_len
        self.max_len = max_len + 2
                    
    def idc_tensor(self, smi):
        tensor = torch.zeros(len(smi)).long()
        for i in range(len(smi)):
            tensor[i] = self.tokens.index(smi[i])
        return tensor
    
    def get_vec(self, smi):
        padding = ''.join([self.p_token for _ in range(self.max_sm_len - len(smi))])
        smi = self.s_token + smi + self.e_token + padding
        vec = self.idc_tensor(smi)
        return vec
 
    def remove_extra_tokens(self, smi):
        smi = smi.replace(self.p_token,'')
        smi = smi.replace(self.s_token,'')
        return smi.replace(self.e_token,'')
    
    def undo_BrCl_singles(self,smi):
        smi = smi.replace('R','Br')
        return smi.replace('L','Cl')
    def do_BrCl_singles(self,smi):
        smi = smi.replace('Br','R')
        return smi.replace('Cl','L')      
        
    def convert_vec_to_smi(self, vec, snip=False):
        smi_arr = np.array(list(self.tokens))[vec.cpu().detach().numpy()]
        smi_list = [''.join(arr) for arr in smi_arr]
        smi_list = [self.undo_BrCl_singles(smi) for smi in smi_list]
        if snip:
            smi_list = [self.remove_extra_tokens(smi) for smi in smi_list]
        return smi_list
    
    def masks(self, seq):
        p_idx = self.tokens.index(self.p_token)
        s_idx = self.tokens.index(self.s_token)
        e_idx = self.tokens.index(self.e_token)
        
        # pad mask: masks pad tokens
        pad_mask = (seq==p_idx)
 
        # avg mask: masks pad,s,e tokens
        avg_mask = ((seq==p_idx)|(seq==e_idx)|(seq==s_idx)).float()
        avg_mask = torch.ones_like(avg_mask) - avg_mask
        
        # sup (superfluous) mask: masks s,e tokens
        sup_mask = torch.ones(self.n_tokens)
        idx = torch.tensor([s_idx, p_idx])
        sup_mask = torch.zeros_like(sup_mask).scatter_(0,idx,sup_mask)
        sup_mask = sup_mask.unsqueeze(0)
        return pad_mask.unsqueeze(0), avg_mask.unsqueeze(0), sup_mask
        
    def __len__(self):
        return len(self.anc_smi)
    
    def __getitem__(self,idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()
            
        anc_smiles = self.do_BrCl_singles(self.anc_smi[idx])
        anc_vec = self.get_vec(anc_smiles)        
        
        masks = self.masks(anc_vec)
        
        aug_smileses = self.aug_smi[self.aug_smi[:,0]==idx][:,1]
        aug_smileses = [self.do_BrCl_singles(sm) for sm in aug_smileses]
        aug_vecs = torch.stack([self.get_vec(sm) for sm in aug_smileses])
        
#         sample = {'anchor': anc_vec,
#                   'augs': aug_vecs}

        sample = {'seq': anc_vec,
                  'pad_mask': masks[0],
                  'avg_mask': masks[1],
                  'out_mask': masks[2]}
        
        return sample