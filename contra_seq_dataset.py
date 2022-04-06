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


class SeqDatum(object):
    def __init__(self, smi, smi_tokens,
             start_token='<', 
             end_token='>', 
             pad_token = 'X',
             max_len = 120,
             use_cuda=None):
     
        
        tokens = list(set(smi_tokens + start_token + end_token + pad_token))
        self.tokens = ''.join(list(np.sort(tokens)))  
        self.n_tokens = len(self.tokens)
               
        self.s_token = start_token
        self.e_token = end_token
        self.p_token = pad_token

        self.max_sm_len = max_len
        self.max_len = max_len + 2
        
        self.smi = smi
        
        # BrCl singled vec  
        _smi = self.do_BrCl_singles(self.smi)
        vec = self.get_vec(_smi)        
        # Masked vectors
        masks = self.masks(vec)

        # TODO: formal getters and setters !!!!
        self.seq_attr = {'smiles':self.smi, 
                         'seq': vec,
                         'pad_mask': masks[0],
                         'avg_mask': masks[1],
                         'out_mask': masks[2]} 
        

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
    
    
    def __getitem__(self):
        return self.seq_attr
        
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
        
        _augs_arr = np.asarray([x[1] for x in self.aug_smi])
        tokens,_ = tokenize( np.hstack((self.anc_smi,_augs_arr)) )  
        tokens = list(set(tokens + start_token + end_token + pad_token))
        self.tokens = ''.join(list(np.sort(tokens)))  
        self.n_tokens = len(self.tokens)
               
        self.s_token = start_token
        self.e_token = end_token
        self.p_token = pad_token

        self.max_len = max_len
                    
    def __len__(self):
        # returns number of anchors
        return len(self.anc_smi)
    
    def __getitem__(self,idx):
        
        if torch.is_tensor(idx):
            idx = idx.tolist()
            
        anc_smi = self.anc_smi[idx]
#         print(anc_smi)
        anc_seq = SeqDatum(anc_smi, self.tokens).__getitem__()
        
        aug_smis = self.aug_smi[self.aug_smi[:,0]==idx][:,1]
#         print(aug_smis)
        aug_seqs = [SeqDatum(sm, self.tokens).__getitem__() for sm in aug_smis]
        
        sample = {'anc': anc_seq,
                  'augs': aug_seqs}
        
        return sample
    
    
    
    
    
    
    
    
    
    
    