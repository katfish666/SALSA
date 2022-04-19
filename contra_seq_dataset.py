from utilities.data_utils import tokenize

import torch
from torch.utils.data import Dataset

import random
import numpy as np
import pandas as pd
import os
import csv

def get_dataset_array(anc_path, aug_path):
    ''' Returns pandas array of all molecules (ancs and augs). '''
    anc_smi = pd.read_csv(anc_path)
    anc_smi['anc_idx'] = anc_smi.index
    anc_smi['atype'] = 'Anc'
    anc_smi = anc_smi[['smiles','atype','anc_idx',]]
    
    aug_smi = pd.read_csv(aug_path)
    aug_smi['atype'] = 'Aug'
    aug_smi = aug_smi[['smiles','atype','anc_idx',]]
    
    tot_smi = pd.concat([anc_smi,aug_smi])

    return tot_smi

def get_anc_map(tot_smi):
    anc_idc = tot_smi['anc_idx'].unique()
    augs = np.asarray(tot_smi['anc_idx'].values)
    anc_map = {}
    for i in anc_idc:
        aug_idc = np.where(augs==i)[0]
        anc_map[i] = aug_idc
    return anc_map


from torch.utils.data import Sampler
from typing import Iterator, List

# Modeled off of ... 
# https://pytorch.org/docs/stable/_modules/torch/utils/data/sampler.html#BatchSampler
class AnchoredSampler(Sampler[List[int]]):
    """
    Args:
        sampler (Sampler or Iterable): Base sampler. 
        batch_size (int): Size of mini-batch.
        drop_last (bool): If ``True``, the sampler will drop the last batch if
            its size would be less than ``batch_size``
    """

    def __init__(self, sampler: Sampler[int], anc_map: dict,
                 batch_size: int, drop_last: bool) -> None:
        self.sampler = sampler
        self.anc_map = anc_map
        self.batch_size = batch_size
        self.drop_last = drop_last

    def __iter__(self) -> Iterator[List[int]]:
        batch = []
        i = 0
        for idx in self.sampler:
            augs = self.anc_map[idx].tolist()
            batch.extend(augs)
            i+=1
            if i % (self.batch_size) == 0:
                yield batch
                batch = []               
        if len(batch) > 0 and not self.drop_last:
            yield batch

    def __len__(self) -> int:
        if self.drop_last:
            return len(self.sampler) // self.batch_size  
        else:
            return (len(self.sampler) + self.batch_size - 1) // self.batch_size  
        
    
        
        
class ContraSeqDataset(Dataset):
    def __init__(self, anc_path, aug_path,
                 start_token='<', end_token='>', pad_token = 'X', 
                 max_len = 120, use_cuda=None):  
        
        super().__init__() 
        
        self.df = get_dataset_array(anc_path, aug_path)
        
        all_smi = np.transpose(self.df['smiles'].values)
#         tokens,_ = tokenize( all_smi )  
        tokens = '#%()+-0123456789<=>BCFHILNOPRSX[]cnos'
        tokens = list(set(tokens + start_token + end_token + pad_token))
        self.tokens = ''.join(list(np.sort(tokens)))  
        self.n_tokens = len(self.tokens)
               
        self.s_token = start_token
        self.e_token = end_token
        self.p_token = pad_token

        self.max_sm_len = max_len
        self.max_len = max_len + 2
           
    def idc_tensor(self, smi):
        tensor = torch.zeros(len(smi)).long()
#         print(smi)
        for i in range(len(smi)):
#             try:
            tensor[i] = self.tokens.index(smi[i])
#             except:
#                 print(smi)
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
#         return pad_mask.unsqueeze(0), avg_mask.unsqueeze(0), sup_mask
        return pad_mask, avg_mask, sup_mask
    
    def __len__(self):
        return len(self.df)
        
    def __getitem__(self, idx):

        if torch.is_tensor(idx):
            idx = idx.tolist()
          
        smi, atype, label = self.df.iloc[idx].values
        
        # BrCl singled vec  
        _smi = self.do_BrCl_singles(smi)
        vec = self.get_vec(_smi)        
        # Masked vectors
        masks = self.masks(vec)
            
        # TODO: formal getters and setters !!!!
        seq_attr = {'seq': vec,
                    'smiles': smi, 
                    'atype': atype,
                    'label': label, 
                    'pad_mask': masks[0],
                    'avg_mask': masks[1],
                    'out_mask': masks[2]} 
        
        return seq_attr
    
    

        


        
        
        
        
        
        
        
        
        
        
        
        
    
    
    