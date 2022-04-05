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
             property_pred = None,
             max_len = 120,
             use_cuda=None):
        
        super().__init__() 
        
        self.anc_smi = np.transpose(pd.read_csv(anc_path)['smiles'].values)
        self.aug_smi = np.asarray(pd.read_csv(aug_path))
        
        augs_arr = np.asarray([x[1] for x in self.aug_smi])
        tokens,_ = tokenize( np.hstack((self.anc_smi,augs_arr)) )  
        all_tokens = list(set(tokens + start_token + end_token + pad_token))
        self.tokens = ''.join(list(np.sort(all_tokens)))        
               
        self.start_token = start_token
        self.end_token = end_token
        self.pad = pad_token

        self.max_sm_len = max_len
        self.max_len = max_len + 2
        
        self.property_pred = property_pred
    
    def idc_tensor(self, smi):
        tensor = torch.zeros(len(smi)).long()
        for i in range(len(smi)):
            tensor[i] = self.tokens.index(smi[i])
        return tensor
    
    def get_vec(self, smi):
        padding = ''.join([self.pad for _ in range(self.max_sm_len - len(smi))])
        smi = self.start_token + smi + self.end_token + padding
        vec = self.idc_tensor(smi)
        return vec
 

    def remove_extra_tokens(self, smi):
        smi = smi.replace(self.pad,'')
        smi = smi.replace(self.start_token,'')
        return smi.replace(self.end_token,'')
    
    def convert_vec_to_smi(self, vec, snip=False):
        smi_arr = np.array(list(self.tokens))[vec.cpu().detach().numpy()]
        smi_list = [''.join(arr) for arr in smi_arr]
        if snip:
            smi_list = [self.remove_extra_tokens(smi) for smi in smi_list]
        return smi_list
        
    def __len__(self):
        return self.data_len
    
    def __getitem__(self,idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()
            
        anc_vec = self.get_vec(self.anc_smi[idx])        
        aug_smileses = self.aug_smi[self.aug_smi[:,0]==idx][:,1]
        aug_vecs = torch.stack([self.get_vec(sm) for sm in aug_smileses])
        
        sample = {'anchor': anc_vec,
                  'augs': aug_vecs}
        
        return sample