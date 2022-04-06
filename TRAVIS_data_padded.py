#< ---- 20 char ---->< ---- 20 char ---->< ---- 20 char ---->< --- 19 char --- >

import torch
from torch.utils.data import Dataset

import random
import numpy as np
import pandas as pd

def tokenize(smiles, tokens=None):
    """
    Returns list of unique tokens, token-2-index dictionary and number of
    unique tokens from the list of SMILES

    Parameters
    ----------
        smiles: list
            list of SMILES strings to tokenize.

        tokens: list, str (default None)
            list of unique tokens

    Returns
    -------
        tokens: list
            list of unique tokens/SMILES alphabet.

        token2idx: dict
            dictionary mapping token to its index.

        num_tokens: int
            number of unique tokens.
    """
    if tokens is None:
        tokens = list(set(''.join(smiles)))
        tokens = list(np.sort(tokens))
        tokens = ''.join(tokens)
    token2idx = dict((token, i) for i, token in enumerate(tokens))
    num_tokens = len(tokens)
    return tokens, token2idx, num_tokens

class VAEDataset(Dataset):
    
    def __init__(self, training_data_path, cols = 'SMILES',
                 tokens = None,
                 start_token='<', 
                 end_token='>', 
                 pad_token = 'X',
                 property_pred = None,
                 max_len = 120,
                 use_cuda=None):
        """
        Constructor for the VAEDataset.

        Parameters
        ----------
        training_data_path: str
            path to file with training dataset. Training dataset must contain
            a column with training strings. The file also may contain other
            columns.
            
        cols: str (default 'SMILES')
            name of column containing training SMILES strings.

        tokens: list (default None)
            list of characters specifying the language alphabet. Of left
            unspecified, tokens will be extracted from data automatically.

        start_token: str (default '<')
            special character that will be added to the beginning of every
            sequence and encode the sequence start.

        end_token: str (default '>')
            special character that will be added to the end of every
            sequence and encode the sequence end.
            
        pad_token: str (default 'X')
            special character that will be added after the end of every
            sequence to account for different sequence lengths.
            
        property_pred: callable (default 'None')
            if not None, callable function yielding predicted properties in
            dataset for model prediction

        max_len: int (default 120)
            maximum allowed length of the sequences. All sequences longer
            than max_len will be excluded from the training data. All sequences
            shorter than max_len will be padded to that length with pad_token.

        use_cuda: bool (default None)
            parameter specifying if GPU is used for computations. If left
            unspecified, GPU will be used if available

        """
    
        super().__init__()
        
        file = np.transpose(pd.read_csv(training_data_path)[cols].values)
        
        if max_len is not None:
            data = np.array([sm for sm in file if len(sm)<=max_len])
        else:
            data = file
            max_len = max([len(sm) for sm in data])
        
        self.max_sm_len = max_len
        self.max_len = max_len + 2
        
        self.file = data
        
        self.start_token = start_token
        self.end_token = end_token
        self.pad_token = pad_token
       
        if tokens is None:
            tokens, _, _ = tokenize(data)
            
        self.tokens = ''.join(list(np.sort(list(set(tokens\
                                                    + start_token\
                                                    + end_token\
                                                    + pad_token)))))
        self.n_tokens = len(self.tokens)
        
        self.data_len = data.shape[0]
        
        self.property_pred = property_pred
        if property_pred is not None:
            self.prop_size = property_pred.num_preds
        
    
    def cap_and_pad_str(self,string):
        pad_str = ''.join([self.pad_token for _ in range(self.max_sm_len\
                                                         - len(string))])
        
        return self.start_token + string + self.end_token + pad_str
    
    def char_tensor(self, string):
        """
        Converts SMILES into tensor of indices wrapped into 
        torch.autograd.Variable.
        Args:
            string (str): input SMILES string
        Returns:
            tokenized_string (torch.autograd.Variable(torch.tensor))
        """
        tensor = torch.zeros(len(string)).long()
        for c in range(len(string)):
            tensor[c] = self.tokens.index(string[c])
        return tensor
       
    def transform(self, string):
        
        string = self.cap_and_pad_str(string) 
        out = self.char_tensor(string)
        
        return out
    
    def mask(self, seq):
        pad_idx = self.tokens.index(self.pad_token)
        start_idx = self.tokens.index(self.start_token)
        end_idx = self.tokens.index(self.end_token)
#         seq_pad_mask = ((seq == pad_idx)|(seq == end_idx))
        seq_pad_mask = (seq == pad_idx)
    
        seq_avg_mask = ((seq == pad_idx)|(seq == end_idx)|\
                        (seq==start_idx)).float()
        seq_avg_mask = torch.ones_like(seq_avg_mask) - seq_avg_mask
    
        return seq_pad_mask,seq_avg_mask
    
    def crossentropy_mask(self):
        pad_idx = self.tokens.index(self.pad_token)
        start_idx = self.tokens.index(self.start_token)
        
        tok_mask = torch.ones(self.n_tokens)
        idx_tens = torch.tensor([start_idx, pad_idx])
        mask = torch.zeros_like(tok_mask).scatter_(0,idx_tens,tok_mask)
    
        return mask
    
    def remove_extra_tokens(self, string):
        string = string.replace(self.pad_token,'')
        string = string.replace(self.start_token,'')
        string = string.replace(self.end_token,'')
        return string
    
    def convert_seq_to_smi(self, seq,cut=False):
        smi_arr = np.array(list(self.tokens))[seq.cpu()]
        if cut:
            smi = [self.remove_extra_tokens(''.join(arr)) for arr in smi_arr]
        else:
            smi = [''.join(arr) for arr in smi_arr]
        return smi
    
    def __len__(self):
        return self.data_len
    
    def __getitem__(self,idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()
            
        seq = self.transform(self.file[idx])
        
        
        masks = self.mask(seq)
        out_mask = self.crossentropy_mask()
                       
        sample = {'seq': seq,
                  'pad_mask': masks[0],
                  'avg_mask': masks[1],
                  'out_mask': out_mask.unsqueeze(0)}
        
        if self.property_pred is not None:
            props = self.property_pred.pred(self.file[idx])
            sample['props'] = props
        
        return sample