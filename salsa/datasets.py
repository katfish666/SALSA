#< ---- 20 char ---->< ---- 20 char ---->< ---- 20 char ---->< --- 18 char --->
import numpy as np
import warnings
warnings.filterwarnings("ignore")
import sys
sys.path.insert(0, '..')

import random
import os
import csv
import pandas as pd

from typing import Iterator, List
import copy
import torch
from torch.utils.data import Dataset, DataLoader, RandomSampler, Sampler

from salsa.utils import get_cansmiles, count_atoms
from salsa.constants import VOCAB, S_TOKEN, E_TOKEN, P_TOKEN, TOKENS
from salsa.constants import MAX_SMI_LEN, MAX_ANC_LEN

random.seed(666)

#< ---- 20 char ---->< ---- 20 char ---->< ---- 20 char ---->< --- 18 char --->

# # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # #
# # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # #
# Inference dataset # # # # # # # # # # # # # # # # # # # # # # # # # #
# # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # #
# # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # #

class InferenceDataset(Dataset):
    def __init__(self, ds, filter=True):        
        super().__init__()

        # print(f"\nConstructing InferenceDataset ...")
        self.dataframe = get_inference_dataset(ds_in=ds, filter=filter)
        
    def __len__(self):
        return len(self.dataframe)      

    def __getitem__(self, idx):        
        if torch.is_tensor(idx):
            idx = idx.tolist()
        smi = self.dataframe.iloc[idx].Smiles 
        vec = _get_vec( _replace_unks(_do_BrCl_singles(smi)) )        
        masks = _get_masks_from_vec(vec)
        return {'seq': vec,
                'smiles': smi, 
                'pad_mask': masks[0],
                'avg_mask': masks[1],
                'out_mask': masks[2], }  


def get_inference_dataset(ds_in, filter=True, keep_cols=True, print_out=False):
    '''
    Args:
        ds_in: Input must be dataframe including a column of SMILES strings 
            (named 'Smiles'), a list of smiles, or a path to a csv containing 
            smiles. It may have additional columns.
        filter: Whether or not to apply Salsa-compatible filter
        keep_cols: Whether or not to maintain all columns of the
            input dataframe. Some rows may be dropped.
    Returns:
        df: Output dataframe with cols 'Origin_id' and 'Smiles'
            (and other provided cols if desired)
    '''     
    
    def print(*args, **kwargs):
        if(print_out):
                return __builtin__.print(*args, **kwargs)
        
    df_in = get_smiles_df_from_input(ds_in)
    
    ## Index in the original dataset. used to see what was filtered.
    df = df_in.reset_index(drop=True, inplace=False)
    df['Origin_id'] = df.index

    if filter:
        df = filter_inf_ds(df)
        df.reset_index(drop=True, inplace=True)

    # Id col !
    df['Id'] = df.index

    # return only id and smiles. 
    if keep_cols:
        other_cols = [x for x in df.columns.values if x not in ['Origin_id','Smiles','Id']]
        cols = ['Id','Origin_id','Smiles'] + other_cols
        df = df[cols]
    else:    
        df = df[['Id','Origin_id','Smiles']]
    
    n_toss = len(df_in) - len(df)
    message = f'''Function 'get_inference_dataset()' returned a (complete) dataset!\
    \n Disgarded compounds: {n_toss:,} ({n_toss})\
    \n Remaining compounds: {len(df):,} ({len(df)})''' 
        
    print(message)
    return df


def filter_inf_ds(ds, #max_len=MAX_SMI_LEN, 
                  max_atoms=35, 
                  atom_filter=False, remove_dups=False, remove_mixs=False):
    '''
    Given a dataset containing a 'Smiles' column!!! ...
    Return a dataset that is agreement with Salsa specs:
        • Canonicalized
        • Filtered by max SMILES length
        • Filtered by max number of heavy atoms (not required) 
    '''
    # df = get_smiles_df_from_input(ds)
    df = ds
    ## Canonicalize smiles
    df['Smiles'] = df.Smiles.apply(lambda x: get_cansmiles(x))
    df = df[df['Smiles'].str.len() > 0]
    ## Remove duplicates
    if remove_dups:
        df = df.drop_duplicates('Smiles',keep='first')
    ## Remove mixtures
    if remove_mixs:
        keep = df.Smiles.apply(lambda x: '.' not in x)
        df = df[keep]
    ## Filter by length of longest allowed SMILES
    # df = df[df['Smiles'].str.len() <= max_len]
    ## Filter by num atoms
    if atom_filter:
        df = df.Smiles.apply(lambda x: count_atoms(x) <= max_atoms)
    # print('Len',len(df))
    return df
   

def get_smiles_df_from_input(ds):
    '''
    Takes in a dataframe, list, np array, or file path name and returns a SMILES dataframe.
    '''
    try:
        if isinstance(ds, list) or isinstance(ds, np.ndarray): # is list or array
            df = pd.DataFrame(ds,columns=['Smiles'])
        elif isinstance(ds, str): # is file path
            if 'csv' not in ds: 
                ds = f'{ds}.csv'
            df = pd.read_csv(ds) 
        elif isinstance(ds, pd.DataFrame): # is dataframe
            df = ds
        return df
    except Exception as e:
        raise (e)


# # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # #
# # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # #
# Training dataset  # # # # # # # # # # # # # # # # # # # # # # # # # #
# # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # #
# # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # #

class TrainingDataset(Dataset):
    def __init__(self, df, n_samp=None, redo_cols=True):        
        super().__init__()
        
        self.dataframe, self.fam_size = get_training_dataset(df_in=df, n_samp=n_samp, 
                                                             redo_cols=redo_cols)
        self.anc_ids = self.dataframe.Anc_id.tolist()
        self.anc_id_set = self.dataframe.Anc_id.unique().tolist()
        
    def __len__(self):
        return len(self.dataframe)

    def __getitem__(self, idx):        
        if torch.is_tensor(idx):
            idx = idx.tolist()
        id, smi, is_anc, anc_id, fam_id = self.dataframe.iloc[idx].values
        # vec = _get_vec( _replace_unks(_do_BrCl_singles(smi)) )       
        vec = convert_smi_to_vec(smi) 
        masks = _get_masks_from_vec(vec)
        
        return {'seq': vec,
                'smiles': smi, 
                'is_anc': is_anc,
                'anc_id': anc_id, 
                'fam_id': fam_id,
                'pad_mask': masks[0],
                'avg_mask': masks[1],
                'out_mask': masks[2],
                } 
       

def get_training_dataset(df_in, n_samp=None, redo_cols=True, seed=666):
    '''
    This function requires a mutated dataframe.
    ASSUMPTIONS:
        Required columns: ['Is_anc', 'Anc_id', 'Smiles', ... ]  
        Row order must be ... Anchor_n followed by Mutants_n_0:n_5
        There must be fam_size-1 mutants per anchor.
    Args:
        df_in: input MUTATED dataset.
        fam_sze: The n_muts+1 if not default (6).
        n_samp: The number of fams (ancs) to sample if desired.
    Returns:
        df
        fam_size: !!!!!
    '''      
    
    df = df_in.copy()
    df.rename({'Anc_idx':'Anc_id', 'Src_idx':'Src_id'},axis=1,
              errors='ignore',inplace=True)

    # Get fam size!!!
    first_anc_id = df.Anc_id.values[0]
    fam_size = len(df[df.Anc_id==first_anc_id])

    # filter out smiles with length more than max    
    big_mutset_ids = set(df[df.Smiles.str.len() > MAX_SMI_LEN].Anc_id)
    n_toss = len(big_mutset_ids)
    df = df[ ~(df.Anc_id.isin(big_mutset_ids)) ]
    
    if n_samp:
        my_seed = seed
        random.seed(my_seed)
        anc_ids = df.Anc_id.unique().tolist()
        anc_ids_samp = random.sample(anc_ids, n_samp)
        df = df[ df.Anc_id.isin(anc_ids_samp) ]
        df.reset_index(inplace=True, drop=True)
    else:
        df.reset_index(inplace=True, drop=True)
                
    if redo_cols:
        # Make new id col !
        df['Id'] = df.index

        # Original anchor id in source dataset !!!!
        # throw away for now ... change code if want later ... 
        # df['Origin_anc_id'] = df['Anc_id']

        # Redo anc_ids and fam_ids
        num_ancs = len(df[df.Is_anc==True])
        tot_mols = num_ancs*(fam_size)

        # [0,0,0,0,0,0,6,6,6,6,6,6,12,12,12,12,12,12,...]
        new_anc_ids = [i for i in range(0, tot_mols, fam_size) for j in range(fam_size)]
        # display(new_anc_ids)
        df['Anc_id'] = new_anc_ids

        # [0,0,0,0,0,0,1,1,1,1,1,1,2,2,2,2,2,2,...]
        fam_ids = [i for i in range(num_ancs) for j in range(fam_size)]
        df['Fam_id'] = fam_ids

        df = df[['Id','Smiles','Is_anc','Anc_id','Fam_id']]

    anc_ids = df.Anc_id.unique().tolist()
    
    if n_samp:
        message = f'''Function 'get_training_dataset()' returned a (sampled) dataset!\
        \n Fam size: {fam_size}\
        \n Sampled fams: {n_samp:,} ({n_samp})\
        \n Total compounds: {len(df):,} ({len(df)})'''
        
    else:
        message = f'''Function 'get_training_dataset()' returned a (complete) dataset!\
        \n Fam size: {fam_size}\
        \n Disgarded fams: {n_toss:,} ({n_toss})\
        \n Remaining fams: {len(anc_ids):,} ({len(anc_ids)})\
        \n Total compounds: {len(df):,} ({len(df)})'''        
    
    print(message)
    return (df, fam_size)



# # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # #
# # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # #
# Misc. functions for cleaning datasets # # # # # # # # # # # # # # # #
# # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # #
# # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # #

from salsa.utils import count_atoms
def filter_source_ds(ds, max_len=MAX_ANC_LEN, max_atoms=35,
                     atom_filter=False):
    '''
    Used for filterinig anchor training sets.
    Given a dataset containing a 'Smiles' column ...
    Return a dataset that is agreement with Salsa specs:
        • Canonicalized
        • Filtered by max SMILES length
        • Filtered by max number of heavy atoms (not required) 
    '''

    if isinstance(ds, pd.DataFrame): df = ds
    else: df = pd.read_csv(ds)   

    ## Canonicalize smiles
    df['Smiles'] = df.Smiles.apply(lambda x: get_cansmiles(x))

    ## Filter by length
    df['Len'] = df.Smiles.apply(lambda x: len(x))
    df = df[df.Len <= max_len]

    ## Filter by num atoms
    if atom_filter:
        df['N_atoms'] = df.Smiles.apply(lambda x: count_atoms(x))
        df = df[df.N_atoms <= max_atoms]

    return df



# # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # #
# # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # #
# SMILES formatting pre-vectorization.  # # # # # # # # # # # # # # # #
# # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # #
# # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # #

def _remove_extra_tokens(smi):
    smi = smi.replace(P_TOKEN,'')
    smi = smi.replace(S_TOKEN,'')
    smi = smi.replace(E_TOKEN,'')
    return smi

def _undo_BrCl_singles(smi):
    smi = smi.replace('R','Br')
    smi = smi.replace('L','Cl')
    return smi

def _do_BrCl_singles(smi):
    smi = smi.replace('Br','R')
    smi = smi.replace('Cl','L')   
    return smi

def _replace_unks(smi):
    smi = ''.join([char if char in VOCAB else '$' for char in smi])
    return smi

def _get_vec(smi):
    padding = ''.join([P_TOKEN for _ in range(MAX_SMI_LEN - len(smi))])
    smi = S_TOKEN + smi + E_TOKEN + padding    
    vec = torch.zeros(len(smi)).long()
    for i in range(len(smi)): 
        vec[i] = TOKENS.index(smi[i])    
    return vec


def _get_masks_from_vec(vec):
    p_idx = TOKENS.index(P_TOKEN)
    s_idx = TOKENS.index(S_TOKEN)
    e_idx = TOKENS.index(E_TOKEN)
    # pad mask: masks pad tokens
    pad_mask = (vec==p_idx)
    # avg mask: masks pad,s,e tokens
    avg_mask = ((vec==p_idx)|(vec==e_idx)|(vec==s_idx)).float()
    avg_mask = torch.ones_like(avg_mask) - avg_mask
    # sup (superfluous) mask: masks s,e tokens
    sup_mask = torch.ones(len(TOKENS))
    idx = torch.tensor([s_idx, p_idx])
    sup_mask = torch.zeros_like(sup_mask).scatter_(0,idx,sup_mask)
    sup_mask = sup_mask.unsqueeze(0)
    return pad_mask, avg_mask, sup_mask  

def convert_vec_to_smi(vec, snip=False):
    # print(TOKENS)
    p_idx = TOKENS.index(P_TOKEN)
    s_idx = TOKENS.index(S_TOKEN)
    e_idx = TOKENS.index(E_TOKEN)
    # print(p_idx, e_idx,s_idx)
    smi_arr = np.array(list(TOKENS))[vec.cpu().detach().numpy()]
    smi_list = [''.join(arr) for arr in smi_arr]
    smi_list = [_undo_BrCl_singles(smi) for smi in smi_list]
    if snip:
        smi_list = [_remove_extra_tokens(smi) for smi in smi_list]
    return smi_list

def convert_smi_to_vec(smi):
    vec = _get_vec( _replace_unks(_do_BrCl_singles(smi)) )  
    return vec 


# # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # #
# # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # #
# Samplers! # # # # # # # # # # # # # # # # # # # # # # # # # # # # # #
# # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # #
# # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # #

# inspo ... pytorch.org/docs/stable/_modules/torch/utils/data/sampler.html#BatchSampler
class RandomFamSampler(Sampler[List[int]]):
    """
    This sampler randomly samples fam sets and puts them in a batch.
    Args:
        anc_ids: list of anc ids from which to sample.
            e.g. [0,0,0,0,0,0,6,6,6,6,6,6,12,12,12,...]
        batch_size: total batch size. to get the number of fams in a batch,
            interger divide by fam_size.
        drop_last (bool): If True, the sampler will drop the last batch if
            its size would be less than batch_size.
    """

    def __init__(self, anc_ids: list, 
                 batch_size: int, 
                 drop_last: bool) -> None:
        
        self.anc_ids = anc_ids
        self.batch_size = batch_size
        self.drop_last = drop_last
        self.len_ds = len(self.anc_ids)

        self.anc_id_set = list(set(anc_ids))
        self.anc_sampler = RandomSampler(self.anc_id_set)

        # Get implicit fam size!!!
        self.fam_size = np.count_nonzero(np.asarray(self.anc_ids)==self.anc_ids[0])
        self.fams_per_batch = self.batch_size // self.fam_size
        self.true_batch_size = self.fams_per_batch * self.fam_size

        message = f'''Instantiated RandomFamSampler.\
        \n Batchsize: {self.true_batch_size}\
        \n Fams per batch: {self.fams_per_batch}'''
        print(message)

    def __iter__(self) -> Iterator[List[int]]:        
        batch = []
        i = 0
        # print(self.anc_ids)
        
        for index in self.anc_sampler:  
            anc_id = self.anc_id_set[index]

            fam_ids = [anc_id+i for i in range(self.fam_size)]
            batch.extend(fam_ids)
            i+=1
            # if i % self.fams_per_batch == 0:
            if len(batch)==self.true_batch_size:
                yield batch
                batch = []    
            # print("THIS ANC:",anc_id)
            # print(batch)           
        if len(batch) > 0 and not self.drop_last:
            yield batch
            
    def __len__(self) -> int:
        if self.drop_last:
            return self.len_ds // self.true_batch_size  
        else:
            return (self.len_ds + self.true_batch_size - 1) // self.true_batch_size  