import random
random.seed(666)

#< ---- 20 char ---->< ---- 20 char ---->< ---- 20 char ---->< --- 18 char --->

# # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # #              
# # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # 
# Fingerprinting functions  # # # # # # # # # # # # # # # # # # # # # # #
# # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # #            
# # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # #   
    
from rdkit import Chem
from rdkit.Chem import rdFingerprintGenerator as rdFpGen
from rdkit import DataStructs
from scipy import sparse
import numpy as np

def get_fp(smiles, fp_type='morgan', counts=False, bits=1024, 
           radius=2, chiral=False, sparsed=False):
    '''
    This function ... 
    Args:
        smiles: SMILES string.
        fp_type: Which fp method?
        counts: count or bit vector? Default is False.
        bits: length of vector. Default is 1024 (industry standard).
        radius: radius of atom environment. Default is 2 (industry standard).
        chiral: whether or not to compute chiral bits.
        sparsed: return sparsed matrix? 
    Returns:
        fp: numpy array of size (bits,)
    '''
    
    mol = Chem.MolFromSmiles(smiles)
    if not mol:
        return False
    
    gen = None
    
    # Extended-connectivity fingerprint (Topological circular)
    if fp_type=='morgan':
        gen = rdFpGen.GetMorganGenerator(radius=radius,fpSize=bits,includeChirality=chiral)      
    # Functional class fingerprint (Topological circular)
    if fp_type=='fcfp':
        gen = rdFpGen.GetMorganGenerator(radius=radius,fpSize=bits,includeChirality=chiral, 
                                         atomInvariantsGenerator=rdFpGen.GetMorganFeatureAtomInvGen())
    # Daylight-esque fingerprint (Topological path-based)
    if fp_type=='rdkit':
        # maxPath = diameter = radius * 2
        gen = rdFpGen.GetRDKitFPGenerator(maxPath=radius*2,fpSize=bits,numBitsPerFeature=2)
    # "based on the atomic environments and shortest path separations of every atom pair" 
    if fp_type=='atom_pair':
        gen = rdFpGen.GetAtomPairGenerator(maxDistance=radius,fpSize=bits,includeChirality=chiral)
    
    if counts==True: _fp = gen.GetCountFingerprint(mol)
    else: _fp = gen.GetFingerprint(mol)

    fp = np.zeros(bits, dtype=np.int32)
    DataStructs.ConvertToNumpyArray(_fp, fp)
    
    if sparsed: return sparse.csr_matrix(fp)
    else: return fp
    
    
from joblib import Parallel, delayed
def get_fps_in_parallel(smiles, fp_type='morgan', counts=False, bits=1024, 
           radius=2, chiral=False, sparsed=False):
    '''
    This function computes Morgan fingerprints in parallel. 
    Args: See 'get_fp()'
    Returns:
        fps: numpy array of size (num. smiles, bits)
    '''
    parallelizer = Parallel(n_jobs=-1, backend= 'multiprocessing' )
    fp_tasks = (delayed(get_fp)(sm,fp_type,counts,bits,radius,chiral) for sm in smiles)
    fps = parallelizer(fp_tasks)
    fps = np.vstack(fps)
    return fps


def get_normed_fps(smiles, counts=False, bits=1024, radius=2):
    '''
    This function computes normalized Morgan fingerprints in parallel. 
    Args: See 'get_fp()'
    Returns:
        fp_name: string indicating bits and radius
        fps: numpy array of size (num. smiles, bits)
    '''
        
    fps = get_fps_in_parallel(smiles,'morgan',counts=counts,bits=bits,radius=radius)
    fps = np.stack(fps)
    norm = np.linalg.norm(fps, ord=2, axis=1)[:,np.newaxis]
    fps = fps/norm     
    
    return fps


import itertools 
from tqdm import tqdm

def get_fp_combos(smiles, bitss, radii):
    '''
    ! ! ! ! ! ! ! ! !
    NEED TO TEST THIS !!!!
    ! ! ! ! ! ! ! !! 
    This function computes a bunch of normed fingerprints. 
    Args: 
        bitss: list of bits
        radii: list of radii
    Returns: name_to_fps: dict of fp_name to fps
    '''
    
    name_to_fps = {}
    # bitss = [256, 512, 1024]
    # radii = [2,4]
    for bits, rad in tqdm(itertools.product(bitss,radii), 
                          total=len(bitss)*len(radii), disable=False):
        method = f'fpnorm_{bits}_r{rad}'
        fps_normed = get_normed_fps(smiles, counts=False, bits=bits, radius=rad)    
        name_to_fps[method] = [x for x in fps_normed]
        print(f"Computed {method} fingerprints!")
        
    return name_to_fps



#< ---- 20 char ---->< ---- 20 char ---->< ---- 20 char ---->< --- 18 char --->

# # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # #              
# # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # 
# Rdkit-related functions # # # # # # # # # # # # # # # # # # # # # # # #
# # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # #            
# # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # #   
import pubchempy as pcp
def get_smiles_from_name(comp_name):
    cs = pcp.get_compounds(comp_name, 'name')
    smi = get_cansmiles(cs[0].isomeric_smiles)
    return f"'{comp_name.capitalize()}': '{smi}'"   

    
from rdkit import Chem
from rdkit import RDLogger
RDLogger.DisableLog('rdApp.*')

def get_cansmiles(smiles, chiral=False):
    try:
        mol = Chem.MolFromSmiles(smiles)
        if mol is None:
            return ''
        return Chem.MolToSmiles(mol, isomericSmiles=chiral)
    except Exception as e:
        return ''


def neutralize_atoms(smi):
    mol = Chem.MolFromSmiles(smi)
    pattern = Chem.MolFromSmarts("[+1!h0!$([*]~[-1,-2,-3,-4]),-1!$([*]~[+1,+2,+3,+4])]")
    at_matches = mol.GetSubstructMatches(pattern)
    at_matches_list = [y[0] for y in at_matches]
    if len(at_matches_list) > 0:
        for at_idx in at_matches_list:
            atom = mol.GetAtomWithIdx(at_idx)
            chg = atom.GetFormalCharge()
            hcount = atom.GetTotalNumHs()
            atom.SetFormalCharge(0)
            atom.SetNumExplicitHs(hcount - chg)
            atom.UpdatePropertyCache()
    return Chem.MolToSmiles(mol)

    
def count_atoms(smiles):
    '''
    Gives a SMILES string, return atom count.
    '''
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return 0
    return len(mol.GetAtoms())
    
def get_atom_cnts(smileses):
    '''
    Given a list of SMILES strings, return dictionary of atom counts
    '''
    atom_to_cnt = {}                                                
    
    for smiles in smileses:
        mol = Chem.MolFromSmiles(smiles)

        for atom in mol.GetAtoms():
            atom_type = atom.GetSymbol()
            if atom_type not in atom_to_cnt.keys():
                atom_to_cnt[atom_type] = 0
            else:
                atom_to_cnt[atom_type] += 1
            
    return atom_to_cnt

import copy
def show_atom_index( mol ):
    atoms = mol.GetNumAtoms()
    new_mol = copy.deepcopy(mol)
    for idx in range( atoms ):
        new_mol.GetAtomWithIdx( idx ).SetProp( 'molAtomMapNumber', 
                                          str( mol.GetAtomWithIdx( idx ).GetIdx() ) )
    return new_mol


#< ---- 20 char ---->< ---- 20 char ---->< ---- 20 char ---->< --- 18 char --->

# # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # #              
# # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # 
# Property prediction functions # # # # # # # # # # # # # # # # # # # # #
# # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # #            
# # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # #   

import sys
import numpy as np
import pandas as pd

import torch
from rdkit import Chem
from rdkit.ML.Descriptors.MoleculeDescriptors import MolecularDescriptorCalculator
from rdkit.Chem import Descriptors

def get_props(smiles, prop_list):

    if 'QED' in prop_list and prop_list[-1]!='QED':
        raise Exception('QED must be last in the list! Lmao.')

    props_no_qed = [p for p in prop_list if p!='QED']
    calc = MolecularDescriptorCalculator(props_no_qed)
    props = [x for x in calc.CalcDescriptors(Chem.MolFromSmiles(smiles))]

    if 'QED' in prop_list:
        qed = Descriptors.qed(Chem.MolFromSmiles(smiles))
        props.append(qed)

    return torch.nan_to_num(torch.tensor(props),0.0).float()

def surface_predictor(smiles):
    calc = MolecularDescriptorCalculator(get_surface_descriptor_subset())
    props = calc.CalcDescriptors(Chem.MolFromSmiles(smiles))
    return torch.nan_to_num(torch.tensor(props),0.0).float()

class property_predictor():
    def __init__(self,pred_func,num_preds,scaler = None):
        super().__init__()
        self.pred_func = pred_func
        self.num_preds = num_preds 
        self.scaler = scaler
            
    def pred(self,smiles):
        with torch.no_grad():
            props = np.nan_to_num(self.pred_func(smiles).numpy())
            if self.scaler is not None:
                props = self.scaler.transform(np.array([props]))[0]     
        return torch.nan_to_num(torch.tensor(props))
        

def get_surface_descriptor_subset():
        """MOE-like surface descriptors
        EState_VSA: VSA (van der Waals surface area) of atoms contributing to a specified bin of e-state
        SlogP_VSA: VSA of atoms contributing to a specified bin of SlogP
        SMR_VSA: VSA of atoms contributing to a specified bin of molar refractivity
        PEOE_VSA: VSA of atoms contributing to a specified bin of partial charge (Gasteiger)
        LabuteASA: Labute's approximate surface area descriptor
        """
        return [
            'SlogP_VSA1','SlogP_VSA10','SlogP_VSA11','SlogP_VSA12','SlogP_VSA2',
            'SlogP_VSA3','SlogP_VSA4','SlogP_VSA5','SlogP_VSA6','SlogP_VSA7',
            'SlogP_VSA8','SlogP_VSA9',
            'SMR_VSA1','SMR_VSA10','SMR_VSA2','SMR_VSA3','SMR_VSA4','SMR_VSA5',
            'SMR_VSA6','SMR_VSA7','SMR_VSA8','SMR_VSA9',
            'EState_VSA1','EState_VSA10','EState_VSA11','EState_VSA2','EState_VSA3',
            'EState_VSA4','EState_VSA5','EState_VSA6','EState_VSA7','EState_VSA8',
            'EState_VSA9',
            'LabuteASA',
            'PEOE_VSA1','PEOE_VSA10','PEOE_VSA11','PEOE_VSA12','PEOE_VSA13','PEOE_VSA14',
            'PEOE_VSA2','PEOE_VSA3','PEOE_VSA4','PEOE_VSA5','PEOE_VSA6','PEOE_VSA7',
            'PEOE_VSA8','PEOE_VSA9',
            'TPSA',
        ]
    
    
#< ---- 20 char ---->< ---- 20 char ---->< ---- 20 char ---->< --- 18 char --->

# # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # #              
# # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # 
# Mischellaneous functions  # # # # # # # # # # # # # # # # # # # # # # #
# # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # #            
# # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # #

import random
random.seed(666)
def capped_sample(pop, samp_size):
    '''
    Simple random.sample wrapper that returns original population if it 
        is smaller than requested sample size. 
        Must be of type DataFrame, list, or array !!!
    Args: 
        pop: Population
        samp_size: Sample size
    Returns:
        samp
    '''

    if isinstance(pop, pd.DataFrame):
        if len(pop) > samp_size: samp = pop.sample(samp_size)
        else: samp = pop
    else:
        if len(pop) > samp_size: samp = random.sample(pop, k=samp_size)
        else: samp = pop
    return samp