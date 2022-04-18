import random
import numpy as np
import pandas as pd
from rdkit import Chem

from rdkit.Chem import PandasTools

from utilities.rdkit_utils import *
from utilities.graph_augs import *

from rdkit import RDLogger  
RDLogger.DisableLog('rdApp.*')

from joblib import Parallel, delayed
import multiprocessing
import random
from property_predictors import surface_predictor

def get_augs(smiles,atom_to_cnt,prop_filter=True,maximum=5):
        
    mol = Chem.MolFromSmiles(smiles)

    atom_idc = [i for i in range(0,(mol.GetNumAtoms()))]
    random.shuffle(atom_idc)
    
    goods = 0 
    aug_smis = []
    for i in atom_idc:
        if goods==maximum:
            break
            
        atom_type = get_weighted_random_atom(atom_to_cnt)
        
        try: 
            mol_aug = add_atom_to_mol(mol, atom_type, i, clean_aroms = True)
            if mol_aug.GetNumAtoms()==0:
                continue
            else:
                sm = Chem.MolToSmiles(mol_aug)
                props = surface_predictor(sm)
                if prop_filter and sum(props)>=1:
                    goods+=1
                    aug_smis.append(sm)
                elif not prop_filter:
                    goods+=1
                    aug_smis.append(sm)
                    
        except Exception as e:
            continue
            
    while len(aug_smis) < maximum:
        try:
            aug_smis.append( random.choice(aug_smis) )
        except:
            aug_smis.append(smiles)
            
    return (smiles, aug_smis)

def get_anc_to_aug_map(df):
    PandasTools.AddMoleculeColumnToFrame(df,'smiles','mol',includeFingerprints=False)
    
    atom_to_cnt = get_atom_cnts(df.smiles)

    parallelizer = Parallel(n_jobs=multiprocessing.cpu_count()-1, backend= 'multiprocessing' )
    augs_tasks = (delayed(get_augs)(smi,atom_to_cnt) for smi in df.smiles)
    smiles_to_augs = parallelizer(augs_tasks)

    smiles_to_augs = {k:v for k,v in smiles_to_augs}
    
    return smiles_to_augs




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

def get_weighted_random_atom(symb_to_count):
    '''
    Given a dictionary of atom types to counts, return a randomly chosen atom,
    weighted according to  the given count ratios.
    '''
    tot = sum(symb_to_count.values())
    symb_to_prob = {}
    for k,v in symb_to_count.items():
        symb_to_prob[k] = v/tot 
        
    a = np.array([x for x in symb_to_prob.keys()])
    p = np.array([x for x in symb_to_count.values()])
    p = p / np.sum(p)
    atom_type = np.random.choice(a=a, p=p)
    
    return(atom_type)

# # # # # # # # # #

elements = ['H', 'He', 'Li', 'Be', 'B', 'C', 'N', 'O', 'F', 'Ne', 'Na', 'Mg',
       'Al', 'Si', 'P', 'S', 'Cl', 'Ar', 'K', 'Ca', 'Sc', 'Ti', 'V', 'Cr',
       'Mn', 'Fe', 'Co', 'Ni', 'Cu', 'Zn', 'Ga', 'Ge', 'As', 'Se', 'Br',
       'Kr', 'Rb', 'Sr', 'Y', 'Zr', 'Nb', 'Mo', 'Tc', 'Ru', 'Rh', 'Pd',
       'Ag', 'Cd', 'In', 'Sn', 'Sb', 'Te', 'I', 'Xe', 'Cs', 'Ba', 'La',
       'Ce', 'Pr', 'Nd', 'Pm', 'Sm', 'Eu', 'Gd', 'Tb', 'Dy', 'Ho', 'Er',
       'Tm', 'Yb', 'Lu', 'Hf', 'Ta', 'W', 'Re', 'Os', 'Ir', 'Pt', 'Au',
       'Hg', 'Tl', 'Pb', 'Bi', 'Po', 'At', 'Rn', 'Fr', 'Ra', 'Ac', 'Th',
       'Pa', 'U', 'Np', 'Pu', 'Am', 'Cm', 'Bk', 'Cf', 'Es', 'Fm', 'Md',
       'No', 'Lr', 'Rf', 'Db', 'Sg', 'Bh', 'Hs', 'Mt', 'Ds ', 'Rg ',
       'Cn ', 'Nh', 'Fl', 'Mc', 'Lv', 'Ts', 'Og']

def tokenize(smileses):
   
    # Fix Br and Cl problem ... 
    tokens = ''.join(smileses)
    tokens = tokens.replace('Br','R')
    tokens = tokens.replace('Cl','L')
    
    tokens = list(set(tokens))
    tokens = list(np.sort(tokens))
    tokens = ''.join(tokens)
    token_to_idx = dict((token, i) for i, token in enumerate(tokens))
    
    return tokens, token_to_idx