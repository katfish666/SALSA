import random
import numpy as np
import pandas as pd
from rdkit import Chem

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