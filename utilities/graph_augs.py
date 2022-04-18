from utilities.rdkit_utils import *

# TODO: change input and return types to Mol instead of RWMol (?)
def add_atom_to_mol(mol, atom_type, to_aidx, clean_aroms=False, clean_it=True): 
    """
    :param mol: Mol object
    :param atom_type: str 
    :param to_aidx: int, which index of molecule to add new atom
    :return: new RWMol object after atom addition
    """
    molgraph = get_rwmol(mol) # molgraph.UpdatePropertyCache()
    update_mol_rep(molgraph, clean_aroms)
    
    atom = Atom(atom_type)
    atom.SetBoolProp("mutability", True)    
    molgraph.AddAtom(atom)
    
    molgraph.AddBond(molgraph.GetNumAtoms() - 1, to_aidx, BondType.SINGLE)
    
    update_mol_rep(molgraph, clean_aroms) 
    mol_new = molgraph.GetMol() 
    
    if clean_it:
        mol_new = Chem.MolFromSmiles( get_cansmiles( Chem.MolToSmiles(mol_new) ))
    return mol_new

import numpy as np
import random
def randomly_add_atom(mol, atom_type='O', at_random=True):
    if not at_random:
        random.seed(666)
    idc = [i for i in range(0,(mol.GetNumAtoms()))]
    random.shuffle(idc)
    for i in idc:
        try:
            mol_new = add_atom_to_mol(mol, atom_type, i)
        except Exception as e:
            print("rand add atom",e)
            continue
        else:
            break
    else:
        return None
    return i,mol_new


def add_atom_everywhere(mol, atom_type='weighted', max_children='max', weights=None):
#     subs = ['Br','I','C','N','S','P','O','F','Cl']
    subs = ['C','O','N','Cl','S','P']
    
    if max_children=='max':
        num_aidc = mol.GetNumAtoms()
    else:
        num_aidc = max_children
        
    aug_graphs = []
    bad_aidc = []
    for i in range(0,num_aidc):

        try:
            if atom_type=='weighted':
                random.shuffle(subs)
                atom_type = subs[0]
                
                mol_new = add_atom_to_mol(mol, atom_type, i)
            else:
                mol_new = add_atom_to_mol(mol, atom_type, i)
            if mol_new.GetNumAtoms()==0:
                bad_aidc.append(i)
            else:
                aug_graphs.append(mol_new)
        except Exception as e:
            print(i,e)
            bad_aidc.append(i)
            continue
    return aug_graphs, bad_aidc