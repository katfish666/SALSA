from rdkit_utils import *

# TODO: change input and return types to Mol instead of RWMol (?)
def add_atom_to_mol(molgraph, atom_type, to_aidx): 
    """
    :param mol: Mol object
    :param atom_type: str 
    :param to_aidx: int, which index of molecule to add new atom
    :return: new RWMol object after atom addition
    """
    update_mol_rep(molgraph)
    molgraph_new = copy_rwmol(molgraph)
    
    atom = Atom(atom_type)
    atom.SetBoolProp("mutability", True)    
    molgraph_new.AddAtom(atom)
    
    molgraph_new.AddBond(molgraph_new.GetNumAtoms() - 1, to_aidx, BondType.SINGLE)
    
    update_mol_rep(molgraph_new)
    return molgraph_new


import numpy as np
import random

# TODO: function that returns ALL possible "one atom addition" augs ... 
def randomly_add_atom(molgraph,atom_type='O',seedy=666):
#     random.seed(seedy)
    idc = [i for i in range(0,(molgraph.GetNumAtoms()))]
    random.shuffle(idc)
    for i in idc:
#         print(i)
        try:
            molgraph_new = add_atom_to_mol(molgraph, atom_type, i)
        except Exception as e:
            continue
        else:
            break
    else:
        return None
    return i,molgraph_new