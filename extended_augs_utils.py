from property_predictors import surface_predictor
from utilities.graph_augs import add_atom_to_mol
from rdkit import Chem
import random

from utilities.data_utils import get_weighted_random_atom 
from utilities.data_utils import get_atom_cnts

def get_ext_augs(smiles,prop_filter=True,maximum=10,tries=3):
    
    atom_to_cnt = {'N': 2696053,
     'C': 17363219,
     'O': 2544692,
     'S': 393100,
     'I': 7265,
     'Br': 56161,
     'Cl': 221403,
     'F': 272005,
     'P': 14095,
     'B': 23}
        
    mol = Chem.MolFromSmiles(smiles)

    atom_idc = [i for i in range(0,(mol.GetNumAtoms()))]
    random.shuffle(atom_idc)
    
    goods = 0 
    aug_smis = []
    
    for attempt in range(tries):
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
                
    return aug_smis