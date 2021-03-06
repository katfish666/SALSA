from rdkit import Chem
from rdkit.Chem.rdchem import RWMol, Atom, BondType

def get_cansmiles(smiles, iso=False):
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return ''
    return Chem.MolToSmiles(mol, isomericSmiles=iso)

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

# https://github.com/wengong-jin/hgraph2graph/blob/master/hgraph/chemutils.py
def copy_atom(atom, atommap=True):
    new_atom = Chem.Atom(atom.GetSymbol())
    new_atom.SetFormalCharge(atom.GetFormalCharge())
    if atommap: 
        new_atom.SetAtomMapNum(atom.GetAtomMapNum())
    return new_atom

def get_rwmol(mol):
    new_mol = Chem.RWMol(Chem.MolFromSmiles(''))
    for atom in mol.GetAtoms():
        
        new_atom = copy_atom(atom)
        new_atom = Chem.Atom(atom.GetSymbol())
        new_atom.SetFormalCharge(atom.GetFormalCharge())
        new_atom.SetAtomMapNum(atom.GetAtomMapNum())
        
        aroms = ['N']
        if atom.GetIsAromatic() and atom.GetSymbol() in aroms:
            new_atom.SetNumExplicitHs(atom.GetTotalNumHs())
        
        new_mol.AddAtom(new_atom)

    for bond in mol.GetBonds():
        a1 = bond.GetBeginAtom().GetIdx()
        a2 = bond.GetEndAtom().GetIdx()
        bt = bond.GetBondType()
        new_mol.AddBond(a1, a2, bt)
    return new_mol

from rdkit.Chem.rdmolops import SanitizeMol
def update_mol_rep(molgraph, clean_aroms=False, sanitize=False):
    
    if sanitize:
        SanitizeMol(molgraph)
        molgraph.ClearComputedProps()
    
    if clean_aroms:
        Chem.Kekulize(molgraph)
        # Setting all atoms to non aromatics
        for i in range(molgraph.GetNumAtoms()):
            molgraph.GetAtomWithIdx(i).SetIsAromatic(False)
        # Setting all bonds to non aromatics
        for i in range(molgraph.GetNumAtoms()):
            for j in range(molgraph.GetNumAtoms()):
                bond = molgraph.GetBondBetweenAtoms(i, j)
                if bond is not None:
                    bond.SetIsAromatic(False)

    for i in range(molgraph.GetNumAtoms()):
        molgraph.GetAtomWithIdx(i).UpdatePropertyCache()

    # Updating RDKit representation
    molgraph.UpdatePropertyCache()
#     Chem.FastFindRings(molgraph)
    