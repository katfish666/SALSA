from rdkit import Chem
from rdkit.Chem.rdchem import RWMol

from rdkit.Chem.rdchem import RWMol, Atom, BondType
def add_atom_to_mol(molgraph, atom_type, to_aidx): 
    """
    :param molgraph: RWMol object
    :param atom_type: str 
    :param to_aidx: int, which index of molecule to add new atom
    :return: new graph after atom addition
    """
    molgraph_new = copy_rwmol(molgraph)
    
    # add atom to mol object
    atom = Atom(atom_type)
    atom.SetBoolProp("mutability", True)    
    molgraph_new.AddAtom(atom)
    
    # create bond from new atom to source molecule
    molgraph_new.AddBond(molgraph_new.GetNumAtoms() - 1, to_aidx, BondType.SINGLE)
    
    update_mol_rep(molgraph_new)
    return molgraph_new


def show_atom_index( mol ):
    atoms = mol.GetNumAtoms()
    for idx in range( atoms ):
        mol.GetAtomWithIdx( idx ).SetProp( 'molAtomMapNumber', 
                                          str( mol.GetAtomWithIdx( idx ).GetIdx() ) )
    return mol

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
        
        new_mol.AddAtom(new_atom)

    for bond in mol.GetBonds():
        a1 = bond.GetBeginAtom().GetIdx()
        a2 = bond.GetEndAtom().GetIdx()
        bt = bond.GetBondType()
        new_mol.AddBond(a1, a2, bt)
    return new_mol

def copy_rwmol(mol):
    new_mol = Chem.RWMol(Chem.MolFromSmiles(''))
    for atom in mol.GetAtoms():
        new_atom = Chem.Atom(atom.GetSymbol())
        new_atom.SetFormalCharge(atom.GetFormalCharge())
        new_atom.SetAtomMapNum(atom.GetAtomMapNum())

        if atom.GetIsAromatic() and atom.GetSymbol() == 'N':
            new_atom.SetNumExplicitHs(atom.GetTotalNumHs())

        new_mol.AddAtom(new_atom)
    for bond in mol.GetBonds():
        a1 = bond.GetBeginAtom().GetIdx()
        a2 = bond.GetEndAtom().GetIdx()
        bt = bond.GetBondType()
        new_mol.AddBond(a1, a2, bt)
    return new_mol 

def update_mol_rep(molgraph, sanitize=False):
    if sanitize:
        SanitizeMol(molgraph)
        self.molgraph.ClearComputedProps()
    
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
    Chem.FastFindRings(molgraph)
    