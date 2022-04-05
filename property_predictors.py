#< ---- 20 char ---->< ---- 20 char ---->< ---- 20 char ---->< --- 18 char --->
import sys
import numpy as np
import pandas as pd

import torch
from rdkit import Chem
from rdkit.ML.Descriptors.MoleculeDescriptors import MolecularDescriptorCalculator

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
        

def rdkit_calc_predictor(smiles):
    props = cheminformatics.calc_rdkit(Chem.MolFromSmiles(smiles))
    return torch.nan_to_num(torch.tensor(props),0.0).float()

def get_surface_descriptor_subset():
        """MOE-like surface descriptors
        EState_VSA: VSA (van der Waals surface area) of atoms contributing to a specified bin of e-state
        SlogP_VSA: VSA of atoms contributing to a specified bin of SlogP
        SMR_VSA: VSA of atoms contributing to a specified bin of molar refractivity
        PEOE_VSA: VSA of atoms contributing to a specified bin of partial charge (Gasteiger)
        LabuteASA: Labute's approximate surface area descriptor
        """
        return [
            'SlogP_VSA1',
            'SlogP_VSA10',
            'SlogP_VSA11',
            'SlogP_VSA12',
            'SlogP_VSA2',
            'SlogP_VSA3',
            'SlogP_VSA4',
            'SlogP_VSA5',
            'SlogP_VSA6',
            'SlogP_VSA7',
            'SlogP_VSA8',
            'SlogP_VSA9',
            'SMR_VSA1',
            'SMR_VSA10',
            'SMR_VSA2',
            'SMR_VSA3',
            'SMR_VSA4',
            'SMR_VSA5',
            'SMR_VSA6',
            'SMR_VSA7',
            'SMR_VSA8',
            'SMR_VSA9',
            'EState_VSA1',
            'EState_VSA10',
            'EState_VSA11',
            'EState_VSA2',
            'EState_VSA3',
            'EState_VSA4',
            'EState_VSA5',
            'EState_VSA6',
            'EState_VSA7',
            'EState_VSA8',
            'EState_VSA9',
            'LabuteASA',
            'PEOE_VSA1',
            'PEOE_VSA10',
            'PEOE_VSA11',
            'PEOE_VSA12',
            'PEOE_VSA13',
            'PEOE_VSA14',
            'PEOE_VSA2',
            'PEOE_VSA3',
            'PEOE_VSA4',
            'PEOE_VSA5',
            'PEOE_VSA6',
            'PEOE_VSA7',
            'PEOE_VSA8',
            'PEOE_VSA9',
            'TPSA',
        ]
def surface_predictor(smiles):
    calc = MolecularDescriptorCalculator(get_surface_descriptor_subset())
    props = calc.CalcDescriptors(Chem.MolFromSmiles(smiles))
    return torch.nan_to_num(torch.tensor(props),0.0).float()