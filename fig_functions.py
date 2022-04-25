import pandas as pd
from rdkit.Chem import PandasTools
from rdkit.Chem import rdMolDescriptors, Descriptors, Crippen
import seaborn as sns 
import matplotlib.pyplot as plt
import random
random.seed(666)

def get_prop_scatters(tag):
    
    p = '/home/kat/Repos/SALSA/'
    
    df = pd.read_csv(f'{p}results/umap_dfs/{tag}.csv')
    PandasTools.AddMoleculeColumnToFrame(df,'Smiles','Mol',includeFingerprints=False)
    df = df[["Smiles","Mol","Label","Atype","x","y"]]

    my_props = [(Descriptors.MolWt, "MolWt"),
                (Descriptors.HeavyAtomCount, "HeavyAtomCount"), 
                (Descriptors.FractionCSP3, "FractionCSP3"), 
                (Descriptors.TPSA, "TPSA"), 
                (Descriptors.NumHAcceptors, "NumHAcceptors"), 
                (Descriptors.NumHDonors, "NumHDonors"), 
                (Descriptors.NumRotatableBonds, "NumRotatableBonds"), 
                (Crippen.MolLogP, "MolLogP"),
               ]

    my_props_names = [x[1] for x in my_props]

    for descriptor, name in my_props:
        props = [descriptor(m) for m in df.Mol] 
        df[name] = props
        
    fig, axs = plt.subplots(ncols=2,nrows=4,figsize=(20,40)) 
    for i,prop in enumerate(my_props_names):
        row = i//2
        col = i%2
        fig = sns.scatterplot(data=df, x='x', y='y', hue=prop,
                              alpha=0.75, s=5, palette='plasma', ax=axs[row][col]) 
        fig.axes.get_xaxis().set_visible(False)
        fig.axes.get_yaxis().set_visible(False) 
    display()