import torch.nn as nn
import torch
from seqAE_model import SeqAutoencoder
from contra_seq_dataset import *
from torch.utils.data import DataLoader, RandomSampler
import copy
import random
import numpy as np
import umap.umap_ as umap
from tqdm.notebook import trange, tqdm

random.seed(666)


def get_df_and_latents_extended(tag, which_train, which_test, test_ood, samp_size, eval_bs, 
                       n_epochs, use_cuda, empty_cuda, cuda_ids, normed_latent=True):
    # Load model. 
    model = SeqAutoencoder(dim_emb=512, heads=8, dim_hidden=32,
                           L_enc=6, L_dec=6, dim_ff=2048, 
                           drpt=0.1, actv='relu', eps=0.6, b_first=True)
    p = f'/home/kat/Repos/SALSA/results/models/{tag}/{n_epochs-1:02}.pt'
    model.load_state_dict(torch.load(p), strict = False)
    print(f"Loaded model weights from {p}")
    if empty_cuda:
        torch.cuda.empty_cache()
    if use_cuda:
        if len(cuda_ids) == 1:
            cuda_id = cuda_ids[0]
            device = torch.device(f"cuda:{cuda_id}")
        elif len(cuda_ids) > 1:
            device =  torch.device("cuda")
            print("Let's use", len(cuda_ids), "GPUs!")
            model = nn.DataParallel(model, device_ids=cuda_ids)
            model.to(device)
    else:
        device = torch.device("cpu")
        model = model.to(device)
    model = model.eval()
    
    # Dataset to evaluate.
    mdir = '/home/kat/Repos/SALSA/data/model_ready/'
    if ood and extended:
        raise ValueError("Must choose OOD or extended.")
    if extended:
        anc_path = f'{mdir}{which_train}/train/anchor_smiles_extended.csv'
        print(f"Loaded evaluation data from {anc_path}")
        ds = ContraSeqDataset(anc_path)
        _df = get_dataset_array(anc_path, aug_path)
        _df.columns = ['Smiles','Atype','Label'] 
        if samp_size < len(_df):
            rand_idc = random.sample(range(0,len(_df)),samp_size)
            df = _df.iloc[rand_idc]
            idc = rand_idc
        else:
            df = _df
            
        # Get latent code from model.
        loader = DataLoader(ds, batch_size=eval_bs, sampler=idc, num_workers=0, pin_memory=True)
        latents = []
        for samp in tqdm(loader, total=len(df)//eval_bs):
            for k,v in samp.items():
                if torch.is_tensor(v):
                    samp[k] = v.to(device)
            latent, _ = model.forward(samp['seq'], samp['pad_mask'], 
                                      samp['avg_mask'], samp['out_mask'], normed_latent)
            latent = latent.cpu().detach().numpy()
            latents.append(latent)

        latents = np.concatenate(latents, axis=0)
    
    return (df, latents)
    
    
    
    
def get_df_and_latents(tag, which_train, which_test, test_ood, samp_size, eval_bs, 
                       n_epochs, use_cuda, empty_cuda, cuda_ids, normed_latent=True):
    
    """
    which_test: 'train' -> infer v from training set. 
                '01' -> test set for v 01.
                
    """
    # Load model. 
    model = SeqAutoencoder(dim_emb=512, heads=8, dim_hidden=32,
                           L_enc=6, L_dec=6, dim_ff=2048, 
                           drpt=0.1, actv='relu', eps=0.6, b_first=True)

    p = f'/home/kat/Repos/SALSA/results/models/{tag}/{n_epochs-1:02}.pt'
    model.load_state_dict(torch.load(p), strict = False)
    print(f"Loaded model weights from {p}")

    if empty_cuda:
        torch.cuda.empty_cache()

    if use_cuda:
        if len(cuda_ids) == 1:
            cuda_id = cuda_ids[0]
            device = torch.device(f"cuda:{cuda_id}")
        elif len(cuda_ids) > 1:
            device =  torch.device("cuda")
            print("Let's use", len(cuda_ids), "GPUs!")
            model = nn.DataParallel(model, device_ids=cuda_ids)
            model.to(device)
    else:
        device = torch.device("cpu")
        model = model.to(device)

    model = model.eval()

    # Dataset to evaluate.
    mdir = '/home/kat/Repos/SALSA/data/model_ready/'
    
    if ood and extended:
        raise ValueError("Must choose OOD or extended.")
            
#     if extended:
#         anc_path = f'{mdir}{which_train}/train/anchor_smiles_extended.csv'
#         ds = ContraSeqDataset(anc_path, aug_path)
#         ds_arr = get_dataset_array(anc_path, aug_path)
        
    if which_test == 'train':
        anc_path = f'{mdir}{which_train}/train/anchor_smiles.csv'
        aug_path = f'{mdir}{which_train}/train/augmented_smiles.csv'
    else:
        if ood:
            tag_caboose = '_ood'
        else:
            tag_caboose = ''
        anc_path = f'{mdir}{which_test}/test/anchor_smiles{tag_caboose}.csv'
        aug_path = f'{mdir}{which_test}/test/augmented_smiles{tag_caboose}.csv'   
    print(f"Loaded evaluation data from {anc_path}")

    ds = ContraSeqDataset(anc_path, aug_path)
    ds_arr = get_dataset_array(anc_path, aug_path)
    anc_map = get_anc_map(ds_arr)
    _df = copy.deepcopy(ds_arr)

    _df.columns = ['Smiles','Atype','Label'] 

    if samp_size < len(anc_map):
        rand = random.sample(range(0,len(anc_map)),samp_size)
        rand_idc = np.concatenate([anc_map[x] for x in rand],axis=0)
        df = _df.iloc[rand_idc]
        idc = rand_idc
    else:
        df = _df
        idc = range(len(df))

    # Get latent code from model.
    loader = DataLoader(ds, batch_size=eval_bs, sampler=idc, num_workers=0, pin_memory=True)
    latents = []
    for samp in tqdm(loader, total=len(df)//eval_bs):
        for k,v in samp.items():
            if torch.is_tensor(v):
                samp[k] = v.to(device)
        latent, _ = model.forward(samp['seq'], samp['pad_mask'], 
                                  samp['avg_mask'], samp['out_mask'], normed_latent)
        latent = latent.cpu().detach().numpy()
        latents.append(latent)

    latents = np.concatenate(latents, axis=0)
    
    return (df, latents)

import seaborn as sns 
import matplotlib.pylab as plt
import umap.umap_ as umap

def get_umap_coords(tag, which_test, test_ood, samp_size, df, latents, n_neighs, min_dist, 
                    save_coords=True, save_plot=True, show_plot=True):
    
    ood = 'ood' if test_ood else ''
    which = 'train' if which_test=='train' else 'test'
    which = 'ood' if test_ood else which
    tag_emb = '_'.join([x for x in [tag, f'{samp_size}n', which] if len(x)>0])
    
    rdir = '/home/kat/Repos/SALSA/results/'
    
    umapper = umap.UMAP(n_neighbors=n_neighs, min_dist=min_dist, 
                        n_components=2, metric='euclidean')
    embedding = umapper.fit_transform(latents)

    df['x'] = embedding[:, 0]
    df['y'] = embedding[:, 1]

    pt = str(min_dist).split('.')[1]
    tag_coords = '_'.join([tag_emb, f'{n_neighs}neigh', f'mindist0pt{pt}'])

    if save_coords:
        csv_out = f'{rdir}umap_dfs/{tag_coords}.csv'
        df.to_csv(csv_out,index=False)
        print(f"Saved smiles and coords to {csv_out}!")
        
    sns.set_theme(style='ticks',font_scale=1.5)
    plt.figure(figsize=(10,10))
    sns.scatterplot(data=df[df['Atype']=='Aug'], x='x', y='y', hue='Atype', 
                    alpha=0.5, s=10, palette={'Aug':'red'})
    sns.scatterplot(data=df[df['Atype']=='Anc'], x='x', y='y', hue='Atype', 
                    alpha=1., s=7, palette={'Anc':'blue'})
        
    if save_plot:
        png_out = f'{rdir}umap_figs/{tag_coords}.png'
        plt.savefig(png_out, bbox_inches='tight')
        print(f"Saved umap plot to {png_out}!")
    if show_plot:
        plt.show()
        
    return df
        
        
        
        
        
        
        