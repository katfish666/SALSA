from contra_seq_dataset import AnchoredSampler, ContraSeqDataset
from contra_seq_dataset import get_dataset_array, get_anc_map
from torch.utils.data import DataLoader, RandomSampler

from utilities.plot_utils import *
from losses import SupConLoss, padce_loss

import os
import torch
import torch.nn as nn
from datetime import datetime
import pandas as pd
import random
random.seed(666)

def get_loss_data(use_losses, run_data, samp, dec_out, latent, BS):
    recon_loss = padce_loss(samp['seq'], dec_out.squeeze(), 
                            samp['pad_mask'], samp['out_mask'])  
    contra_loss = SupConLoss()(latent, labels=torch.tensor(range(BS)))
    run_data['Recon'].append(recon_loss.item())
    run_data['SupCon'].append(contra_loss.item())
        
    if set(use_losses)=={'Recon'}:
        loss = recon_loss
    elif set(use_losses)=={'SupCon'}:
        loss = contra_loss
    elif set(use_losses)=={'Recon','SupCon'}:
        loss = recon_loss + contra_loss
    
    return (loss, run_data)



def get_ds_and_loader(ds_v, bs=32, samp='max'):
    '''
    BS is the "batch_size", i.e. the number of anchors.
    '''
    
    anc_path = f'data/model_ready/{ds_v}/train/anchor_smiles.csv'
    aug_path = f'data/model_ready/{ds_v}/train/augmented_smiles.csv'

    ds = ContraSeqDataset(anc_path, aug_path)
    ds_arr = get_dataset_array(anc_path, aug_path)
    anc_map = get_anc_map(ds_arr)
    
    samp_idc = list(anc_map.keys())
    
    if samp!='max':
        if samp < len(samp_idc):
            samp_idc = random.sample(samp_idc, samp)
#     print(len(samp_idc))
    sampler = AnchoredSampler(sampler = RandomSampler(samp_idc), 
                              anc_map = anc_map, batch_size = bs, drop_last = True)
    loader = DataLoader(ds, batch_sampler=sampler, num_workers=0, pin_memory=True)
    return (ds, loader)


def fit(model, device, optimizer, loader, use_losses, v, bs=32, n_epochs=1, 
        normed_latent=True, do_plot=False, save_step=1):
    
    start = datetime.now()
    s = start
    
    today = datetime.today().strftime('%Y%m%d%H')
    tag = f'{today}_{v}' #'.pt'
    
    model_dir = os.path.join('results/models',tag)    
    if not os.path.exists(model_dir):
        os.mkdir(model_dir)
        print(model_dir)
    else:
        print("DIRECTORY EXISTS! ...",model_dir)
    
    run_data = {k:[] for k in use_losses}
    
    for i in range(n_epochs):
        print(f"Epoch {i} running ...")
        ii = f'0{i}' if i < 10 else i
        model_path = f'{model_dir}/{ii}.pt'
        
        for samp in loader:
#             print(len(samp['seq']))

            optimizer.zero_grad()

            for k,v in samp.items():
                if torch.is_tensor(v):
                    samp[k] = v.to(device)
                    
#             print("fit samp seq:",samp['seq'].shape)
            latent, dec_out = model.forward(samp['seq'], samp['pad_mask'], 
                                            samp['avg_mask'], samp['out_mask'], 
                                            normed=normed_latent)
#             print(latent.shape, dec_out.shape)
            latent = torch.stack(torch.split(latent, 6), dim=0) # (BS, 6, 32)     
#             print(latent.shape)

            loss, run_data = get_loss_data(use_losses, run_data, samp, dec_out, latent, bs)
            loss.backward()
            optimizer.step()

            if do_plot:
                live_plot(run_data, bs, n_epochs, figsize=(12.5,5))

        # Report progress.
        e = datetime.now()
        lap = e - s
        s = e
        print(f"Epoch done. Runtime: {lap.seconds//60%60} mins {lap.seconds%60} secs.")
        
        # Save shit.
        df_run = pd.DataFrame.from_dict(run_data)
        df_run.to_csv(f'results/training_logs/losses_{tag}.csv')
        
        try:
            state_dict = model.module.state_dict()
        except AttributeError:
            state_dict = model.state_dict()                

        if save_step==1:
            torch.save(state_dict, model_path)
        elif (i+1) % save_step==0:
            torch.save(state_dict, model_path)
            
        if i+1 == n_epochs:
            lap = e - start 
            h, m, s = lap.seconds//3600, lap.seconds//60%60, lap.seconds%60
            print(f"All done. Total runtime: {h} hr {m}  min {s} sec.")
            torch.save(state_dict, model_path)
                
    return model


# def _get_loss_data(use_losses, run_data, samp, dec_out, latent, BS):
#     if 'Recon' in use_losses: 
#         recon_loss = padce_loss(samp['seq'], dec_out.squeeze(), 
#                                 samp['pad_mask'], samp['out_mask'])  
#         run_data['Recon'].append(recon_loss.item())
#     if 'SupCon' in use_losses: 
#         contra_loss = SupConLoss()(latent, labels=torch.tensor(range(BS)))
#         run_data['SupCon'].append(contra_loss.item())
        
#     if set(use_losses)=={'Recon'}:
#         loss = recon_loss
#     elif set(use_losses)=={'SupCon'}:
#         loss = contra_loss
#     elif set(use_losses)=={'Recon','SupCon'}:
#         loss = recon_loss + contra_loss
    
#     return (loss, run_data)