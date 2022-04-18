from contra_seq_dataset import AnchoredSampler, ContraSeqDataset
from contra_seq_dataset import get_dataset_array, get_anc_map
from torch.utils.data import DataLoader, RandomSampler

from utils.plot_utils import *
from losses import SupConLoss, padce_loss

import os
import time
from datetime import datetime
import torch
import torch.nn as nn

def get_loss_data(use_losses, run_data, samp, dec_out, latent, BS):
    if 'Recon' in use_losses: 
        recon_loss = padce_loss(samp['seq'], dec_out.squeeze(), 
                                samp['pad_mask'], samp['out_mask'])  
        run_data['Recon'].append(recon_loss.item())
    if 'SupCon' in use_losses: 
        contra_loss = SupConLoss()(latent, labels=torch.tensor(range(BS)))
        run_data['SupCon'].append(contra_loss.item())
        
    if set(use_losses)=={'Recon'}:
        loss = recon_loss
    elif set(use_losses)=={'SupCon'}:
        loss = contra_loss
    elif set(use_losses)=={'Recon','SupCon'}:
        loss = recon_loss + contra_loss
    
    return (loss, run_data)

def get_ds_and_loader(ds_v, bs=32):
    '''
    BS is the "batch_size", i.e. the number of anchors.
    '''
    
    anc_path = f'data/model_ready/{ds_v}/anchor_smiles.csv'
    aug_path = f'data/model_ready/{ds_v}/augmented_smiles_balanced.csv'

    ds = ContraSeqDataset(anc_path, aug_path)
    ds_arr = get_dataset_array(anc_path, aug_path)
    anc_map = get_anc_map(ds_arr)
    
    sampler = AnchoredSampler(sampler = RandomSampler(list(anc_map.keys())), 
                              anc_map = anc_map, batch_size = bs, drop_last = True)
    loader = DataLoader(ds, batch_sampler=sampler, num_workers=0, pin_memory=True)
    
    return (ds, loader)


def fit(model, device, optimizer, loader, use_losses, v, bs=32, n_epochs=1, 
        normed_latent=True, do_plot=False, save_fifths=True):
    
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

            optimizer.zero_grad()

            for k,v in samp.items():
                if torch.is_tensor(v):
                    samp[k] = v.to(device)
            latent, dec_out = model.forward(samp['seq'], samp['pad_mask'], 
                                            samp['avg_mask'], samp['out_mask'], 
                                            normed=normed_latent)
            latent = torch.stack(torch.split(latent, 6), dim=0) # (BS, 6, 32)     

            loss, run_data = get_loss_data(use_losses, run_data, samp, dec_out, latent, bs)
            loss.backward()
            optimizer.step()

            if do_plot:
                live_plot(run_data, bs, n_epochs, figsize=(12.5,5))
                
        e = datetime.now()
        lap = e - s
        s = e
        print(f"Epoch done. Runtime: {lap.seconds//60%60} mins {lap.seconds%60} secs.")

        try:
            state_dict = model.module.state_dict()
        except AttributeError:
            state_dict = model.state_dict()                

        if (i+1) % 5==0:
            if save_fifths:
                torch.save(state_dict, model_path)
        if i+1 == n_epochs:
            lap = e - start 
            h, m, s = lap.seconds//3600, lap.seconds//60%60, lap.seconds%60
            print(f"All done. Total runtime: {h} hr {m}  min {s} sec.")
            torch.save(state_dict, model_path)
                
    return run_data


#         run_data = train_epoch(model, optimizer, device, loader, use_losses, 
#                                bs, n_epochs, normed_latent, do_plot)    
# def train_epoch(model, optimizer, device, loader, use_losses, bs=32, n_epochs=1, 
#                 normed_latent=True, do_plot=False):
    
#     for samp in loader:

#         optimizer.zero_grad()

#         for k,v in samp.items():
#             if torch.is_tensor(v):
#                 samp[k] = v.to(device)
#         latent, dec_out = model.forward(samp['seq'], samp['pad_mask'], 
#                                         samp['avg_mask'], samp['out_mask'], 
#                                         normed=normed_latent)
#         latent = torch.stack(torch.split(latent, 6), dim=0) # (BS, 6, 32)     

#         loss, run_data = get_loss(use_losses, run_data, samp, dec_out, latent, BS)
#         loss.backward()
#         optimizer.step()

#         if do_plot:
#             live_plot(run_data, bs, n_epochs, figsize=(12.5,5))
            
#     return run_data
    