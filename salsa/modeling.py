import sys
sys.path.insert(0, '..')
import random
import numpy as np
import pandas as pd
import os
import sys
import csv
random.seed(666)

from tqdm import tqdm
import torch
import torch.nn as nn
from datetime import datetime
from collections import OrderedDict
import warnings
warnings.filterwarnings("ignore")

from salsa.constants import NAMED_LOSSES


#< ---- 20 char ---->< ---- 20 char ---->< ---- 20 char ---->< --- 18 char --->

# # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # #
# # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # #
# Training functions  # # # # # # # # # # # # # # # # # # # # # # # # # # # #
# # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # #
# # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # #

from torch.cuda.amp import GradScaler 
from torch import autocast 
def fit(model, sampler, loader, use_cuda=True, tag=None, 
        lr=0.00001, losses=NAMED_LOSSES, n_epochs=1, save_step=10000,
        model_path = None):
    ''' 
        This function fits a new SALSA model.
    '''   

    model, device = get_model_and_device(model, use_cuda=use_cuda)
    model.train()
    torch.set_grad_enabled(True)

    print(f"Preparing to train model with tag '{tag}' ...")
    loss_csv = f'{model_path}/keep/losses_{tag}.csv'
    
    start = datetime.now()
    s = start    
    print(loss_csv)
    with open(loss_csv,'w',newline='') as output_file:
        
        writer = csv.writer(output_file, delimiter=',')
        # writer.writerow(['Loss'] + NAMED_LOSSES)  
        # print(['Loss'] + NAMED_LOSSES)
        writer.writerow(['Loss'] + losses)  
        print(['Loss'] + losses)
        loss_data = []
        
        optimizer = torch.optim.Adam(model.parameters(), lr=lr) 
        scaler = GradScaler()

        for i in range(n_epochs):
            # print(i,range(n_epochs))
            # print(len(loader))
            for j,samp in tqdm(enumerate(loader), total=len(loader), desc=f"Epoch {i}"):
                # print(j)

                optimizer.zero_grad()

                with autocast(device_type='cuda',dtype=torch.float16):

                    for k,v in samp.items():
                        if torch.is_tensor(v):
                            samp[k] = v.to(device)

                    _, inter_latent, dec_out = model.forward(**samp)
                    loss, run_data = get_loss_data(losses, samp, inter_latent, 
                                                dec_out, sampler.fams_per_batch)
                loss = loss.to(device).float()
                loss_data.append(run_data)
                writer.writerow([loss.item()] + run_data)
                # print([loss.item()] + run_data)
                
                # loss.backward()
                # optimizer.step()
                scaler.scale(loss).backward()
                scaler.step(optimizer)
                scaler.update()
                
                # Save models at specified steps.
                if j%save_step==0 or j==(len(loader)-1):
                    save_model(model, model_path, loader, i, j, save_step)
              
            # Report progress.
            e = datetime.now()
            lap = e - s
            s = e
            hrs, mins, secs = get_time(lap.seconds)
            print(f"Epoch done. Runtime: {hrs} hrs {mins}  mins {secs} secs.") 

            if i+1 == n_epochs:
                lap = e - start
                hrs, mins, secs = get_time(lap.seconds)
                print(f"All done. Total runtime: {hrs} hrs {mins}  mins {secs} secs.")
                
    return model



#< ---- 20 char ---->< ---- 20 char ---->< ---- 20 char ---->< --- 18 char --->

# # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # #
# # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # #
# Accessory modeling functions  # # # # # # # # # # # # # # # # # # # # # # #
# # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # #
# # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # #


from salsa.modules import SupConLoss, CrossEntropyLoss, get_aligniform_loss

def get_loss_data(losses, samp, inter_latent, dec_out, fams_per_batch):
    ''' This function computes loss. 
    '''
    # loss_vals = torch.full((len(NAMED_LOSSES),1), False)
    # loss_dict = OrderedDict(zip(NAMED_LOSSES, loss_vals))
    loss_vals = torch.full((len(losses),1), False)
    loss_dict = OrderedDict(zip(losses, loss_vals))
    if 'SupCon' in losses:
        fam_ids = torch.tensor(range(fams_per_batch))
        loss_dict['SupCon'] = SupConLoss()(features=inter_latent, labels=fam_ids)
    if 'Recon' in losses:
        loss_dict['Recon'] = CrossEntropyLoss(dec_out, **samp)  
    if 'Aligniform' in losses:
        loss_dict['Aligniform'] = get_aligniform_loss(latent=inter_latent)
    loss = sum(loss_dict[k] for k in list(set(losses)))
    loss_items = {k:v.item() for k,v in loss_dict.items()}
    return (loss, [v for _,v in loss_items.items()])



def get_model_and_device(model, use_cuda=False):
    ''' This function loads the model to the proper device. 
    '''
    device = torch.device("cuda" if use_cuda else "cpu")  
    if isinstance(use_cuda, list):
        if len(use_cuda)==1:
            _id = use_cuda[0]
            device = torch.device(f"cuda:{_id}") 
        elif len(use_cuda)>1 and 0 in use_cuda:
            model = nn.DataParallel(model, device_ids=use_cuda)
        model.to(device)
    else: 
        model = model.to(device)  
    return model, device



def save_model(model, model_path, loader, i, j, save_step=10000):
    ''' This function saves model weights (state dictionary) to a .pt file. 
    '''
    try:
        state_dict = model.module.state_dict()
    except AttributeError:
        state_dict = model.state_dict()
    pad = len(str(len(loader)))
    fname = f'{i:02}_{j:0{pad}}.pt'
    if j % save_step == 0:
        path_to_wgts = os.path.join(model_path,'checkpoints',fname)
    if j == (len(loader)-1):
        path_to_wgts = os.path.join(model_path,'checkpoints',fname)
        path_to_wgts = os.path.join(model_path,'keep',fname)
    torch.save(state_dict, path_to_wgts)


def get_time(seconds):
    ''' This function returns run time in (hr, min, sec) format. 
    '''
    seconds_left = seconds
    hours = seconds_left // 3600
    seconds_left = seconds_left % 3600
    minutes = seconds_left // 60
    seconds_left = minutes % 60
    seconds = seconds_left
    return hours, minutes, seconds