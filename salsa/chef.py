import pandas as pd
import sys
sys.path.insert(0, '..')
sys.path.insert(0, '../..')
sys.path.insert(0, '../salsa/')
import warnings
warnings.filterwarnings("ignore")
from datetime import datetime
import random
import yaml
import os
import numpy as np
from datetime import datetime
random.seed(666)

from salsa.datasets import TrainingDataset, InferenceDataset, RandomFamSampler, DataLoader
from salsa.modules import SalsaNet
from salsa.modeling import fit, get_model_and_device
from tqdm import tqdm
import torch
import torch.nn as nn

def prnt(string, print_out):
    if print_out:
        print(string)
    else:
        return

# # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # #
# # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # #
# Cook & Serve functions  # # # # # # # # # # # # # # # # # # # # # # #
# # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # #
# # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # 

#< ---- 20 char ---->< ---- 20 char ---->< ---- 20 char ---->< --- 18 char --->


from salsa.modules import count_params
def serve_salsa(input_ds, model_version, use_cuda=False, batch_size=1, 
                df_mode='salsa', weight_dir='keep',machine='gpubeast', 
                epoch='00', step=None, use_tqdm=False, print_out=False):
    '''
    This is a wrapper function that computes SALSA latents and 
        returns them in a dataframe.
    Args:
        input_ds: Input dataset. Can be either:
            1) dataframe with 'Smiles' column 
            2) path to csv file
            3) list of SMILES strings
            4) np array of SMILES strings
        model_version: Which model version to use? 
            Default is 27_202305301701, a 32-dim salsa.
        use_cuda: Which gpus to use if any (e.g. False, [0], [1,2,3], etc.). 
            False = cpu is used.
        batch_size: The batch size.
        df_mode: Return original df or InferenceDataset df? 
            'salsa' returns salsa df (id, origin_id, smiles, latent).
            'original' returns original input df columns and salsa df columns.
        weight_dir: Which dir to pull weights from?
        machine: 'gpubeast' or 'longleaf'
        epoch: Which epoch? indexing from 0. (i.e. 1st epoch = 0, 2nd epoch = 1).
        step: Exact step in epoch?
    Returns:
        df_w_latents: Dataframe with additional column of latents.
        subversion: Which subversion of salsa was used.
        salsa_latents: Matrix of salsa latents.
        train_ds_name: Name of the training set.
    '''     

    if machine=='gpubeast':
        model_dir = f'/data/{model_version}/{weight_dir}/'
        yml_dir = f'/data/{model_version}/keep/'
    else:
        model_dir = f'../models/{model_version}/{weight_dir}/'
        yml_dir = f'../models/{model_version}/keep/'
            
    ## Get epoch and step if not provided by user
    fs = os.listdir(f'{model_dir}')
    fs = [f for f in fs if '.pt' in f]
    if epoch is None:
        epochs = sorted(set([x[:2] for x in fs]))
        epoch = epochs[-1]
    if step is None:
        steps = sorted([f[3:f.find('.pt')] for f in fs if f[:2]==epoch])
        step = steps[-1]
        
    subversion = f'{model_version}-{epoch}-{step}'

    # msgs = [f" > {k}: {v}" for k,v in locals().items() if k not in ['input_ds','steps','epochs','fs']]
    # print(('\n').join(msgs))
    
    # print(f"\nConstructing InferenceDataset ...")
    ds_salsa = InferenceDataset(input_ds) 
    df = ds_salsa.dataframe     

    ## Correct batch size !!
    if use_cuda: bs = batch_size - (batch_size % len(use_cuda))
    elif not use_cuda: bs = batch_size

    ## Loader.
    prnt(f"\nCreating DataLoader with (corrected) batch size of {bs} ...", print_out)
    loader = DataLoader(ds_salsa, batch_size=bs, shuffle=False, drop_last=False)
    
    ## Read params and load model weights.
    prnt(f"Preparing to serve salsa! {model_version} epoch {epoch} step {step} ...",print_out)
    param_file = f'{yml_dir}params.yaml'  
    with open(param_file, "r") as f:
        js = f.read()
        params = yaml.safe_load(js)
    model = SalsaNet(**params)
    weights_path = f'{model_dir}{epoch:02}_{step}.pt'

    ## Get the TrainingDataset version ...
    train_ds_name = params['train_csv'].replace('.csv','')

    ## Param count.
    cnt = count_params(model)
    prnt(f" Parameter count: {cnt:,}", print_out)

    ## Compute latent codes.
    salsa_latents = get_salsa_latents(model, weights_path, loader, use_cuda, use_tqdm)
    prnt(f"\nYou have been served salsa! Subversion: {subversion}", print_out)
    
    ## Return df with latent codes.
    if df_mode=='original':
        df_w_latents = df
    elif df_mode=='salsa':
        df_w_latents = ds_salsa.dataframe
    df_w_latents[subversion] = [x for x in salsa_latents]

    return (df_w_latents, salsa_latents, subversion, train_ds_name)



#< ---- 20 char ---->< ---- 20 char ---->< ---- 20 char ---->< --- 18 char --->

# def cook_salsa(train_csv, model_dir, use_cuda, version, batch_size, lr, losses, 
#                n_epochs, save_step, dim_emb, dim_salsa, n_heads, L_enc, L_dec, 
#                dim_ff, proj_head, pool_func, actv, layer_norm_eps):

def cook_salsa(train_csv, model_dir=None, use_cuda='0', version='test', 
               batch_size=1, lr=0.00001, losses='Recon-SupCon', 
               n_epochs=1, save_step=10000, dim_emb=512, dim_salsa=32, 
               n_heads=8, L_enc=8, L_dec=8, dim_ff=2048, 
               proj_head=False, pool_func='avg', actv='relu', 
               layer_norm_eps=0.6, random_smis=False, warm_start_path=None,
               notes=""):
    '''
    Args:
        train_csv: 
        model_dir
    Returns:
        model: the model
    '''     

    if model_dir is None:
        model_dir = '~/Repos/Salsa/models/'

    msgs = [f" > {k}: {v}" for k,v in locals().items()]
    print(('\n').join(msgs))

    my_params = {
        'train_csv':train_csv, 'model_dir':model_dir, 'use_cuda':use_cuda,
        'version':version, 'batch_size':batch_size, 'lr':lr,
        'losses':losses, 'n_epochs':n_epochs, 'save_step':save_step,
        'dim_emb':dim_emb, 'dim_salsa':dim_salsa, 'n_heads':n_heads,
        'L_enc':L_enc, 'L_dec':L_dec, 'dim_ff':dim_ff,
        'proj_head':proj_head, 'pool_func':pool_func, 'actv':actv,
        'layer_norm_eps':layer_norm_eps,'random_smis':random_smis,
        'warm_start_path':warm_start_path, 'notes':notes,
        }

    ## Unjoin listed params.
    if isinstance(use_cuda,str):
        use_cuda = [int(x) for x in use_cuda.split('-')]
    losses = losses.split('-')
    # print(use_cuda, losses)

    ## Get model tag
    t = datetime.today().strftime('%Y%m%d%H%M')
    tag = f'{version}_{t}'

    ## Create directory
    model_path = os.path.join(model_dir,tag)
    if not os.path.exists(model_path):
        # print(os.getcwd())
        os.mkdir(model_path)
        os.mkdir(os.path.join(model_path,'checkpoints'))
        os.mkdir(os.path.join(model_path,'keep'))
        message = f"Created new directory '{model_path}'!"
    else: 
        message = f"Directory '{model_path}' exists!"
    print(message)

    ## Dump params into yaml for later loading.
    yml = yaml.dump(my_params)
    with open(f'{model_path}/keep/params.yaml','w') as f:
        f.write(yml)

    ## Load training dataset
    ddir = '~/Repos/Salsa/data/train/' 

    ## Saveguard for if '.csv' is in train_csv arg !!
    train_csv = train_csv.replace('.csv','')
    _df = pd.read_csv(f'{ddir}{train_csv}.csv')
    ds_salsa = TrainingDataset(_df)

    ## Get fam size!
    fam_size = ds_salsa.fam_size

    ## Correct batch size ...
    factor = len(use_cuda)*fam_size
    bs = batch_size - (batch_size % factor)

    ## Sampler, loader
    sampler = RandomFamSampler(ds_salsa.anc_ids, batch_size=bs, drop_last=True)
    loader = DataLoader(ds_salsa, batch_sampler=sampler, pin_memory=True)
    print(f"Created DataLoader with (corrected) batch size of {bs}!")

    ## Fit model
    model = SalsaNet(dim_emb, dim_salsa, n_heads, L_enc, L_dec, dim_ff, fam_size,
                     proj_head, pool_func, actv)
    if warm_start_path is not None:
        model.load_state_dict(torch.load(warm_start_path), strict = False)
    fit(model, sampler, loader, use_cuda, tag, 
        lr, losses, n_epochs, save_step, model_path) 
    
    return model
    



#< ---- 20 char ---->< ---- 20 char ---->< ---- 20 char ---->< --- 18 char --->



def write_salsa_recipe(cols=None, notebook=False, save_to_desktop=True):
    import yaml
    import dataframe_image as dfi
    from io import BytesIO
    from IPython.display import display
    from PIL import Image
    '''
    Args:
        cols: selected params in order.
    Returns:
        writes to disk: df as csv.
        writes to disk: df as excel file.
        df: returns input df of models' params.
    '''
    date = datetime.today().strftime('%Y%m%d%H%M')
    models_dir = f'/data/'
    fs = os.listdir(f'{models_dir}')
    fs = [f for f in fs if len(f)==15]
    
    params_list = []
    for v in fs:
        if os.path.isfile(f'{models_dir}{v}/checkpoints/params.yaml'):
            fname = f'{models_dir}{v}/checkpoints/params.yaml'
        elif os.path.isfile(f'{models_dir}{v}/keep/params.yaml'):
            fname = f'{models_dir}{v}/keep/params.yaml'
        else:
            # raise Exception (f"No params file! version: {v}")
            continue
    
        with open(fname, "r") as f:
            js = f.read()
            params = yaml.safe_load(js)
            params = {k:v for k,v in params.items() if k!='model_dir'}
            params['version_name'] = v
            if 'layer_norm_eps' not in params.keys():
                params['layer_norm_eps'] = 0.00001
            if 'warm_start_path' not in params.keys():
                params['warm_start_path'] = 'None'
            if 'random_smis' not in params.keys():
                params['random_smis'] = 'False'
            if 'notes' not in params.keys():
                params['notes'] = ''
            params_list.append(params)
            
    df = pd.DataFrame(params_list)

    if cols is not None:
        df = df[cols]
    
    df = df.sort_values('version_name',inplace=False)
    df.reset_index(inplace=True,drop=True)
    
    # if notebook:
        # display(df)
    
    ## Save as csv.
    df.to_csv(f'../models/model_params_{date}.csv',index=False)
    
    ## Save as excel.
    df.to_excel(f'../models/excels/model_params_{date}.xlsx', index=None, header=True)
    
    ## Save as png.
    buf = BytesIO()
    dfi.export(df, buf)
    png = buf.getvalue()
    fname = f'../figs/param_tables/{date}.png'
    with open(fname,'wb') as fd:
        fd.write(png)

    if notebook:
        display(Image.open(fname))

    import subprocess
    ## scp png to MacBook desktop
    if save_to_desktop:
        args = ['scp',fname,'kat@kathryns-air-5.wireless-1x.unc.edu:/Users/kat/Desktop/']
        subprocess.run(args) 
  
    return df



#< ---- 20 char ---->< ---- 20 char ---->< ---- 20 char ---->< --- 18 char --->

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


# # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # #
# # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # #
# Inference utils functions # # # # # # # # # # # # # # # # # # # # # # 
# # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # #
# # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # 

#< ---- 20 char ---->< ---- 20 char ---->< ---- 20 char ---->< --- 18 char --->

def get_salsa_latents(model, weights_path, loader, use_cuda, use_tqdm=True):
    ''' 
        This function returns SALSA latent codes for a given model and
        its weights (loaded from weights_path).
    '''   
    
    model.load_state_dict(torch.load(weights_path), strict = False)
    model, device = get_model_and_device(model, use_cuda=use_cuda)
    model.eval()

    salsa_list = []
    for samp in tqdm(loader, total=len(loader), disable=not use_tqdm):
        for k,v in samp.items():
            if torch.is_tensor(v): samp[k] = v.to(device)
        _, inter_latent = model.forward(**samp)
        salsa_latent = inter_latent.cpu().detach().numpy()
        salsa_list.append(salsa_latent)
    
    salsa_latents = np.concatenate(salsa_list)

    return salsa_latents



def load_model(version, use_cuda, weight_dir='keep', machine='gpubeast', 
              epoch=None, step=None):

    if machine=='gpubeast':
        model_dir = f'/data/{version}/{weight_dir}/'
        yml_dir = f'/data/{version}/keep/'
    else:
        model_dir = f'../models/{version}/{weight_dir}/'
        yml_dir = f'../models/{version}/keep/'

    ## Get epoch and step if not provided by user
    fs = os.listdir(f'{model_dir}')
    fs = [f for f in fs if '.pt' in f]
    if epoch is None:
        # print(fs)
        epochs = sorted(set([x[:2] for x in fs]))
        epoch = epochs[-1]
    if step is None:
        steps = sorted([f[3:f.find('.pt')] for f in fs if f[:2]==epoch])
        step = steps[-1]
        
    subversion = f'{version}-{epoch}-{step}'

    msgs = [f" > {k}: {v}" for k,v in locals().items() if k not in ['input_ds','steps','epochs','fs']]
    print(('\n').join(msgs))

    ## Read params and load model weights.
    print(f"Preparing salsa vendor! {version} epoch {epoch} step {step} ...")
    param_file = f'{yml_dir}params.yaml'  
    with open(param_file, "r") as f:
        js = f.read()
        params = yaml.safe_load(js)
    model = SalsaNet(**params)
    weights_path = f'{model_dir}{epoch:02}_{step}.pt'

    ## Param count.
    cnt = count_params(model)
    print(f" Parameter count: {cnt:,}")

    model.load_state_dict(torch.load(weights_path), strict = False)
    model, device = get_model_and_device(model, use_cuda=use_cuda)
    model.eval()

    return (subversion, model, device, weights_path)


def store_salsa(date, version, ds_name, feats):
    path = f'../data/latents/{date}__{version}__{ds_name}.txt'
    print(f"Storing salsa at path:\n {path}")
    np.savetxt(path,feats)
