import torch.nn as nn
import torch
from seqAE_model import SeqAutoencoder
from torch.utils.data import DataLoader, RandomSampler
from contra_seq_dataset import ContraSeqDataset, get_dataset_array, get_anc_map
import copy
import random
import numpy as np
from tqdm.notebook import trange, tqdm
random.seed(666)

def get_df_and_latents(tag, which_train, which_test, test_ood, samp_size, eval_bs, 
                       n_epochs, normed_latent, use_cuda, empty_cuda, cuda_ids):
    # Load model.
    
    model = SeqAutoencoder(max_len = 122, dim_emb=512, heads=8, dim_hidden=32,
                           L_enc=6, L_dec=6, dim_ff=2048, 
                           drpt=0.1, actv='relu', eps=0.6, b_first=True)
    p = f'results/models/{tag}/{n_epochs-1:02}.pt'
    model.load_state_dict(torch.load(p), strict = False)
    print(f"Loaded model weights {p}")

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
    
    if which_test == 'train':
        anc_path = f'data/model_ready/{which_train}/train/anchor_smiles.csv'
        aug_path = f'data/model_ready/{which_train}/train/augmented_smiles.csv'
    else:
        if test_ood:
            tag_caboose = '_ood'
        else:
            tag_caboose = ''
        anc_path = f'data/model_ready/{which_test}/test/anchor_smiles{tag_caboose}.csv'
        aug_path = f'data/model_ready/{which_test}/test/augmented_smiles{tag_caboose}.csv'   
    print(f'Eval set anchors: {anc_path}')

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