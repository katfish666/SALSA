import matplotlib.pyplot as plt
from IPython.display import clear_output

def live_plot(data_dict, bs, e, figsize=(10,3), save=False, path=None):
    clear_output(wait=True)
    plt.figure(figsize=figsize)
    for label,data in data_dict.items():
        plt.plot(data, label=label)
    title=f'Batch size: {bs}, Epochs: {e}'
    plt.title(title)
    plt.grid(True)
    plt.xlabel('Run')
    plt.legend(loc='upper right') # the plot evolves to the right
    if save:
        plt.savefig(path, bbox_inches='tight')
    plt.show();
    
import pandas as pd
import seaborn as sns 
import matplotlib.pyplot as plt

def draw_loss_plot(tag, use_losses, bs, e, y_lim=(0,5), figsize=(12.5,5), save=True):
    
    sns.set_theme(style='ticks',font_scale=1.25)
    losses = pd.read_csv(f'results/training_logs/losses_{tag}.csv',usecols=use_losses)

    plt.figure(figsize=figsize)
    for label,data in losses.iteritems():
        if label=='Recon':
            plt.plot(data, label=label, color="peru")
        elif label=='SupCon':
            plt.plot(data, label=label, color="steelblue")
    plt.ylim(*y_lim)
    plt.title(f'Batch size: {bs}, Epochs: {e}')
    plt.grid(True)
    plt.xlabel('Run')
    plt.legend(loc='upper right')
    
    if save:
        plt.savefig(f'results/training_logs/loss_plot_{tag}.png', bbox_inches='tight')
    plt.show()
    
import torch
import torch.nn as nn
from seqAE_model import SeqAutoencoder
from contra_seq_dataset import ContraSeqDataset

def eval_loss_plot(tag, which_train, use_losses, bs, n_epochs, 
                   use_cuda=False, empty_cuda=False, cuda_ids=[]):
    
    anc_path = f'data/model_ready/{which_train}/train/anchor_smiles.csv'
    aug_path = f'data/model_ready/{which_train}/train/augmented_smiles.csv'

    ds = ContraSeqDataset(anc_path, aug_path)
    model = SeqAutoencoder(n_tokens = ds.n_tokens, max_len = 122,
                           dim_emb=512, heads=8, dim_hidden=32,
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
    device =  torch.device("cuda" if use_cuda else "cpu")
    # device = torch.device("cuda:3")
    if use_cuda==True and torch.cuda.device_count() > 1:
        print("Let's use", torch.cuda.device_count(), "GPUs!")
        model = nn.DataParallel(model, device_ids=devices)
        model.to(device)
    else:
        model = model.to(device)

    model.eval()

    
    
    
    # # # # # # # # # # # 
tag = '2022041804_04' 

which_train = '01'
which_test = '01'

use_losses = ['SupCon','Recon']
bs = 32
n_epochs = 30
normed_latent = True
use_cuda = False
# # # # # # # # # # # 