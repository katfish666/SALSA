import warnings
warnings.filterwarnings("ignore")

import torch
import torch.nn as nn
from seqAE_model import SeqAutoencoder
from training_functions import *

# ds, loader = get_ds_and_loader('00',32)

model = SeqAutoencoder(n_tokens = ds.n_tokens, max_len = 122,
                       dim_emb=512, heads=8, dim_hidden=32,
                       L_enc=6, L_dec=6, dim_ff=2048, 
                       drpt=0.1, actv='relu', eps=0.6, b_first=True)

torch.cuda.empty_cache()
use_cuda = True
device =  torch.device("cuda" if use_cuda else "cpu")

if use_cuda and torch.cuda.device_count() > 1:
    print("Let's use", torch.cuda.device_count()-1, "GPUs!")
    model = nn.DataParallel(model, device_ids = [0, 1, 2])
    model.to(device)
else:
    model = model.to(device)
    
optimizer = torch.optim.Adam(model.parameters(), lr = 0.00001) 

model.train()
torch.set_grad_enabled(True)

from training_functions import *

use_losses = ['Recon']
v = '02'

today = datetime.today().strftime('%Y%m%d%H')
tag = f'{today}_{v}'

run_data = fit(model, device, optimizer, loader, use_losses, v, bs=16, n_epochs=50, 
               normed_latent=True, do_plot=False, save_fifths=True)

df_run = pd.DataFrame.from_dict(run_data)
df_run.to_csv(f'results/training_logs/losses_{tag}.csv')