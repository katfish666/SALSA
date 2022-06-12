import torch.nn as nn
import torch
from seqAE_model import SeqAutoencoder
from contra_seq_dataset import *
from torch.utils.data import DataLoader, RandomSampler
from tqdm import tqdm

def get_latents(model_tag, ds_set, ds_cut, load_bs=32):
    tag = model_tag
    cut = ds_cut
    bs = load_bs

    n_epochs = 30
    use_cuda = True
    empty_cuda = True
    cuda_ids = [0,1,2,3]
    model = SeqAutoencoder(dim_emb=512, heads=8, dim_hidden=32,
                           L_enc=6, L_dec=6, dim_ff=2048, 
                           drpt=0.1, actv='relu', eps=0.6, b_first=True)

    p = f'/home/kat/Repos/SALSA/results/models/{tag}/{n_epochs-1:02}.pt'
    model.load_state_dict(torch.load(p), strict = False)
    if empty_cuda:
        torch.cuda.empty_cache()
    if use_cuda:
        if len(cuda_ids) == 1:
            cuda_id = cuda_ids[0]
            device = torch.device(f"cuda:{cuda_id}")
        elif len(cuda_ids) > 1:
            device =  torch.device("cuda")
            print(f"Using {len(cuda_ids)} GPUs!")
            model = nn.DataParallel(model, device_ids=cuda_ids)
            model.to(device)
    else:
        device = torch.device("cpu")
        model = model.to(device)
    model = model.eval()
    print(f"Loaded model weights from {p}!")

    p = f'data/model_ready/{ds_set}/{cut}/anchor_smiles.csv'
    ds = ContraSeqDataset(p)
    df = get_dataset_array(p)

    loader = DataLoader(ds, batch_size=bs, sampler=range(len(df)), 
                        num_workers=0, pin_memory=True)
    latents = []
    for samp in tqdm(loader, total=len(df)//bs):
        for k,v in samp.items():
            if torch.is_tensor(v):
                samp[k] = v.to(device)
        latent, _ = model.forward(samp['seq'], samp['pad_mask'], 
                                  samp['avg_mask'], samp['out_mask'])
        latent = latent.cpu().detach().numpy()
        latents.append(latent)
    latents = np.concatenate(latents, axis=0)
    
    return latents


from sklearn.metrics import confusion_matrix
from sklearn.metrics import average_precision_score
from sklearn.metrics import roc_auc_score

def get_metrics(y_test, y_pred, y_prob, which_metrics=['Sn','Sp','PPV','BAcc']):
    
    conf_matrix = confusion_matrix(y_test, y_pred)
    
    TN, FP, FN, TP = conf_matrix.ravel()

    # Sensitivity, hit rate, recall, or true positive rate
    TPR = TP/(TP+FN)
    # Specificity or true negative rate
    TNR = TN/(TN+FP) 
    # Precision or positive predictive value
    PPV = TP/(TP+FP)
    # Negative predictive value
    NPV = TN/(TN+FN)
    # Fall out or false positive rate
    FPR = FP/(FP+TN)
    # False negative rate
    FNR = FN/(TP+FN)
    # False discovery rate
    FDR = FP/(TP+FP)

    BACC = (TPR+TNR)/2
    ACC = (TP+TN)/(TP+FP+FN+TN)
    
    AP = average_precision_score(y_test, y_pred)
    AUROC = roc_auc_score(y_test, y_prob)

#     my_metrics = ['sensitivity','specificity','precision','accuracy']
#     my_scores = [TPR,TNR,PPV,ACC]
    
    return {'Sn':TPR,'Sp':TNR,'Pr':PPV,'BAcc':BACC,
            "Average Precision":AP, "AUROC":AUROC}

#     for metric, score in zip(my_metrics,my_scores):
#         print(f'{metric:<10} {round(score,3)}')
        
        
