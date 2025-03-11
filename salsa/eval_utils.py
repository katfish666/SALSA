import numpy as np
import warnings
warnings.filterwarnings("ignore")
import sys
sys.path.insert(0, '..')
from tqdm import tqdm


#< ---- 20 char ---->< ---- 20 char ---->< ---- 20 char ---->< --- 18 char --->

# # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # #              
# # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # 
# Interpolation functions # # # # # # # # # # # # # # # # # # # # # # # #
# # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # #            
# # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # #   
from salsa.chef import serve_salsa
from salsa.datasets import convert_vec_to_smi
from salsa.utils import get_cansmiles
import torch
from rdkit.Chem.Draw import IPythonConsole
import collections

# from salsa.eval_utils import get_mid_interps
def s_interp(pt1,pt2, dist=0.5):
    return torch.nn.functional.normalize(pt1 + dist*(pt2-pt1), dim = -1)

def serve_latents(smis, version):
    smis = [get_cansmiles(smi) for smi in smis]
    _, latents, _, _ = serve_salsa(smis, version, use_cuda=[1], batch_size=1)
    latents = torch.from_numpy(latents)
    return latents

def serve_gen_smis(model, latent, device, generate_cnt):
    seq = model.generate(latent, device, generate_cnt=generate_cnt)
    gen_smis = convert_vec_to_smi(vec=seq, snip=True)
    gen_smis = [get_cansmiles(x) for x in gen_smis]
    gen_smis = [x for x in gen_smis if x]
    return gen_smis

def get_mid_interps(endpoint_smis, model, version, device, generate_cnt=1, 
                    top_cnt=1, drop_endpoint_gens=True):

    endpoint_latents = serve_latents(endpoint_smis, version)
    latent = s_interp(*endpoint_latents,dist=0.5)
    
    interp_smis = []
    while len(interp_smis)==0: 
        interp_smis = serve_gen_smis(model, latent, device, generate_cnt)

    if drop_endpoint_gens:
        interp_smis = [s for s in interp_smis if s not in endpoint_smis]
        
    counts = collections.Counter(interp_smis)
    top_smis = [x[0] for x in counts.most_common(top_cnt)]
        
    return top_smis, list(set(interp_smis))



#< ---- 20 char ---->< ---- 20 char ---->< ---- 20 char ---->< --- 18 char --->

# # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # 
# # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # #
# Save UMAP transforms  # # # # # # # # # # # # # # # # # # # # # # # # # # #
# # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # #
# # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # #

import umap.umap_ as umap
import joblib
from salsa.chef import serve_salsa

from datetime import datetime
DATE = datetime.today().strftime('%Y%m%d%H%M')

def save_umap_transform(ds_path, version, use_cuda=[0,1,2,3], 
                        batch_size=150, n_neighs=32, min_dist=0.1):
    
    # Compute SALSA latents.
    df = pd.read_csv(f'{ds_path}')
    df, latents, subv, _ = serve_salsa(input_ds=df, model_version=version, 
                                       use_cuda=use_cuda, batch_size=batch_size)
    


    # UMAP transform.
    umap_transformer = umap.UMAP(n_neighbors=n_neighs,
                                 min_dist=min_dist, 
                                 n_components=2, 
                                 metric='euclidean')
    umap_transformer.fit(latents)
    
    # Save transform
    ds_name = ds_path[ds_path.rfind('/')+1:-4]
    umap_tag = f'umapper-{n_neighs}-{min_dist}'.replace('.','pt')
    out_umap = f'../data/umap_transforms/{DATE}__{subv}__{ds_name}.sav'
    joblib.dump(umap_transformer, out_umap) 
    print(f"Saved umap transform! Path: {out_umap}")
    
    # Save latents
    out_latents = f'../data/latents/{DATE}__{subv}__{ds_name}.txt'
    np.savetxt(out_latents,latents) 
    print(f"Saved latents! Path: {out_latents}") 
    
    return out_umap, out_latents


#< ---- 20 char ---->< ---- 20 char ---->< ---- 20 char ---->< --- 18 char --->

# # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # 
# # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # #
# GED-EuD supermutant benchmark # # # # # # # # # # # # # # # # # # # # # # #
# # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # #
# # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # #

from scipy.spatial import distance

def get_method_to_dists_dict(methods, df, depth=5):
    
    method_to_dists = {}

    for method in methods:

        depths = np.arange(1,depth+1)

        ## Get all anchors. 
        df_anc = df[df.Is_anc==True]
        anc_ids = df_anc.Anc_id.unique().tolist()

        ## Get supermut df.
        df_supmuts = df[df.Is_anc==False]  

        step_dist_dict = {}

        for step in depths:

            step_dist_dict[step] = []

            for anc_id in tqdm(anc_ids,total=len(anc_ids),desc=f"{method}: depth {step}"):

                anc = df_anc[df_anc.Anc_id==anc_id]
                anc_feat = anc[method].values[0]

                muts_at_step = df_supmuts[(df_supmuts.Anc_id==anc_id) & (df_supmuts.Depth==step)]

                for i,mut_at_step in muts_at_step.iterrows():
                    mut_feat = mut_at_step[method]

                    dist = distance.euclidean(anc_feat, mut_feat)
                    step_dist_dict[step].append(dist)

        method_to_dists[method] = step_dist_dict
    
    return method_to_dists


from sklearn.metrics import roc_curve
from sklearn.metrics import auc

def get_method_to_aurocs_dict(method_to_dists):
    meth_to_aucs = {}
    for method,dists in method_to_dists.items():
        alg = f'{method}' 
        roc_aucs = []
        depths = list(range(1,5+1))
        for idx2 in depths[1:]:
            idx1 = idx2-1
            y_score1 = dists[idx1]
            y_score2 = dists[idx2]
            y_score = np.hstack([y_score1, y_score2])
            y_true1 = np.full(len(y_score1), idx1)
            y_true2 = np.full(len(y_score2), idx2)
            y_true = np.hstack([y_true1, y_true2])
            fpr, tpr, _ = roc_curve(y_true, y_score, pos_label=idx1)
            roc_auc = auc(tpr, fpr)
            roc_aucs.append(roc_auc)
        meth_to_aucs[alg] = roc_aucs   
    return meth_to_aucs



from scipy import stats
import pandas as pd
def get_mean_corr_df(method_to_dists, method_to_name):
    ''' wow this needs to be debugged urgently. this function depends on method_to_dists to 
        being in the same order (ordered by method) as method_to_name. so watch yourself !!!
    '''

    geds = [i for i in range(1,6)]

    cols = ['Method',"Spearman ρ", "ρ std", "Kendall τ", "τ std"]
    rows = []
    for k,v in method_to_dists.items():
        
        dff = pd.DataFrame(v)
        rhos = dff.apply(lambda x: stats.spearmanr(geds, x)[0] ,axis=1 )
        taus = dff.apply(lambda x: stats.kendalltau(geds, x)[0] ,axis=1 )
        
        rho = np.mean(rhos)
        tau = np.mean(taus)
        rho_sd = np.std(rhos)
        tau_sd = np.std(taus)
        rows.append([method_to_name[k],rho,rho_sd,tau,tau_sd])
        
    df_out = pd.DataFrame(rows,columns=cols,index=method_to_name.values())  
    df_out.index.name='Method'
    df_out.drop('Method',inplace=True,axis=1)
    df_out = df_out.round(4)
    return df_out


#< ---- 20 char ---->< ---- 20 char ---->< ---- 20 char ---->< --- 18 char --->

# # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # 
# # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # #
# kNN-Recall benchmark  # # # # # # # # # # # # # # # # # # # # # # # # # # #
# # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # #
# # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # #

from sklearn.neighbors import NearestNeighbors

def get_recall(tot_pred, true):
    pred = [x for x in tot_pred if x in true]
    recall = len(pred) / len(true)
    return recall


def get_nn_dict(df, methods, neigh_radii=[10,20,40,80]):
    rs = ', '.join([str(x) for x in neigh_radii])
    ms = ', '.join(methods)
    print(f"Computing {{{rs}}} nearest neighbors for methods: {ms}")
    
    nns = {}
    for method in methods:
        latents = np.stack(df[method],axis=0)
        nns[method] = {}

        for n in neigh_radii:
            neigh = NearestNeighbors(n_neighbors=n)
            neigh.fit(latents)
            nns[method][n] = []

            anc_ids = df[df.Is_anc==False].Anc_id.unique()

            for anc_id in tqdm(anc_ids,total=len(anc_ids), desc=f'{method},{n}'):
                anc = np.expand_dims(latents[anc_id,:], axis=0)
                neighbors = neigh.kneighbors(anc, return_distance=False)[0]
                recall = get_recall(neighbors, true = [anc_id+1+j for j in range(neigh_radii[0])])
                nns[method][n].append(recall) 
                
    return nns