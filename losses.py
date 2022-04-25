#< ---- 20 char ---->< ---- 20 char ---->< ---- 20 char ---->< --- 18 char --->

import numpy as np
import torch
import torch.nn as nn
from torch.autograd import Variable
import torch.nn.functional as F

import math
import random

from tqdm.auto import trange, tqdm


def padce_loss(tgt, model_out, pad_mask,out_mask):
#     pad_mask = pad_mask.squeeze()
#     out_mask = out_mask.squeeze()
#     print(tgt.shape, model_out.shape, pad_mask.shape, out_mask.shape)

    pad_mask = torch.ones_like(pad_mask.float()) - pad_mask.float()
    pad_mask = torch.flatten(pad_mask[:,1:]).float()
    weights = torch.ones_like(out_mask[0,0]) - out_mask[0,0]
    criterion = nn.CrossEntropyLoss(reduction = 'none',weight = weights)
    
    unred_loss = criterion(torch.flatten(model_out[:,:-1],
                                         end_dim = -2),
                           torch.flatten(tgt[:,1:]))
    
    return torch.matmul(unred_loss,pad_mask)/torch.sum(pad_mask)

def normal_kl_divergence(mean, log_var):
    mean_sq = (mean*mean).sum(axis = -1).mean()
    tr_var = torch.exp(log_var).sum(axis =-1).mean()
    
    dim = mean.shape[-1]
    
    tr_log_var = log_var.sum(axis =-1).mean()
    
    return 0.5*(mean_sq + tr_var - dim - tr_log_var)


import torch
import torch.nn as nn

class SupConLoss(nn.Module): 
    def __init__(self, temp=0.07, contrast_mode='all', base_temp=0.07):
        super(SupConLoss, self).__init__()
        self.temperature = temp
        self.contrast_mode = contrast_mode
        self.base_temperature = base_temp

    def forward(self, features, labels=None, mask=None):
        """
        Args:
            features: hidden vector of shape [bsz, n_views, ...].
            labels: ground truth of shape [bsz].
            mask: contrastive mask of shape [bsz, bsz], mask_{i,j}=1 if sample j
                has the same class as sample i. Can be asymmetric.
        Returns:
            A loss scalar.
        """
        device = (torch.device('cuda') if features.is_cuda
                  else torch.device('cpu'))
        if len(features.shape) > 3:
            features = features.view(features.shape[0], features.shape[1], -1)

        batch_size = features.shape[0]
#         print(labels)
        labels = labels.contiguous().view(-1, 1)
#         print(labels)
        mask = torch.eq(labels, labels.T).float().to(device)
#         print(mask)

        contrast_count = features.shape[1]
        contrast_feature = torch.cat(torch.unbind(features, dim=1), dim=0)
        
        if self.contrast_mode == 'one':
            anchor_feature = features[:, 0]
            anchor_count = 1
        elif self.contrast_mode == 'all':
            anchor_feature = contrast_feature
            anchor_count = contrast_count
        
        
        # compute logits
        anchor_dot_contrast = torch.div(
            torch.matmul(anchor_feature, contrast_feature.T),
            self.temperature)
        # for numerical stability
        logits_max, _ = torch.max(anchor_dot_contrast, dim=1, keepdim=True)
        logits = anchor_dot_contrast - logits_max.detach()

        # tile mask
        mask = mask.repeat(anchor_count, contrast_count)
        # mask-out self-contrast cases
        logits_mask = torch.scatter(torch.ones_like(mask), 1,
                                    torch.arange(batch_size * anchor_count).\
                                    view(-1, 1).to(device), 0 )
        mask = mask * logits_mask

        # compute log_prob
        exp_logits = torch.exp(logits) * logits_mask
        log_prob = logits - torch.log(exp_logits.sum(1, keepdim=True))

        # compute mean of log-likelihood over positive
        mean_log_prob_pos = (mask * log_prob).sum(1) / mask.sum(1)

        # loss
        loss = - (self.temperature / self.base_temperature) * mean_log_prob_pos
        loss = loss.view(anchor_count, batch_size).mean()

        return loss