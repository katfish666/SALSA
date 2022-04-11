#< ---- 20 char ---->< ---- 20 char ---->< ---- 20 char ---->< --- 18 char --->

import numpy as np
import math

import torch
import torch.nn as nn
import torch.nn.functional as F

class SeqAE(nn.Module):
    
    def __init__(self,n_tokens,max_len = 122,
                 d_model=512,nhead=8,d_latent=32,num_encoder_layers=6,
                 num_decoder_layers=6,dim_feedforward=2048,dropout=0.1,
                 activation='relu',layer_norm_eps=0.6,batch_first=True):
        
        super().__init__()
        
        self.n_tokens = n_tokens
        self.max_len = max_len
        self.d_latent = d_latent
        self.d_model = d_model
        
        self.pos_enc = PositionalEncoding(d_model, dropout=dropout)
        self.embedder = nn.Embedding(n_tokens, d_model)
        encoder_layer = nn.TransformerEncoderLayer(d_model=d_model,
                                                   nhead=nhead,
                                                   dim_feedforward=\
                                                   dim_feedforward,
                                                   dropout=dropout,
                                                   activation=activation,
                                                   layer_norm_eps=\
                                                   layer_norm_eps,
                                                   batch_first=batch_first)
        
        self.encoder = nn.TransformerEncoder(encoder_layer,
                                             num_layers=num_encoder_layers)
        
        self.latent_mean = nn.Linear(d_model,d_latent)
        self.latent_upsample = nn.Linear(d_latent,d_model*max_len)
        
        decoder_layer = nn.TransformerDecoderLayer(d_model=d_model,
                                                   nhead=nhead,
                                                   dim_feedforward=\
                                                   dim_feedforward,
                                                   dropout=dropout,
                                                   activation=activation,
                                                   layer_norm_eps=\
                                                   layer_norm_eps,
                                                   batch_first=batch_first)
        
        self.decoder = nn.TransformerDecoder(decoder_layer,
                                             num_layers=num_decoder_layers)
        
        self.token_decode = nn.Linear(d_model,n_tokens)
    
    def forward(self, seq, pad_mask=None, avg_mask=None, out_mask=None,
                variational = False, bottleneck = True):
        
        causal_mask = get_subsequent_mask(seq)
        
        emb_seq = self.pos_enc(self.embedder(seq))
        
        enc_out = self.encoder(emb_seq,
                               mask=causal_mask,
                               src_key_padding_mask=pad_mask)
        
        if avg_mask is None:
            avg_mask = torch.ones_like(seq)
            
        enc_sum = (avg_mask.unsqueeze(2)*enc_out).sum(axis = 1)
        enc_avg = enc_sum/(avg_mask.sum(axis = 1).unsqueeze(1))
        
        ltnt_mean = self.latent_mean(enc_avg)
        
        if bottleneck:
            ltnt_draw = ltnt_mean    
            ltnt_code = self.latent_upsample(ltnt_draw).reshape(-1,
                                                                self.max_len,
                                                                self.d_model)
        else:
            ltnt_mean = enc_out
            ltnt_code = enc_out
            
        dec_mask = causal_mask
        mem_mask = causal_mask

        dec_out = self.decoder(emb_seq,ltnt_code,
                               tgt_mask = dec_mask,
                               memory_mask = mem_mask,
                               tgt_key_padding_mask = pad_mask,
                               memory_key_padding_mask = pad_mask)
        
        if out_mask is None:
            out_mask = torch.zeros(self.n_tokens).to(seq.device)
            
        dec_out = self.token_decode(dec_out).masked_fill(out_mask==1,-1e9)
        
        return ltnt_mean, dec_out
    
class PositionalEncoding(nn.Module):

    def __init__(self, d_model, dropout=0.1, max_len=5000):
        
        super().__init__()
        self.dropout = nn.Dropout(p=dropout)

        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() \
                             * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0)
        self.register_buffer('pe', pe)

    def forward(self, x):
        x = x + self.pe[:,:x.size(1)]
        return self.dropout(x)

def get_subsequent_mask(seq):
    sz_b, len_s = seq.size()
    subsequent_mask = torch.triu(torch.ones((len_s, len_s),device=seq.device),
                                 diagonal=1).bool()
    return subsequent_mask
                        
def count_params(model):
    return sum([param.nelement() for param in model.parameters()])