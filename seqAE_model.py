import numpy as np
import math
import torch
import torch.nn as nn
import torch.nn.functional as F

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
    
def get_downstream_mask(seq):
    _, ls = seq.size()
    dvc = seq.device
    mask = torch.triu(torch.ones((ls, ls), device=dvc), diagonal=1).bool()
    return mask

def count_params(model):
    return sum([p.nelement() for p in model.parameters()])

from torch.nn import TransformerDecoderLayer as DecLayer
from torch.nn import TransformerDecoder as Decoder
from torch.nn import TransformerEncoderLayer as EncLayer
from torch.nn import TransformerEncoder as Encoder

class SeqAutoencoder(nn.Module):
    
    def __init__(self, n_tokens, max_len = 122,
                 dim_emb=512, heads=8, dim_hidden=32,
                 L_enc=6, L_dec=6, dim_ff=2048, 
                 drpt=0.1, actv='relu', eps=0.6, b_first=True):
        '''
        dim_emb: dimensions of input embedding
        heads: number of attention heads
        dim_hidden: dimensions of latent space
        L_enc: number of layers in encoder
        L_dec: number of layers in decoder
        dim_ff: dimensions of feed forward network
        '''
        super().__init__()
        
        self.n_tokens = n_tokens
        self.max_len = max_len
        self.dim_hidden = dim_hidden
        self.dim_emb = dim_emb
        
        # Initial embedding and subsequent positional encoder
        self.embedder = nn.Embedding(n_tokens, dim_emb)
        self.pos_enc = PositionalEncoding(dim_emb, dropout=drpt)

        # Encoder
        enc_layer = EncLayer(d_model=dim_emb, nhead=heads,
                             dim_feedforward=dim_ff, dropout=drpt,
                             activation=actv, layer_norm_eps=eps,
                             batch_first=b_first)
        self.enc = Encoder(enc_layer, num_layers=L_enc)
        
        # Reparameterize
        self.linear = nn.Linear(dim_emb,dim_hidden) # mu
#         self.sigma = nn.Linear(dim_emb,dim_hidden)
        self.sample = nn.Linear(dim_hidden,dim_emb*max_len)
        
        # Decoder
        dec_layer = DecLayer(d_model=dim_emb, nhead=heads,
                             dim_feedforward=dim_ff, dropout=drpt,
                             activation=actv, layer_norm_eps=eps,
                             batch_first=b_first)
        self.dec = Decoder(dec_layer, num_layers=L_dec)
        self.decode_out = nn.Linear(dim_emb, n_tokens)
        
    def forward(self, seq, pad_mask=None, avg_mask=None, out_mask=None,
                normed=False, bottleneck=True):
        
        # What to do with dummy seqs?
        if seq==None:
            return torch.tensor([0])
                
        if len(seq.shape)==1:
            seq = seq.unsqueeze(0)
        
        # Masks
        mask = get_downstream_mask(seq) # casual mask ... 
        dec_mask = mask
        mem_mask = mask
        if avg_mask is None:
            avg_mask = torch.ones_like(seq)
        if out_mask is None:
            out_mask = torch.zeros(self.n_tokens).to(seq.device)
        
        # Encode
        emb_seq = self.pos_enc( self.embedder(seq) )
        enc_out = self.enc(src=emb_seq, mask=mask, src_key_padding_mask=pad_mask)
        # out -> (bs, 120, 512)

        # Situate the latent vector
        if bottleneck: 
        # this is the actual bottlenecking !!!!
            enc_sum = (avg_mask.unsqueeze(2)*enc_out).sum(axis = 1)
            enc_avg = enc_sum/(avg_mask.sum(axis = 1).unsqueeze(1))
            # out -> (bs, 512)
            latent_vec = self.linear(enc_avg)
            
            if normed:
                latent_vec = F.normalize(latent_vec, dim=-1)           
            
            # out -> (bs, 32)            
            latent_out = self.sample(latent_vec)
            latent_out = latent_out.reshape(-1, self.max_len, self.dim_emb)
        else:
            latent_out = enc_out



        # Decode
        dec_out = self.dec(tgt=emb_seq, memory=latent_out,
                           tgt_mask=dec_mask, memory_mask=mem_mask,
                           tgt_key_padding_mask=pad_mask,
                           memory_key_padding_mask=pad_mask)
        dec_out = self.decode_out(dec_out).masked_fill(out_mask==1,-1e9)

        return latent_vec, dec_out
            
       
    
    
    
    
    
    
    
    
    