import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
class Transformer(nn.Module):
    def __init__(self, encoder, decoder):
        super(Transformer, self).__init__()
        self.encoder = encoder
        self.decoder = decoder

    def encode(self, src, src_mask):
        out = self.encoder(src, src_mask)
        return out
    
    def decode(self, z, c):
        out = self.decoder(z, c)
        return out
    
    def forward(self, src, tgt, src_mask):
        c = self.encode(src, src_mask)
        y = self.decode(tgt,c)
        return y
    
    def make_pad_mask(self, Q, K, pad_idx = 1):
        query_seq_len, key_seq_len = Q.size(1), K.size(1)
        key_mask = K.ne(pad_idx).unsqueeze(1).unsqueeze(2) 
        #(n_batch, 1, 1, key_seq_len)
        # elementwise compare 후 bool type matrix 반환
        key_mask = key_mask.repeat(1, 1, query_seq_len, 1) 
        #(n_batch, 1, query_seq_len, key_seq_len)
        # data 복제한 tensor 생성

        query_mask = Q.ne(pad_idx).unsqueeze(1).unsqueeze(3)
        query_mask = query_mask.repeat(1, 1, 1, key_seq_len)
        mask = key_mask & query_mask    # cross attention 일 경우 하나라도 mask 되어 있으면 mask될 부분에 포함
        mask.requires_grad = False
        return mask
    
    def make_src_mask(self, src):
        pad_mask = self.make_pad_mask(src, src)
        return pad_mask