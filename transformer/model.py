import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
class Transformer(nn.Module):
    def __init__(self, src_embed, tgt_embed, encoder, decoder, generator):
        super(Transformer, self).__init__()
        self.src_embed = src_embed
        self.tgt_embed = tgt_embed
        self.encoder = encoder
        self.decoder = decoder
        #decoder로 만들어진 embedding을 기반으로 자연스러운 문장 창조
        self.generator = generator

    def encode(self, src, src_mask):
        out = self.encoder(self.src_embed(src), src_mask)
        return out
    
    def decode(self, tgt, encoder_out, tgt_mask, src_tgt_mask):
        out = self.decoder(self.src_embed(tgt), encoder_out, tgt_mask, src_tgt_mask)
        return out
    
    def forward(self, src, tgt, src_mask):
        src_mask = self.make_src_mask(src)
        tgt_mask = self.make_tgt_mask(tgt)
        src_tgt_mask = self.make_src_tgt_mask(src, tgt)
        encoder_out = self.encode(src, src_mask)
        decoder_out = self.decode(tgt, encoder_out, tgt_mask, src_tgt_mask)
        out = self.generator(decoder_out)
        out = F.log_softmax(out, dim=-1)
        return out, decoder_out
    
    def make_pad_mask(self, Q, K, pad_idx = 1):
        query_seq_len, key_seq_len = Q.size(1), K.size(1)
        key_mask = K.ne(pad_idx).unsqueeze(1).unsqueeze(2) 
        # unsqueeze() 1인 차원 생성, ne(1)?
        #(n_batch, 1, 1, key_seq_len)
        # elementwise compare 후 bool type matrix 반환
        key_mask = key_mask.repeat(1, 1, query_seq_len, 1) 
        #(n_batch, 1, query_seq_len, key_seq_len)
        # data 복제한 tensor 생성

        query_mask = Q.ne(pad_idx).unsqueeze(1).unsqueeze(3)
        query_mask = query_mask.repeat(1, 1, 1, key_seq_len)
        mask = key_mask & query_mask    # cross attention 일 경우 하나라도 mask 되어 있으면 mask될 부분에 포함
        mask.requires_grad = False  # gradient descendent에 사용되지 않음으로
        return mask
    
    def make_src_mask(self, src):
        pad_mask = self.make_pad_mask(src, src)
        return pad_mask    
    
    def make_subsequent_mask(query, key):
        # enables decoder make attention for ith token, using only 1~i-1th token.
        # used for first (self) attention layer in decoder
        query_seq_len, key_seq_len = query.size(1), key.size(1)
        tril = np.tril(np.ones((query_seq_len, key_seq_len)), k=0).astype('uint8')
        # can the length of query and key be different? no, not in this implementation
        mask = torch.tensor(tril, dtype=torch.bool, requires_grad = False, device=query.device)
        return mask
    
    def make_tgt_mask(self, tgt):
        #tgt, input in decoder
        pad_mask = self.make_pad_mask(tgt, tgt)
        seq_mask = self.make_subsequent_mask(tgt, tgt)
        mask = pad_mask & seq_mask
        return mask
    
    def make_src_tgt_mask(self, src, tgt):
        pad_mask = self.make_pad_mask(tgt, src)
        return pad_mask
