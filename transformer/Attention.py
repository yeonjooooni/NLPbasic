import torch
import math
import torch.nn as nn
import torch.nn.functional as F
import copy
class MultiHeadAttentionLayer(nn.Module):
    def __init__(self, d_model, h, qkv_fc, out_fc, dr_rate=0):
        super(MultiHeadAttentionLayer, self).__init__()
        #qkv_fc, out_fc same size (d_embed, d_model)
        # h = # of head
        self.d_model = d_model
        self.h = h
        # input이 같은 token sequence(n_batch, seq_len, d_embed)지만 다른 fc가 곱해져, 다른 embedding을 얻음
        # fc:(d_embed, d_model) 
        self.q_fc = copy.deepcopy(qkv_fc)
        self.k_fc = copy.deepcopy(qkv_fc)
        self.v_fc = copy.deepcopy(qkv_fc)
        self.out_fc = out_fc

    def forward(self, Q, K, V, mask=None):
        n_batch = Q.size(0)
        #mainly described in 3.2.2
        def transform(x, fc):
            # this part calculates Q, K, V using sentence and weight matrix
            # And also transforms the original structure eligible for multi-head structure
            out = fc(x)
            # out = (n_batch, seq_len, d_model)
            # 왜 이거 -1? 다른 부분 이용하여 -1 부분 dimension 계산
            out = out.view(n_batch, -1, self.h, self.d_model//self.h)
            # .view change the dimension
            # this part enables the parallelization
            # out = (n_batch, seq_len, h, d_k)
            out = out.transpose(1,2)
            # out = (n_batch, h, seq_len, d_k)
            return out
        Q = transform(Q, self.q_fc) #(n_batch, h, seq_len, d_k)
        K = transform(K, self.k_fc)
        V = transform(V, self.v_fc)

        out = self.calculate_attention(Q, K, V, mask)
        out = out.transpose(1, 2)
        out = out.contiguous().view(n_batch, -1, self.d_model)
        # contiguous for memory efficiency
        out = self.out_fc(out)
        return out

    def calculate_attention(Q, K, V, mask):
        # q, k, v have same size, (n_batch, seq_len,d_k) --> (n_batch, h, seq_len, d_k), 
        # input is not a sentence, rather batch
        # mask: (n_batch, seq_len, seq_len) --> (n_batch, 1, seq_len, seq_len)
        # d_k: input dimension, each token have d_k size 1d array
        # mainly described in 3.2.1
        d_k = K.shape[-1]
        attention_score = torch.matmul(Q, K.transpose(-2,-1)) 
        # k : (n_batch, h, d_k, seq_len), attention : (n_batch, h, seq_len, seq_len)
        attention_score = attention_score / math.sqrt(d_k) # prevent gradient vanishing
        if mask is None:
            # pad token에 attention을 부여하지 않기 위해서 필요
            attention_score = attention_score.masked_fill(mask==0, -1e9)    #-1000000000
        attention_prob = F.softmax(attention_score, dim=-1) 
        out = torch.matmul(attention_prob, V)
        return out
