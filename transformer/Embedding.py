import torch
import torch.nn as nn
import math
class TransformerEmbedding(nn.Module):
    # described in 3.4
    def __init__(self, token_embed, pos_embed):
        super(TransformerEmbedding, self).__init__()
        self.embedding = nn.Sequential(token_embed, pos_embed)
        #nn.Sequential: make input pass through multiple layers sequentially

    def forward(self, x):
        out = self.embedding(x)
        return out

class TokenEmbedding(nn.Module):
    def __init__(self, d_embed, vocab_size):
        super(TokenEmbedding, self).__init__()
        self.embedding = nn.Embedding(vocab_size, d_embed)
        # nn.Embedding: 
        # input:size of dicionary(sentence), embedding_dim(size of embedding vector(output))
        # simple lookup table that store embedding of fixed dictionary
        self.d_embed = d_embed
    def forward(self, x):
        out = self.embedding(x)*math.sqrt(self.d_embed)
        return out
    
class PositionalEncoding(nn.Module):
    # described in 3.5
    # used to normalize the sentence indices
    def __init__(self, d_embed, max_len = 256, device = torch.device("cpu")):
        super(PositionalEncoding, self).__init__()
        encoding = torch.zeros(max_len, d_embed)
        encoding.requires_grad = False
        position = torch.arange()
        #1차원 크기 텐서 반환
        div_term = torch.exp(torch.arange(0, d_embed, 2)*-(math.log(10000.0)/d_embed))
        encoding[:, 0::2] = torch.sin(position*div_term)
        encoding[:, 1::2] = torch.cos(position*div_term)
        self.encoding = encoding.unsqueeze(0).to(device)

    def forward(self, x):
        _, seq_len, _ = x.size()
        pos_embed = self.encoding[:, :seq_len, :]
        out = x + pos_embed
        return out