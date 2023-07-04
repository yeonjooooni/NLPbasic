import copy
import torch.nn as nn
from layers import *
class Encoder(nn.Module):
    def __init__(self, encoder_layer, n_layer):
        super(Encoder, self).__init__()
        self.n_layer = n_layer
        self.layers = nn.ModuleList([copy.deepcopy(encoder_layer) for _ in range(self.n_layer)])
        
    def forward(self, src, src_mask):
        out = src
        for layer in self.layer:
            out = layer(out, src_mask)
        return out

class EncoderLayer(nn.Module):
    def __init__(self, self_attention, position_ff):
        # 2 layers with multi-head attention, FFN
        # basic units for whole encoder
        # 부모 클래스 상속받은 상태의 인스턴스 생성을 의미
        super(EncoderLayer, self).__init__()
        self.self_attention = self_attention
        self.position_ff = position_ff
        self.residuals = [ResidualConnectionLayer() for _ in range(2)]
        # two residuals, one for multi-head attention layer, one for positionff
    # input attention, mask
    def forward(self, src, src_mask):
        out = src
        out = self.residuals[0](out, lambda out: self.self_attention(Q=out, K=out, V=out, mask=src_mask))
        # 지정해줘야 할 다른 parameter들 때문에 lambda 사용
        ut = self.residuals[1](out, self.position_ff)
        return out

