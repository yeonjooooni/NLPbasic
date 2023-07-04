import copy
import torch.nn as nn
from layers import *
class Decoder(nn.Module):
    def __init__(self, decoder_layer, n_layer):
        super(Decoder, self).__init__()
        self.n_layer = n_layer
        self.layers = nn.ModuleList([copy.deepcopy(decoder_layer) for _ in range(self.n_layer)])
        # can set multiple modules at the same time
    
    def forward(self, tgt, encoder_out, tgt_mask, src_tgt_mask):
        out = tgt
        for layer in self.layers:
            out = layer(out, encoder_out, tgt_mask, src_tgt_mask)
        return out

class DecoderLayer(nn.Module):
    # 3 layers, 2 identical as encoder
    # calculate multihead attention over output of encoder stack
    # context, sentence as an input
    def __init__(self, self_attention, cross_attention, position_ff):
        super(DecoderLayer, self).__init__()
        self.self_attention = self_attention
        self.cross_attention = cross_attention
        self.position_ff = position_ff
        self.residuals = [ResidualConnectionLayer() for _ in range(3)]

    def forward(self, tgt, encoder_out, tgt_mask, src_tgt_mask):
        out = tgt
        out = self.residuals[0](out, lambda out: self.self_attention(Q=out, K=out, V=out, mask=tgt_mask))
        out = self.residuals[1](out, lambda out: self.cross_attention(Q=out, K=encoder_out, V=encoder_out, mask=src_tgt_mask))
        out = self.residuals[2](out, self.position_ff)
        return out