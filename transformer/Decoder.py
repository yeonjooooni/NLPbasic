import copy
import torch.nn as nn
class Decoder(nn.Module):
    # 3 layers, 2 identical as encoder
    # calculate multihead attention over output of encoder stack
    