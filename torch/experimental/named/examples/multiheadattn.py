import math
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.experimental.named as named
from collections import OrderedDict

torch.manual_seed(0)
named.load_named_lib()

# New ops:
# split_dims(tensor, namedshape): splits a dimension into multiple new ones.
# join_dims(tensor, dims): inverse of split_dims
# shift(tensor, from_dims, to_dims):
#   permutes from_dims to to_dims. Essentially a safer version of transpose
# align_as(tensor, other):
#   Aligns the name of 'tensor' to match that of 'other'
# dot(tensor, other, tensor_dims, other_dims):
#   does a matmul on tensor_dims and other_dims, ignoring all other dimensions.
#   All other dims of the tensor must match.


# Preconditions on names:
# query, key, value, need to have ['T', 'D_k'] as a subsequence of names.
# mask needs to have ['T', 'T_other'] as a subsequence of names.
#
# All other dimensions are ignored. This is nice because we can pass in
# q, k, v with batch dimension (N) and/or head dimension (H).
def attention(query, key, value, mask=None, dropout=None):
    # Necessary to make things work out
    key.rename_(T='T2')
    value.rename_(T='T2')
    if mask is not None:
        mask.rename_(T_other='T2')

    d_k = query.size('D_k')
    scores = torch.dot(query, key.shift(['T2', 'D_k'], ['D_k', 'T2']),
                       ['T', 'D_k'], ['D_k', 'T2']) / math.sqrt(d_k)
    if mask is not None:
        scores = scores.masked_fill((mask == 0).align_as(scores), -1e9)
    p_attn = F.softmax(scores, dim='T2')
    if dropout is not None:
        p_attn = dropout(p_attn)
    return torch.dot(p_attn, value, ['T', 'T2'], ['T2', 'D_k']), p_attn


class MultiHeadedAttention(nn.Module):
    def __init__(self, h, d_model, dropout=0.1):
        "Take in model size and number of heads."
        super(MultiHeadedAttention, self).__init__()
        assert d_model % h == 0
        # We assume d_v always equals d_k
        self.d_k = d_model // h
        self.h = h
        self.linears = nn.ModuleList([nn.Linear(d_model, d_model) for i in range(4)])
        self.attn = None
        self.dropout = nn.Dropout(p=dropout)

    def forward(self, query, key, value, mask=None):
        # 1) Do all the linear projections in batch from d_model => h x d_k
        query, key, value = \
            [l(x).split_dim('D_model', OrderedDict({'H': H, 'D_k': D}))
             for l, x in zip(self.linears, (query, key, value))]

        # 2) Apply attention on all the projected vectors in batch.
        x, self.attn = attention(query, key, value, mask=mask,
                                 dropout=self.dropout)

        # 3) "Concat" by joining dimensions and apply a final linear.
        x = x.contiguous().join_dims(['H', 'D_k'], 'D_model')
        return self.linears[-1](x)


N, T, D, H = 2, 3, 5, 7
size = (N, T, D * H)
names = ('N', 'T', 'D_model')


def get_inputs(names=False):
    kwargs = {}
    kwargs_mask = {}
    if names:
        kwargs['names'] = ('N', 'T', 'D_model')
        kwargs_mask['names'] = ('N', 'T', 'T_other')

    Q = torch.randn(size, **kwargs)
    K = torch.randn(size, **kwargs)
    V = torch.randn(size, **kwargs)
    mask = torch.ones(N, T, T, **kwargs_mask)
    return Q, K, V, mask

Q, K, V, mask = get_inputs(names=True)
model = MultiHeadedAttention(H, D * H, dropout=0)
out = model(Q, K, V, mask)
print(out)
