import math
import torch
import torch.nn as nn
import torch.nn.functional as F

import torch.experimental.named as named
from collections import OrderedDict

torch.manual_seed(0)
named.load_named_lib()


class MultiHeadAttention(nn.Module):
    def __init__(self, n_heads, dim, dropout=0):
        super(MultiHeadAttention, self).__init__()
        self.n_heads = n_heads
        self.dim = dim

        self.attn_dropout = nn.Dropout(p=dropout)  # --attention-dropout
        self.q_lin = nn.Linear(dim, dim)
        self.k_lin = nn.Linear(dim, dim)
        self.v_lin = nn.Linear(dim, dim)
        # TODO: merge for the initialization step
        nn.init.xavier_normal_(self.q_lin.weight)
        nn.init.xavier_normal_(self.k_lin.weight)
        nn.init.xavier_normal_(self.v_lin.weight)
        # and set biases to 0
        self.out_lin = nn.Linear(dim, dim)

        nn.init.xavier_normal_(self.out_lin.weight)

    def forward(self, query, key=None, value=None, mask=None):
        query.check('N', 'T', 'D')
        if mask.dim() is 2:
            mask.check('N', 'T')  # selfattn
        else:
            mask.check('N', 'T', 'T_k')  # enc attn

        batch_size, query_len, dim = query.size()
        assert dim == self.dim, \
            f'Dimensions do not match: {dim} query vs {self.dim} configured'
        assert mask is not None, 'Mask is None, please specify a mask'
        n_heads = self.n_heads
        dim_per_head = dim // n_heads
        scale = math.sqrt(dim_per_head)

        def prepare_head(tensor):
            tensor.check('N', 'T', 'D')
            bsz, seq_len, _ = tensor.size()
            tensor = tensor.split_dim('D', OrderedDict({'H': n_heads, 'D_h': dim_per_head})) \
                           .shift(['T', 'H'], ['H', 'T']) \
                           .contiguous()
            return tensor.check('N', 'H', 'T', 'D_h')

        # q, k, v are the transformed values
        if key is None and value is None:
            # self attention
            key = value = query
        elif value is None:
            # key and value are the same, but query differs
            # self attention
            key.check('N', 'T', 'D')
            value = key
        _, key_len, dim = key.size()

        q = prepare_head(self.q_lin(query))

        # Distinguish between query_len and key_len
        k = prepare_head(self.k_lin(key)).rename(T='T_k')
        v = prepare_head(self.v_lin(value)).rename(T='T_k')

        dot_prod = q.matmul(k.shift(['T_k', 'D_h'], ['D_h', 'T_k']))
        dot_prod.check('N', 'H', 'T', 'T_k')
        attn_mask = (mask == 0).align_as(dot_prod)
        dot_prod.masked_fill_(attn_mask, -float(1e20))

        attn_weights = F.softmax(dot_prod / scale, dim='T_k')
        attn_weights = self.attn_dropout(attn_weights)  # --attention-dropout

        attentioned = attn_weights.matmul(v).check('N', 'H', 'T', 'D_h')
        attentioned = (
            attentioned
            .shift(['H', 'T'], ['T', 'H']).contiguous()
            .join_dims(['H', 'D_h'], 'D')
            .check('N', 'T', 'D')
        )

        out = self.out_lin(attentioned)

        return out.check('N', 'T', 'D')


n, t, d, h = 7, 5, 2 * 3, 3
N, T, D, H = 'N', 'T', 'D', 'H'
query = torch.randn(n, t, d, names=(N, T, D))
mask = torch.ones(n, t, names=(N, T))
attn = MultiHeadAttention(h, d)
out = attn(query, mask=mask)
print(out.shape)
print(out.names)  # 'N', 'T', 'D'
