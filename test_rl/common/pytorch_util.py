from __future__ import print_function

import os
import sys
import numpy as np
import torch
import random
from torch.autograd import Variable
from torch.nn.parameter import Parameter
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from tqdm import tqdm

def get_torch_version():
    return float('.'.join(torch.__version__.split('.')[0:2]))

def glorot_uniform(t):
    if len(t.size()) == 2:
        fan_in, fan_out = t.size()
    elif len(t.size()) == 3:
        # out_ch, in_ch, kernel for Conv 1
        fan_in = t.size()[1] * t.size()[2]
        fan_out = t.size()[0] * t.size()[2]
    else:
        fan_in = np.prod(t.size())
        fan_out = np.prod(t.size())

    limit = np.sqrt(6.0 / (fan_in + fan_out))
    t.uniform_(-limit, limit)

def orthogonal_gru(t):
    assert len(t.size()) == 2
    assert t.size()[0] == 3 * t.size()[1]
    hidden_dim = t.size()[1]

    x0 = torch.Tensor(hidden_dim, hidden_dim)
    x1 = torch.Tensor(hidden_dim, hidden_dim)
    x2 = torch.Tensor(hidden_dim, hidden_dim)

    version = get_torch_version()
    if version >= 0.4:
        nn.init.orthogonal_(x0)
        nn.init.orthogonal_(x1)
        nn.init.orthogonal_(x2)
    else:
        nn.init.orthogonal(x0)
        nn.init.orthogonal(x1)
        nn.init.orthogonal(x2)

    t[0:hidden_dim, :] = x0
    t[hidden_dim:2*hidden_dim, :] = x1
    t[2*hidden_dim:3*hidden_dim, :] = x2

def _param_init(m):
    if isinstance(m, Parameter):
        glorot_uniform(m.data)
        print('a Parameter inited')
    elif isinstance(m, nn.Linear):
        m.bias.data.zero_()
        glorot_uniform(m.weight.data)
        print('a Linear inited')
    elif isinstance(m, nn.GRU):
        for k in range(m.num_layers):
            getattr(m,'bias_ih_l%d'%k).data.zero_()
            getattr(m,'bias_hh_l%d'%k).data.zero_()
            glorot_uniform(getattr(m,'weight_ih_l%d'%k).data)
            orthogonal_gru(getattr(m,'weight_hh_l%d'%k).data)
        print('a GRU inited')
    elif isinstance(m, nn.GRUCell):
        getattr(m,'bias_ih').data.zero_()
        getattr(m,'bias_hh').data.zero_()
        glorot_uniform(getattr(m,'weight_ih').data)
        orthogonal_gru(getattr(m,'weight_hh').data)        
        print('a GRUCell inited')

def weights_init(m):
    for p in m.modules():
        if isinstance(p, nn.ParameterList):
            for pp in p:
                _param_init(pp)
        else:
            _param_init(p)

    for name, p in m.named_parameters():
        if not '.' in name: # top-level parameters
            _param_init(p)

class MySpMM(torch.autograd.Function):

    @staticmethod
    def forward(ctx, sp_mat, dense_mat):
        ctx.save_for_backward(sp_mat, dense_mat)

        return torch.mm(sp_mat, dense_mat)

    @staticmethod
    def backward(ctx, grad_output):        
        sp_mat, dense_mat = ctx.saved_variables
        grad_matrix1 = grad_matrix2 = None

        assert not ctx.needs_input_grad[0]
        if ctx.needs_input_grad[1]:
            grad_matrix2 = Variable(torch.mm(sp_mat.data.t(), grad_output.data))
        
        return grad_matrix1, grad_matrix2

def gnn_spmm(sp_mat, dense_mat):
    return MySpMM.apply(sp_mat, dense_mat)

def to_num(x):
    version = get_torch_version()
    if version >= 0.4:
        return x.item()
    else:
        return torch.squeeze(x).data[0]
