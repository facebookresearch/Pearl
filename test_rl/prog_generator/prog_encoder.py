from __future__ import print_function

import os
import sys
import numpy as np
import torch
import json
import random
from tqdm import tqdm
from torch.autograd import Variable
from torch.nn.parameter import Parameter
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from itertools import chain


from code2inv.common.constants import NUM_EDGE_TYPES, RULESET, MAX_CHILD
from code2inv.common.cmd_args import cmd_args
from code2inv.common.pytorch_util import weights_init
from code2inv.common.ssa_graph_builder import checkallnone


class LogicEncoder(nn.Module):
    def __init__(self, latent_dim):
        super(LogicEncoder, self).__init__()
        self.latent_dim = latent_dim


        if RULESET:
            self.char_embedding = Parameter(torch.Tensor(sum([len(RULESET[rule]) for rule in RULESET]), latent_dim))

        def new_gate():
            lh = nn.Linear(self.latent_dim, self.latent_dim)
            rh = nn.Linear(self.latent_dim, self.latent_dim)
            return lh, rh
        

        self.ilh, self.irh = new_gate()        
        self.lflh, self.lfrh = new_gate()
        self.rflh, self.rfrh = new_gate()
        self.ulh, self.urh = new_gate()

        self.i_gates = [ nn.Linear(self.latent_dim, self.latent_dim) for _ in range(MAX_CHILD) ]
        self.f_gates = [ [ nn.Linear(self.latent_dim, self.latent_dim) for _ in range(MAX_CHILD) ] for _ in range(MAX_CHILD) ]
        self.u_gates = [ nn.Linear(self.latent_dim, self.latent_dim) for _ in range(MAX_CHILD) ]

        self.ix = nn.Linear(self.latent_dim, self.latent_dim)
        self.fx = nn.Linear(self.latent_dim, self.latent_dim)
        self.ux = nn.Linear(self.latent_dim, self.latent_dim)

        self.cx, self.ox = new_gate()
        self.oend = nn.Linear(self.latent_dim * 3, self.latent_dim)

        weights_init(self)

    def forward(self, node_embedding, init_embedding, root):
        if root is None or checkallnone(root):
            return init_embedding
            
        if root.state is None:
            self.subexpr_embed(node_embedding, root)
        s = torch.cat((init_embedding, root.state[0], root.state[1]), dim=1)
        o = F.tanh(self.oend(s))
        return o
        

    def _get_token_embed(self, node_embedding, node):
        idx = 0
        for rule in RULESET:
            for opt in RULESET[rule]:
                if len(opt) == 1 and node.name == opt[0]:
                    x_embed = torch.index_select(self.char_embedding, 0, Variable(torch.LongTensor([idx])))
                    return x_embed
                elif node.name in opt and node.name not in RULESET:
                    x_embed = torch.index_select(self.char_embedding, 0, Variable(torch.LongTensor([idx])))
                    return x_embed
                idx += 1
        if node.name != "":
            # Not a rule nor a terminal
            # must be a const
            assert node.pg_node is not None
            return torch.index_select(node_embedding, 0, Variable(torch.LongTensor([node.pg_node.index])))
        elif node.name == "" and node.rule is not None and node.rule != "":
            idx = 0
            for rule in RULESET:
                if node.rule == rule:
                    x_embed = torch.index_select(self.char_embedding, 0, Variable(torch.LongTensor([idx])))
                    return x_embed
                idx += 1

    def subexpr_embed(self, node_embedding, node):
        if node.state is not None:
            return node.state
            
        x_embed = self._get_token_embed(node_embedding, node)
        
        if len(node.children) == 0: # leaf node         
            
            c = self.cx(x_embed)
            o = F.sigmoid(self.ox(x_embed))
            h = o * F.tanh(c)
            node.state = (c, h)
        else:
            children_states = []
            for child in node.children:
                self.subexpr_embed(node_embedding, child)
                children_states.append(child.state)

            i_gate_vals = []
            for idx in range(len(node.children)):
                i_gate_vals.append(self.i_gates[idx](children_states[idx][1]))

            i = F.sigmoid(self.ix(x_embed) + sum(i_gate_vals))
            fx = self.fx(x_embed)

            f_vals = []

            for idx in range(len(node.children)):
                f_gate_vals = []
                for idx2 in range(len(node.children)):
                    f_gate_vals.append(self.f_gates[idx][idx2](children_states[idx2][1]))
                f_vals.append(fx + sum(f_gate_vals))

            u_gate_vals = []

            for idx in range(len(node.children)):
                u_gate_vals.append(self.u_gates[idx](children_states[idx][1]))

            update = F.tanh(self.ux(x_embed) + sum(u_gate_vals))
            c = i * update + sum([ f_vals[idx] * children_states[idx][0] for idx in range(len(node.children)) ])
            h = F.tanh(c)

            node.state = (c, h)
