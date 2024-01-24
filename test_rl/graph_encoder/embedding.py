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

from code2inv.common.ssa_graph_builder import ProgramGraph
from code2inv.common.constants import NUM_EDGE_TYPES
from code2inv.common.cmd_args import cmd_args
from code2inv.common.pytorch_util import weights_init, gnn_spmm, get_torch_version
from code2inv.graph_encoder.s2v_lib import S2VLIB, S2VGraph
from code2inv.common.pytorch_util import glorot_uniform


class ParamEmbed(nn.Module):
    def __init__(self, latent_dim, num_nodes):
        super(ParamEmbed, self).__init__()
        self.latent_dim = latent_dim
        self.num_nodes = num_nodes

        node_embed = torch.Tensor(num_nodes, latent_dim)
        glorot_uniform(node_embed)
        self.node_embed = Parameter(node_embed)

    def forward(self, graph, istraining=True):
        # assert len(graph) == 1
        assert graph.pg.num_nodes() == self.num_nodes
        return self.node_embed


class LSTMEmbed(nn.Module):
    def __init__(self, latent_dim, num_node_feats):
        super(LSTMEmbed, self).__init__()
        self.latent_dim = latent_dim        
        self.num_node_feats = num_node_feats        

        self.w2v = nn.Embedding(num_node_feats, latent_dim)
        self.lstm = nn.LSTMCell(latent_dim, latent_dim)

    def forward(self, graph_list, istraining=True):    
        embed_list = []     
        if type(graph_list) is not list:
            graph_list = [graph_list]
        for g in graph_list:
            indices = torch.tensor(g.token_idx, dtype=torch.long)            
            embeddings = self.w2v(indices)
            hx = embeddings.new_zeros(1, self.latent_dim, requires_grad=False)
            cx = embeddings.new_zeros(1, self.latent_dim, requires_grad=False)

            list_embeddings = []
            for i in range(len(g.token_idx)):                
                hx, cx = self.lstm(embeddings[i].view(1, -1), (hx, cx))        
                list_embeddings.append(hx)

            node_embeddings = []
            for node in g.pg.raw_variable_nodes:            
                node_embeddings.append( list_embeddings[ g.pg.var_pos[node] ])
            for node in g.pg.const_nodes:
                node_embeddings.append( list_embeddings[ g.pg.const_pos[node] ])
            node_embeddings.append( hx )

            node_embeddings = torch.cat(node_embeddings, dim=0)
            embed_list.append(node_embeddings)
        
        embed_list = torch.cat(embed_list, dim=0)
        return embed_list
        
class EmbedMeanField(nn.Module):
    def __init__(self, latent_dim, num_node_feats, max_lv = 3):
        super(EmbedMeanField, self).__init__()
        self.latent_dim = latent_dim        
        self.num_node_feats = num_node_feats        

        self.max_lv = max_lv

        self.w_n2l = nn.Linear(num_node_feats, latent_dim)

        self.conv_param_list = []
        self.merge_param_list = []
        for i in range(self.max_lv):
            self.conv_param_list.append(nn.Linear(latent_dim, NUM_EDGE_TYPES * latent_dim))
            self.merge_param_list.append( nn.Linear(NUM_EDGE_TYPES * latent_dim, latent_dim) )

        self.conv_param_list = nn.ModuleList(self.conv_param_list)
        self.merge_param_list = nn.ModuleList(self.merge_param_list)

        self.state_gru = nn.GRUCell(latent_dim, latent_dim)

        weights_init(self)

    def forward(self, graph_list, istraining=True): 
        node_feat = S2VLIB.ConcatNodeFeats(graph_list)        
        sp_list = S2VLIB.PrepareMeanField(graph_list)
        version = get_torch_version()
        if not istraining:
            if version >= 0.4:
                torch.set_grad_enabled(False)
            else:
                node_feat = Variable(node_feat.data, volatile=True)
        
        h = self.mean_field(node_feat, sp_list)

        if not istraining: # recover
            if version >= 0.4:
                torch.set_grad_enabled(True)

        return h

    def mean_field(self, node_feat, sp_list):
        input_node_linear = self.w_n2l(node_feat)
        input_message = input_node_linear
        input_potential = F.tanh(input_message)

        lv = 0
        cur_message_layer = input_potential
        while lv < self.max_lv:
            conv_feat = self.conv_param_list[lv](cur_message_layer)
            chunks = torch.split(conv_feat, self.latent_dim, dim=1)
            
            msg_list = []
            for i in range(NUM_EDGE_TYPES):
                t = gnn_spmm(sp_list[i], chunks[i])
                msg_list.append( t )
            
            msg = F.tanh( torch.cat(msg_list, dim=1) )
            cur_input = self.merge_param_list[lv](msg)# + input_potential

            cur_message_layer = cur_input + cur_message_layer
            # cur_message_layer = self.state_gru(cur_input, cur_message_layer)
            cur_message_layer = F.tanh(cur_message_layer)
            lv += 1

        return cur_message_layer

if __name__ == '__main__':
    random.seed(cmd_args.seed)
    np.random.seed(cmd_args.seed)
    torch.manual_seed(cmd_args.seed)    

    s2v_graphs = []
    pg_graphs = []
    with open(cmd_args.data_root + '/list.txt', 'r') as f:
        for row in f:            
            with open(cmd_args.data_root + '/' + row.strip() + '.json', 'r') as gf:
                graph_json = json.load(gf)
                pg_graphs.append(ProgramGraph(graph_json))
    for g in pg_graphs:
        s2v_graphs.append( S2VGraph(g) )
    
    print(len(s2v_graphs))
    # mf = EmbedMeanField(128, len(node_type_dict))
    if cmd_args.ctx == 'gpu':
        mf = mf.cuda()

    embedding = mf(s2v_graphs[0:2])
    embed2 = mf(s2v_graphs[0:1])
    embed3 = mf(s2v_graphs[1:2])
    ee = torch.cat([embed2, embed3], dim=0)
    diff = torch.sum(torch.abs(embedding - ee))
    print(diff)

    r = range(len(s2v_graphs))
    for i in tqdm(range(1000)):
        random.shuffle(r)
        glist = []
        for j in range(20):
            glist.append(s2v_graphs[r[j]])

        embedding = mf(glist)
