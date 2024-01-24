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

from code2inv.common.ssa_graph_builder import ProgramGraph, GraphNode
from code2inv.common.constants import *
from code2inv.common.cmd_args import cmd_args, tic, toc
from code2inv.common.checker import stat_counter, boogie_result

from code2inv.graph_encoder.embedding import EmbedMeanField, LSTMEmbed, ParamEmbed
from code2inv.prog_generator.rl_helper import rollout, actor_critic_loss
from code2inv.prog_generator.tree_decoder import GeneralDecoder

from code2inv.graph_encoder.s2v_lib import S2VLIB, S2VGraph


class GraphSample(S2VGraph):
    def __init__(self, pg, vc_list, node_type_dict):
        super(GraphSample, self).__init__(pg, node_type_dict)
        self.sample_index = 0
        self.db = None
        self.vc_list = vc_list


if __name__ == '__main__':
    random.seed(cmd_args.seed)
    np.random.seed(cmd_args.seed)
    torch.manual_seed(cmd_args.seed)
    tic()
    params = []

    graph = None
    node_type_dict = {}
    vc_list = []

    with open(cmd_args.input_graph, 'r') as graph_file:
        graph = ProgramGraph(json.load(graph_file))
        for node in graph.node_list:
                if not node.node_type in node_type_dict:
                    v = len(node_type_dict)
                    node_type_dict[node.node_type] = v

    
    if graph is not None:
        if cmd_args.encoder_model == 'GNN':
            encoder = EmbedMeanField(cmd_args.embedding_size, len(node_type_dict), max_lv=cmd_args.s2v_level)
        elif cmd_args.encoder_model == 'LSTM':
            encoder = LSTMEmbed(cmd_args.embedding_size, len(node_type_dict))
        elif cmd_args.encoder_model == 'Param':
            g_list = GraphSample(graph, vc_list, node_type_dict)
            encoder = ParamEmbed(cmd_args.embedding_size, g_list.pg.num_nodes())
        else:
            raise NotImplementedError

        decoder = GeneralDecoder(cmd_args.embedding_size)
        
        if cmd_args.init_model_dump is not None:
            encoder.load_state_dict(torch.load(cmd_args.init_model_dump + '.encoder'))
            decoder.load_state_dict(torch.load(cmd_args.init_model_dump + '.decoder'))

        
            
        params.append( encoder.parameters() )
        params.append( decoder.parameters() )
        
        optimizer = optim.Adam(chain.from_iterable(params), lr=cmd_args.learning_rate)

        for epoch in range(cmd_args.num_epochs):
            best_reward = -5.0
            best_root = None
            tested_roots = []
                    
            acc_reward = 0.0
            pbar = tqdm(range(100), file=sys.stdout)
            for k in pbar:
                
                g_list = GraphSample(graph, vc_list, node_type_dict)
                node_embedding_batch = encoder(g_list)

                total_loss = 0.0
                embedding_offset = 0
                
                for b in range(cmd_args.rl_batchsize):
                    g = GraphSample(graph, vc_list, node_type_dict)
                    node_embedding = node_embedding_batch
                    nll_list, value_list, reward_list, root, _ = rollout(g, node_embedding, decoder, use_random = True, eps = 0.05)

                    tested_roots.append(root)
                    if reward_list[-1] > best_reward:
                        best_reward = reward_list[-1]
                        best_root = root

                    acc_reward += np.sum(reward_list) / cmd_args.rl_batchsize                          
                    loss = actor_critic_loss(nll_list, value_list, reward_list)      
                    total_loss += loss

                optimizer.zero_grad()
                loss = total_loss / cmd_args.rl_batchsize
                loss.backward()
                optimizer.step()
                pbar.set_description('avg reward: %.4f' % (acc_reward / (k + 1)))
            
            g = GraphSample(graph, vc_list, node_type_dict)
            node_embedding = encoder(g)

            while True:
                _, _, _, root, trivial = rollout(g, node_embedding, decoder, use_random = True, eps = 0.0)
                if trivial == False:
                    break
            
            print('epoch: %d, average reward: %.4f, Random: %s, result_r: %.4f' % (epoch, acc_reward / 100.0, root, boogie_result(g, root)))
            print("best_reward:", best_reward, ", best_root:", best_root)
            stat_counter.report_global()
            if cmd_args.save_dir is not None:
                torch.save(encoder.state_dict(), cmd_args.save_dir + '/epoch-%d.encoder' % epoch)
                torch.save(decoder.state_dict(), cmd_args.save_dir + '/epoch-%d.decoder' % epoch)
                torch.save(encoder.state_dict(), cmd_args.save_dir + '/epoch-latest.encoder')
                torch.save(decoder.state_dict(), cmd_args.save_dir + '/epoch-latest.decoder')
