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
from code2inv.common.cmd_args import cmd_args
from code2inv.common.pytorch_util import get_torch_version
from code2inv.common.dataset import PickleDataset
from code2inv.common.checker import boogie_result, report_ice_stats, report_tested_stats, stat_counter
from code2inv.graph_encoder.embedding import EmbedMeanField


from code2inv.prog_generator.prog_encoder import LogicEncoder
from code2inv.prog_generator.rl_helper import RLEnv, ExprNode, rollout, actor_critic_loss
from code2inv.prog_generator.tree_decoder import GeneralDecoder


def test_loop():
    tqdm.write('testing')
    g_list = []
    for idx in dataset.test_indices:
        g_list.append(dataset.sample_graphs[idx])

    node_embedding_batch = encoder(g_list, istraining=False)

    version = get_torch_version()
    if version >= 0.4:
        torch.set_grad_enabled(False)
   
    embedding_offset = 0
    r = 0
    for b in range(len(g_list)):
        g = g_list[b]
        node_embedding = node_embedding_batch[embedding_offset : embedding_offset + g.pg.num_nodes(), :]
        embedding_offset += g.pg.num_nodes()
        nll_list, value_list, reward_list, root, _ = rollout(g, node_embedding, decoder, use_random = True, eps = 0.0)  
        r += sum(reward_list)
    tqdm.write('avg test_rl reward: %.4f\n' % (r / len(g_list)))
    if version >= 0.4:
        torch.set_grad_enabled(True)
    if cmd_args.phase == 'test_rl' and len(dataset.test_indices) == len(stat_counter.reported):
        sys.exit()

if __name__ == '__main__':
    random.seed(cmd_args.seed)
    np.random.seed(cmd_args.seed)
    torch.manual_seed(cmd_args.seed)    
    
    dataset = PickleDataset()

    params = []

    if cmd_args.encoder_model == 'GNN':
        encoder = EmbedMeanField(cmd_args.embedding_size, len(dataset.node_type_dict), max_lv=cmd_args.s2v_level)
    elif cmd_args.encoder_model == 'LSTM':
        encoder = LSTMEmbed(cmd_args.embedding_size, len(dataset.node_type_dict))
    elif cmd_args.encoder_model == 'Param':
        g_list = GraphSample(graph, vc_list, dataset.node_type_dict)
        encoder = ParamEmbed(cmd_args.embedding_size, g_list.pg.num_nodes())
    else:
        raise NotImplementedError

    decoder = GeneralDecoder(cmd_args.embedding_size)
    if cmd_args.init_model_dump is not None and os.path.isfile(cmd_args.init_model_dump + '.encoder'):
        encoder.load_state_dict(torch.load(cmd_args.init_model_dump + '.encoder'))
        decoder.load_state_dict(torch.load(cmd_args.init_model_dump + '.decoder'))
        
    params.append( encoder.parameters() )
    params.append( decoder.parameters() )
    
    optimizer = optim.Adam(chain.from_iterable(params), lr=cmd_args.learning_rate)    
    
    for epoch in range(cmd_args.num_epochs):
        print("NUM EPOCHS", cmd_args.num_epochs)
        best_reward = -5.0
        best_root = None
        tested_roots = []
        if cmd_args.phase == 'train' or not cmd_args.tune_test:
            test_loop()
        if cmd_args.phase == 'test_rl' and cmd_args.tune_test == 0:
            continue
        acc_reward = 0.0
        pbar = tqdm(range(100))
        for k in pbar:
            if cmd_args.phase == 'test_rl' and len(stat_counter.reported) == len(dataset.test_indices):
                sys.exit()
            
            g_list = dataset.sample_minibatch(cmd_args.rl_batchsize, replacement = True)
            node_embedding_batch = encoder(g_list)

            total_loss = 0.0
            embedding_offset = 0
            for b in range(cmd_args.rl_batchsize):
                if len(g_list) > 1:
                    g = g_list[b]
                    node_embedding = node_embedding_batch[embedding_offset : embedding_offset + g.pg.num_nodes(), :]
                    embedding_offset += g.pg.num_nodes()
                else:
                    g = g_list[0]
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
        
        g = dataset.sample_minibatch(1, replacement = True)[0]
        node_embedding = encoder(g)
        while True:
            _, _, _, root, trivial = rollout(g, node_embedding, decoder, use_random = True, eps = 0.0)
            print("ROOT", root)
            if trivial == False:
                break
        print('epoch: %d, average reward: %.4f, Random: %s, result_r: %.4f' % (epoch, acc_reward / 100.0, root, boogie_result(g, root)))
        print("best_reward:", best_reward, ", best_root:", best_root)
        if cmd_args.save_dir is not None and cmd_args.phase == 'train':
            torch.save(encoder.state_dict(), cmd_args.save_dir + '/epoch-%d.encoder' % (epoch + 1))
            torch.save(decoder.state_dict(), cmd_args.save_dir + '/epoch-%d.decoder' % (epoch + 1))
            torch.save(encoder.state_dict(), cmd_args.save_dir + '/epoch-latest.encoder')
            torch.save(decoder.state_dict(), cmd_args.save_dir + '/epoch-latest.decoder')
