import json
import sys
import os
import random
import numpy as np

from code2inv.common.cmd_args import cmd_args
from code2inv.common.ssa_graph_builder import ProgramGraph
from code2inv.common.seq_graph_builder import SeqTokenGraph
import gzip
import pickle

from code2inv.graph_encoder.s2v_lib import S2VLIB, S2VGraph


class GraphSample(S2VGraph):
    def __init__(self, sample_index, db, pg, node_type_dict):
        super(GraphSample, self).__init__(pg, node_type_dict)
        self.sample_index = sample_index
        self.db = db

class SeqSample(object):
    def __init__(self, sample_index, db, pg, node_type_dict):
        self.pg = pg
        self.sample_index = sample_index
        self.db = db        

        self.token_idx = []
        for t in pg.raw_token_list:
            assert t in node_type_dict
            self.token_idx.append( node_type_dict[t] )

class Dataset(object):
    def __init__(self):
        self.pg_list = []
        self.sample_graphs = []
        self.file_names = []
        self.ordered_pre_post = []
        self.setup(GraphSample)

    def load_pg_list(self, fname):
        with open(cmd_args.data_root + '/graph/' + fname + '.json', 'r') as gf:
            graph_json = json.load(gf)
            self.pg_list.append( ProgramGraph(graph_json) )

    def setup(self, classname):
        print("running setup")
        with open(cmd_args.data_root + '/' + cmd_args.file_list, 'r') as f:
            for row in f:
                self.file_names.append(row.strip())
                self.load_pg_list(row.strip())
                tpl = []
                for i in range(1, 5):
                    if cmd_args.only_use_z3:
                        with open(cmd_args.data_root + '/smt2/' + row.strip() + '.smt.%d' % i, 'r') as gf:
                            tpl.append(gf.read())
                    else:
                        with open(cmd_args.data_root + '/boogie/' + row.strip() + '.bpl.%d' % i, 'r') as gf:
                            tpl.append(gf.read())
                self.ordered_pre_post.append( tpl )

        self.build_node_type_dict()

        for i in range(len(self.pg_list)):
            g = self.pg_list[i]
            self.sample_graphs.append( classname(i, self, g, self.node_type_dict) )
 
        self.sample_idxes = list(range(len(self.sample_graphs)))
        random.shuffle(self.sample_idxes)
        self.sample_pos = 0

    def build_node_type_dict(self):
        self.node_type_dict = {}
        
        for g in self.pg_list:
            for node in g.node_list:
                if not node.node_type in self.node_type_dict:
                    v = len(self.node_type_dict)
                    self.node_type_dict[node.node_type] = v        

    def sample_minibatch(self, num_samples, replacement=False):        
        if cmd_args.single_sample is not None:
            return [self.sample_graphs[cmd_args.single_sample]]

        g_list = []
        if replacement:
            for i in range(num_samples):
                idx = np.random.randint(len(self.sample_graphs))
                g_list.append( self.sample_graphs[idx] )
        else:
            assert num_samples <= len(self.sample_idxes)
            if num_samples == len(self.sample_idxes):
                return self.sample_graphs

            if self.sample_pos + num_samples > len(self.sample_idxes):
                random.shuffle(self.sample_idxes)
                self.sample_pos = 0

            for i in range(self.sample_pos, self.sample_pos + num_samples):
                g_list.append( self.sample_graphs[ self.sample_idxes[i] ] )
            self.sample_pos == num_samples

        return g_list
    
class SeqGraphDataset(Dataset):
    def __init__(self):
        self.pg_list = []
        self.sample_graphs = []
        self.file_names = []
        self.ordered_pre_post = []
        
        self.setup(SeqSample)

    def load_pg_list(self, fname):
        with open(cmd_args.data_root + '/token_files/' + fname + '.token', 'r') as gf:
            graph_json = json.load(gf)
            self.pg_list.append( SeqTokenGraph(graph_json) )

    def build_node_type_dict(self):
        self.node_type_dict = {}

        for g in self.pg_list:
            for token in g.raw_token_list:
                if not token in self.node_type_dict:
                    v = len(self.node_type_dict)
                    self.node_type_dict[token] = v    

        print(self.node_type_dict)

class PickleDataset(object):
    def __init__(self):
        self.pg_list = []
        self.sample_graphs = []
        self.ordered_pre_post = []

        self.train_indices = []
        self.test_indices = []
        self.single_sample_train = []
        self.single_sample_test = []

        self.setup()

    def setup(self):
        with open(cmd_args.data_root + '/' + cmd_args.file_list, 'r') as f:
            cur_sample_idx = 0
            for row in f:
                if cmd_args.single_sample is None or cur_sample_idx == cmd_args.single_sample:
                    filename = cmd_args.data_root + '/' + row.strip() + '.pickle'
                    with gzip.open(filename, 'rb') as f:
                        loaded_object = pickle.load(f)
                    num_samples = len(loaded_object)
                    num_train = int( num_samples * cmd_args.train_frac )
                    local_idx = 0
                    for x in loaded_object:
                        local_idx += 1
                        if local_idx <= num_train:
                            self.train_indices.append( len(self.pg_list) )
                            if cmd_args.single_sample is not None and cmd_args.single_sample == cur_sample_idx:
                                self.single_sample_train.append( len(self.pg_list) )
                        else:
                            self.test_indices.append( len(self.pg_list) )
                            if cmd_args.single_sample is not None and cmd_args.single_sample == cur_sample_idx:
                                self.single_sample_test.append( len(self.pg_list) )
                        graph_json = json.loads(x[0])
                        self.pg_list.append( ProgramGraph(graph_json))
                        self.ordered_pre_post.append( x[1] )
                cur_sample_idx += 1
        if cmd_args.single_sample is not None:
            assert len(self.single_sample_test) and len(self.single_sample_train)
            self.train_indices = self.single_sample_train
            self.test_indices = self.single_sample_test

        self.build_node_type_dict()

        for i in range(len(self.pg_list)):
            g = self.pg_list[i]
            self.sample_graphs.append( GraphSample(i, self, g, self.node_type_dict) )

        if cmd_args.phase == 'train':
            self.sample_idxes = self.train_indices
        else:
            self.sample_idxes = self.test_indices
            
        random.shuffle(self.sample_idxes)
        self.sample_pos = 0
    
    def build_node_type_dict(self):
        self.node_type_dict = {}

        for g in self.pg_list:
            for node in g.node_list:
                if not node.node_type in self.node_type_dict:
                    v = len(self.node_type_dict)
                    self.node_type_dict[node.node_type] = v        

    def sample_minibatch(self, num_samples, replacement=False):
        g_list = []
        if replacement:
            for i in range(num_samples):
                idx = np.random.randint(len(self.sample_idxes))
                g_list.append( self.sample_graphs[ self.sample_idxes[idx] ] )
        else:
            assert num_samples <= len(self.sample_idxes)

            if self.sample_pos + num_samples > len(self.sample_idxes):
                random.shuffle(self.sample_idxes)
                self.sample_pos = 0

            for i in range(self.sample_pos, self.sample_pos + num_samples):
                g_list.append( self.sample_graphs[ self.sample_idxes[i] ] )
            self.sample_pos == num_samples

        return g_list
