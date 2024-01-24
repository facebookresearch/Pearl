import ctypes
import numpy as np
import os
import sys
import torch
import json
from torch.autograd import Variable

sys.path.append('%s/../common' % os.path.dirname(os.path.realpath(__file__)))

from constants import NUM_EDGE_TYPES
from ssa_graph_builder import ProgramGraph
from cmd_args import cmd_args

class _s2v_lib(object):

    def __init__(self, args):
        dir_path = os.path.dirname(os.path.realpath(__file__))
        self.lib = ctypes.CDLL('%s/build/dll/libs2v.so' % dir_path)

        self.lib.n2n_construct.restype = ctypes.c_int

        if sys.version_info[0] > 2:
            args = [arg.encode() for arg in args]  # str -> bytes for each element in args
        arr = (ctypes.c_char_p * len(args))()
        arr[:] = args
        self.lib.Init(len(args), arr)

    def _getGraphOrList(self, s2v_graphs):
        if type(s2v_graphs) is not list:
            single_graph = s2v_graphs
        elif len(s2v_graphs) == 1:
            single_graph = s2v_graphs[0]
        else:
            single_graph = None
        return single_graph

    def PrepareMeanField(self, s2v_graphs):
        n2n_sp_list = []
        single_graph = self._getGraphOrList(s2v_graphs)
        for i in range(NUM_EDGE_TYPES):
            if single_graph is not None:
                n2n_sp = single_graph.n2n_sp_list[i]
            else:                
                num_edges = 0
                for g in s2v_graphs:
                    num_edges += len(g.typed_edge_list[i])                
                n2n_idxes = torch.LongTensor(2, num_edges)
                n2n_vals = torch.FloatTensor(num_edges)

                num_nodes = 0
                nnz = 0
                for j in range(len(s2v_graphs)):
                    g = s2v_graphs[j]
                    n2n_idxes[:, nnz : nnz + len(g.typed_edge_list[i])] = g.n2n_sp_list[i]._indices() + num_nodes
                    n2n_vals[nnz : nnz + len(g.typed_edge_list[i])] = g.n2n_sp_list[i]._values()
                    num_nodes += g.pg.num_nodes()
                    nnz += len(g.typed_edge_list[i])
                assert nnz == num_edges

                n2n_sp = torch.sparse.FloatTensor(n2n_idxes, n2n_vals, torch.Size([num_nodes, num_nodes]))
            if cmd_args.ctx == 'gpu':
                n2n_sp = n2n_sp.cuda()

            n2n_sp_list.append(Variable(n2n_sp))

        return n2n_sp_list

    def ConcatNodeFeats(self, s2v_graphs):
        single_graph = self._getGraphOrList(s2v_graphs)

        if single_graph is not None:
            feat = single_graph.node_feat
        else:
            feat_list = []
            for g in s2v_graphs:
                feat_list.append(g.node_feat)
            
            feat = torch.cat(feat_list, dim=0)
        if cmd_args.ctx == 'gpu':
            feat = feat.cuda()
        return Variable(feat)

dll_path = '%s/build/dll/libs2v.so' % os.path.dirname(os.path.realpath(__file__))
if os.path.exists(dll_path):
    S2VLIB = _s2v_lib(sys.argv)
else:
    S2VLIB = None

class S2VGraph(object):
    def __init__(self, pg, node_type_dict):
        self.pg = pg

        self.typed_edge_list = [None] * NUM_EDGE_TYPES
        for i in range(NUM_EDGE_TYPES):
            self.typed_edge_list[i] = []

        for e in self.pg.edge_list:
            self.typed_edge_list[e[2]].append((e[0], e[1]))
        
        self.n2n_sp_list = []
        for i in range(NUM_EDGE_TYPES):
            edges = self.typed_edge_list[i]
            degrees = np.zeros(shape=(pg.num_nodes()), dtype=np.int32)
            for e in edges:
                degrees[e[1]] += 1
                            
            edges.sort(key = lambda x : (x[1], x[0]))
            
            num_edges = len(edges)
            n2n_idxes = torch.LongTensor(2,  num_edges)
            n2n_vals = torch.FloatTensor(num_edges) 
            if num_edges > 0:
                x, y = zip(*edges)            
                edge_pairs = np.ndarray(shape=(num_edges, 2), dtype=np.int32)
                edge_pairs[:, 0] = x
                edge_pairs[:, 1] = y
                edge_pairs = edge_pairs.flatten()
                
                S2VLIB.lib.n2n_construct(pg.num_nodes(),
                                        num_edges,
                                        ctypes.c_void_p(degrees.ctypes.data), 
                                        ctypes.c_void_p(edge_pairs.ctypes.data), 
                                        ctypes.c_void_p(n2n_idxes.numpy().ctypes.data), 
                                        ctypes.c_void_p(n2n_vals.numpy().ctypes.data), 
                                        )
                                        
                n2n_sp = torch.sparse.FloatTensor(n2n_idxes, n2n_vals, torch.Size([pg.num_nodes(), pg.num_nodes()]))
                
                self.n2n_sp_list.append(n2n_sp)
        
        self.node_feat = torch.zeros(pg.num_nodes(), len(node_type_dict))
        
        for i in range(pg.num_nodes()):
            node_type = pg.node_list[i].node_type
            self.node_feat[i, node_type_dict[node_type]] = 1.0
        
if __name__ == '__main__':
    s2v_graphs = []
    with open(cmd_args.data_root + '/list.txt', 'r') as f:
        for row in f:            
            with open(cmd_args.data_root + '/' + row.strip(), 'r') as gf:
                graph_json = json.load(gf)
                g = ProgramGraph(graph_json)

                s2v_graphs.append( S2VGraph(g) )

    sp_list = S2VLIB.PrepareMeanField(s2v_graphs[0:2])

    new_feat = S2VLIB.ConcatNodeFeats(s2v_graphs[0:2])
    print(new_feat)