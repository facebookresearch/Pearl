import json
import sys

from code2inv.common.cmd_args import cmd_args
from code2inv.common.constants import AST_EDGE_TYPE, CONTROL_EDGE_TYPE, VAR_LINK_TYPE
from code2inv.common.checker import z3_check_implication


from code2inv.common.ssa_graph_builder import GraphNode


class SeqTokenGraph(object):
    def __init__(self, token_list):        
        self.node_list = []                
        self.raw_variable_nodes = {}
        self.const_nodes = {}
        self.unique_nodes = {}
        
        self.var_pos = {}
        self.const_pos = {}

        self.raw_token_list = [None] * len(token_list)

        if sys.version_info.major == 2:
            for i in range(len(token_list)):
                if type(token_list[i]) is unicode:
                    token_list[i] = str(token_list[i])
        
        for i in range(len(token_list)):
            token = token_list[i]
            if type(token) is dict:
                k, v = list(token.items())[0]
                if k == 'Var': # variable node
                    var_name = '_'.join(v.split('_')[0:-1])
                    if not var_name in self.raw_variable_nodes:
                        n_vars = len(self.raw_variable_nodes)
                        self.raw_variable_nodes[var_name] = self.add_node('var-%d' % n_vars, var_name)
                    self.var_pos[var_name] = i
                    self.raw_token_list[i] = self.raw_variable_nodes[var_name].node_type

        for i in range(len(token_list)):
            token = token_list[i]
            if type(token) is dict:
                k, v = list(token.items())[0]
                if k == 'Const': # constant
                    if not v in self.const_nodes:
                        n_const = len(self.const_nodes)
                        self.const_nodes[v] = self.add_node('const-%d' % n_const, v)
                    self.const_pos[v] = i
                    self.raw_token_list[i] = self.const_nodes[v].node_type

        for i in range(len(token_list)):
            token = token_list[i]
            if type(token) is str:
                self.raw_token_list[i] = token
            else:
                assert type(token) is dict
                k, v = list(token.items())[0]
                if k == 'UNK':
                    self.raw_token_list[i] = 'unk'

        for t in self.raw_token_list:
            assert t is not None

        self.eof_node = self.add_node('eof', 'eof')
        assert len(self.node_list) == len(self.raw_variable_nodes) + len(self.const_nodes) + 1
        
    def add_node(self, node_type, name = None):                
        idx = len(self.node_list)
        node = GraphNode(idx, node_type=node_type, name=name)
        self.node_list.append(node)
        if name is not None:
            key = node_type + '-' + name
            assert key not in self.unique_nodes
            self.unique_nodes[key] = node
        return node
