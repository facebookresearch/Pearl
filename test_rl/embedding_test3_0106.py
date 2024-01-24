import json
from graphviz import Digraph
import numpy as np
from torch import nn
from z3 import *

import torch
import numpy as np
from torch.nn.parameter import Parameter
from pearl.SMTimer.KNN_Predictor import Predictor

predictor = Predictor('KNN')

file_path = '/home/lz/baidudisk/smt/gnu_angr.tar.gz/single_test/arch/arch15998'
with open(file_path, 'r') as file:
    # 读取文件所有内容到一个字符串
    smtlib_str = file.read()
# 解析字符串
try:
    # 将JSON字符串转换为字典
    dict_obj = json.loads(smtlib_str)
    # print("转换后的字典：", dict_obj)
except json.JSONDecodeError as e:
    print("解析错误：", e)
#
smtlib_str = dict_obj['script']
assertions = parse_smt2_string(smtlib_str)

variables = set()


# Visit each assertion to extract variables
def visit(expr):
    if is_const(expr) and expr.decl().kind() == Z3_OP_UNINTERPRETED:
        # Add only uninterpreted functions (which represent variables)
        variables.add(str(expr))
    else:
        # Recursively visit children for composite expressions
        for child in expr.children():
            visit(child)

solver = Solver()
for a in assertions:
    solver.add(a)

# Extract variables from each assertion
for a in assertions:
    visit(a)

# Print all variables
print("变量列表：")
for v in variables:
    print(v)

# Node definition for the graph
NUM_EDGE_TYPES = 3
NUM_EDGE_TYPES = 3  # 根据你的图的边的种类来定义

NODE_TYPE_ENUM = {
    "BoolExpr": 0,       # 布尔表达式
    "ArithExpr": 1,      # 算术表达式
    "Variable": 2,       # 变量
    "Constant": 3,       # 常量
    "Quantifier": 4,     # 量词
    "FuncAndRelation": 5 # 函数和关系
}
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
class GraphNode:
    def __init__(self, idx, ast_node=None, node_type=None, semantic_info=None):
        self.idx = idx
        self.ast_node = ast_node  # Z3 AST节点
        self.node_type = node_type  # 节点类型
        self.semantic_info = semantic_info  # 节点的语义信息
        self.edges_out = []  # 出边列表
        self.edges_in = []   # 入边列表

    def add_out_edge(self, dest_node, edge_type):
        self.edges_out.append((dest_node, edge_type))

    def add_in_edge(self, src_node, edge_type):
        self.edges_in.append((src_node, edge_type))

class Z3ASTGraph:
    def __init__(self, z3_ast):
        self.node_list = []
        self.edge_list = []
        self.ast_node_to_graph_node = {}  # Z3 AST节点到图节点的映射
        self._build_graph(z3_ast)

    def _build_graph(self, z3_ast):
        if isinstance(z3_ast, z3.AstVector):
            for ast in z3_ast:
                self._traverse_ast_and_add(ast)
        else:
            self._traverse_ast_and_add(z3_ast)

    def _traverse_ast_and_add(self, ast_node):
        if ast_node in self.ast_node_to_graph_node:
            return self.ast_node_to_graph_node[ast_node]

        idx = len(self.node_list)
        node_type, semantic_info = self._determine_node_type_and_info(ast_node)
        graph_node = GraphNode(idx, ast_node, node_type, semantic_info)
        self.node_list.append(graph_node)
        self.ast_node_to_graph_node[ast_node] = graph_node

        if hasattr(ast_node, 'children'):
            for child_ast_node in ast_node.children():
                child_graph_node = self._traverse_ast_and_add(child_ast_node)
                self._add_directed_edge(graph_node, child_graph_node, 0)  # 默认边类型为0

        return graph_node

    def _add_directed_edge(self, src_node, dst_node, edge_type):
        src_node.add_out_edge(dst_node, edge_type)
        dst_node.add_in_edge(src_node, edge_type)
        self.edge_list.append((src_node.idx, dst_node.idx, edge_type))

    def _determine_node_type_and_info(self, node):
        # 确定节点类型和提取语义信息
        if z3.is_bool(node):
            return "BoolExpr", node.decl().name() if hasattr(node, 'decl') else ""
        elif z3.is_arith(node):
            return "ArithExpr", node.decl().name() if hasattr(node, 'decl') else ""
        elif z3.is_quantifier(node):
            return "Quantifier", str(node)
        elif z3.is_var(node):
            return "Variable", node.decl().name() if hasattr(node, 'decl') else ""
        elif z3.is_const(node):
            return "Constant", node.decl().name() if hasattr(node, 'decl') else ""
        else:
            return "Unknown", str(node)  # 默认类型

    def num_nodes(self):
        return len(self.node_list)

    def num_edges(self):
        return len(self.edge_list)

class Graph2Vec:
    def __init__(self, graph):
        self.graph = graph

        # 初始化边类型列表
        self.typed_edge_list = [[] for _ in range(NUM_EDGE_TYPES)]
        for src, dst, edge_type in self.graph.edge_list:
            edge_type = int(edge_type)  # 确保edge_type是整数
            self.typed_edge_list[edge_type].append((src, dst))

        # 初始化节点特征向量
        num_node_types = len(NODE_TYPE_ENUM)  # 节点类型数量
        self.node_feat = torch.zeros(graph.num_nodes(), num_node_types)
        for i, node in enumerate(graph.node_list):
            type_index = NODE_TYPE_ENUM.get(node.node_type, 0)
            self.node_feat[i, type_index] = 1.0  # one-hot编码
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
        # assert graph.pg.num_nodes() == self.num_nodes
        return self.node_embed

graph = Z3ASTGraph(assertions)
# graph.display()
# print(graph.node_list)
# graph.visualize()
# compare_ast_and_graph(assertions,graph)
node_type_dict = NODE_TYPE_ENUM

    # 步骤4: 使用Graph2Vec转换图到向量表示
graph2vec = Graph2Vec(graph)
# 步骤5: 输出转换结果
print("节点特征向量:")
print(graph2vec.node_feat.shape)
latent_dim = 128  # 假设潜在维度为128
encoder = ParamEmbed(latent_dim, graph.num_nodes())

# 步骤4: 编码图
node_embeddings = encoder(graph)

print("节点嵌入向量:")
print(node_embeddings)
print(node_embeddings.shape)