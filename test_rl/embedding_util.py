import json
from graphviz import Digraph
import numpy as np
from torch import nn
from z3 import *
import gensim.downloader as api
from torch.nn.parameter import Parameter

from common.pytorch_util import weights_init
from pearl.SMTimer.KNN_Predictor import Predictor
import torch.nn.functional as F
import torch
import numpy as np

predictor = Predictor('KNN')


# Visit each assertion to extract variables



# Node definition for the graph
NUM_EDGE_TYPES = 3
NUM_EDGE_TYPES = 2  # 根据你的图的边的种类来定义

NODE_TYPE_ENUM = {
    "Variable-Int": 0,  # 布尔表达式
    "Variable-Real": 1,  # 算术表达式
    "Constant": 2,  # 变量
    "BoolExpr": 3,  # 常量
    "ArithExpr": 4,  # 量词
    "Quantifier": 5,  # 函数和关系
    "Operation": 6,  # 函数和关系
    "Unknown": 7  # 函数和关系
}
EDGE_TYPE_ENUM = {
    "ParentChild": 0,
    "Sibling": 1,
    # 根据需求可以添加更多边的类型
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
        self.edges_in = []  # 入边列表

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
        # 处理AstVector或单一AST节点
        if isinstance(z3_ast, z3.AstVector):
            for ast in z3_ast:
                self._traverse_ast_and_add(ast)
        else:
            self._traverse_ast_and_add(z3_ast)

    def _traverse_ast_and_add(self, expr, parent_node=None, level=0):
        # 创建图节点
        idx = len(self.node_list)
        node_type, semantic_info = self._determine_node_type_and_info(expr)
        graph_node = GraphNode(idx, expr, node_type, semantic_info)
        self.node_list.append(graph_node)
        self.ast_node_to_graph_node[expr] = graph_node

        # 如果有父节点，添加父子关系边
        if parent_node is not None:
            self._add_directed_edge(parent_node, graph_node, EDGE_TYPE_ENUM["ParentChild"])

        # 根据Z3表达式的类型递归遍历子节点
        if isinstance(expr, (
        z3.BoolRef, z3.ArithRef, z3.BitVecRef, z3.ArrayRef)) and expr.decl().kind() != z3.Z3_OP_UNINTERPRETED:
            for child_expr in expr.children():
                self._traverse_ast_and_add(child_expr, graph_node, level + 1)
        elif isinstance(expr, z3.QuantifierRef):
            self._traverse_ast_and_add(expr.body(), graph_node, level + 1)

    def _add_directed_edge(self, src_node, dst_node, edge_type):
        src_node.add_out_edge(dst_node, edge_type)
        dst_node.add_in_edge(src_node, edge_type)
        self.edge_list.append((src_node.idx, dst_node.idx, edge_type))

    def _determine_node_type_and_info(self, node):
        # 根据Z3节点类型确定图中节点的类型和提取语义信息
        if z3.is_const(node):
            # 检查节点是否表示整数或实数
            if node.sort().kind() == z3.Z3_INT_SORT:
                return "Variable-Int", node.decl().name()
            elif node.sort().kind() == z3.Z3_REAL_SORT:
                return "Variable-Real", node.decl().name()
            else:
                return "Constant", node.decl().name()  # 其他常量类型
        else:
            if z3.is_bool(node):
                return "BoolExpr", str(node)
            elif z3.is_arith(node):
                return "ArithExpr", str(node)
            elif z3.is_quantifier(node):
                return "Quantifier", str(node)
            # elif z3.is_var(node):#与程序中的变量不同，可能需要考虑修改
            #     print('************************************')
            #     print(str(node))
            #     return "Variable", str(node)
            # elif z3.is_const(node):
            #     return "Constant", str(node.decl())
            elif node.decl().kind() == z3.Z3_OP_UNINTERPRETED:
                return "Operation", str(node.decl().name())
            else:
                return "Unknown", str(node)

    def num_nodes(self):
        return len(self.node_list)

    def num_edges(self):
        return len(self.edge_list)

    def visualize(self, filename='ast_graph'):
        dot = Digraph(comment='The Z3 AST Graph')

        # 添加所有节点
        for node in self.node_list:
            label = f"Idx: {node.idx}\nType: {node.node_type}\nInfo: {node.semantic_info}"
            dot.node(str(node.idx), label)

        # 添加所有边
        for src, dst, edge_type in self.edge_list:
            edge_label = list(EDGE_TYPE_ENUM.keys())[edge_type]
            dot.edge(str(src), str(dst), label=edge_label)

        dot.render(filename, view=True)  # 保存并显示图形


word_vectors = api.load("glove-wiki-gigaword-50")  # 你可以选择不同的模型


class Graph2Vec:
    def __init__(self, graph):
        self.graph = graph
        self.latent_dim = 50  # 词嵌入的维度，根据选择的模型而定
        self.num_node_types = len(NODE_TYPE_ENUM)  # 节点类型数量，确保这个值正确

        # 初始化节点特征向量
        self.node_feat = torch.zeros(graph.num_nodes(), self.latent_dim + self.num_node_types)

        for i, node in enumerate(graph.node_list):
            # 获取节点的语义信息，并转换为向量
            semantic_vector = self._get_embedding(node.semantic_info)
            # 获取节点的类型信息，并转换为独热编码
            type_vector = self._get_type_vector(node.node_type)
            # 拼接向量
            self.node_feat[i] = torch.cat((torch.tensor(semantic_vector), type_vector), 0)

    def _get_embedding(self, text):
        # 将文本分解为单词，并获取每个单词的词嵌入向量
        # 然后取平均作为整个文本的向量表示
        words = text.split()
        embeddings = [word_vectors[word] for word in words if word in word_vectors]
        if len(embeddings) == 0:
            print('****************************')
            return np.zeros(self.latent_dim)  # 如果没有单词有嵌入，则返回0向量
        return np.mean(embeddings, axis=0)

    def _get_type_vector(self, node_type):
        # 创建独热编码表示节点类型
        type_vector = torch.zeros(self.num_node_types)
        type_index = NODE_TYPE_ENUM[node_type]
        type_vector[type_index] = 1
        return type_vector


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


class Attention(nn.Module):
    def __init__(self, hidden_size):
        super(Attention, self).__init__()
        self.hidden_size = hidden_size
        self.attn = nn.Linear(self.hidden_size * 2, hidden_size)
        self.v = nn.Parameter(torch.rand(hidden_size))
        stdv = 1. / (self.v.size(0) ** 0.5)
        self.v.data.normal_(mean=0, std=stdv)

    def forward(self, hidden, encoder_outputs):
        # hidden: [batch_size, 1, hidden_size]
        # encoder_outputs: [batch_size, seq_len, hidden_size]
        batch_size = encoder_outputs.size(0)
        seq_len = encoder_outputs.size(1)

        # 重复hidden张量，使其第二维（seq_len）与encoder_outputs匹配
        hidden = hidden.repeat(1, seq_len, 1)  # [batch_size, seq_len, hidden_size]

        # 现在可以安全地连接hidden和encoder_outputs
        energy = torch.tanh(self.attn(torch.cat([hidden, encoder_outputs], 2)))
        energy = energy.transpose(1, 2)  # [batch_size, hidden_size, seq_len]
        v = self.v.repeat(batch_size, 1).unsqueeze(1)  # [batch_size, 1, hidden_size]
        energy = torch.bmm(v, energy)  # [batch_size, 1, seq_len]
        return F.softmax(energy, dim=2)


class VariableConstantDecoder(nn.Module):
    def __init__(self, input_size, hidden_size, num_variables, num_constants):
        super(VariableConstantDecoder, self).__init__()
        self.hidden_size = hidden_size
        self.lstm = nn.LSTM(input_size, hidden_size, batch_first=True)
        self.attention = Attention(hidden_size)
        self.variable_predictor = nn.Linear(hidden_size, num_variables)  # 变量数量
        self.constant_predictor = nn.Linear(hidden_size, num_constants)  # 常数数量

    def forward(self, input_seq):
        # 确保输入是三维张量
        if input_seq.dim() == 2:
            input_seq = input_seq.unsqueeze(0)  # 添加batch维度
        if input_seq.dim() == 1:
            input_seq = input_seq.unsqueeze(0)  # 添加batch维度
            input_seq = input_seq.unsqueeze(0)
        lstm_out, _ = self.lstm(input_seq)
        attn_weights = self.attention(lstm_out[:, -1, :].unsqueeze(0), lstm_out)
        context = attn_weights.bmm(lstm_out)

        variable_pred = self.variable_predictor(context)
        constant_pred = self.constant_predictor(context)

        # 使用softmax函数对输出进行归一化
        return F.softmax(variable_pred, dim=2), F.softmax(constant_pred, dim=2)


# class LogicEncoder(nn.Module):
#     def __init__(self, latent_dim):
#         super(LogicEncoder, self).__init__()
#         self.latent_dim = latent_dim
#
#         # if RULESET:
#         #     self.char_embedding = Parameter(torch.Tensor(sum([len(RULESET[rule]) for rule in RULESET]), latent_dim))
#
#         def new_gate():
#             lh = nn.Linear(self.latent_dim, self.latent_dim)
#             rh = nn.Linear(self.latent_dim, self.latent_dim)
#             return lh, rh
#
#         self.ilh, self.irh = new_gate()
#         self.lflh, self.lfrh = new_gate()
#         self.rflh, self.rfrh = new_gate()
#         self.ulh, self.urh = new_gate()
#
#         self.i_gates = [nn.Linear(self.latent_dim, self.latent_dim) for _ in range(MAX_CHILD)]
#         self.f_gates = [[nn.Linear(self.latent_dim, self.latent_dim) for _ in range(MAX_CHILD)] for _ in
#                         range(MAX_CHILD)]
#         self.u_gates = [nn.Linear(self.latent_dim, self.latent_dim) for _ in range(MAX_CHILD)]
#
#         self.ix = nn.Linear(self.latent_dim, self.latent_dim)
#         self.fx = nn.Linear(self.latent_dim, self.latent_dim)
#         self.ux = nn.Linear(self.latent_dim, self.latent_dim)
#
#         self.cx, self.ox = new_gate()
#         self.oend = nn.Linear(self.latent_dim * 3, self.latent_dim)
#
#         weights_init(self)
#
#     def forward(self, node_embedding, init_embedding, root):
#         if root is None or checkallnone(root):
#             return init_embedding
#
#         if root.state is None:
#             self.subexpr_embed(node_embedding, root)
#         s = torch.cat((init_embedding, root.state[0], root.state[1]), dim=1)
#         o = F.tanh(self.oend(s))
#         return o
#
#     def _get_token_embed(self, node_embedding, node):
#         idx = 0
#         for rule in RULESET:
#             for opt in RULESET[rule]:
#                 if len(opt) == 1 and node.name == opt[0]:
#                     x_embed = torch.index_select(self.char_embedding, 0, Variable(torch.LongTensor([idx])))
#                     return x_embed
#                 elif node.name in opt and node.name not in RULESET:
#                     x_embed = torch.index_select(self.char_embedding, 0, Variable(torch.LongTensor([idx])))
#                     return x_embed
#                 idx += 1
#         if node.name != "":
#             # Not a rule nor a terminal
#             # must be a const
#             assert node.pg_node is not None
#             return torch.index_select(node_embedding, 0, Variable(torch.LongTensor([node.pg_node.index])))
#         elif node.name == "" and node.rule is not None and node.rule != "":
#             idx = 0
#             for rule in RULESET:
#                 if node.rule == rule:
#                     x_embed = torch.index_select(self.char_embedding, 0, Variable(torch.LongTensor([idx])))
#                     return x_embed
#                 idx += 1
#
#     def subexpr_embed(self, node_embedding, node):
#         if node.state is not None:
#             return node.state
#
#         x_embed = self._get_token_embed(node_embedding, node)
#
#         if len(node.children) == 0:  # leaf node
#
#             c = self.cx(x_embed)
#             o = F.sigmoid(self.ox(x_embed))
#             h = o * F.tanh(c)
#             node.state = (c, h)
#         else:
#             children_states = []
#             for child in node.children:
#                 self.subexpr_embed(node_embedding, child)
#                 children_states.append(child.state)
#
#             i_gate_vals = []
#             for idx in range(len(node.children)):
#                 i_gate_vals.append(self.i_gates[idx](children_states[idx][1]))
#
#             i = F.sigmoid(self.ix(x_embed) + sum(i_gate_vals))
#             fx = self.fx(x_embed)
#
#             f_vals = []
#
#             for idx in range(len(node.children)):
#                 f_gate_vals = []
#                 for idx2 in range(len(node.children)):
#                     f_gate_vals.append(self.f_gates[idx][idx2](children_states[idx2][1]))
#                 f_vals.append(fx + sum(f_gate_vals))
#
#             u_gate_vals = []
#
#             for idx in range(len(node.children)):
#                 u_gate_vals.append(self.u_gates[idx](children_states[idx][1]))
#
#             update = F.tanh(self.ux(x_embed) + sum(u_gate_vals))
#             c = i * update + sum([f_vals[idx] * children_states[idx][0] for idx in range(len(node.children))])
#             h = F.tanh(c)
#
#             node.state = (c, h)

#聚合向量信息
class AttentionModule(nn.Module):
    def __init__(self, feature_dim, attention_dim):
        super(AttentionModule, self).__init__()
        self.feature_dim = feature_dim
        self.attention_dim = attention_dim
        self.attention_weights = nn.Linear(feature_dim, attention_dim)
        self.context_vector = nn.Linear(attention_dim, 1, bias=False)

    def forward(self, x):
        # x 的维度是 [n, feature_dim]

        # 为了兼容单个样本，我们增加一个维度来模仿批量处理
        # 新的维度是 [1, n, feature_dim]
        x = x.unsqueeze(0)

        # 计算注意力分数
        attn_score = self.attention_weights(x) # [1, n, attention_dim]
        attn_score = torch.tanh(attn_score)
        attn_score = self.context_vector(attn_score) # [1, n, 1]

        # 应用 softmax 以获取注意力分布
        attn_distribution = F.softmax(attn_score, dim=1) # [1, n, 1]

        # 用注意力分布加权平均
        aggregated_output = torch.sum(x * attn_distribution, dim=1) # [1, feature_dim]

        # 移除用于批量处理的额外维度，得到 [feature_dim]
        aggregated_output = aggregated_output.squeeze(0)

        return aggregated_output

# 示例
n = 10 # 假设有 10 个向量
feature_dim = 128
attention_dim = 32 # 注意力层的维度可以自行设置

input_tensor = torch.randn(n, feature_dim) # 随机生成一个输入张量

# 创建模块并应用
attention_module = AttentionModule(feature_dim, attention_dim)
output_tensor = attention_module(input_tensor)

print(output_tensor.shape) # 输出的维度应该是 [128]