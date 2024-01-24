import json

from z3 import *

from torch.nn.parameter import Parameter
from embedding_util import Z3ASTGraph, Graph2Vec, ParamEmbed, VariableConstantDecoder, glorot_uniform, AttentionModule
from pearl.SMTimer.KNN_Predictor import Predictor
import torch.nn.functional as F
from pearl.action_representation_modules import action_representation_module
from pearl.action_representation_modules.one_hot_action_representation_module import \
    OneHotActionTensorRepresentationModule

from pearl.neural_networks.common.value_networks import EnsembleQValueNetwork
from pearl.replay_buffers.sequential_decision_making.bootstrap_replay_buffer import BootstrapReplayBuffer
from pearl.policy_learners.sequential_decision_making.bootstrapped_dqn import BootstrappedDQN
from pearl.utils.functional_utils.experimentation.set_seed import set_seed
from pearl.action_representation_modules.identity_action_representation_module import IdentityActionRepresentationModule
from pearl.history_summarization_modules.lstm_history_summarization_module import LSTMHistorySummarizationModule
from pearl.policy_learners.sequential_decision_making.deep_q_learning import DeepQLearning
from pearl.replay_buffers.sequential_decision_making.fifo_off_policy_replay_buffer import FIFOOffPolicyReplayBuffer
from pearl.utils.functional_utils.train_and_eval.online_learning import online_learning
from pearl.pearl_agent import PearlAgent
from pearl.tutorials.single_item_recommender_system_example.env_model import SequenceClassificationModel
from pearl.tutorials.single_item_recommender_system_example.env import RecEnv
import torch
import matplotlib.pyplot as plt
import numpy as np
from env import ConstraintSimplificationEnv
def visit(expr):
    if is_const(expr) and expr.decl().kind() == Z3_OP_UNINTERPRETED:
        # Add only uninterpreted functions (which represent variables)
        variables.add(str(expr))
    else:
        # Recursively visit children for composite expressions
        for child in expr.children():
            visit(child)
# predictor = Predictor('KNN')

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

graph = Z3ASTGraph(assertions)
# graph.visualize()
# print(graph.node_list)
# graph.visualize()
# compare_ast_and_graph(assertions,graph)
node_type_dict = NODE_TYPE_ENUM

# 步骤4: 使用Graph2Vec转换图到向量表示
graph2vec = Graph2Vec(graph)
# 步骤5: 输出转换结果
print("节点特征向量:")
print(graph2vec.node_feat.shape)
# node_embed = glorot_uniform(graph2vec.node_feat)
node_embed = Parameter(graph2vec.node_feat)
latent_dim = 58  # 假设潜在维度为58
attention_dim = int(node_embed.shape[0])
print(node_embed.shape)
attention_module = AttentionModule(latent_dim, attention_dim)
output_tensor = attention_module(node_embed)
# encoder = ParamEmbed(latent_dim, graph.num_nodes())
#
# # 步骤4: 编码图
# node_embeddings = encoder(graph)
# print("节点嵌入向量:")
# print(node_embeddings)
# print(node_embeddings.shape)
input_size = 58  # 输入特征的维度
hidden_size = 58  # LSTM隐藏层的维度
decoder = VariableConstantDecoder(input_size, hidden_size, len(variables), len(variables))

# 假设输入向量维度为[377, 128]
# input_seq = torch.rand(1, 377, 128)  # Batch size为1
print(output_tensor.shape)
variable_pred, constant_pred = decoder(output_tensor)

predicted_variable_index = torch.argmax(variable_pred, dim=2).item()  # 获取概率最高的变量索引
predicted_constant_index = torch.argmax(constant_pred, dim=2).item()  # 获取概率最高的常数索引
print(predicted_variable_index, predicted_constant_index)

set_seed(0)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
# model = SequenceClassificationModel(100).to(device)
# model.load_state_dict(torch.load("/home/lz/PycharmProjects/Pearl/pearl/tutorials/single_item_recommender_system_example/env_model_state_dict.pt"))
# actions = torch.load("/home/lz/PycharmProjects/Pearl/pearl/tutorials/single_item_recommender_system_example/news_embedding_small.pt")
# env = RecEnv(list(actions.values())[:100], model)
# observation, action_space = env.reset()

# 创建环境实例
# 创建环境实例
env = ConstraintSimplificationEnv(attention_module, decoder, assertions, len(variables), len(variables))
observation, action_space = env.reset()
# action_representation_module = OneHotActionTensorRepresentationModule(
#     max_number_actions=len(env.variables),
# )
action_representation_module = IdentityActionRepresentationModule(
    max_number_actions=action_space.n,
    representation_dim=action_space.action_dim,
)
# experiment code
number_of_steps = 100000
record_period = 400
# 创建强化学习代理
print(len(env.variables))
agent = PearlAgent(
    policy_learner=BootstrappedDQN(
        q_ensemble_network=EnsembleQValueNetwork(
            # 配置网络参数
            state_dim=latent_dim,
            action_dim=len(env.variables),
            ensemble_size=10,
            output_dim=1,
            hidden_dims=[58, 58],
            prior_scale=0.3,
        ),
        action_space=env.action_space,
        training_rounds=50,
        action_representation_module=action_representation_module,
    ),
    history_summarization_module=LSTMHistorySummarizationModule(
        observation_dim=latent_dim,
        action_dim=len(env.variables),
        hidden_dim=58,
        history_length=len(env.variables),  # 和完整结点数相同
    ),
    replay_buffer=BootstrapReplayBuffer(100_000, 1.0, 10),
    device_id=-1,
)

# 训练代理
info = online_learning(
    agent=agent,
    env=env,
    number_of_steps=number_of_steps,
    print_every_x_steps=100,
    record_period=record_period,
    learn_after_episode=True,
)
torch.save(info["return"], "BootstrappedDQN-LSTM-return.pt")
plt.plot(record_period * np.arange(len(info["return"])), info["return"], label="BootstrappedDQN-LSTM")
plt.legend()
plt.show()
