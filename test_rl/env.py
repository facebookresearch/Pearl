# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
#
# pyre-ignore-all-errors

import random
from abc import ABC

import numpy as np
import sys

import torch
from torch.nn.parameter import Parameter
from z3 import *
import embedding_util
from pearl.SMTimer.KNN_Predictor import Predictor

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

import torch
import numpy as np
import random
from pearl.api.action_result import ActionResult
from pearl.api.environment import Environment
from pearl.utils.instantiations.spaces.discrete_action import DiscreteActionSpace

sys.path.append('/home/lz/PycharmProjects/Pearl/test_rl')
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


class CustomEnvironment(Environment):
    def __init__(self, model, encoder, decoder, graph_builder, num_vars, num_consts):
        self.model = model.to(device)
        self.encoder = encoder.to(device)
        self.decoder = decoder.to(device)
        self.graph_builder = graph_builder
        self.num_vars = num_vars
        self.num_consts = num_consts
        self.T = 20  # 最大步数
        self.t = 0  # 当前步数

    def reset(self, smtlib_str, seed=None):
        self.t = 0
        # 使用graph_builder从smtlib字符串构建图
        graph = self.graph_builder.build_graph_from_smtlib(smtlib_str)
        # 使用encoder获取节点嵌入
        node_embeddings = self.encoder(graph)
        # 将node_embeddings传递给decoder，以预测变量和常数
        # self.variable_preds, self.constant_preds = self.decoder(node_embeddings)
        self.action_space = DiscreteActionSpace(list(range(self.num_vars * self.num_consts)))
        return [0.0], self.action_space

    def step(self, action):
        # 将动作转换为变量和常数的选择
        var_index = action // self.num_consts
        const_index = action % self.num_consts
        chosen_var = self.variable_preds[var_index]
        chosen_const = self.constant_preds[const_index]

        # 假设model可以使用chosen_var和chosen_const进行计算并返回奖励
        reward = self.model(chosen_var, chosen_const)
        true_reward = np.random.binomial(1, reward)

        self.t += 1
        terminated = self.t >= self.T
        return ActionResult(
            observation=[float(true_reward)],
            reward=float(true_reward),
            terminated=terminated,
            truncated=False,
            info={},
            available_action_space=self.action_space
        )


class ConstraintSimplificationEnv(Environment):

    def __init__(self, encoder, decoder, z3ast, num_variables, num_constants):
        self.decoder = decoder
        self.encoder = encoder
        self.z3ast = z3ast
        self.z3ast_original = z3ast
        self.num_variables = num_variables
        self.num_constants = num_constants
        self.state = None
        self.variables = set()
        self.actions = []
        self.concrete_finish = False
        self.concrete_count = 0
        self.counterexamples_list = []
        self.finish = False
        self.used_variables = []
        # 记录state输入了多少次
        self.state_count = 0
        self.predictor = Predictor('KNN')

    def reset(self, seed=None):
        self.concrete_finish = False
        self.concrete_count = 0
        self.finish = False
        self.used_variables = []
        # 从原始的ast开始构建s

        graph = embedding_util.Z3ASTGraph(self.z3ast_original)
        # node_type_dict = NODE_TYPE_ENUM
        graph2vec = embedding_util.Graph2Vec(graph)
        # 步骤5: 输出转换结果
        print("节点特征向量:")
        print(graph2vec.node_feat.shape)
        # node_embed = embedding_util.glorot_uniform(graph2vec.node_feat)
        node_embed = Parameter(graph2vec.node_feat)
        self.state = self.encoder(node_embed)
        variables = set()
        for a in self.z3ast:
            visit(a, variables)
        self.variables = list(variables)
        # 之后要修改成变量+常量
        # for i in range(len(self.variables)):
        #     self.actions.append(torch.tensor(i))
        #     # 先不使用字典了
        self.actions = self.strings_to_onehot(self.variables)
        self.actions.to(device)
        print('++++++++++++++++++++')
        print(self.actions)
        print(self.actions.shape)
        # self.variables = {index: item for index, item in enumerate(self.variables)}
        self.action_space = DiscreteActionSpace(self.actions)
        return self.state, self.action_space

    def action_space(self):
        """Returns the action space of the environment."""
        pass

    def step(self, action):
        reward = 0
        # variable_pred = self.variables[action]
        action = self.action_space.actions_batch[action]
        print('////////////////////////////')
        print(action)
        print(type(action))
        print(self.onehot_to_indices(action))
        variable_pred = self.variables[self.onehot_to_indices(action)[0]]
        # 在一次执行过程中，action不能重复
        if self.concrete_count == 0:
            self.counterexamples_list.append([])
        if variable_pred not in self.used_variables:
            self.used_variables.append(variable_pred)
            self.concrete_count += 1
            #数值这部分需要修改
            min_int32 = -2147483648
            max_int32 = 2147483647

            # 生成一个随机的32位整数
            random_int = random.randint(min_int32, max_int32)


            self.counterexamples_list[-1].append([variable_pred, random_int])

            solver = Solver()
            for a in self.z3ast:
                solver.add(a)
            exec(f"{variable_pred} = Int('{variable_pred}')")
            # 修改，添加取值部分内容


            solver.add(eval(variable_pred) == random_int)
            reward += self.calculate_reward(solver)
            self.z3ast = solver.assertions()
            graph = embedding_util.Z3ASTGraph(self.z3ast)
            # node_type_dict = NODE_TYPE_ENUM
            graph2vec = embedding_util.Graph2Vec(graph)
            # 步骤5: 输出转换结果
            print("节点特征向量:")
            print(graph2vec.node_feat.shape)
            # node_embed = embedding_util.glorot_uniform(graph2vec.node_feat)
            node_embed = Parameter(graph2vec.node_feat)
            self.state = self.encoder(node_embed)

            if self.concrete_count == len(self.variables):
                self.concrete_finish = True
                self.reset()
                # 判断这里需不需要直接reset
        else:
            reward += -10
            print(action)
            device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
            self.actions = [act.to(device) for act in self.actions]
            print(self.actions)
            for i in self.actions:
                i.to(device)
            action.to(device)
            self.actions = [tensor1 for tensor1 in self.actions if not any(torch.equal(tensor1, tensor2) for tensor2 in action)]
            self.action_space=DiscreteActionSpace(self.actions)
        return ActionResult(
            observation=self.state,
            reward=float(reward),
            terminated=self.concrete_finish,
            truncated=self.finish,
            info={},
            available_action_space=self.action_space, )

    @staticmethod
    def strings_to_onehot(string_list):
        # 创建一个从字符串到索引的映射
        str_to_index = {string: index for index, string in enumerate(string_list)}

        # 创建One-Hot编码的张量
        one_hot_tensors = []
        for string in string_list:
            # 创建一个全0的向量
            one_hot_vector = torch.zeros(len(string_list), dtype=torch.float32)
            # 将对应位置置1
            one_hot_vector[str_to_index[string]] = 1.0
            one_hot_vector.to(device)
            one_hot_tensors.append(one_hot_vector)
        one_hot_matrix = torch.stack(one_hot_tensors)

        return one_hot_matrix
        # return one_hot_tensors

    @staticmethod
    def onehot_to_indices(one_hot_tensors):
        # 将One-Hot编码的张量转换回索引
        return [torch.argmax(tensor).item() for tensor in one_hot_tensors]
    @staticmethod
    def counter_reward_function(total_length, unique_count):

        """
        Calculate the reward based on the total length of the list and the number of unique in it.

        Args:
        - total_length (int): The total length of the list.
        - unique_count (int): The number of unique in the list.

        Returns:
        - float: The calculated reward.
        """
        # Define the base reward values
        R_positive = 1
        R_negative = -1

        # Define the scaling factor for negative reward
        alpha = 1 / math.sqrt(total_length) if total_length > 0 else 1

        # Check if there are any unique strings
        if unique_count > 0:
            # Calculate the positive reward, scaled based on the list length
            reward = R_positive / math.log(1 + total_length)
        else:
            # Apply the negative reward, scaled by alpha
            reward = R_negative * alpha

        return reward

    def calculate_reward(self, solver):
        reward = 0
        count = 0
        solver.set("timeout", 60000)
        # 判断新产生的序列和之前有没有重复
        #判断是否存在反例
        if len(self.counterexamples_list) > 1:
            if self.counterexamples_list[-1] in self.counterexamples_list[:len(self.counterexamples_list) - 1]:
                reward += -1
            else:
                # 判断新的序列和之前是否有重复（字符串重复）
                # for i in range(len(self.counterexamples_list) - 1):
                #     # if self.are_lists_equal(self.counterexamples_list[i],self.counterexamples_list[-1]):
                #     if ' '.join(self.counterexamples_list[-1]) in ' '.join(self.counterexamples_list[i]):
                #         count += 1
                last_joined = ' '.join(
                    ' '.join(str(item) for item in inner_list) for inner_list in self.counterexamples_list[-1])
                for i in range(len(self.counterexamples_list) - 1):
                    current_joined = ' '.join(
                        ' '.join(str(item) for item in inner_list) for inner_list in self.counterexamples_list[i])
                    if last_joined in current_joined:
                        count += 1
                reward += self.counter_reward_function(len(self.counterexamples_list) - 1,
                                                       len(self.counterexamples_list) - 1 - count)
            print(self.counterexamples_list)
        # 后续实现一些子集求解
        query_smt2 = solver.to_smt2()
        # print(query_smt2)
        predicted_solvability = self.predictor.predict(query_smt2)
        if predicted_solvability == 0:
            # 提高一下reward数值
            reward += 2
            r = solver.check()
            stats = solver.statistics()
            if z3.sat == r:

                self.finish = True

                print("求解时间:", stats.get_key_value('time'))
            else:
                # reward += 1 / stats.get_key_value('time') * 100
                reward += -5

        return reward

    def are_lists_equal(self, list1, list2):
        if len(list1) != len(list2):
            return False

        for item1, item2 in zip(list1, list2):
            if item1 != item2:
                return False

        return True

    def render(self) -> None:
        """Renders the environment. Default implementation does nothing."""
        return None

    def close(self) -> None:
        """
        Closes environment, taking care of any cleanup needed.
        Default implementation does nothing.
        """
        return None


class ConstraintSimplificationEnv_v2(Environment):

    def __init__(self, encoder, decoder, z3ast, num_variables, num_constants):
        self.decoder = decoder
        self.encoder = encoder
        self.z3ast = z3ast
        self.z3ast_original = z3ast
        self.num_variables = num_variables
        self.num_constants = num_constants
        self.state = None
        self.variables = set()
        self.actions = []
        self.concrete_finish = False
        self.concrete_count = 0
        self.counterexamples_list = []
        self.finish = False
        self.used_variables = []
        # 记录state输入了多少次
        self.state_count = 0
        self.predictor = Predictor('KNN')

    def reset(self, seed=None):
        self.concrete_finish = False
        self.concrete_count = 0
        self.finish = False
        self.used_variables = []
        # 从原始的ast开始构建s

        graph = embedding_util.Z3ASTGraph(self.z3ast_original)
        # node_type_dict = NODE_TYPE_ENUM
        graph2vec = embedding_util.Graph2Vec(graph)
        # 步骤5: 输出转换结果
        print("节点特征向量:")
        print(graph2vec.node_feat.shape)
        # node_embed = embedding_util.glorot_uniform(graph2vec.node_feat)
        node_embed = Parameter(graph2vec.node_feat)
        self.state = self.encoder(node_embed)
        variables = set()
        for a in self.z3ast:
            visit(a, variables)
        self.variables = list(variables)
        # 之后要修改成变量+常量
        # for i in range(len(self.variables)):
        #     self.actions.append(torch.tensor(i))
        #     # 先不使用字典了
        self.actions = self.strings_to_onehot(self.variables)
        self.actions.to(device)
        print('++++++++++++++++++++')
        print(self.actions)
        print(self.actions.shape)
        # self.variables = {index: item for index, item in enumerate(self.variables)}
        self.action_space = DiscreteActionSpace(self.actions)
        return self.state, self.action_space

    def action_space(self):
        """Returns the action space of the environment."""
        pass

    def step(self, action):
        reward = 0
        # variable_pred = self.variables[action]
        action = self.action_space.actions_batch[action]
        print('////////////////////////////')
        print(action)
        print(type(action))
        print(self.onehot_to_indices(action))
        variable_pred = self.variables[self.onehot_to_indices(action)[0]]
        # 在一次执行过程中，action不能重复
        if self.concrete_count == 0:
            self.counterexamples_list.append([])
        if variable_pred not in self.used_variables:
            self.used_variables.append(variable_pred)
            self.concrete_count += 1
            #数值这部分需要修改
            min_int32 = -2147483648
            max_int32 = 2147483647

            # 生成一个随机的32位整数
            random_int = random.randint(min_int32, max_int32)


            self.counterexamples_list[-1].append([variable_pred, random_int])

            solver = Solver()
            for a in self.z3ast:
                solver.add(a)
            exec(f"{variable_pred} = Int('{variable_pred}')")
            # 修改，添加取值部分内容


            solver.add(eval(variable_pred) == random_int)
            reward += self.calculate_reward(solver)
            self.z3ast = solver.assertions()
            graph = embedding_util.Z3ASTGraph(self.z3ast)
            # node_type_dict = NODE_TYPE_ENUM
            graph2vec = embedding_util.Graph2Vec(graph)
            # 步骤5: 输出转换结果
            print("节点特征向量:")
            print(graph2vec.node_feat.shape)
            # node_embed = embedding_util.glorot_uniform(graph2vec.node_feat)
            node_embed = Parameter(graph2vec.node_feat)
            self.state = self.encoder(node_embed)

            if self.concrete_count == len(self.variables):
                self.concrete_finish = True
                self.reset()
                # 判断这里需不需要直接reset
        else:
            reward += -10
            print(action)
            device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
            self.actions = [act.to(device) for act in self.actions]
            print(self.actions)
            for i in self.actions:
                i.to(device)
            action.to(device)
            self.actions = [tensor1 for tensor1 in self.actions if not any(torch.equal(tensor1, tensor2) for tensor2 in action)]
            self.action_space=DiscreteActionSpace(self.actions)
        return ActionResult(
            observation=self.state,
            reward=float(reward),
            terminated=self.concrete_finish,
            truncated=self.finish,
            info={},
            available_action_space=self.action_space, )

    @staticmethod
    def strings_to_onehot(string_list):
        # 创建一个从字符串到索引的映射
        str_to_index = {string: index for index, string in enumerate(string_list)}

        # 创建One-Hot编码的张量
        one_hot_tensors = []
        for string in string_list:
            # 创建一个全0的向量
            one_hot_vector = torch.zeros(len(string_list), dtype=torch.float32)
            # 将对应位置置1
            one_hot_vector[str_to_index[string]] = 1.0
            one_hot_vector.to(device)
            one_hot_tensors.append(one_hot_vector)
        one_hot_matrix = torch.stack(one_hot_tensors)

        return one_hot_matrix
        # return one_hot_tensors

    @staticmethod
    def onehot_to_indices(one_hot_tensors):
        # 将One-Hot编码的张量转换回索引
        return [torch.argmax(tensor).item() for tensor in one_hot_tensors]
    @staticmethod
    def counter_reward_function(total_length, unique_count):

        """
        Calculate the reward based on the total length of the list and the number of unique in it.

        Args:
        - total_length (int): The total length of the list.
        - unique_count (int): The number of unique in the list.

        Returns:
        - float: The calculated reward.
        """
        # Define the base reward values
        R_positive = 1
        R_negative = -1

        # Define the scaling factor for negative reward
        alpha = 1 / math.sqrt(total_length) if total_length > 0 else 1

        # Check if there are any unique strings
        if unique_count > 0:
            # Calculate the positive reward, scaled based on the list length
            reward = R_positive / math.log(1 + total_length)
        else:
            # Apply the negative reward, scaled by alpha
            reward = R_negative * alpha

        return reward

    def calculate_reward(self, solver):
        reward = 0
        count = 0
        solver.set("timeout", 60000)
        # 判断新产生的序列和之前有没有重复
        #判断是否存在反例
        if len(self.counterexamples_list) > 1:
            if self.counterexamples_list[-1] in self.counterexamples_list[:len(self.counterexamples_list) - 1]:
                reward += -1
            else:
                # 判断新的序列和之前是否有重复（字符串重复）
                # for i in range(len(self.counterexamples_list) - 1):
                #     # if self.are_lists_equal(self.counterexamples_list[i],self.counterexamples_list[-1]):
                #     if ' '.join(self.counterexamples_list[-1]) in ' '.join(self.counterexamples_list[i]):
                #         count += 1
                last_joined = ' '.join(
                    ' '.join(str(item) for item in inner_list) for inner_list in self.counterexamples_list[-1])
                for i in range(len(self.counterexamples_list) - 1):
                    current_joined = ' '.join(
                        ' '.join(str(item) for item in inner_list) for inner_list in self.counterexamples_list[i])
                    if last_joined in current_joined:
                        count += 1
                reward += self.counter_reward_function(len(self.counterexamples_list) - 1,
                                                       len(self.counterexamples_list) - 1 - count)
            print(self.counterexamples_list)
        # 后续实现一些子集求解
        query_smt2 = solver.to_smt2()
        # print(query_smt2)
        predicted_solvability = self.predictor.predict(query_smt2)
        if predicted_solvability == 0:
            # 提高一下reward数值
            reward += 2
            r = solver.check()
            stats = solver.statistics()
            if z3.sat == r:

                self.finish = True

                print("求解时间:", stats.get_key_value('time'))
            else:
                # reward += 1 / stats.get_key_value('time') * 100
                reward += -5

        return reward

    def are_lists_equal(self, list1, list2):
        if len(list1) != len(list2):
            return False

        for item1, item2 in zip(list1, list2):
            if item1 != item2:
                return False

        return True

    def render(self) -> None:
        """Renders the environment. Default implementation does nothing."""
        return None

    def close(self) -> None:
        """
        Closes environment, taking care of any cleanup needed.
        Default implementation does nothing.
        """
        return None


def visit(expr, variables):
    if is_const(expr) and expr.decl().kind() == Z3_OP_UNINTERPRETED:
        # Add only uninterpreted functions (which represent variables)
        # print(type(self.variables))
        variables.add(str(expr))
    else:
        # Recursively visit children for composite expressions
        for child in expr.children():
            visit(child, variables)
