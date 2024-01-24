from __future__ import print_function
import tokenize
import io
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
from tqdm import tqdm
import importlib
# import sys
# sys.path.append('/home/yy/SMTimer')
from code2inv.SMTimer.KNN_Predictor import Predictor
from z3 import *

from code2inv.common.cmd_args import cmd_args, toc
import random
from collections import Counter
import z3
import sys
import time
import numpy as np
from tqdm import tqdm
import os
import importlib
checker_module = importlib.import_module(cmd_args.inv_checker)

from code2inv.common.ssa_graph_builder import ProgramGraph, GraphNode, ExprNode
from code2inv.common.constants import *
from code2inv.common.cmd_args import cmd_args
from code2inv.common.checker import boogie_result, z3_precheck, z3_precheck_expensive, stat_counter
from code2inv.prog_generator.tree_decoder import genExprTree, GeneralDecoder, InvariantTreeNode, fully_expanded_node
checker_module = importlib.import_module(cmd_args.inv_checker)

class RLEnv(object):
    def __init__(self, s2v_graph, decoder):
        self.s2v_graph = s2v_graph
        self.pg = s2v_graph.pg
        self.decoder = decoder
        #ji lu fanli
        self.cexs = []
        self.reset()


    def reset(self):
        try:
            self.concrete_finish = False
            self.concrete_count = 0
            # new struct to store concrete var and value
            #self.root_value = None
            # the root node to store constraint
            self.root = None
            self.inv_candidate = None
            self.terminal = False
            self.trivial_subexpr = False
            self.expr_in_and = set()
            self.expr_in_or = set()
            self.used_core_vars = set()
        except RecursionError:
            print("ERROR- Non Terminating grammar found")
            exit(-1)

    def num_vars(self):
        return len(self.pg.raw_variable_nodes)

    def num_consts(self):
        return len(self.pg.const_nodes)

    def pg_nodes(self):
        return self.pg.node_list

    def update_used_core_var(self, var_node):
        if var_node.name in self.pg.core_vars:
            self.used_core_vars.add(var_node.name)

    def constraint_satisfied(self):
        #possible need to edit
        return len(self.used_core_vars) == len(self.pg.core_vars)

    def insert_subexpr(self, node, subexpr_node, eps = 0.05):
        if node is None:
            node = genExprTree(RULESET, "S")
            return self.insert_subexpr(node, subexpr_node)
        elif node.name == "" and node.rule == "p" and node.expanded is None:
            node = subexpr_node
            node.expanded = True
            return node, True
        elif node.name == "" and node.rule == "p" and node.expanded == True:
            return node, False
        elif node.rule is not None and len(node.children) == 0:
            w = nn.Linear(cmd_args.embedding_size, len(RULESET[node.rule]))
            logits = w(self.decoder.latent_state)
            
            ll = F.log_softmax(logits, dim=1)

            if self.use_random:
                scores = torch.exp(ll) * (1 - eps) + eps / ll.shape[1]
                picked = torch.multinomial(scores, 1)
            else:            
                _, picked = torch.max(ll, 1)
            picked = picked.view(-1)        

            node = genExprTree(RULESET, node.rule, picked)
            return self.insert_subexpr(node, subexpr_node)

        elif len(node.children) > 0:
            last_junct = ""
            for i in range(len(node.children)):
                node.children[i], node_update = self.insert_subexpr(node.children[i], subexpr_node)
                if node_update:
                    return node, node_update
            return node, False
        else:
            return node, False


    def step(self, subexpr_node, node_embedding, use_random, eps,cex):
        print('**************************')
        print(subexpr_node.expr_str())
        print(subexpr_node.rule)
        print(self.used_core_vars)
        print(self.pg.raw_variable_nodes)
        print('++++++++++++++++++')
        self.use_random = use_random
        reward = 0.0        
        self.inv_candidate, updated = self.insert_subexpr(self.inv_candidate, subexpr_node)
        self.root = self.inv_candidate.clone_expanded()

        self.terminal = fully_expanded_node(self.inv_candidate)
        self.root.state = None
        # if self.inv_candidate.check_rep_pred():
        #     self.trivial_subexpr = True
        # else:
        #     try:
        #         if checker_module.is_trivial(cmd_args.input_vcs, str(subexpr_node)):
        #             self.trivial_subexpr = True
        #         else:
        #             reward += 0.5
        #     except Exception as e:
        #         reward += 0.5
        #
        # if self.trivial_subexpr:
        #     reward += -2.0
        #     self.terminal = True

        if self.terminal:
            if not self.concrete_finish: #self.constraint_satisfied():
                try:
                    # r = boogie_result(self.s2v_graph, self.root)
                    r,cex = self.ce_check(cex)
                    reward += r
                except Exception as e:
                    if str(e) == "Not implemented yet":
                        raise e
                    reward += -6.0
            else:
                reward += -4.0
        return reward, self.trivial_subexpr, cex

    def condense(inv_tokens):
        op_list = ["+", "-", "*", "/", "%", "<", "<=", ">", ">=", "==", "!=", "and", "or"]
        un_op_list = ["+", "-"]
        old_list = list(inv_tokens)
        new_list = list(inv_tokens)
        while True:
            for idx in range(len(old_list)):
                if old_list[idx] in un_op_list:
                    if idx == 0 or old_list[idx - 1] in op_list or old_list[idx - 1] == "(":
                        new_list[idx] = old_list[idx] + old_list[idx + 1]
                        new_list[idx + 1:] = old_list[idx + 2:]
                        break
            if old_list == new_list:
                break
            else:
                old_list = list(new_list)
        return new_list

    def infix_postfix(infix_token_list):
        opStack = []
        postfix = []
        opStack.append("(")
        infix_token_list.append(")")

        for t in infix_token_list:
            # print(t)
            if t not in p:
                postfix.append(t)
            elif t == "(":
                opStack.append(t)
            elif t == ")":
                while opStack[-1] != "(":
                    postfix.append(opStack.pop(-1))
                opStack.pop(-1)
            elif t in p:
                while len(opStack) > 0 and p[opStack[-1]] >= p[t]:
                    postfix.append(opStack.pop(-1))
                opStack.append(t)
            # print(postfix, opStack)
        return postfix

    def postfix_prefix(postfix_token_list):
        stack = []
        for t in postfix_token_list:
            if t not in p:
                stack.append(t)
            else:
                sub_stack = []
                sub_stack.append("(")
                sub_stack.append(t)
                op1 = stack.pop(-1)
                op2 = stack.pop(-1)
                sub_stack.append(op2)
                sub_stack.append(op1)

                sub_stack.append(")")
                stack.append(sub_stack)
        return stack

    def stringify_prefix_stack(prefix_stack):
        s = ""
        for e in prefix_stack:
            if type(e) == list:
                s += stringify_prefix_stack(e)
            else:
                s += e + " "
        return s

    # to be modify que ren
    # @property
    def ce_check(self, cex):
        reward = 0
        if not self.concrete_finish:
            self.concrete_count += 1
        # if '&&' in str(self.root):
        #     tmp = str(self.root).rsplit('&&')[-1]
        #     # for tmp_var in tmp:
        #     tmp_var = tmp
        #     tmp_var = tmp_var.replace('(', '').replace(')', '').strip()
        #     tmp_var = tmp_var.split('==')
        #     tmp_var = [item.strip() for item in tmp_var]
        #     self.cexs.append([tmp_var])
        # else:
        # file_path = '/home/yy/constraiant/constraint.txt'
        # with open(file_path, 'r', encoding='utf-8') as file:
        #     cons = file.read()
        # cons = cons.replace("&&", "and", -1)
        # cons = cons.replace("||", "or", -1)
        # b = io.StringIO(cons)
        # t = tokenize.generate_tokens(b.readline)
        # cons_tokenized = []
        # for a in t:
        #     if a.string != "":
        #         cons_tokenized.append(a.string)
        # cons = stringify_prefix_stack(postfix_prefix(infix_postfix(condense(cons_tokenized))))
        # cons = cons.replace("==", "=", -1)
        #jilu z3 bianliang
        vars = {}
        # print(type(self.used_core_vars[0]))
        # 创建求解器实例
        solver = Solver()
        # for name in self.pg.core_vars:
        # # 创建求解器实例:
        #     # 使用 exec 执行动态代码，创建变量
        #     exec(f"{name} = Int('{name}')")
        #     vars[name] = eval(name)
        vars = {var_name: Int(var_name) for var_name in self.pg.core_vars}


        #向求解器添加约束条件
        x = Int('x')
        y = Int('y')
        z = Int('z')
        solver.add(((x * 37) % 100) + y == z)
        solver.add(x ** 2 + y ** 2 > 50)
        solver.add(If(x > y, z ** 2 == 16, z ** 2 < 10))
        solver.add(If(((z * 37) % 100) > 30, y ** 2 == x + 10, y ** 2 == x - 5))
        solver.add(x * y - z != 0)

        solver.set(auto_config=False)
        res = []
        #pan duan hai xu yao you hua
        print('999999999999999999999999999999999999999999999')
        predictor = Predictor('KNN')
        print(type(predictor))
        print(type(self.root))
        if not self.concrete_finish:
            if '&&' in str(self.root):
                tmp = str(self.root).rsplit('&&')[-1]
                # for tmp_var in tmp:
                tmp_var = tmp
                tmp_var = tmp_var.replace('(', '').replace(')', '').strip()
                tmp_var = tmp_var.split('==')
                tmp_var = [item.strip() for item in tmp_var]
                cex.append([tmp_var])
            else:
                tmp_var = str(self.root).replace('(', '').replace(')', '').strip().split('==')
                tmp_var = [item.strip() for item in tmp_var]
                # cex = list(set(cex))
            # for i in cex:
            #     if [tmp_var].__eq__(i):
            #         return -10
            cex.append([tmp_var])
            try:
                # if predicted_solvability == 0:

                # shiyongz3zhijieqiujie
                solver.set("timeout", 600000)
                # print(cex[-1][0]+'=='+cex[-1][1])
                solver.add(vars[cex[-1][0][0]] == int(cex[-1][0][1]))
                #zhuan huan ge shi
                query_smt2 = solver.to_smt2()
                print(query_smt2)
                predicted_solvability = predictor.predict(query_smt2)
                print(predicted_solvability)
                # r = solver.check()
                # if not z3.unknown == r:
                if predicted_solvability == 0:
                    reward += 1
                    r = solver.check()
                    stats = solver.statistics()
                    if z3.sat == r:
                        cex.pop()
                        self.concrete_finish = True

                        print("求解时间:", stats.get_key_value('time'))
                    else:
                        # reward += 1 / stats.get_key_value('time') * 100
                        reward += -20
                else:
                    reward += -2
            except Z3Exception as e:
                print("Z3求解器发生异常：", e)
        # cex.append(ce)
        print(cex)
        return reward, cex
    def available_var_indices(self, list_vars):
        list_indices = []
        if list_vars and len(list_vars) > 0:
            for var in self.pg.raw_variable_nodes:
                if var in list_vars:
                    list_indices.append(self.pg.raw_variable_nodes[var].index)
            list_indices.sort()

            if len(list_indices) == 0:
                list_indices = list(range(len(self.pg.raw_variable_nodes)))

            return list_indices
        else:
            return list(range(len(self.pg.raw_variable_nodes)))

    def is_finished(self):
        return self.terminal

def rollout(g, node_embedding, decoder, use_random, eps, cex):
    nll_list = []
    value_list = []
    reward_list = []
    trivial = False
    env = RLEnv(g, decoder)
    while not env.is_finished():
        try:
            and_or, subexpr_node, nll, vs, latent_state = decoder(env, node_embedding, use_random=use_random, eps = eps)

            reward, trivial,cex = env.step(subexpr_node, node_embedding, use_random, eps, cex)
            nll_list.append(nll)
            value_list.append(vs)
            
            root = env.root
            reward_list.append(reward)
        except Exception as e:
            # print("EXCEPTION", e)
            nll_list.append(decoder.nll)
            value_list.append(decoder.est)
            reward_list.append(-6.0)
            pass

    if not env.trivial_subexpr:
        if cmd_args.decoder_model == 'AssertAware':            
            assert env.constraint_satisfied()

    return nll_list, value_list, reward_list, root, cex, trivial

def actor_critic_loss(nll_list, value_list, reward_list):
    r = 0.0
    rewards = []
    for t in range(len(reward_list) - 1, -1, -1):
        r = r + reward_list[t] # accumulated future reward
        rewards.insert(0, r / 10.0)
            
    policy_loss = 0.0
    targets = []
    for t in range(len(reward_list)):        
        reward = rewards[t] - value_list[t].data[0, 0]                
        policy_loss += nll_list[t] * reward
        targets.append(Variable(torch.Tensor([ [rewards[t]] ])))

    policy_loss /= len(reward_list)

    value_pred = torch.cat(value_list, dim=0)
    targets = torch.cat(targets, dim=0)    
    value_loss = F.mse_loss(value_pred, targets)

    loss = policy_loss + value_loss

    return loss
