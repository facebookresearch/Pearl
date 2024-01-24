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

from code2inv.common.ssa_graph_builder import ProgramGraph, GraphNode, ExprNode, checkallnone
from code2inv.common.constants import *
from code2inv.common.cmd_args import cmd_args
from code2inv.common.pytorch_util import weights_init, to_num

from code2inv.prog_generator.prog_encoder import LogicEncoder


class IDecoder(nn.Module):
    def __init__(self, latent_dim):
        super(IDecoder, self).__init__()
        self.latent_dim = latent_dim

        self.decision = nn.Linear(latent_dim, 3)

        self.state_gru = nn.GRUCell(latent_dim, latent_dim)

        self.and_embedding = Parameter(torch.Tensor(1, latent_dim))
        self.or_embedding = Parameter(torch.Tensor(1, latent_dim))

        self.value_pred_w1 = nn.Linear(self.latent_dim, self.latent_dim)
        self.value_pred_w2 = nn.Linear(self.latent_dim, 1)

        if cmd_args.attention:
            self.first_att = nn.Linear(self.latent_dim, 1)

        weights_init(self)

    def choose_action(self, state, cls_w, use_random, eps):
        if type(cls_w) is Variable or type(cls_w) is Parameter or type(cls_w) is torch.Tensor:
            logits = F.linear(state, cls_w, None)
        elif type(cls_w) is torch.nn.modules.linear.Linear:
            logits = cls_w(state)
        else:
            raise NotImplementedError()
        # print(state)
        # print(cls_w)
        # print(logits)
        ll = F.log_softmax(logits, dim=1)

        if use_random:
            scores = torch.exp(ll) * (1 - eps) + eps / ll.shape[1]
            picked = torch.multinomial(scores, 1)
        else:
            _, picked = torch.max(ll, 1)
        picked = picked.view(-1)
        self.nll += F.nll_loss(ll, picked)
        return picked

    def update_state(self, input_embedding):
        self.latent_state = self.state_gru(input_embedding, self.latent_state)

    # 直接编码即可，只有一个node
    def direct_decode(self, env, use_random, eps):
        left_node = self.choose_operand(env, self.variable_embedding, use_random, eps)
        right_node = self.choose_operand(env, self.var_const_embedding, use_random, eps)
        # 修改语法规则，这里应该得到的是变量=常量这样的结果
        w = self.token_w[0: len(LIST_PREDICATES), :]
        # 好像是操作符？这里应该需要进行修改
        picked = self.choose_action(self.latent_state, w, use_random, eps)
        self.update_state(self.char_embedding(picked))
        cur_node = ExprNode(LIST_PREDICATES[picked.data.cpu()[0]])
        cur_node.children.append(left_node)
        cur_node.children.append(right_node)

    def recursive_decode(self, pg_node_list, lv, use_random, eps):
        if lv == 0:  # subtree expression root node, decode the first symbol
            # left child, which is a variable
            picked = self.choose_action(self.latent_state, self.variable_embedding, use_random, eps)
            self.update_state(torch.index_select(self.variable_embedding, 0, picked))
            left_node = ExprNode(pg_node_list[picked.data.cpu()[0]])

            # recursively construct right child
            right_node = self.recursive_decode(pg_node_list, 1, use_random, eps)

            # root
            w = self.token_w[0: len(LIST_PREDICATES), :]
            picked = self.choose_action(self.latent_state, w, use_random, eps)
            self.update_state(self.char_embedding(picked))
            cur_node = ExprNode(LIST_PREDICATES[picked.data.cpu()[0]])
            cur_node.children.append(left_node)
            cur_node.children.append(right_node)
        else:
            if lv < MAX_DEPTH:
                w_op = self.token_w[len(LIST_PREDICATES):, :]
                classifier = torch.cat((w_op, self.var_const_embedding), dim=0)
                picked = self.choose_action(self.latent_state, classifier, use_random, eps)
                self.update_state(torch.index_select(classifier, 0, picked))

                idx = picked.data.cpu()[0]
                if idx < len(LIST_OP):  # it is an op node
                    cur_node = ExprNode(LIST_OP[idx])
                    # assert binary operator
                    cur_node.children.append(self.recursive_decode(pg_node_list, lv + 1, use_random, eps))
                    cur_node.children.append(self.recursive_decode(pg_node_list, lv + 1, use_random, eps))
                else:
                    cur_node = ExprNode(pg_node_list[idx - len(LIST_OP)])
            else:  # can only choose variable or const
                picked = self.choose_action(self.latent_state, self.var_const_embedding, use_random, eps)
                self.update_state(torch.index_select(self.var_const_embedding, 0, picked))

                cur_node = ExprNode(pg_node_list[picked.data.cpu()[0]])

        return cur_node

    def embed_tree(self, node_embedding, root):
        raise NotImplementedError()

    def forward(self, env, node_embedding, use_random, eps=0.05):
        state_embedding = self.embed_tree(node_embedding, env.root)
        self.latent_state = state_embedding
        est = self.value_pred_w2(F.relu(self.value_pred_w1(state_embedding)))

        self.nll = 0.0
        subexpr_node = None
        if env.root is None:
            act = 1
        else:
            first_decision = self.choose_action(self.latent_state, self.decision, use_random, eps)
            act = first_decision.data.cpu()[0]
            if act == 0 or (act == 1 and env.and_budget() == 0) or (act == 2 and env.or_budget() == 0):
                return None, subexpr_node, self.nll, est

        self.variable_embedding = node_embedding[0: env.num_vars(), :]
        self.var_const_embedding = node_embedding[0: env.num_vars() + env.num_consts(), :]

        if act == 1:
            self.update_state(self.and_embedding)

        self.update_state(self.or_embedding)
        subexpr_node = self.recursive_decode(env.pg_nodes(), 0, use_random, eps)

        if act == 1:
            and_or = '&&'
        else:
            and_or = '||'

        return and_or, subexpr_node, self.nll, est

    def get_init_embedding(self, node_embedding, state):
        if cmd_args.attention:
            allnone = True

            if type(state) == ExprNode:
                allnone = checkallnone(state)

            if state is None or allnone:
                logits = self.first_att(node_embedding)
            else:
                logits = torch.sum(node_embedding * self.latent_state, dim=1, keepdim=True)
            weights = F.softmax(logits, dim=0)
            init_embedding = torch.sum(weights * node_embedding, dim=0, keepdim=True)
        else:
            init_embedding = torch.mean(node_embedding, dim=0, keepdim=True)
        return init_embedding


class CFGTreeDecoder(IDecoder):
    def __init__(self, latent_dim):
        super(CFGTreeDecoder, self).__init__(latent_dim)

        self.tree_embed = LogicEncoder(self.latent_dim)

    def embed_tree(self, node_embedding, root):
        init_embedding = self.get_init_embedding(node_embedding, root)
        return self.tree_embed(node_embedding, init_embedding, root)


class CFGRNNDecoder(IDecoder):
    def __init__(self, latent_dim):
        super(CFGRNNDecoder, self).__init__(latent_dim)

    def embed_tree(self, node_embedding, root):
        init_embedding = self.get_init_embedding(node_embedding, root)

        if root is None:
            return init_embedding
        else:
            return self.latent_state + init_embedding


class AssertAwareDecoder(IDecoder):
    def __init__(self, latent_dim):
        super(AssertAwareDecoder, self).__init__(latent_dim)
        self.tree_grow_decision = nn.Linear(latent_dim, 2)

        weights_init(self)

    def count_var_leaves(self, env, expr):
        if len(expr.children) == 0:
            if expr.name in env.pg.raw_variable_nodes:
                return 1
            return 0
        cnt = 0
        for c in expr.children:
            cnt += self.count_var_leaves(env, c)
        return cnt

    def choose_operand(self, env, node_embedding, use_random, eps):
        var_list = env.available_var_indices([])
        if len(var_list) == env.num_vars():  # can freely choose any variable
            selector = node_embedding
        else:  # have to choose from core variables
            selector = node_embedding[var_list, :]

        picked = self.choose_action(self.latent_state, selector, use_random, eps)
        self.update_state(torch.index_select(selector, 0, picked))

        idx = to_num(picked)
        if idx < len(var_list):  # otherwise, we have chosen a constant
            idx = var_list[idx]
            env.update_used_core_var(env.pg_nodes()[idx])
        return ExprNode(env.pg_nodes()[idx])

    def recursive_decode(self, env, lv, use_random, eps):
        if lv == 0:  # subtree expression root node, decode the first symbol
            # left child, which is a variable
            left_node = self.choose_operand(env, self.variable_embedding, use_random, eps)

            self.cur_token_used += 1  # occupy one token slot

            # recursively construct right child
            right_node = self.recursive_decode(env, 1, use_random, eps)

            assert self.cur_token_used == 2 ** (MAX_DEPTH - 1) + 1
            # root            
            w = self.token_w[0: len(LIST_PREDICATES), :]
            picked = self.choose_action(self.latent_state, w, use_random, eps)
            self.update_state(self.char_embedding(picked))
            cur_node = ExprNode(LIST_PREDICATES[picked.data.cpu()[0]])
            cur_node.children.append(left_node)
            cur_node.children.append(right_node)
        else:
            if lv < MAX_DEPTH:
                # can I just choose a variable, instead of an operator? 
                classifier = None
                tmp = 2 ** (MAX_DEPTH - lv)
                if env.core_var_budget(self.cur_token_used + tmp - 1) < 0:
                    decision = 0
                else:
                    sp = self.choose_action(self.latent_state, self.tree_grow_decision, use_random, eps)
                    decision = to_num(sp)

                if decision == 0:  # op node
                    classifier = self.token_w[len(LIST_PREDICATES):, :]
                    picked = self.choose_action(self.latent_state, classifier, use_random, eps)
                    self.update_state(torch.index_select(classifier, 0, picked))
                    idx = to_num(picked)
                    cur_node = ExprNode(LIST_OP[idx])
                    cur_node.children.append(self.recursive_decode(env, lv + 1, use_random, eps))
                    cur_node.children.append(self.recursive_decode(env, lv + 1, use_random, eps))
                else:  # leaf const/var node
                    self.cur_token_used += 2 ** (MAX_DEPTH - lv) - 1
                    cur_node = self.choose_operand(env, self.var_const_embedding, use_random, eps)
                    self.cur_token_used += 1
            else:  # can only choose variable or const
                cur_node = self.choose_operand(env, self.var_const_embedding, use_random, eps)

                self.cur_token_used += 1  # it is a leaf

        return cur_node

    def forward(self, env, node_embedding, use_random, eps=0.05):
        state_embedding = self.embed_tree(node_embedding, env.root)
        self.latent_state = state_embedding
        est = self.value_pred_w2(F.relu(self.value_pred_w1(state_embedding)))

        self.nll = 0.0
        self.cur_token_used = 0
        subexpr_node = None

        self.variable_embedding = node_embedding[0: env.num_vars(), :]
        self.var_const_embedding = node_embedding[0: env.num_vars() + env.num_consts(), :]

        self.update_state(self.or_embedding)

        subexpr_node = self.recursive_decode(env, 0, use_random, eps)

        return None, subexpr_node, self.nll, est, self.latent_state


class AssertAwareTreeLSTMDecoder(AssertAwareDecoder):
    def __init__(self, latent_dim):
        super(AssertAwareTreeLSTMDecoder, self).__init__(latent_dim)

        self.tree_embed = LogicEncoder(self.latent_dim)

    def embed_tree(self, node_embedding, root):
        init_embedding = self.get_init_embedding(node_embedding, root)
        return self.tree_embed(node_embedding, init_embedding, root)


class AssertAwareRNNDecoder(AssertAwareDecoder):
    def __init__(self, latent_dim):
        super(AssertAwareRNNDecoder, self).__init__(latent_dim)

    def embed_tree(self, node_embedding, root):
        init_embedding = self.get_init_embedding(node_embedding, root)
        if root is None:
            return init_embedding
        else:
            return self.latent_state + init_embedding


precedence_order = [("(", ")"), Z3_OP, Z3_CMP, "||", "&&"]


def lowerPrecedence(op1, op2):
    # returns if op1 is lower in precedence than op2
    if op1 == op2:
        return False
    else:
        for elem in precedence_order:
            if type(elem) == tuple:
                for e in elem:
                    if op1 == e:
                        return True
                    elif op2 == e:
                        return False
            elif op1 == elem:
                return True
            elif op2 == elem:
                return False
    return True


def genExprTree(ruleset: dict, rule: str, ind=-1):
    rules = ruleset[rule]
    if len(rules) > 1 and ind == -1:
        return InvariantTreeNode("", rule)
    elif len(rules) == 1 and ind == -1 and rule == "var":
        return InvariantTreeNode("", rule)
    elif len(rules) == 1:
        ind = 0

    assert ind < len(rules)

    node = InvariantTreeNode("", rule)

    if len(rules[ind]) == 1 and rules[ind][0] not in ruleset:
        node.name = rules[ind][0]
        return node

    elem_list = []

    for element in rules[ind]:
        if element in ruleset:
            elem_list.append(genExprTree(ruleset, element))
        else:
            elem_list.append(InvariantTreeNode(element))

    node.children = elem_list

    if node.name == "" and len(node.children) == 1:
        node = node.children[0]

    return node


class InvariantTreeNode(ExprNode):
    def __init__(self, pg_node, rule=None):
        super(InvariantTreeNode, self).__init__(pg_node)
        self.rule = rule
        self.expanded = None

    def clone_expanded(self):
        node = InvariantTreeNode(self.pg_node, self.rule)
        node.name = self.name
        node.expanded = self.expanded
        for child in self.children:
            if child.name == "" and len(child.children) == 0:
                continue
            elif child.name == "var" or child.name == "const":
                continue
            else:
                x = child.clone_expanded()
                if x.name == "" and len(x.children) == 0:
                    pass
                else:
                    node.children.append(x)
        return node

    def __repr__(self):
        if self.name == "" and self.rule is not None:
            if len(self.children) == 0:
                return "(UNEXP RULE " + self.rule + ")"
            else:
                return " ".join(["(" + str(x) + ")" if len(x.children) > 0 else str(x) for x in self.children])
        return super(InvariantTreeNode, self).__repr__()

    def __str__(self):
        if self.name == "" and self.rule is not None:
            if len(self.children) == 0:
                return "(UNEXP RULE " + self.rule + ")"
            else:
                return " ".join([str(x) if len(x.children) > 0 else str(x) for x in self.children])
        return super(InvariantTreeNode, self).__str__()

    def expr_str(self):
        if self.name == "" and self.rule is not None:
            if len(self.children) != 0:
                tmp_list = [" ( " + x.expr_str() + " ) " if len(x.children) > 0 and x.expr_str() != "" else x.expr_str()
                            for x in self.children]
                expr_list = []
                removed_bin = False
                for i in range(len(tmp_list) - 1, -1, -1):
                    if tmp_list[i] == "":
                        pass
                    elif removed_bin:
                        removed_bin = False
                    elif tmp_list[i] == "const" or tmp_list[i] == "var":
                        pass
                    elif tmp_list[i] in ("&&", "||") and all(x == ")" for x in expr_list):
                        pass
                    elif (tmp_list[i] in Z3_OP or tmp_list[i] in Z3_CMP) and all(x == ")" for x in expr_list):
                        removed_bin = True
                    elif tmp_list[i] == "(" and expr_list[0] == ")":
                        expr_list.remove(")")
                    else:
                        expr_list.insert(0, tmp_list[i])
                return " ".join(expr_list)
            else:
                return ""
        return self.name

    def to_expr_node(self, program_graph: ProgramGraph):
        expr_str = self.expr_str().split()
        infix = []
        for p in expr_str:
            if p in Z3_CMP or p in Z3_OP or p in (
            "&&", "||") or p in program_graph.const_nodes or p in program_graph.raw_variable_nodes:
                infix.append(p)
            else:
                infix = infix + list(p)
        postfixexpr = []
        stack = ["("]

        infix.append(")")
        for element in infix:
            if element == "(":
                stack.append(element)
            elif element == ")":
                while stack[-1] != "(":
                    postfixexpr.append(stack.pop())
                stack.pop()
            else:
                while lowerPrecedence(element, stack[-1]) and stack[-1] != "(":
                    postfixexpr.append(stack.pop())
                stack.append(element)
        stack = []
        try:
            for element in postfixexpr:
                if element in Z3_CMP or element in Z3_OP or element in ("&&", "||"):
                    n = ExprNode(element)
                    a1 = stack.pop()
                    a2 = stack.pop()
                    n.children = [a2, a1]
                    stack.append(n)
                else:
                    n = ExprNode(element)
                    for node in program_graph.node_list:
                        if node.name == element:
                            n.pg_node = node
                            break
                    stack.append(n)
            return stack[0]
        except Exception as e:
            return None

    def complete_expr(self):
        if self.name == "" and len(self.children) == 0:
            return False
        elif self.name == "const" or self.name == "var":
            return False
        else:
            for child in self.children:
                if not child.complete_expr():
                    return False
            return True

    def check_rep_pred(self):
        if self.name == "":
            if self.rule == "p":
                return False
            elif len(self.children) == 0:
                return False
            else:
                pred_children = []
                for child in self.children:
                    if child.rule is None or child.rule == "":
                        continue
                    elif child.rule == "p" and len(child.children) > 0:
                        pred_children.append(child)
                    elif child.rule != "p":
                        if child.check_rep_pred():
                            return True
                for i in range(len(pred_children)):
                    for j in range(i + 1, len(pred_children)):
                        if str(pred_children[i]) == str(pred_children[j]):
                            return True
                return False
        else:
            return False


def fully_expanded_node(node):
    if node.name == "" and node.rule is not None:
        if node.rule == "p":
            if node.expanded is not None:
                return True
            else:
                return False
        elif len(node.children) > 0:
            exp = True
            for i in node.children:
                exp = exp and fully_expanded_node(i)
            return exp
        else:
            return False
    else:
        return True


class GeneralDecoder(AssertAwareDecoder):
    def embed_tree(self, node_embedding, root):
        init_embedding = self.get_init_embedding(node_embedding, root)
        return self.tree_embed(node_embedding, init_embedding, root)

    def __init__(self, latent_dim):
        super(GeneralDecoder, self).__init__(latent_dim)

        self.start = "p"
        self.ruleset = RULESET

        self.top_act_w = Parameter(torch.Tensor(sum([len(self.ruleset[rule]) for rule in self.ruleset]), latent_dim))
        self.char_embedding = nn.Embedding(sum([len(self.ruleset[rule]) for rule in self.ruleset]), latent_dim)
        self.tree_embed = LogicEncoder(self.latent_dim)
        weights_init(self)

        self.root = genExprTree(self.ruleset, "p")

    def forward(self, env, node_embedding, use_random, eps=0.05):
        self.root = genExprTree(self.ruleset, "p")
        state_embedding = self.embed_tree(node_embedding, env.root)
        self.latent_state = state_embedding
        self.est = self.value_pred_w2(F.relu(self.value_pred_w1(state_embedding)))

        self.nll = 0.0
        self.cur_token_used = 0
        subexpr_node = None

        self.variable_embedding = node_embedding[0: env.num_vars(), :]
        self.var_const_embedding = node_embedding[0: env.num_vars() + env.num_consts(), :]
        self.const_embedding = node_embedding[env.num_vars(): env.num_vars() + env.num_consts(), :]

        self.root = self.updateTreeWRTNode(env, self.root, node_embedding, use_random, eps)

        return None, self.root, self.nll, self.est, self.latent_state

    def updateTreeWRTNode(self, env, node, node_embedding, use_random, eps, depth=0):
        if depth == MAX_DEPTH:
            raise Exception("MAX DEPTH REACHED")
        if node.name == "" and node.rule is not None and len(node.children) == 0 and node.rule != "var":
            s = 0
            for rule in self.ruleset:
                if rule == node.rule:
                    w = self.top_act_w[s: s + len(self.ruleset[rule]), :]
                    break

                s += len(self.ruleset[rule])

            ind = self.choose_action(self.latent_state, w, use_random, eps)

            embedidx = 0
            for rule in self.ruleset:
                if rule != node.rule:
                    embedidx += len(self.ruleset[rule])
                else:
                    break

            embedidx += ind

            node = genExprTree(self.ruleset, node.rule, ind)
            self.update_state(self.char_embedding(embedidx))

            if node.name in ("", "var", "const"):
                return self.updateTreeWRTNode(env, node, node_embedding, use_random, eps, depth + 1)
            else:
                return node

        elif node.name == "" and node.rule is not None and node.rule == "var":
            var_list = [var[0] for var in self.ruleset["var"]]

            node = self.choose_var(env, self.variable_embedding, use_random, eps, var_list)
            node.rule = "var"
            return node
        elif node.name == "var":
            var_list = []
            node = self.choose_var(env, self.variable_embedding, use_random, eps, var_list)
            node.rule = "var"
            return node
        elif (node.name == "" and node.rule is not None):
            total_update = False
            for i in range(len(node.children)):
                if node.children[i].name == "const" or node.children[i].name == "var" or (
                        node.children[i].name == "" and node.children[i].rule == "var"):
                    node.children[i] = self.updateTreeWRTNode(env, node.children[i], node_embedding, use_random, eps,
                                                              depth + 1)
                elif node.children[i].rule is not None:
                    node.children[i] = self.updateTreeWRTNode(env, node.children[i], node_embedding, use_random, eps,
                                                              depth + 1)

            return node
        elif node.name == "const":
            const_picked = env.num_vars() + self.choose_action(self.latent_state, self.const_embedding, True, 0.05)
            self.update_state(torch.index_select(self.var_const_embedding, 0, const_picked))
            node = InvariantTreeNode(env.pg_nodes()[const_picked], "const")
            return node
        elif node.name == "" and node.rule is not None and node.rule == "const":
            const_indices = []
            const_choices = [c[0] for c in self.ruleset["const"]]
            for const in env.pg.const_nodes:
                if const in const_choices:
                    const_indices.append(env.pg.const_nodes[const])
            if len(const_indices) > 0:
                selector = self.const_embedding[const_indices, :]
            else:
                selector = self.const_embedding

            const_picked = env.num_vars() + self.choose_action(self.latent_state, self.const_embedding, True, 0.05)
            self.update_state(torch.index_select(self.var_const_embedding, 0, const_picked))
            node = InvariantTreeNode(env.pg_nodes()[const_picked], "const")
            return node
        else:
            return node

    def choose_var(self, env, node_embedding, use_random, eps, var_list):
        var_list = env.available_var_indices(var_list)
        selector = node_embedding[var_list, :]

        picked = self.choose_action(self.latent_state, selector, use_random, eps)
        self.update_state(torch.index_select(selector, 0, picked))

        idx = to_num(picked)
        idx = var_list[idx]
        env.update_used_core_var(env.pg_nodes()[idx])
        return InvariantTreeNode(env.pg_nodes()[idx])
