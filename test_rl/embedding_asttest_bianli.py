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
def traverse_z3_expr(expr, level=0):
    indent = ' ' * 2 * level
    # 打印当前节点的基本信息
    print(f"{indent}Level {level} [{expr.__class__.__name__}]: {expr}")

    # 根据不同类型处理节点
    if isinstance(expr, BoolRef) or isinstance(expr, ArithRef) or isinstance(expr, BitVecRef) or isinstance(expr, ArrayRef):
        # 打印操作符
        if expr.decl().kind() != Z3_OP_UNINTERPRETED:
            print(f"{indent}Operator: {expr.decl()}")

        # 递归遍历子节点
        for arg in expr.children():
            traverse_z3_expr(arg, level + 1)

    elif isinstance(expr, FuncDeclRef):
        print(f"{indent}Function: {expr.name()}")
        # 遍历参数类型
        for i in range(expr.arity()):
            print(f"{indent}Arg {i}: {expr.domain(i)}")
        print(f"{indent}Return Type: {expr.range()}")

    elif isinstance(expr, QuantifierRef):
        print(f"{indent}Quantifier")
        # 递归遍历子表达式
        traverse_z3_expr(expr.body(), level + 1)

    elif isinstance(expr, IntNumRef) or isinstance(expr, RealNumRef):
        print(f"{indent}Constant value: {expr.as_string()}")

    else:
        print(f"{indent}Unhandled expression type: {type(expr)}")
def traverse(expr, level=0):
    # 打印当前节点信息
    print(f"{' ' * level * 2}Level {level}: {expr}")

    # 如果是变量或常量，直接返回
    if is_const(expr):
        if expr.decl().kind() == Z3_OP_UNINTERPRETED:
            print(f"{' ' * level * 2}Found variable: {expr}")
        else:
            print(f"{' ' * level * 2}Found constant: {expr}")
        return

    # 递归遍历子节点
    for i in range(expr.num_args()):
        traverse(expr.arg(i), level + 1)
def traverse_z3_expr_onlyleaf(expr, level=0):
    list_type.append([type(expr),level])
    indent = ' ' * 2 * level
    # 检查是否为叶子节点（没有子节点的节点）
    if not expr.children():
        # 叶子节点：打印变量或常量
        if is_const(expr) or isinstance(expr, IntNumRef) or isinstance(expr, RealNumRef):
            print(f"{indent}Leaf [{expr.__class__.__name__}]: {expr}")
    else:
        # 非叶子节点：打印操作符
        print(f"{indent}Operator [{expr.decl().kind()}]: {expr.decl()}")

        # 递归遍历子节点
        for arg in expr.children():
            traverse_z3_expr(arg, level + 1)
def visit(expr):
    if is_const(expr) and expr.decl().kind() == Z3_OP_UNINTERPRETED:
        # Add only uninterpreted functions (which represent variables)
        variables.add(str(expr))
    else:
        # Recursively visit children for composite expressions
        for child in expr.children():
            visit(child)
def read_lines_from_file(file_path):
    try:
        with open(file_path, 'r', encoding='utf-8') as file:
            # Reading each line and storing it in a list
            lines = file.readlines()
            # Stripping newline characters from each line
            lines = [line.strip() for line in lines]
        return lines
    except Exception as e:
        return str(e)

# Sample Execution (The function won't actually read a file here as it requires a file path)
# lines = read_lines_from_file('path_to_file.txt')
# print(lines)
file_list = read_lines_from_file('/home/lz/baidudisk/busybox')
file_list1 = read_lines_from_file('/home/lz/code2inv_rl/code2inv_rl/code2inv/angr/gnu_KLEE.txt')
file_list2 = read_lines_from_file('/home/lz/code2inv_rl/code2inv_rl/code2inv/angr/gnu_angr.tar.gz.txt')
file_list = file_list + file_list1 + file_list2
list_type = []
for file_path in file_list:
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



    for a in assertions:
        print('********************************************')
        # print(a)
        print(type(a))
        if type(a) not in list_type:
            list_type.append(type(a))

        traverse_z3_expr_onlyleaf(a)
        # traverse(a)
print(list_type)
