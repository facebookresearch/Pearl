import json
import time

from z3 import *

from pearl.SMTimer.KNN_Predictor import Predictor

# SMT-LIB格式的字符串
file_path = '/home/lz/baidudisk/smt/gnu_angr.tar.gz/single_test/arch/arch15998'


# file_path = '/home/lz/下载/QF_BV_Hierarchy(1)/QF_BV/2018-Goel-hwbench/15967301/QF_BV_fischer.2.prop1_cc_ref_max.smt2'
# file_path = '/home/lz/下载/QF_BV_Hierarchy(1)/QF_BV/2018-Goel-hwbench/15967301/QF_BV_fischer.2.prop1_cc_ref_max.smt2'
def solve_and_measure_time(solver, timeout):
    solver.set("timeout", timeout)
    start_time = time.time()
    result = solver.check()
    elapsed_time = time.time() - start_time
    if result == sat:
        return "求解成功", solver.model(), elapsed_time
    elif result == unknown:
        return "求解超时", None, elapsed_time
    else:
        return "求解失败", None, elapsed_time


# 使用with语句安全打开文件
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
# print(smtlib_str)
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
for var_name in variables:
    # 假设所有变量都是整数，使用Int构造函数创建整数变量
    if var_name == 'unconstrained_ret_mbrtowc_739_64':
        exec(f"{var_name} = Int('{var_name}')")
        solver.add(eval(var_name) == 1)

# unconstrained_ret_mbrtowc_739_64 = Int(variables[1])
predictor = Predictor('KNN')
# 后续实现一些子集求解
query_smt2 = solver.to_smt2()
print(query_smt2)
predicted_solvability = predictor.predict(query_smt2)
print(predicted_solvability)
# result = solver.check()
# print(result)
# print(solver.model())
# 2分钟超时限制（单位：毫秒）
timeout = 120000

# 尝试求解，并测量时间
result, model, time_taken = solve_and_measure_time(solver, timeout)
print(f"未具体化变量 - 结果: {result}, 时间: {time_taken:.2f} 秒, 模型: {model}")
