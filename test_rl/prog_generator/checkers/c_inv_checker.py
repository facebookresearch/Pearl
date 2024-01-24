import z3
import sys
import tokenize
import io
import logging
from code2inv.prog_generator.chc_tools.chctools.horndb import *
from code2inv.prog_generator.chc_tools.chctools.solver_utils import *
from code2inv.prog_generator.chc_tools.chctools.chcmodel import load_model_from_file, define_fun_to_lambda
from z3 import *

p = {}
p["%"] = 5
p["*"] = 5
p["/"] = 5
p["+"] = 4
p["-"] = 4
p["("] = 0
p[")"] = 0
p[">="] = 2
p[">"] = 2
p["=="] = 2
p["<="] = 2
p["<"] = 2
p["and"] = 1
p['or'] = 1

def condense(inv_tokens):
    op_list = ["+", "-", "*", "/", "%", "<", "<=", ">", ">=", "==", "!=", "and", "or"]
    un_op_list = ["+", "-"]
    old_list = list(inv_tokens)
    new_list = list(inv_tokens)
    while True:
        for idx in range(len(old_list)):
            if old_list[idx] in un_op_list:
                if idx == 0 or old_list[idx-1] in op_list or old_list[idx-1] == "(":
                    new_list[idx] = old_list[idx] + old_list[idx+1]
                    new_list[idx+1:] = old_list[idx+2:]
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


def inv_checker(vc_file: str, inv: str, assignments):
    inv = inv.replace("&&", "and", -1)
    inv = inv.replace("||", "or", -1)
    b = io.StringIO(inv)
    t = tokenize.generate_tokens(b.readline)
    inv_tokenized = []
    for a in t:
        if a.string != "":
            inv_tokenized.append(a.string)
    
    var_list = set()
    for token in inv_tokenized:
        if token[0] in "ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz" and token not in ("and", "or"):
            var_list.add(token)
    
    for assignment in assignments:
        v = assignment[0]
        val = assignment[1]
        if v in var_list:
            exec(v + "=" + val)
            var_list.discard(v)
    
    for var in var_list:
        exec(var +"=1")

    try:
        return eval(inv)
    except:
        return False

def inv_solver(vc_file: str, inv: str):
    inv = inv.replace("&&", "and", -1)
    inv = inv.replace("||", "or", -1)
    b = io.StringIO(inv)
    t = tokenize.generate_tokens(b.readline)
    inv_tokenized = []
    for a in t:
        if a.string != "":
            inv_tokenized.append(a.string)
    inv = stringify_prefix_stack(postfix_prefix(infix_postfix(condense(inv_tokenized))))
    inv = inv.replace("==", "=", -1)

    sol = z3.Solver()
    sol.set(auto_config=False)
    res = []

    vc_sections = [""]
    with open(vc_file, 'r') as vc:
        for vc_line in vc.readlines():
            if "SPLIT_HERE_asdfghjklzxcvbnmqwertyuiop" in vc_line:
                vc_sections.append("")
            else:
                vc_sections[-1] += vc_line
    assert len(vc_sections) == 5

    tpl = [vc_sections[0]]

    for i in range(2, 5):
        tpl.append(vc_sections[1] + vc_sections[i])
    res = []
    for i in range(3):
        s = tpl[0] + inv + tpl[i+1]
        sol.reset()
        try:
            sol.set("timeout", 10000)
            decl = z3.parse_smt2_string(s)
            sol.add(decl)
            r = sol.check()
            if z3.sat == r:
                m = sol.model()
                ce = {}
                if i == 0 or i == 2:
                    for x in m:
                        v = str(x)
                        if "_" in v:
                            continue
                        ce[str(x)] = str(m[x])
                else:
                    m1, m2 = {}, {}
                    for x in m:
                        v = str(x)
                        const = str(m[x])
                        if "_" in v:
                            continue
                        elif v.endswith("!"):
                            m2[ v[:-1] ] = const
                        else:
                            m1[v] = const
                    ce = (m1, m2)
                res.append(ce)
            elif z3.unknown == r:
                if i == 0:
                    w = "pre"
                elif i == 1:
                    w = "loop"
                elif i == 2:
                    w = "post"
                logging.warning("inv- " + inv + " solution unknown in " + w) 
                raise Exception("SOL UNKNOWN")
            else:
                res.append(None)
        except Exception as e:
            # print("Encountered Exception in solver", e)
            res.append("EXCEPT")
    return res

'''
def is_trivial(vc_file : str, pred : str):
    inv = pred.replace("&&", "and", -1)
    inv = inv.replace("||", "or", -1)
    tmp_token_exclusive__b = io.StringIO(inv)
    tmp_token_exclusive__t = tokenize.generate_tokens(tmp_token_exclusive__b.readline)
    inv_tokenized = []
    for tmp_token_exclusive__a in tmp_token_exclusive__t:
        if tmp_token_exclusive__a.string != "":
            inv_tokenized.append(tmp_token_exclusive__a.string)

    var_list = set()
    for token in inv_tokenized:
        if token[0] in "ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz" and token not in ("and", "or"):
            var_list.add(str(token))
    
    # inv = stringify_prefix_stack(postfix_prefix(infix_postfix(inv_tokenized)))
    # inv = inv.replace("==", "=", -1)
    
    for v in var_list:
        exec("%s = z3.Int('%s')" % (v, v))
    
    try:
        __tmp_x_for_eval__ = eval(inv)
        if __tmp_x_for_eval__ == True or __tmp_x_for_eval__ == False:
            return True
        res = str(z3.simplify(__tmp_x_for_eval__))
    except Exception as e:
        # print("Exception", e, inv)
        return False

    if res == 'True' or res == 'False':
        return True
    else:
        return False
'''
