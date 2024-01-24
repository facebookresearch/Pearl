from z3 import *
from code2inv.prog_generator.chc_tools.chctools.horndb import *
from code2inv.prog_generator.chc_tools.chctools.solver_utils import *
from code2inv.prog_generator.chc_tools.chctools.chcmodel import load_model_from_file, define_fun_to_lambda
import tokenize
import io
import sys
from pysmt.smtlib.parser import SmtLibZ3Parser, SmtLibCommand
import pysmt.solvers.z3 as pyz3
import traceback

def validate_rule(rule, model, db):
    inv_args = {}
    for x in model.get_fn_collection():
        inv_args[x] = []
    with pushed_solver(Solver()) as s:
        s.set("timeout", 10000)
        uninterp_sz = rule.uninterp_size()
        for idx, term in enumerate(rule.body()):
            if idx < uninterp_sz:
                try:
                    eval_term = model.eval(term)
                    inv_args[term.decl().name()].append(term.children())
                except:
                    if term.decl().name() == db.get_queries()[0].body()[0].decl().name():
                        eval_term = BoolVal(False)
                    else:
                        eval_term = BoolVal(True)
                s.add(eval_term)
            else:
                s.add(term)
        if not rule.is_query():
            try:
                eval_term = model.eval(rule.head())
                inv_args[rule.head().decl().name()].append(rule.head().children())
            except:
                if rule.head().decl().name() == db.get_queries()[0].body()[0].decl().name():
                    eval_term = BoolVal(False)
                else:
                    eval_term = BoolVal(True)
            s.add(Not(eval_term))
        res = s.check()
        return res, s, inv_args

def find_inv_fn(db: HornClauseDb):
    inv_fun = []
    for rule in db.get_rules():
        head_name = rule.head().decl().name()
        
        for pred in rule.body():
            pred_name = pred.decl().name()
            if pred_name == head_name:
                inv_fun.append((pred_name, [ str(x)[:str(x).rfind("_")] for x in pred.children() ]))
    return inv_fun

def load_model_from_smt2_str(smt2_str):
    model = FolModel()
    b = io.StringIO(smt2_str)
    parser = SmtLibZ3Parser()
    for cmd in parser.get_command_generator(b):
        if type(cmd) == SmtLibCommand and cmd.name == 'define-fun':
            name = cmd.args[0]
            lmbd = define_fun_to_lambda(parser.env, cmd)
            model[name] = lmbd
            
    return model

def inv_checker(vc_file: str, inv: str, assignments):
    try:
        db = load_horn_db_from_file(vc_file)
        inv_fn = find_inv_fn(db)
        inv_fn_string = "( define-fun " + inv_fn[0][0] + " ( "
        assignment_order = []
        # print(assignments)
        for rel in db.get_rels():
            if rel.name() == inv_fn[0][0]:
                for i in range(rel.arity()):
                    inv_fn_string += "( " + "V" + str(i) + " " + str(rel.domain(i)) + " ) "
                    assignment_found = False
                    for assignment in assignments:
                        if "V" + str(i) == assignment[0]:
                            assignment_order.append(assignment[1])
                            assignment_found = True
                            break
                    if not assignment_found:
                        assignment_order.append("1")
                inv_fn_string += ") " + str(rel.range()) + "\n\t" + inv + "\n)\n"
                inv_fn_string += "( assert ( " + rel.name() + " "
                
                for a in assignment_order:
                    inv_fn_string += str(a) + " "
                inv_fn_string += ") )\n"
                break
        # print(inv_fn_string)        
        decl = parse_smt2_string(inv_fn_string)
        sol = Solver()
        sol.add(decl)
        return sol.check() == sat
    except Exception as e:
        print("Encountered Exception inv_checker", e)
        return False

def inv_solver(vc_file: str, inv: str):
    try:
        db = load_horn_db_from_file(vc_file)
        
        inv_fn = find_inv_fn(db)
        
        tmp_inv = str(inv)
        inv_fn_string = "( define-fun " + inv_fn[0][0] + " ( "
        for rel in db.get_rels():
            if rel.name() == inv_fn[0][0]:
                for i in range(rel.arity()):
                    inv_fn_string += "( " + "V" + str(i) + " " + str(rel.domain(i)) + " ) "
                    
                inv_fn_string += ") " + str(rel.range()) + "\n\t" + tmp_inv + "\n)\n"
                break
            
        model = load_model_from_smt2_str(inv_fn_string)

        res = []
        for rule in db.get_rules():
            try:
                if len(rule.body()) > 0:
                    r, s, inv_args = validate_rule(rule, model, db)
                    
                    inv_fn = list(inv_args.keys())[0]
                    if r == sat:
                        m = s.model()
                        # print(m)
                        if len(inv_args[inv_fn]) == 2:
                            m1, m2 = {}, {}
                            idx = 0
                            for arg in inv_args[inv_fn][0]:
                                val = str(m[arg])
                                if val != "None":
                                    m1["V"+str(idx)] = val
                                idx += 1
                            idx = 0
                            for arg in inv_args[inv_fn][1]:
                                val = str(m[arg])
                                if val != "None":
                                    m2["V"+str(idx)] = val
                                idx += 1
                            res.append((m1, m2))
                        else:
                            m1 = {}
                            idx = 0
                            
                            for arg in inv_args[inv_fn][0]:
                                val = str(m[arg])
                                if val != "None":
                                    m1["V"+str(idx)] = val
                                idx += 1
                            res.append(m1)
                    else:
                        res.append(None)
            except Exception as e:
                print("EXCEPTION ENCOUNTERED inv_solver", e)
                res.append("EXCEPT")
        # print("res", res)
        return res
    except Exception as e:
        print("EXCEPTION ENCOUNTERED inv_solver", e)
        return ["EXCEPT"] * 3

# def is_trivial(vc_file : str, pred : str):
#     db = load_horn_db_from_file(vc_file)
#     inv_fn = find_inv_fn(db)
#     check_str = "( assert ( forall ("
#     for v in inv_fn[0][1]:
#         check_str += "(" + v + " Int ) "
#     check_str += ")"
#     check_str += "(" + pred + ")))"
#     try:
#         sol = Solver()
#         decl = parse_smt2_string(check_str)
#         sol.add(decl)
#         return sol.check() == sat
#     except:
#         return False
