from __future__ import print_function
from subprocess import check_output
from code2inv.common.constants import AC_CODE, INVALID_CODE, ENTRY_FAIL_CODE, INDUCTIVE_FAIL_CODE, POST_FAIL_CODE, ALWAYS_TRUE_EXPR_CODE, ALWAYS_FALSE_EXPR_CODE, NORMAL_EXPR_CODE

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

code_ce_dict = {}
ICE_KEYS = ("T:", "I:", "F:")

class StatsCounter(object):
    def __init__(self):
        self.stats_dict = {}
        self.reported = set()

    def add(self, pid, name, delta=1):
        if not pid in self.stats_dict:
            self.stats_dict[pid] = Counter()
        c = self.stats_dict[pid]
        c[name] += delta

    def report(self, pid):
        
        if not pid in self.stats_dict:
            self.stats_dict[pid] = Counter()
        c = self.stats_dict[pid]
        
        tqdm.write('z3_report pid: %s stats: %s' % (str(pid), str(c)))
        dur = toc()
        
        tqdm.write('z3_report time: %.2f pid: %s stats: %s' % (dur, str(pid), str(c)))
        
    def report_str(self, pid):
        if not pid in self.stats_dict:
            self.stats_dict[pid] = Counter()
        c = self.stats_dict[pid]
        report = ""
        # report += 'z3_report pid: %s stats: %s' % (str(pid), str(c))
        dur = toc()
        
        report += 'z3_report time: %.2f pid: %s stats: %s' % (dur, str(pid), str(c))

        return report

    def report_str_once(self, pid):
        if pid in self.reported:
            return
        self.reported.add(pid)
        
        return self.report_str(pid)

    def report_once(self, pid):
        if pid in self.reported:
            return
        self.reported.add(pid)
        
        self.report(pid)

    def report_global(self):
        t = Counter()
        for key in self.stats_dict:
            c = self.stats_dict[key]
            for k in c:
                t[k] += c[k]
        tqdm.write('z3_global_stats: %s' % str(t))

stat_counter = StatsCounter()

class CounterExample(object):
    def __init__(self, src = None, ice = None):
        if src is not None and ice is not None:
            #self.parse_boogie_ice(src,ice)
            self.parse_z3_ice(src, ice)
    
    def __repr__(self):
        return self.ice_str
    
    def parse_z3_ice(self,src, ice_model):
        kind,model = ice_model
        self.kind = kind
        if self.kind == "pre" or self.kind == "post":
            self.config = {}
            for k in model:
                v = str(k)
                if "_" in v:
                    #ignore variable with ssa subscript
                    continue
                self.config[str(k)] = str(model[k])
        elif self.kind == "loop":
            m1,m2 = {},{}
            for k in model:
                v = str(k)
                const = str(model[k])
                if "_" in v:
                    #ignore variable with ssa subscript
                    continue
                elif v.endswith("!"):
                    m2[ v[:-1] ] = const
                else:
                    m1[v] = const
            self.config = (m1,m2)
        else:
            raise Exception("parse_z3_ice gets unexpected kind")

        self.ice_str = self.to_ice_str()

    def to_ice_str(self):
        res = None
        if self.kind == "pre":
            res = "T:{" + ",".join( ["%s=%s" % (k,self.config[k]) for k in sorted(self.config.keys()) ]) + "}"
        elif self.kind == "post":
            res = "F:{" + ",".join( ["%s=%s" % (k,self.config[k]) for k in sorted(self.config.keys()) ]) + "}"
        elif self.kind == "loop":
            res = "I:{" + ",".join( ["%s=%s" % (k,self.config[0][k]) for k in sorted(self.config[0].keys()) ]) + ";"
            res += ",".join( ["%s=%s" % (k,self.config[1][k]) for k in sorted(self.config[1].keys()) ] ) +"}"
        return res
            
    def parse_boogie_ice(self,src,ice_str):
        self.src = src
        self.ice_str = ice_str
        if ice_str.startswith("T:"):
            self.kind = "pre"
            self.config = self.extract(ice_str[3:-1])
        elif ice_str.startswith("F:"):
            self.kind = "post"
            self.config = self.extract(ice_str[3:-1])
        elif ice_str.startswith("I:"):
            self.kind = "loop"
            sep_index = ice_str.find(';')
            assert sep_index > 0
            A = self.extract( ice_str[3:sep_index] )
            B = self.extract( ice_str[sep_index+1 : -1] )
            self.config = (A,B) 
        else:
            raise Exception("invalid ice_str")

    def extract(self, config_str):
        d = {}
        for s in config_str.split(","):
            var,val = s.split('=')
            d[var] = val
        return d

    def helper(self, dat, expr_root):
        try:
            assignments = []
            for var in dat:
                # print(dat)
                assignments.append((var, dat[var]))
            
            r = checker_module.inv_checker(cmd_args.input_vcs, str(expr_root), assignments)
            
            return r
        except ZeroDivisionError:
            return False

    # to be modify fanli de queren tiaojian
    def check(self, expr_root):
        if self.kind == "pre":
            # we should get a SAT result for positive example
            if self.helper(self.config, expr_root):
                return "good"
            else:
                return "bad"

        elif self.kind == "post":
            # we should get an UNSAT result for negative example
            if not self.helper(self.config, expr_root):
                return "good"
            else:
                return "bad"
        elif self.kind == "loop":
            if not self.helper(self.config[0], expr_root):
                # the first part is not sat, we don't care the other part
                return "good"
            else:
                #now the other part has to be sat
                if self.helper(self.config[1], expr_root):
                    return "good"
                else:
                    return "bad"
                
        raise Exception("check crashed")
        


    def show(self):
        print("kind:", self.kind)
        print("config:", self.config)

class ReplayMem(object):
    def __init__(self, memory_size):
        self.memory_size = memory_size

        self.ce_list = []

        self.count = 0
        self.current = 0
        self.hist_set = set()

    def add(self, ce):
        if ce.ice_str in self.hist_set:
            return
        self.hist_set.add(ce.ice_str)
        if len(self.ce_list) <= self.current:
            self.ce_list.append(ce)
        else:
            self.hist_set.remove(self.ce_list[self.current].ice_str)
            self.ce_list[self.current] = ce
        
        self.count = max(self.count, self.current + 1)
        self.current = (self.current + 1) % self.memory_size
    
    def sample(self, num_samples):
        if num_samples >= self.count:
            return self.ce_list

        sampled_ce = []

        for i in range(num_samples):
            idx = random.randint(0, self.count - 1)
            sampled_ce.append(self.ce_list[idx])
        
        return sampled_ce
    
    def __repr__(self):
        return str(self.ce_list)

class CEHolder(object):
    def __init__(self, sample):
        self.sample = sample
        self.ce_per_key = {}

    def add_ce(self, key, ce):                
        if not key in self.ce_per_key:
            self.ce_per_key[key] = ReplayMem(cmd_args.replay_memsize)
        
        mem = self.ce_per_key[key]
        mem.add(ce)

    def eval(self, key, expr_root):
        if not cmd_args.use_ce:
            return 1.0
        if not key in self.ce_per_key:
            return 0.0
        mem = self.ce_per_key[key]
        samples = mem.sample(cmd_args.ce_batchsize)
        assert len(samples)

        stat_counter.add(self.sample.sample_index, 'ce-' + key, len(samples))
        s = 0.0
        
        for ce in samples:            
            if ce.check(expr_root) == "good":
                s += 1.0
        return s / len(samples)

    def eval_count(self, expr_root):
        ct = 0
        
        for key in self.ce_per_key:
            mem = self.ce_per_key[key]
            samples = mem.sample(cmd_args.ce_batchsize)
            assert len(samples)
            for ce in samples:            
                if ce.check(expr_root) == "good":
                    ct += 1
        return ct

    def eval_detail(self, key, expr_root):
        if not key in self.ce_per_key:
            return

        mem = self.ce_per_key[key]
        samples = mem.sample(cmd_args.ce_batchsize)
        assert len(samples)
        

        print("key:", key)
        for ce in samples:            
            print(">>  ", ce.check(expr_root), "  ", ce.config)

def get_z3_ice(tpl, expr_root):
    if cmd_args.input_vcs is not None:
        input_vcs = cmd_args.input_vcs
    else:
        # very hacky solution for training with the pickles which needs to be improved
        vc_content = ""
        for _t in range(len(tpl)):
            splitter = "SPLIT_HERE_asdfghjklzxcvbnmqwertyuiop"
            # print("TPL", _t)
            # print(tpl[_t])
            b = tpl[_t].split("(assert")
            if len(b) == 1:
                vc_content += b[0]
            elif len(b) == 2:
                if _t == 1:
                    vc_content += splitter + "\n" + b[0] + "\n" + splitter + "\n(assert " + b[1]
                else:
                    vc_content += splitter + "\n(assert " + b[1]
        if cmd_args.single_sample is not None:
            input_vcs = "tmp_vc_" + str(cmd_args.single_sample) + ".smt2"
            with open(input_vcs, "w") as vc_file:
                vc_file.write(vc_content)
        else:
            with open("tmp_vc.smt2", "w") as vc_file:
                vc_file.write(vc_content)
        
            input_vcs = "tmp_vc.smt2"
    inv = str(expr_root)
    sol = z3.Solver()
    sol.set(auto_config=False)

    keys = ICE_KEYS
    kinds = ["pre", "loop", "post"]

    order = np.arange(3)
    if cmd_args.inv_reward_type == 'any':
        np.random.shuffle(order)
        
    cexs = checker_module.inv_solver(input_vcs, inv)
    # print('++++++++++++++++++++++++++')
    # print(cexs)
    print("CEX", cexs)
    res = []
    for i in order:
        if cexs[i] == "EXCEPT":
            print("EXCEPTION")
            raise Exception("ERROR WHILE SOLVING")
        elif cexs[i] is not None:
            cex = CounterExample()
            if i == 0:
                cex.kind = "pre"
            elif i == 1:
                cex.kind = "loop"
            elif i == 2:
                cex.kind = "post"
            cex.config = cexs[i]
            cex.ice_str = cex.to_ice_str()
            res.append((0, keys[i], cex))
            break

    if len(res) == 0:
        return (1, None, None)
    return res[0]


def get_boogie_ice(tpl, expr_root):
    inv = str(expr_root)
    status = 0
    try:
        s1 = ''
        s2 = ''
        s3 = ''
        if cmd_args.inv_reward_type == 'any':
            s3 = tpl[0] + inv + tpl[3]
            if sys.version_info[0] < 3:
                out = check_output( ["mono", cmd_args.boogie_exe, "-noinfer", "-mlHoudini:ice", s3] )    
            else:
                out = check_output( ["mono", cmd_args.boogie_exe, "-noinfer", "-mlHoudini:ice", s3], encoding='utf8' )
        else:
            assert cmd_args.inv_reward_type == 'ordered'

            s1 = tpl[0] + inv + tpl[1]
            s2 = tpl[0] + inv + tpl[2]
            s3 = tpl[0] + inv + tpl[3]

            if sys.version_info[0] < 3:
                out = check_output( ["mono", cmd_args.boogie_exe, "-noinfer", "-mlHoudini:ice", s1,s2,s3] )    
            else:
                out = check_output( ["mono", cmd_args.boogie_exe, "-noinfer", "-mlHoudini:ice", s1,s2,s3], encoding='utf8')
    except:
        print("\n")
        print("error case: %s" % str(expr_root))
        print("-------------s1------------")
        print(s1)
        print("-------------s2------------")
        print(s2)
        print("-------------s3------------")
        print(s3)
        status = 1

    key = None
    if status == 0:        
        if "parse errors" in out:
            status = -1
        elif "BP5004" in out and "T:" in out:
            key = 'T:'
        elif "BP5005" in out and "I:" in out:
            key = 'I:'
        elif "BP5001" in out and "F:" in out:
            key = 'F:'
        elif "0 error" in out:
            status = 1
        else:
            status = -2

    if status < 0:
        # print("status:", status, "out: ", out)
        raise Exception("boogie returns unexpected result")

    ce = None
    if key is not None:
        s_index = out.find(key)
        assert s_index >= 0
        e_index = out.find("}") + 1
        ce = CounterExample( str(expr_root), out[s_index: e_index])

    return (status, key, ce)
    
def report_tested_stats(g, roots):
    if not g.sample_index in code_ce_dict:
        code_ce_dict[g.sample_index] = CEHolder(g)        
    holder = code_ce_dict[g.sample_index]

    stats = [ holder.eval_count(rt) for rt in roots  ]
    arr = np.array( stats )
    print("mean: ", np.mean(arr), " std: ", np.std(arr), "min: ", np.min(arr), "max: ", np.max(arr), "median: ", np.median(arr))

def report_ice_stats(g, best_expr = None):
    if not g.sample_index in code_ce_dict:
        code_ce_dict[g.sample_index] = CEHolder(g)        
    holder = code_ce_dict[g.sample_index]

    ct = {}
    for key in ICE_KEYS:                
        if key in holder.ce_per_key:
            if best_expr is not None:
                holder.eval_detail(key, best_expr)
            ct[key] = len(holder.ce_per_key[key].ce_list)

    print("counter examples info: ", ct)


def reward_0(holder, lambda_holder_eval, lambda_new_ce, scores):
    try:
        # always query boogie
        status, key, ce = lambda_new_ce()
        
        # compute reward
        result = -3.0

        if status > 0:
            result = 3.0

        if key == 'T:':
            scores[0] *= 0.5
            if cmd_args.inv_reward_type == 'ordered':
                scores[1] = scores[2] = 0.0
        elif key == 'I:':
            scores[1] *= 0.5
            if cmd_args.inv_reward_type == 'ordered':
                scores[0] = 1.0
                scores[2] = 0.0
        elif key == 'F:':
            scores[2] *= 0.5
            if cmd_args.inv_reward_type == 'ordered':
                scores[0] = scores[1] = 1.0

        if key is not None:
            holder.add_ce(key, ce)
            result += sum(scores)
    except z3.z3types.Z3Exception:
        result = -6.0
        
    return result

def reward_1(sample_index, holder, lambda_holder_eval, lambda_new_ice):
    ct = 0
    s = 0
    scores = []
    for key in ICE_KEYS:
        
        score = lambda_holder_eval(key)
        
        if key in holder.ce_per_key:
            ct += len(holder.ce_per_key[key].ce_list)
            s += 0.99
        scores.append(score)      
    t = sum(scores)
    
    if ct > 5 and t < s:
        return -3.0 + t * 0.49
    stat_counter.add(sample_index, 'actual_z3')
    
    # otherwise, call the old reward
    return reward_0(holder, lambda_holder_eval, lambda_new_ice, scores)

def boogie_result(g, expr_root):
    stat_counter.add(g.sample_index, 'boogie_result')
    if not g.sample_index in code_ce_dict:
        code_ce_dict[g.sample_index] = CEHolder(g)        
    holder = code_ce_dict[g.sample_index]
    
    lambda_holder_eval = lambda key: holder.eval(key, expr_root)
    if cmd_args.only_use_z3:
        if cmd_args.single_sample is not None:
            lambda_new_ice = lambda: get_z3_ice( g.db.ordered_pre_post[g.sample_index], expr_root)
        else:
            lambda_new_ice = lambda: get_z3_ice( g.vc_list, expr_root)
    else:
        if cmd_args.single_sample is not None:
            lambda_new_ice = lambda: get_boogie_ice( g.db.ordered_pre_post[g.sample_index], expr_root)
        else:
            lambda_new_ice = lambda: get_boogie_ice( g.vc_list, expr_root)

    res = reward_1(g.sample_index, holder, lambda_holder_eval, lambda_new_ice)
    
    if res > 0:
        tqdm.write("found a solution for " + str(g.sample_index) + " , sol: " + str(expr_root))
        # saving the expr_root object in a pickle
        # stat_counter.report_once(g.sample_index)
        r = stat_counter.report_str_once(g.sample_index)
        tqdm.write(r)
        if cmd_args.exit_on_find:
            if cmd_args.save_smt is not None:
                filename = cmd_args.save_smt
                tqdm.write("Writing invariant smt2 to " + filename)
                with open(filename, 'w') as inv_fil:
                    inv_fil.write(expr_root.to_smt2())
                tqdm.write("Written smt2")
            if cmd_args.op_file is not None and cmd_args.op_file != "":
                filename = cmd_args.op_file
                tqdm.write("Writing logs to " + filename)
                with open(filename, 'w') as inv_file:
                    inv_file.write(str(expr_root) + "\n" + r + "\n")
            
            exit()

    return res

    
def z3_precheck_expensive(pg, statement):
    for v in pg.raw_variable_nodes:
        exec("%s = z3.Int('%s')" % (v, v))

    sol = z3.Solver()
    e = eval(statement)
    sol.add(e)
    if sol.check() == z3.unsat:
        return ALWAYS_FALSE_EXPR_CODE
    sol.reset()
    sol.add( z3.Not(e) )
    if sol.check() == z3.unsat:
        return ALWAYS_TRUE_EXPR_CODE

    return NORMAL_EXPR_CODE


def z3_precheck(pg, statement):
    for v in pg.raw_variable_nodes:
        exec("%s = z3.Int('%s')" % (v, v))
    
    try:
        result = str(z3.simplify( eval(statement) ))
    except:
        return ALWAYS_FALSE_EXPR_CODE
    
    if result == 'True':
        return ALWAYS_TRUE_EXPR_CODE
    if result == 'False':
        return ALWAYS_FALSE_EXPR_CODE
    return NORMAL_EXPR_CODE

def z3_check_implication(pg, a, b):
    s = "z3.Implies( " + a + ", " + b + ")"
    if z3_precheck_expensive(pg, s) == ALWAYS_TRUE_EXPR_CODE:
        return True
    
    s = "z3.Implies( " + b + ", " + a + ")"
    if z3_precheck_expensive(pg, s) == ALWAYS_TRUE_EXPR_CODE:
        return True

    return False
