import argparse
import logging

import re
from easyprocess import EasyProcess

import os
import json
import time
import multiprocessing as mp
import traceback

from z3 import Solver as z3Solver
from z3 import parse_smt2_string
from pysmt.smtlib.parser import SmtLibParser
from pysmt.shortcuts import Solver
from six.moves import cStringIO


def parse_arg():
    parser = argparse.ArgumentParser()
    parser.add_argument('--input', default='./data/gnucore/query2', help="the path of SMT files")
    parser.add_argument('--parser', default='pysmt', help="parser for SMT scripts, only 'pysmt' or 'z3'")
    parser.add_argument('--solver', default='z3', help="SMT solver, including 'all', 'msat', 'cvc4', 'yices', "
                                                       "'btor', 'z3', make sure you install it before using")
    parser.add_argument('--logic', default='QF_UFBV', help="not used by now")
    parser.add_argument('--pool_size', type=int, default=7, help="multiprocess pool size")
    parser.add_argument('--log_file', default='adjustment.log', help="result saving file")
    parser.add_argument('--ite_num', type=int, default=1, help="the times of running")
    parser.add_argument('--prefix', default='', help="prefix of SMT file, to filter other files")
    parser.add_argument('--end', default='', help="suffix of SMT file, to filter other files")
    parser.add_argument('--timeout', type=int, default=300, help="solving timeout")
    args = parser.parse_args()
    return args

logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)

solver_list = ["msat", "cvc4", "yices", "btor", "z3"]


def smtfun(data, filename, timeout=300, solver_name="z3"):
    base_dir = os.getcwd()
    # to debug the parsing and solving error, you may use the code below which is a copy of solve.py
    #
    # signal.signal(signal.SIGALRM, handler)
    # if solver_name in solver_list:
    #     solver = Solver(name=solver_name, logic='BVt')
    # else:
    #     return
    # smt_parser = SmtLibParser()
    # try:
    #     script = smt_parser.get_script(cStringIO(data))
    #     a = script.evaluate(solver)
    #     print(a)
    # except Exception as e:
    #     traceback.print_exc()
    #     return

    # pysmt solving can not set timeout, so we use new Process to terminate infinite solving
    s = time.time()
    try:
        result = EasyProcess(
            'python {} {} {}'.format(
                os.path.join(base_dir, "solve.py"),
                filename,
                solver_name
            )
        ).call(timeout=timeout)
        if result.stderr:
            print(result.stderr)
        result = result.stdout
        print(result)

        if result == "":
            res = "unknown"
            e = time.time()
            t = str(e - s)
        else:
            result, t = result.split(" ")
            res = result
    except:
        # traceback.print_exc()
        e = time.time()
        res = "error"
        t = str(e - s)
    st = filename + ', solver: ' + solver_name + ', result: ' + str(res) + ', time:' + t
    logger.info(st)

def z3fun(data, filename, timeout=300000):
    try:
        script = parse_smt2_string(data)
        z3s = z3Solver(ctx=script.ctx)
        z3s.set('timeout', timeout)
        z3s.from_string(data)
    except Exception as e:
        # traceback.print_exc()
        # print(filename)
        return
    s = time.time()
    res = z3s.check()
    e = time.time()

    st = filename + ', solver: z3, result: ' + str(res) + ', time:' + str(e - s)
    logger.info(st)
    return res


def process_data(data):
    try:
        data = json.loads(data)
        atime = float(data["time"])
        old_data = data
        if 'smt_script' in data.keys():
            data = data["smt_script"]
        else:
            data = data["script"]
    except:
        atime = 0
    data = data.split('\n')
    start = 0
    end = len(data)
    for i in range(len(data)):
        if "check-sat" in data[i]:
            end = i
        if "time:" in data[i]:
            try:
                atime = float(data[i].split(":")[-1])
            except:
                pass
    data = '\n'.join(data[start:end + 1])
    return data, atime

def excluded_list():
    try:
        with open("radjust_time/adjustment.log", "r") as f:
            data = f.read()
        time_list = data.split('\n')
    except IOError:
        return []
    fn_list = []
    for time in time_list:
        try:
            if time == "":
                continue
            fn = re.match('[^,]*(?=,)', time).group()
            if "/" in fn:
                fn_list.append(fn.split("/")[-1])
        except:
            pass
    return fn_list


def main(args):
    formatter = logging.Formatter("%(message)s")
    fh = logging.FileHandler(args.log_file, mode='w')
    fh.setLevel(logging.INFO)
    fh.setFormatter(formatter)
    logger.addHandler(fh)
    # excluded = excluded_list()
    excluded = []
    multi = args.pool_size > 1
    for i in range(args.ite_num):
        if multi:
            pool = mp.Pool(processes=args.pool_size)
        for root, dir, files in os.walk(args.input):
            for file in files:
                if file in excluded:
                    continue
                if file.startswith(args.prefix) and file.endswith(args.end):
                    with open(os.path.join(root, file), "r") as f:
                        raw_data = f.read()
                    data, _ = process_data(raw_data)
                    if data == "":
                        continue
                    if args.parser == "pysmt":
                        if args.solver == "all":
                            for solver_name in solver_list:
                                if multi:
                                    pool.apply_async(smtfun, (data, os.path.join(root, file), args.timeout, solver_name))
                                else:
                                    smtfun(data, file, args.timeout * 1000)
                        elif args.solver in solver_list:
                            if multi:
                                pool.apply_async(smtfun, (data, os.path.join(root, file), args.timeout, args.solver))
                            else:
                                z3fun(data, os.path.join(root, file), args.timeout)
                        else:
                            print("this solver is not supported")
                    elif args.parser == "z3":
                        pool.apply_async(z3fun, (data, file, args.timeout * 1000))
                    else:
                        print("only z3 and pysmt is supported")
        if multi:
            pool.close()
            pool.join()

def getlogger():
    return logger

if __name__ == '__main__':
    args = parse_arg()
    main(args)