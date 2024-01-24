#!/usr/bin/env python3

import os
import sys
from subprocess import check_output
from tqdm import tqdm

LOG_ROOT="logs"

def R(fpath):
    with open(fpath, 'r') as fin:
        return fin.read().splitlines()


if len(sys.argv) != 4:
    print("usage: ", sys.argv[0], "slurmscript.sub", "benchlist", "logname")
    exit()

slurmscript = str(sys.argv[1])
bs = R(sys.argv[2])

logname = sys.argv[3]
log_dir = os.path.join(LOG_ROOT, logname)
if os.path.exists(log_dir):
    print("log_dir already exists: ", log_dir)
    exit()

os.makedirs(log_dir)

for b in tqdm(bs):
    try:
        out_f = os.path.join("tmp", b + ".out")
        err_f = os.path.join("tmp", b + ".err")
        log_f = os.path.join(log_dir, b + "-log")

        qsub_envs = "bench=" + b + ",log=" + log_f
        cmd = ["qsub", "-o", out_f, "-e", err_f, "-v", qsub_envs, "-V",slurmscript]
        print(cmd)
        out = check_output(cmd)
        print(b + ' is submitted')
    except Exception as e:
        print("error occur: ", e)

