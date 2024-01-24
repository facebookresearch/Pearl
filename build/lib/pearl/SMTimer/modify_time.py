import argparse
from collections import defaultdict
import re

from preprocessing import dgl_dataset,Tree_Dataset,Vocab,Constants,Vector_Dataset
import torch.nn.utils.rnn as rnn_utils
import torch as th
import numpy as np

parser = argparse.ArgumentParser()
parser.add_argument('--data', default='./data/gnucore/feature/gnucore_train')
parser.add_argument('--time', default='radjust_time/serial_gnucore.log')
parser.add_argument('--time_selection', default='z3')
args = parser.parse_args()

# train_dataset = th.load(args.data)
with open(args.time, "r") as f:
    data = f.read()
time_list = data.split('\n')
time_dict1 = defaultdict(float)
for time in time_list:
    try:
        st = float(re.search("\d+\.\d+", time).group())
        if "error" in time:
            st = -1
        fn = time.split(",")[0].split("/")[-1]
        time_dict1[fn] = st
    except Exception as e:
        print(e)

time_dict2 = defaultdict(float)
with open("radjust_time/remove_last.log", "r") as f:
    data = f.read()
time_list = data.split('\n')
for time in time_list:
    try:
        st = float(re.search("\d+\.\d+", time).group())
        if "error" in time:
            st = -1
        fn = time.split(",")[0].split("/")[-1]
        time_dict2[fn] = st
    except Exception as e:
        print(e)

count1, count2, count3 = 0, 0, 0
sep = defaultdict(int)
for i in time_dict1.keys():
    a, b = time_dict1[i], time_dict2[i]
    if time_dict1[i] > 300 and time_dict2[i] < 30:
        count1 += 1
    if time_dict1[i] > 300 and time_dict2[i] > 30 and time_dict2[i] < 300:
        count2 += 1
    if time_dict1[i] > 300 and time_dict2[i] > 300:
        count3 += 1
    if time_dict1[i] > 300:
        sep[b // 30] += 1
print(count1, count2, count3)
print(sep)
# if len(train_dataset) != len(times):
#     print("size did not match")
# else:
#     for ind, qt in enumerate(train_dataset):
#         if args.time_selection == "adjust":
#             qt.adjust_time = times[ind]
# th.save(train_dataset, args.data + "m")


# ssh://lsc@10.177.75.217:22/home/lsc/lsc/query-solvability/bin/python3 -u /home/lsc/treelstm.pytorch/modify_time.py
# Using backend: pytorch
# 'NoneType' object has no attribute 'group'
# 'NoneType' object has no attribute 'group'
# 922 330 726
# defaultdict(<class 'int'>, {10.0: 726, 0.0: 922, 9.0: 16, 2.0: 52, 1.0: 126, 6.0: 24, 8.0: 14, 3.0: 44, 4.0: 21, 5.0: 17, 7.0: 16})
#
# Process finished with exit code 0
