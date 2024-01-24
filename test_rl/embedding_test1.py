import json

import numpy as np

from pearl.SMTimer.KNN_Predictor import Predictor
from z3 import *
predictor = Predictor('KNN')

file_path = '/home/lz/baidudisk/smt/buzybox_angr.tar.gz/single_test/setconsole/setconsole1167042'
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
predicted_solvability = predictor.predict(smtlib_str)
print(predicted_solvability)

try:
    dataset = Predictor.dataset.generate_feature_dataset([smtlib_str], time_selection="z3")
except (KeyError, IndexError) as e:
    print(e)
print(dataset)
x = np.array(dataset[-1].feature).reshape(-1, 300)
print(x)