import argparse

import json
import os
import numpy as np

from util import construct_data_from_json

np.set_printoptions(suppress=True)
# import torch as th

from dgl_treelstm.KNN import KNN
from preprocessing import Vector_Dataset,op
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
import warnings
import time

warnings.filterwarnings('ignore')

def main(args):
    data = load_dataset(args)

    if args.model_selection == "all":
        sknn = True
        iknn = True
    elif args.model_selection == "knn":
        sknn = True
        iknn = False
    else:
        sknn = False
        iknn = True

    # knn classifier
    test_dataset = None
    train_dataset = None
    if args.cross_project:
        train_dataset = data
        output_dir = os.path.join(args.eva_input, 'train')
        construct_data_from_json(output_dir)
        # test_dataset = th.load(output_dir)
        data = test_dataset
    test_filename = list(set([x.filename for x in data]))
    if "smt-comp" in args.input:
        test_filename = list(set([x.filename for x in data]))
        test_filename = list(set(x.split("_")[0] for x in test_filename))
    dataset = Vector_Dataset()
    dataset.fs_list = data

    # test_filename = ["expand"]
    total_num = 0
    incremental_total_result = []
    sklearn_total_result = []
    truth = []
    s = time.time()
    print(len(data))

    # some data analyse for the data
    # cor(data)
    if args.odds_ratio:
        odds_ratio_test(data)
        # return

    for fn in test_filename:
        if "smt-comp" in args.input:
            # extrame data amount, remove if you want
            # if fn != "Sage2":
            #     continue
            fn = list(map(lambda x:x.filename, filter(lambda x: x.filename.split("_")[0] == fn, data)))
            train_slice, test_dataset = dataset.split_with_filename(fn)
            fn = fn[0].split("_")[0]
        else:
            train_slice, test_dataset = dataset.split_with_filename([fn])
        if not args.cross_project:
            train_dataset = train_slice
        y_test = np.array([1 if i.gettime(args.time_selection) > args.time_limit_setting else 0 for i in test_dataset])
        if sum(y_test) == 0 or len(y_test) < 10:
            continue
        print(fn, len(y_test), sum(y_test))
        # continue
        total_num += len(y_test)
        if iknn:
            incremental_predict = simple_KNN(args, test_dataset, train_dataset)
            incremental_total_result.extend(incremental_predict)

        if sknn:
            sklearn_predict = sklearn_KNN(args, test_dataset, train_dataset)
            sklearn_total_result.extend(sklearn_predict)

        truth.extend(y_test)
    e = time.time()
    print("time", e - s, "data number", len(truth))
    print("total result:")
    if iknn:
        acc = accuracy_score(truth, incremental_total_result)
        pre = precision_score(truth, incremental_total_result)
        rec = recall_score(truth, incremental_total_result)
        f1 = f1_score(truth, incremental_total_result)
        print('incremental test_rl accuracy: {:.3}, precision: {:.3}, recall: {:.3}, f1 score: {:.3}'.format(acc, pre, rec, f1))

    if sknn:
        acc = accuracy_score(truth, sklearn_total_result)
        acc = accuracy_score(sklearn_total_result, truth)
        pre = precision_score(truth, sklearn_total_result)
        rec = recall_score(truth, sklearn_total_result)
        f1 = f1_score(truth, sklearn_total_result)
        print('test_rl accuracy: {:.3}, precision: {:.3}, recall: {:.3}, f1 score: {:.3}'.format(acc, pre, rec, f1))

# correlation coefficient for different operators
def cor(train_dataset):
    train_dataset = list(filter(lambda x: sum(x.feature) != 0, train_dataset))
    x = np.array([i.feature for i in train_dataset])
    x = np.power(10, x) - 1
    x = x[:,:150] + x[:,150:]
    x = np.log10(x + 1)
    y = np.array([1 if i.gettime(args.time_selection) > args.time_limit_setting else 0 for i in train_dataset])
    for i in range(73):
        if sum(x[:,i]) == 0:
            continue
        data = np.corrcoef(x[:,i], y)
        print(i, op[i], data[0,1])

# odds_ratio experiment for different operators
def odds_ratio_test(train_dataset):
    # train_dataset = list(filter(lambda x:sum(x.feature) != 0, train_dataset))
    x = np.array([i.feature for i in train_dataset])
    y = np.array([1 if i.gettime(args.time_selection) > args.time_limit_setting else 0 for i in train_dataset])
    x = x[:, :150] + x[:, 150:]
    for i in range(150):
        # x_with = np.array(list(map(lambda x: x[i] == 0 and x[i + 150] == 0, x)))
        x_with = np.array(list(map(lambda x: x[i] == 0, x)))
        index = np.argwhere(x_with == False).reshape(-1)
        xp = x[index]
        y_wp, y_w = sum(y[index]), len(index)
        index = np.argwhere(x_with == True).reshape(-1)
        y_wop, y_wo = sum(y[index]), len(index)
        try:
            # print(y_wp, y_w)
            # print(y_wop, y_wo)
            if y_w == 0:
                if i < len(op):
                    print(i, op[i], "absent of operator")
                continue
            if y_wo < 10:
                if i < len(op):
                    print(i, op[i], "too little scripts without the operator")
                elif i >= 111:
                    continue
                    # print(i, "var", "unsuitable")
                continue
            if y_w < 10:
                if i < len(op):
                    print(i, op[i], "too little scripts with the operator")
                elif i >= 111:
                    continue
                    # print(i, "var", "unsuitable")
                continue
            if i < len(op):
                print(i, op[i], (y_wp / y_w) / (y_wop / y_wo))
            elif i >= 111:
                break
                # print(i, "var", (y_wp / y_w) / (y_wop / y_wo))
            else:
                print(i, (y_wp / y_w) / (y_wop / y_wo))
        except ZeroDivisionError:
            pass

# bare KNN without incrementation, for comparision, better efficiency since the use of ball_tree
def sklearn_KNN(args, test_dataset, train_dataset):
    clf = KNeighborsClassifier(3, algorithm="ball_tree")
    y_test = np.array([1 if i.gettime(args.time_selection) > args.time_limit_setting else 0 for i in test_dataset])
    x_train = np.array([i.feature for i in train_dataset])
    y_train = np.array([1 if i.gettime(args.time_selection) > args.time_limit_setting else 0 for i in train_dataset])
    clf.fit(x_train, y_train)
    x_test = np.array([i.feature for i in test_dataset])
    y_test_pred = clf.predict(x_test)
    acc = accuracy_score(y_test, y_test_pred)
    pre = precision_score(y_test, y_test_pred)
    rec = recall_score(y_test, y_test_pred)
    f1 = f1_score(y_test, y_test_pred)
    print('test_rl accuracy: {:.3}, precision: {:.3}, recall: {:.3}, f1 score: {:.3}'.format(acc, pre, rec, f1))
    return y_test_pred

def simple_KNN(args, test_dataset, train_dataset):
    clf = KNN(k=3)
    y_test = np.array([1 if i.gettime(args.time_selection) > args.time_limit_setting else 0 for i in test_dataset])
    x_train = np.array([i.feature for i in train_dataset])
    y_train = np.array([1 if i.gettime(args.time_selection) > args.time_limit_setting else 0 for i in train_dataset])
    x_test = np.array([i.feature for i in test_dataset])
    # tf = TfidfTransformer()
    # x_train = np.power(10, x_train) - 1
    # x_train = tf.fit_transform(x_train)
    # x_train.todense()
    # x_train = x_train.toarray()
    # x_train = np.log(x_train[:,:150] + x_train[:,150:] + 1)
    # x_test = np.power(10, x_test) - 1
    # x_test = tf.transform(x_test)
    # x_test.todense()
    # x_test = x_test.toarray()
    # x_test = np.log(x_test[:,:150] + x_test[:,150:] + 1)
    clf.fit(x_train, y_train)
    clf.filename = np.array([i.filename for i in train_dataset])
    filename = np.array([i.filename for i in test_dataset])
    if "smt2" in clf.filename[0]:
        index = np.argsort(filename)
        x_test = x_test[index]
        y_test = y_test[index]
        reverse_index = [0] * len(index)
        for ind,i in enumerate(index):
            reverse_index[i] = ind
    if "fast" in args.model_selection:
        y_test_pred = clf.fast_incremental_predict(x_test, y_test)
    else:
        if "mask" in args.model_selection:
            clf.mask = True
        if "error" in args.model_selection:
            clf.accept_error = True
        y_test_pred = clf.incremental_predict(x_test, y_test)

    acc, pre, rec, fls = clf.score(y_test, y_test_pred)
    print('incremental test_rl accuracy: {:.3}, precision: {:.3}, recall: {:.3}, f1 score: {:.3}'.format(acc, pre, rec, fls))
    if "smt2" in clf.filename[0]:
        y_test_pred = y_test_pred[reverse_index]
    return y_test_pred

def load_dataset(args):
    dataset_type = Vector_Dataset
    output_dir = os.path.join( args.input)
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    if os.path.isdir(output_dir):
        train_file = os.path.join(output_dir, 'train')
    else:
        train_file = output_dir
    if os.path.isfile(train_file):
        train_dataset = construct_data_from_json(train_file)
        # train_dataset = th.load(train_file)
    else:
        qd = dataset_type(feature_number_limit=2)
        train_dataset = qd.generate_feature_dataset(args.data_source, args.time_selection)

    if not os.path.isfile(train_file):
        #output the predict data as json, allow the prediction from anywhere
        solver_selection = "z3" if args.time_selection == "original" else args.time_selection
        output = {
            "x" : [i.feature.tolist() for i in train_dataset], "adjust" : [i.gettime(solver_selection) for i in train_dataset],
            "original" : [i.gettime("original") for i in train_dataset],"filename" : [i.filename for i in train_dataset]
        }
        with open(train_file, "w") as f:
            json.dump(output, f)
        # th.save(train_dataset, train_file)
    return train_dataset


def parse_arg():
    # global args
    parser = argparse.ArgumentParser()
    parser.add_argument('--odds_ratio', action='store_true', help="print odds_ratio for all features")
    parser.add_argument('--data_source', default='gnucore/fv2', help="scripts saving directory")
    parser.add_argument('--input', default='gnucore/training', help="saving directory of feature after "
                            "extraction, avoid duplicate preprocess")
    parser.add_argument('--time_selection', default='original', help="the time label you want to use, allow "
     "'original', 'z3', more type need data from different solvers e.g., 'msat', you may collect on your own")
    parser.add_argument('--cross_project', action='store_true', help="default test_rl use the program from same "
                        "dataset, use this option allow you to test_rl program from other dataset")
    parser.add_argument('--eva_input', default='busybox/fv2', help="cross project test_rl scripts saving directory")
    parser.add_argument('--time_limit_setting', type=int, default=300, help="the timeout threshold for solving, "
                                                                            "must less than 300")
    parser.add_argument('--model_selection', default="all", help="select the KNN running mode, 'knn' for bare KNN"
                "'increment-knn' for the adaptive approach, 'all' for the comparsion of two ways,"
                "further setting including 'increment-knn-fast', 'increment-knn-mask', 'increment-knn-error'")
    args = parser.parse_args()
    print(args)
    return args


if __name__ == '__main__':
    args = parse_arg()
    main(args)