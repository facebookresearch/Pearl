import time

import argparse
import numpy as np
import json

import os
import sys
# from matplotlib import pyplot
from torch.utils.data import DataLoader

from preprocessing import Constants
from util import construct_data_from_json
from dgl_treelstm.KNN import KNN
from dgl_treelstm.nn_models import *
from dgl_treelstm.metric import *
from preprocessing import Vocab
from preprocessing import varTree
from dgl_treelstm.dgl_dataset import dgl_dataset
from check_time import process_data
from train import pad_feature_batcher, batcher
from preprocessing.Vector_Dataset import Vector_Dataset
from preprocessing.Tree_Dataset import Tree_Dataset
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, plot_roc_curve, roc_curve, precision_recall_curve
import warnings
from dataset_filename_seperation import get_dataset_seperation

warnings.filterwarnings('ignore')

# simulation for different models
class Simulation:
    def __init__(self, model, time_selection="adjust", threshold=200):
        self.model = model
        # this threshold can be adaptive, which is updated in the simulation to find a better border for timeout
        self.threshold = threshold
        # save data for adaptive threshold
        self.time_record = {"timeout":[], "solvable":[]}
        # this is the actual time setting, which is fixed for comparision the ground truth
        self.time_out_setting = 200
        self.time_selection = time_selection
        if isinstance(self.model, TreeLSTM):
            self.model_type = "TreeLSTM"
            self.preprocess = Tree_Dataset.generate_feature_dataset
        elif isinstance(self.model, KNN):
            self.model_type = "KNN"
            self.preprocess = Vector_Dataset.generate_feature_dataset
        elif isinstance(self.model, LSTM):
            self.model_type = "LSTM"
            self.preprocess = Vector_Dataset.generate_feature_dataset

    def load_model(self, input):
        raise NotImplementedError
        if self.model_type == "KNN":
            dataset = th.load("data/gnucore/fv2/gnucore_train")
            x = [i.feature for i in dataset]
            y = [1 if i.gettime("adjust") > 300 else 0 for i in dataset]
            self.model.fit(x, y)
        elif self.model_type == "LSTM":
            model = th.load("checkpoints/gnucore/pad_feature_l_z.pkl")["model"]
            self.model.load_state_dict(model)
        elif self.model_type == "TreeLSTM":
            model = th.load("checkpoints/g_tree_feature_t_z_r_200.pkl")["model"]
            self.model.load_state_dict(model)

    def script_to_feature(self, data):
        raise NotImplementedError
        # feature = self.preprocess(script)
        if isinstance(data, varTree):
            dataloader = dgl_dataset([data], None)
            iterator = iter(dataloader)
            data = next(iterator)
            feature = data.logic_tree
            solve_time = data.gettime("original")
        elif self.model_type == "LSTM":
            dataloader = DataLoader(dataset=[data], batch_size=1, collate_fn=pad_feature_batcher('cpu', 'original'),
                                         shuffle=False, num_workers=0)
            iterator = iter(dataloader)
            data = next(iterator)
            feature = rnn_utils.pack_padded_sequence(data.feature, data.data_len, enforce_sorted=False,
                                                           batch_first=True)
            solve_time = data.label
        else:
            feature, solve_time = data.logic_tree, data.gettime("original")
        return feature, solve_time

    def predict(self, feature, truth):
        raise NotImplementedError
        if self.model_type == "KNN":
            predict_result = self.model.incremental_predict(feature, truth)
            skip = predict_result
        elif self.model_type == "LSTM":
            self.model.eval()
            with th.no_grad():
                predict_result = self.model(feature)
            skip = predict_result > self.threshold
        else:
            self.model.eval()
            with th.no_grad():
                h = th.zeros((1, 150))
                c = th.zeros((1, 150))
                predict_result = self.model(feature, h, c)
            skip = predict_result > self.threshold
        return predict_result, skip

    def modify_threshold(self, result, truth):
        if self.model_type == "KNN":
            return
        if result < self.threshold and truth > self.time_out_setting:
            self.time_record["timeout"].append(result)
        elif result < self.threshold and truth < self.time_out_setting:
            self.time_record["solvable"].append(result)
        if result < self.threshold and truth > self.time_out_setting:
            self.dynamic_threshold()
            print("decrease threshold to ", str(self.threshold))
        return

    def dynamic_threshold(self):
        timeout_list = np.array(self.time_record["timeout"])
        solvable_list = self.time_record["solvable"]
        try:
            solvable_limit = max(np.percentile(solvable_list, 95), np.mean(solvable_list), 60)
            suitable_timeout = list(filter(lambda x: x > solvable_limit, timeout_list))
            if len(suitable_timeout) == 0:
                suitable_timeout = [solvable_limit]
            suitable_min_timeout = min(suitable_timeout)
            suitable_min_timeout = min(suitable_min_timeout, self.threshold)
            if isinstance(suitable_min_timeout, th.Tensor):
                suitable_min_timeout = suitable_min_timeout.item()
            max_solvable = max(filter(lambda x:x <= suitable_min_timeout, self.time_record["solvable"]))
            if isinstance(max_solvable, th.Tensor):
                max_solvable = max_solvable.item()
            self.threshold = max(suitable_min_timeout - 1, (suitable_min_timeout + max_solvable) / 2,
                                 self.threshold - 50, 60)
        except (IndexError,ValueError):
            pass

class KNN_Simulation(Simulation):
    def __init__(self, model, time_selection="adjust", threshold=200):
        super(KNN_Simulation, self).__init__(model, time_selection, threshold)
        self.model_type = "KNN"
        self.preprocess = Vector_Dataset.generate_feature_dataset
        self.separate_test = False

    def load_model(self, input):
        # dataset = th.load(input)
        dataset = construct_data_from_json(input)
        # test_filename = ["echo", "ginstall", "expr", "tail", "seq", "split", "test_rl", "yes", "chgrp", "date", "expand",
        #                  "head", "nohup", "printf", "sha1sum", "stat", "timeout", "uniq", "nice", "pr"]
        # test_filename = ["expand"]
        # dataset = list(filter(lambda x:x.filename not in test_filename, dataset))
        x = [i.feature for i in dataset]
        if "smt-comp" in input:
            fn = [x.filename.split("_")[0] for x in dataset]
        else:
            fn = [i.filename for i in dataset]
        y = [1 if i.gettime(self.time_selection) > self.time_out_setting else 0 for i in dataset]
        self.model.fit(x, y)
        self.model.filename = np.array(fn)

    def script_to_feature(self, data):
        if not self.separate_test:
            if ".smt2" in data.filename:
                fn = data.filename.split("_")[0]
            else:
                fn = data.filename
            self.model.remove_test(fn)
            self.separate_test = True
        feature, solve_time = data.feature, data.gettime(self.time_selection)
        return feature, solve_time

    def predict(self, feature, truth):
        predict_result = self.model.incremental_predict(feature, truth)
        skip = predict_result
        return predict_result, skip

class LSTM_Simulation(Simulation):
    def __init__(self, model, time_selection="adjust", threshold=200):
        super(LSTM_Simulation, self).__init__(model, time_selection, threshold)
        self.model_type = "LSTM"
        self.preprocess = Vector_Dataset.generate_feature_dataset

    def load_model(self, input):
        model = th.load(input, map_location='cpu')["model"]
        self.model.load_state_dict(model)

    def script_to_feature(self, data):
        dataloader = DataLoader(dataset=[data], batch_size=1, collate_fn=pad_feature_batcher('cpu', self.time_selection),
                                     shuffle=False, num_workers=0)
        iterator = iter(dataloader)
        data = next(iterator)
        feature = rnn_utils.pack_padded_sequence(data.feature, data.data_len, enforce_sorted=False,
                                                       batch_first=True)
        solve_time = data.label.item()
        return feature, solve_time

    def predict(self, feature, truth):
        self.model.eval()
        with th.no_grad():
            predict_result = self.model(feature)
        skip = predict_result > self.threshold
        return predict_result, skip

class TreeLSTM_Simulation(Simulation):
    def __init__(self, model, time_selection="adjust", threshold=200):
        super(TreeLSTM_Simulation, self).__init__(model, time_selection, threshold)
        self.model_type = "TreeLSTM"
        self.preprocess = Tree_Dataset.generate_feature_dataset

    def load_model(self, input):
        model = th.load(input, map_location='cpu')["model"]
        # model = th.load("checkpoints/g_tree+feature_t_z_r_200.pkl")["model"]
        self.model.load_state_dict(model)

    def script_to_feature(self, data):
        smt_vocab_file = './data/gnucore/smt.vocab'
        smt_vocab = Vocab(filename=smt_vocab_file,
                          data=[Constants.UNK_WORD])
        data = dgl_dataset([data], None, vocab=smt_vocab, time_selection=self.time_selection, time_threshold=self.threshold)
        dataloader = DataLoader(dataset=data, batch_size=1, collate_fn=batcher("cpu"),
                                shuffle=False, num_workers=0)
        iterator = iter(dataloader)
        data = next(iterator)
        feature = data.graph
        solve_time = data.label[0].item()
        return data, solve_time

    def predict(self, feature, truth):
        self.model.eval()
        n = feature.graph.number_of_nodes()
        with th.no_grad():
            h = th.zeros((n, 150))
            c = th.zeros((n, 150))
            predict_result = self.model(feature, h, c)
        skip = predict_result[0] > self.threshold
        return predict_result[0], skip

# result saving structure
class Evalution:
    def __init__(self, pred=np.array([]), truth=np.array([]), time_out_setting=200):
        self.pred = self.get_numpy(pred)
        self.truth = self.get_numpy(truth)
        self.classify_result = np.array([])
        self.time_out_setting = time_out_setting

    def get_numpy(self, data):
        if isinstance(data, th.Tensor):
            data = data.cpu().numpy()
        else:
            data = data
        return data

    def add(self, pred, truth, classify_result):
        self.pred = np.append(self.pred, self.get_numpy(pred))
        self.truth = np.append(self.truth, self.get_numpy(truth))
        self.classify_result = np.append(self.classify_result, self.get_numpy(classify_result))

    def score(self):
        truth = [1 if x > self.time_out_setting else 0 for x in self.truth]
        acc = accuracy_score(truth, self.classify_result)
        pre = precision_score(truth, self.classify_result)
        rec = recall_score(truth, self.classify_result)
        f1 = f1_score(truth, self.classify_result)
        return acc, pre, rec, f1

# time calculation
class Time_Section:
    def __init__(self):
        self.original_time = 0
        self.predict_time = 0
        # overall time for simulation comparision(without solving phase 1 which manually added)
        self.final_time = 0
        self.preprocessing_time = 0

    def update(self, predict_result, solve_time):
        self.original_time += solve_time
        # for the first solving phase t1=1s
        self.final_time += 1
        # skip if predicted timeout
        if not predict_result:
            self.final_time += solve_time

    def add_prediction_time(self, predict_used_time, preprocessing_time):
        self.preprocessing_time = preprocessing_time
        self.predict_time = predict_used_time
        self.final_time = self.final_time + predict_used_time + preprocessing_time

# load the test_rl data, script to feature just like the training, we do not saving the result because the program number
# we also want to include the processing time into final time
def load_data(model, input):
    dataset = None
    if model == "Tree-LSTM":
        dataset = Tree_Dataset(treeforassert=True, feature_number_limit=100)
    elif model == "lstm":
        dataset = Vector_Dataset(feature_number_limit=50)
    elif model == "KNN":
        dataset = Vector_Dataset(feature_number_limit=2)
    else:
        dataset = Tree_Dataset(feature_number_limit=100)
    if "smt-comp" in input:
        test_filename = input.split("/")[-1]
        input = "/".join(input.split("/")[:-1])
        dataset.fs_list = dataset.generate_feature_dataset(input, fileprefix=test_filename)
        if len(dataset.fs_list) == 0:
            print("smt-comp file are not separated with filename, but please use the similar structure, more information in simulation_smt-comp.md")
        # test_filename1 = [x.filename for x in dataset.fs_list]
        # test_file = list(filter(lambda x:x.split("_")[0] == test_filename, test_filename1))
        # dataset.fs_list = dataset.split_with_filename(test_file)[1]
        input = input + "/" + test_filename
    else:
        if "klee" in input:
            # the klee processing is time-consuming because of the SMT scripts structure, so we saved the result for next time
            # for other dataset we extract feature every time it simulates.
            data_input = "data/klee/" + input.split("/")[-1] + model_name
            try:
                if model == "KNN":
                    dataset = construct_data_from_json(data_input)
                else:
                    dataset = th.load(data_input)
            except (TypeError,FileNotFoundError):
                dataset.generate_feature_dataset(input)
                if model != "KNN":
                    th.save(dataset, data_input)
        else:
            dataset.generate_feature_dataset(input)
    return dataset.fs_list, input

# mainly for cross dataset prediction for adaptive KNN model, rely on my model naming pattern
def identify_dataset(input, dataset):
    for i in ["busybox", "smt-comp", "klee"]:
        if i in input:
            return i
    if "g_" in input or "gnucore/" in input:
        return "gnucore"
    if "b_" in input:
        return "busybox"
    if "s_" in input:
        return "smt-comp"
    if "k_" in input:
        return "klee"
    return "gnucore"

# our baseline result, not usable without result from PCC
def make_PCC_output(input, output_result):
    if os.path.exists(input):
        with open(input, "r") as f:
            data = json.load(f)
        serial_result = sorted(data["result"], key=lambda x:(len(x[0]), x[0]))
    else:
        serial_result = []
        for i in range(1,4):
            with open(input[:-5] + "_" + str(i) + ".json", "r") as f:
                data = json.load(f)
            serial_result.extend(sorted(data["result"], key=lambda x: (len(x[0]), x[0])))
    od = serial_result
    for i in ["arch", "chgrp", "csplit", "dirname", "fmt", "id", "md5sum", "mv", "pinky", "readlink", "seq",
             "sleep", "tac", "tsort", "uptime", "base64", "chmod", "cut", "du", "fold", "join", "mkdir",
             "nice", "pr", "rm", "setuidgid", "sort", "tail", "tty", "users", "basename", "chroot", "date", "expand", "ginstall",
             "link", "mkfifo", "nl", "printenv", "rmdir", "sha1sum", "split", "test_rl", "uname", "vdir",
             "cat", "comm", "df", "expr", "head", "ln", "mknod", "od", "printf", "runcon", "shred", "stat", "touch", "unexpand", "wc",
             "chcon", "cp", "dir", "factor", "hostname", "ls", "mktemp", "pathchk", "ptx", "shuf", "su",
             "tr", "unlink", "who", "ifconfig", "rpm", "Sage2", "klogd", "mcm", "lfsr"]:
        serial_result = list(filter(lambda x: x[0].startswith(i), od))
        if len(serial_result) == 0:
            continue
        print(i)
        truth = [x[2] for x in serial_result]
        if isinstance(truth[0], list):
            truth = list(map(lambda x:0 if x[0] else 300, truth))
        pred = [x[1] for x in serial_result]
        dt_simulation = Simulation(None)
        dt_simulation.model_type = "DNN"
        if isinstance(pred[0], int):
            classify_result = pred
        else:
            threshold_list = []
            for i in range(len(truth)):
                dt_simulation.modify_threshold(pred[i], truth[i])
                threshold_list.append(dt_simulation.threshold)
            classify_result = [1.0 if pred[i] > threshold_list[i] else 0.0 for i in range(len(pred))]
            # classify_result = [1.0 if x > data["time_limit_setting"] else 0.0 for x in pred]
        original_time = sum(truth)
        pred_truth_tuple = list(
            zip(range(len(pred)), pred, truth, classify_result))
        pred_truth_diff_tuple = list(filter(lambda a: a[3] != (a[2] > data["time_limit_setting"]), pred_truth_tuple))
        pred_truth_tuple = list(filter(lambda a: a[3] != 0, pred_truth_tuple))
        final_time = original_time - sum([x[2] for x in pred_truth_tuple])
        truth = [1 if x > data["time_limit_setting"] else 0 for x in truth]
        acc = accuracy_score(truth, classify_result)
        pre = precision_score(truth, classify_result)
        rec = recall_score(truth, classify_result)
        f1 = f1_score(truth, classify_result)
        print_output = {"train_dataset": "gnucore", "test_dataset": "gnucore", "pred_truth_diff_tuple": pred_truth_diff_tuple,
                        "original_time": original_time,
                        "total_time": final_time, "input": input, "pos_num":sum(truth), "tp": sum(truth)*rec,
                    "fn": sum(truth)*(1 - rec), "fp": sum(truth)*rec/pre - sum(truth)*rec}
        print(print_output)
        output = {"train_dataset": "gnucore", "test_dataset": "gnucore", "predicted_result": pred,
                  "acutal_solving_time": truth, "original_time": original_time, "total_time": final_time,
                  "metrics": {"acc": acc, "pre": pre, "rec": rec, "f1": f1, "pos_num":sum(truth), "tp": sum(truth)*rec,
                    "fn": sum(truth)*(1 - rec), "fp": sum(truth)*rec/pre - sum(truth)*rec},
                  "time_out_setting": data["time_limit_setting"],
                  "model": "PCC", "input": input}
        output = json.dumps(output, indent=4)
        # print(print_output)
        print('test_rl accuracy: {:.3}, precision: {:.3}, recall: {:.3}, f1 score: {:.3}'.format(acc, pre, rec, f1))
        # fpr, tpr, thresholds = roc_curve(truth, pred)
        # pyplot.plot(fpr, tpr, lw=1, label="lstm")
        # # print(fpr, tpr, thresholds)
        # pyplot.xlim([0.00, 1.0])
        # pyplot.ylim([0.00, 1.0])
        # pyplot.xlabel("False Positive Rate")
        # pyplot.ylabel("True Positive Rate")
        # pyplot.title("ROC")
        # pyplot.legend(loc="lower right")
        # pyplot.savefig(r"./ROC.png")
        # pyplot.show()
        if output_result:
            try:
                outpur_path = "_".join(["gnucore", input.split("/")[-1], "DNN"]) + ".json"
                with open("simulation_result/" + outpur_path, "w")as f:
                    f.write(output)
            except:
                with open("simulation_result/output.json", "w")as f:
                    f.write(output)

# output the result for a single program
# to do: not support for adaptive threshold for regression simulation
def make_output(dsn1, dsn2, input, simulation, result, time_section, output_result=True, plot_picture=True):
    pred_truth_tuple = list(zip(range(len(result.pred)), result.pred.tolist(), result.truth.tolist(), result.classify_result))
    pred_truth_tuple = list(filter(lambda a:a[3] != (a[2] > simulation.time_out_setting), pred_truth_tuple))
    truth = [1 if x > simulation.time_out_setting else 0 for x in result.truth]
    acc = accuracy_score(truth, result.classify_result)
    pre = precision_score(truth, result.classify_result)
    rec = recall_score(truth, result.classify_result)
    f1 = f1_score(truth, result.classify_result)
    confusion_matrix = np.zeros((2, 2))
    for t, p in zip(truth, result.classify_result):
        confusion_matrix[t][int(p)] += 1
    # print_output = {"train_dataset": dsn1, "test_dataset": dsn2, "pred_truth_diff_tuple": pred_truth_tuple,
    #                 "original_time": time_section.original_time,
    #                 "predict_time":time_section.predict_time + time_section.preprocessing_time,
    #                 "total_time": time_section.final_time, "input":input, "pos_num":sum(truth), "tp": sum(truth)*rec,
    #                 "fn": sum(truth)*(1 - rec), "fp": sum(truth)*rec/pre - sum(truth)*rec}
    print_output = {"timeout_query_num":sum(truth), "true-positive number": confusion_matrix[1][1],
                    "false-negative number": confusion_matrix[1][0], "false-positive number": confusion_matrix[0][1]}
    output = {"train_dataset": dsn1, "test_dataset": dsn2, "predicted_result": result.pred.tolist(),
              "acutal_solving_time": result.truth.tolist(), "original_time": time_section.original_time, "predict_time":
              time_section.predict_time + time_section.preprocessing_time, "total_time": time_section.final_time,
              "metrics":{"acc": acc, "pre": pre, "rec": rec, "f1": f1}, "time_out_setting": simulation.time_out_setting,
              "model":simulation.model_type, "input":input, "pos_num":sum(truth), "tp": confusion_matrix[1][1],
                    "fn": confusion_matrix[1][0], "fp": confusion_matrix[0][1]}
    if not len(result.truth):
        return
    output = json.dumps(output, indent=4)
    print("train dataset:" + dsn1)
    # print("test_rl dataset:" + dsn2)
    print("test_rl program:" + input)
    print("prediction truth difference tuple(index, predicted result, truth, classification result):")
    print(pred_truth_tuple)
    print("original solving time:" + str(int(time_section.original_time)) + "s")
    print("prediction time:" + str(int(time_section.predict_time + time_section.preprocessing_time)) + "s")
    print("solving time with the predictor:" + str(int(time_section.final_time)) + "s")
    print(print_output)
    print('test_rl accuracy: {:.3}, precision: {:.3}, recall: {:.3}, f1 score: {:.3}'.format(acc, pre, rec, f1))
    # if simulation.model_type != 'KNN':
    #     fpr, tpr, thresholds = roc_curve(result.truth > simulation.time_out_setting, result.pred)
    #     pyplot.plot(fpr, tpr, lw=1, label=simulation.model_type)
    #     # print(fpr, tpr, thresholds)
    #     pyplot.xlim([0.00, 1.0])
    #     pyplot.ylim([0.00, 1.0])
    #     pyplot.xlabel("False Positive Rate")
    #     pyplot.ylabel("True Positive Rate")
    #     pyplot.title("ROC")
    #     pyplot.legend(loc="lower right")
    #     pyplot.savefig(r"./ROC.png")
    #     pyplot.show()
    if output_result:
        try:
            if args.model_name == "KNN":
                identify = ""
            elif args.classification:
                identify = "_c"
            elif args.adapt:
                identify = "_m"
            else:
                identify = "_r"
            outpur_path = "_".join([dsn1, input.split("/")[-1], simulation.model_type]) + identify + ".json"
            with open("simulation_result/" + outpur_path, "w")as f:
                f.write(output)
        except:
            with open("simulation_result/output.json", "w")as f:
                f.write(output)

# automatic partition selection since we use cross validation to generate three piece of result for a model
# used for the hardcoded switch
def choose_input(dataset, input, load_path):
    fn = get_dataset_seperation(dataset)
    f1, f2, f3 = fn[0], fn[1], fn[2]
    input = input.split("/")[-1]
    if dataset == "smt-comp":
        input = input.split("_")[0]
    if os.path.exists(load_path):
        return load_path
    if input in f1:
        load_path = ".".join([load_path.split(".")[0] + "_0", load_path.split(".")[1]])
    elif input in f2:
        load_path = ".".join([load_path.split(".")[0] + "_1", load_path.split(".")[1]])
    elif input in f3:
        load_path = ".".join([load_path.split(".")[0] + "_2", load_path.split(".")[1]])
    else:
        load_path = ""
    return load_path

# simulate the solving in real order, in the simulation, the predicted timeout solving would be skipped,
# the time different is taken as the time saved.
# the simulation may not reflect the real situation since wrongly skip path means the change of path selection, but if
# you give it a low priority, then these paths are just deferred, you may execute more paths in the same time budget.
def simulation_for_single_program(test_directory, args):
    s = time.time()
    input_index = args.input_index
    load_path = args.load_file
    # some setting process since all simulation use one entry
    if not args.regression:
        regression = False
    else:
        input_list[int(input_index)] = input_list[int(input_index)].replace("_r_", "_c_")
        regression = True
    if model_name == "KNN":
        knn = KNN()
        simulation = KNN_Simulation(knn, time_selection=args.time_selection)
        if not input_index:
            input_index = 8
    elif model_name == "lstm":
        lstm = LSTM(150, regression, False)
        simulation = LSTM_Simulation(lstm, time_selection=args.time_selection)
        if not input_index:
            input_index = 0
    else:
        tree_lstm = TreeLSTM(133, 150, 150, 1, 0.5, regression, False, cell_type='childsum', pretrained_emb=None)
        simulation = TreeLSTM_Simulation(tree_lstm, time_selection=args.time_selection)
        if not input_index:
            input_index = 2
    # setting timeout threshold
    # for original time, we collect the data with timeout with 100s, larger than it would be useless
    simulation.time_out_setting = args.threshold
    if test_directory == None:
        test_directory = input_list[int(input_index)]
    serial_data, test_input = load_data(model_name, test_directory)
    time_section = Time_Section()
    result = Evalution(time_out_setting=args.threshold)
    # for cross project, identify dataset name
    dsn1 = identify_dataset(input_list[int(input_index)], None)
    dsn2 = identify_dataset(test_input, serial_data)
    # load the model for different approach
    if load_path == None:
        load_path = input_list[int(input_index)]
    if model_name != "KNN":
        load_path = choose_input(dsn1, test_input, load_path)
    simulation.load_model(load_path)
    s1 = time.time()
    aindex = 0
    # simulation system, but not actual solving since the solving time is consuming, and time may be different
    for data in serial_data:
        data_index = len(result.truth)
        feature, solve_time = simulation.script_to_feature(data)
        predict_result, skip = simulation.predict(feature, 1 if solve_time > simulation.time_out_setting else 0)
        if len(result.pred) % 500 == 0:
            print(len(result.pred))
        if model_name != "KNN" and regression and args.adapt:
            pass
            simulation.modify_threshold(predict_result, solve_time)
        if model_name != "KNN" and not regression:
            pred = th.argmax(F.log_softmax(predict_result), 1)
            skip = pred == 1
            predict_result = 1 if skip else 0
        time_section.update(skip, solve_time)
        result.add(predict_result, solve_time, skip)
        aindex += 1
    e = time.time()
    time_section.add_prediction_time(e - s1, s1 - s)
    make_output(dsn1, dsn2, test_directory, simulation, result, time_section, True, True)

def parse_arg():
    parser = argparse.ArgumentParser()
    parser.add_argument('--model_name', default="KNN", help="model type, allow 'lstm', 'tree-lstm', 'KNN'")
    parser.add_argument('--test_directory', default=None, help="the script saving directory for test_rl program")
    parser.add_argument('--load_file', default=None, help="the path to model for evaluation")
    parser.add_argument('--input_index', type=int, default=8, help="short-way for switch evaluation model,"
                                "hardcoded, not recommanded to change for use")
    parser.add_argument('--time_selection', default='original', help="the time label you want to use, allow "
    "'original', 'adjust', the 'adjust' stand for 'z3' by now, modify when you experiment with other solver")
    parser.add_argument('--regression', action='store_true', help="used for time prediction(regression),"
        "not use for timeout constraint classification(classification)")
    parser.add_argument('--adapt', action='store_true', help="an adaptive time threshold for neural network "
        "models used for regression, because the predicted timeout threshold varies for different programs")
    parser.add_argument('--threshold', type=int, default=200, help="the timeout threshold for solving")
    parser.add_argument('--batch-size', type=int, default=64, help="some lstm setting in case you change the model")
    parser.add_argument('--x-size', type=int, default=300)
    parser.add_argument('--h-size', type=int, default=150)
    parser.add_argument('--epochs', type=int, default=40)
    parser.add_argument('--num_classes', type=float, default=2)
    args = parser.parse_args()
    print()
    print("Simulation start:")
    print(args)
    return args

if __name__ == '__main__':
    args = parse_arg()
    model_name = args.model_name
    input_index = args.input_index
    # hardcoded short-way for switch evaluation model
    input_list = ["checkpoints/simulation/g_serial_pad_feature_l_z_r_200.pkl",#0
                  "checkpoints/simulation/g_serial_tree_feature_t_z_r_200.pkl",#1
                  "checkpoints/simulation/g_tree+feature_t_z_r_200.pkl",#2
                  "checkpoints/simulation/b_serial_pad_feature_l_z_r_200.pkl",#3
                  "checkpoints/simulation/b_serial_tree_feature_t_z_r_200.pkl",#4
                  "checkpoints/simulation/b_tree+feature_t_z_r_200.pkl",#5
                  "checkpoints/simulation/s_serial_pad_feature_l_z_r_200.pkl",#6
                  "checkpoints/simulation/s_tree_feature_t_z_r_200.pkl",#7
                  "data/gnucore/fv2_serial/train",#8
                  "data/busybox/fv2_serial/train",#9
                  "data/smt-comp/fv2_serial/train",#10
                  "data/klee/fv2_serial/train",#11
                  "checkpoints/simulation/k_serial_pad_feature_l_z_r_200.pkl",#12
                  "checkpoints/simulation/k_serial_tree_feature_l_z_r_200.pkl"]#13
    if args.load_file == None and (args.input_index > 13 or args.input_index < 0):
        print("these paths are hardcoded shortway for specific directory name")
        print(input_list)
        exit(0)
    # test_rl for all programs in a dataset, the home directory is "data/gnucore/single_test"
    # test_input_list = []
    # for root, dir, files in os.walk("data/gnucore/single_test"):
    #     if not root.endswith("single_test"):
    #         test_input_list.append(root)

    # for i in test_input_list:
    #     input = i
    #     simulation_for_single_program(test_directory, input_index)

    if args.test_directory:
        test_directory = args.test_directory
    else:
        test_directory = "data/example/arch"
    # some test_rl
    # test_directory = "data/smt-comp/QF_BV/Sage"
    # test_directory = "data/klee/arch-43200/solver-queries.smt2"
    simulation_for_single_program(test_directory, args)

    # make_PCC_output("data/PCC_result/mcm_c.json", False)

    # regression simulation, not remember much, different time threshold
    # input = "checkpoints/smt-comp/serial_pad_feature_evaluation_c.pkl"
    # if os.path.exists(input):
    #     serial_result = th.load(input)["result"]
    # else:
    #     serial_result = []
    #     for i in range(1, 4):
    #         a = th.load(input[:-4] + "_" + str(i) + ".pkl")["result"]
    #         serial_result.extend(a)
    # result = serial_result
    # pred = np.array(list(map(lambda x:x[0], result)))
    # truth = np.array(list(map(lambda x:x[1], result)))
    # for a in [40,50,60,100,150,200,250]:
    #     if truth.dtype == "int64":
    #         t, p = truth, pred
    #     else:
    #         t, p = truth > a, pred > a
    #         print("threshold", a)
    #     acc = accuracy_score(t, p)
    #     pre = precision_score(t, p)
    #     rec = recall_score(t, p)
    #     f1 = f1_score(t, p)
    #     print('test_rl accuracy: {:.3}, precision: {:.3}, recall: {:.3}, f1 score: {:.3}'.format(acc, pre, rec, f1))
    #     if truth.dtype == "int64":
    #         break
    # try:
    #     fpr, tpr, thresholds = precision_recall_curve(truth > a, pred)
    #     pyplot.plot(tpr, fpr, lw=1, label="lstm")
    #     # print(fpr)
    #     # print(tpr)
    #     # print(thresholds)
    #     i = np.searchsorted(thresholds, a)
    #     print(fpr[i], tpr[i], thresholds[i])
    #     pyplot.xlim([0.00, 1.0])
    #     pyplot.ylim([0.00, 1.0])
    #     pyplot.xlabel("False Positive Rate")
    #     pyplot.ylabel("True Positive Rate")
    #     pyplot.title("ROC")
    #     pyplot.legend(loc="lower right")
    #     pyplot.savefig(r"./ROC.png")
    #     pyplot.show()
    # except (IndexError, ValueError):
    #     pass