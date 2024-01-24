import json
import os
import random
import sys
import gc
# signal used to stop extremely long time of parse, no supported on Windows, remove if needed
import traceback
import signal

from .feature_extraction import Script_Info, feature_extractor
from .feature_structure import AST
from pearl.SMTimer.preprocessing.abstract_tree_extraction import abstract_tree_extraction

gnucore_file_list = ["[", "chmod", "dd", "expr", "hostid", "md5sum", "nproc", "ptx", "sha224sum", "stdbuf", "touch", "unlink", "b2sum", "chown", "df",
"factor","id","mkdir","numfmt","pwd","sha256sum","stty","tr","uptime","base32","chroot","dir","false","join","mkfifo",
"od","readlink","sha384sum","sum","true","users","base64","cksum","dircolors","fmt","kill","mknod","paste","realpath",
"sha512sum","sync","truncate","vdir","basename","comm","dirname","fold","link","mktemp","pathchk","rm","shred","tac",
"tsort","wc","basenc","cp", "du", "getlimits", "ln", "mv", "pinky", "rmdir", "shuf", "tail", "tty", "who", "cat", "csplit", "echo",
                     "ginstall", "logname", "nice", "pr", "runcon", "sort", "tee", "uname", "whoami", "chcon", "cut", "env", "groups", "ls", "nl",
                     "printenv", "seq", "split", "test_rl", "unexpand", "yes", "chgrp", "date", "expand", "head", "make-prime-list", "nohup", "printf",
                     "sha1sum", "stat", "timeout", "uniq"]

test_filename = ["echo", "ginstall", "expr", "tail", "seq", "split", "test_rl", "yes", "chgrp", "date", "expand", "head",
            "nohup", "printf", "sha1sum", "stat", "timeout", "uniq", "nice", "pr"]

def handler(signum, frame):
    signal.alarm(1)
    raise TimeoutError

# input all kinds of scripts and return expression tree
class Dataset:
    def __init__(self, feature_number_limit=100, treeforassert=False, save_address=None):
        self.script_list = []
        self.Script_Info_list = []
        self.fs_list = []
        self.is_json = True
        self.input_filename_list = []
        self.script_filename_list = []
        self.treeforassert = treeforassert
        self.feature_number_limit = feature_number_limit
        self.klee = False
        self.selected_file = False
        self.save_address = save_address

    # read data from file directory or script, preprocess scripts into abstract trees
    def generate_feature_dataset(self, input, time_selection=None, fileprefix=None):
        self.script_list = []
        if isinstance(input, list):
            self.script_list = input
            self.input_filename_list = [""]
        elif isinstance(input, str) and '\n' in input:
            self.script_list = [input]
            self.input_filename_list = [""]
        else:
            self.load_from_directory(input, fileprefix)
            if "klee" in input:
                self.klee = True
            if len(self.input_filename_list) == 0:
                return
            self.read_from_file(self.input_filename_list[0], "")
            self.judge_json(self.script_list[0])
        output_ind = 0
        selected_filename = []
        for ind, string in enumerate(self.input_filename_list):
            if string != "":
                self.script_list = []
                self.read_from_file(string, "")
            for script_string in self.script_list:
                script = Script_Info(script_string, self.is_json)
                try:
                    if time_selection != "original":
                        check_time_selection = time_selection
                    else:
                        check_time_selection = "z3"
                    if script.solving_time_dic[check_time_selection][0] < 0:
                        continue
                    if not self.selected_file and len(self.input_filename_list) > 35000:
                        if script.solving_time < 20 and script.solving_time_dic[check_time_selection][0] < 10:
                            if ind % 10 != 0:
                                continue
                    selected_filename.append(string.split("/")[-1])
                except:
                    pass
                signal.signal(signal.SIGALRM, handler)
                signal.alarm(1)
                try:
                    ret = self.parse_data(script, time_selection)
                    print(ret)
                    self.fs_list.append(ret)
                    self.script_filename_list.append(string.split("/")[-1])
                except TimeoutError:
                    signal.alarm(0)
                    # traceback.print_exc()
                    print("preprocess over time", len(self.fs_list))
                except (KeyError, AttributeError):
                    traceback.print_exc()
                    continue
                finally:
                    signal.alarm(0)
                output_ind = self.print_and_write(output_ind)
        if not self.selected_file and len(selected_filename) != None and len(self.script_filename_list) > 30000:
            with open(os.path.dirname(input) + "/selected_file.txt", "w") as f:
                for i in selected_filename:
                    f.write(i + "\n")
        del self.script_list
        print("feature extraction finished.")
        return self.fs_list

    def print_and_write(self, output_ind):
        if len(self.fs_list) % 500 == 0:
            print("not implement, don't matter if memory is enough.")

    # should not be used
    def parse_data(self, script, time_selection):
        print("not implement")
        if not self.treeforassert and not self.klee:
            # my own parse for angr and smt-comp has been abandoned,to construct tree for asserts,please refer to pysmt
            featurestructure = feature_extractor(script, time_selection, self.feature_number_limit)
            featurestructure.treeforassert = self.treeforassert
        else:
            featurestructure = abstract_tree_extraction(script, time_selection, self.feature_number_limit)
            featurestructure.treeforassert = self.treeforassert
        featurestructure.script_to_feature()
        ast = AST(featurestructure)
        print(ast)
        del featurestructure.logic_tree, featurestructure.feature_list
        del featurestructure
        del script
        return ast

    # for data augment, to be mention, cut, combine and replace operator is not enough to generate high quality scripts
    def augment_scripts_dataset(self, input):
        if isinstance(input, list):
            self.script_list = input
        elif isinstance(input, str) and '\n' in input:
            self.script_list = [input]
        else:
            self.load_from_directory(input)
        self.judge_json(self.script_list[0])
        for string in self.script_list:
            script = Script_Info(string, self.is_json)
            self.Script_Info_list.append(script)
        return self.Script_Info_list

    # only accept files with single script or specific design of KLEE "QF_AUFBV", to avoid running out of memory
    # we only record the input file name into
    def load_from_directory(self, input, fileprefix=None):
        if not input or input == "":
            return
        if os.path.isdir(input):
            selected_file = None
            try:
                with open(os.path.dirname(input) + "/selected_file.txt") as f:
                    selected_file = f.read().split("\n")
                self.selected_file = True
            except:
                selected_file = None
            for root, dirs, files in os.walk(input):
                files.sort(key=lambda x: (len(x), x))
                for file in files:
                    if fileprefix != None:
                        if file.startswith(fileprefix):
                            self.input_filename_list.append(os.path.join(root, file))
                        continue
                    if selected_file and file not in selected_file:
                        continue
                    if file.endswith("txt"):
                        continue
                    # if os.path.getsize(os.path.join(root, file)) > 512 * 1024:
                    #     continue
                    self.input_filename_list.append(os.path.join(root, file))
                    # self.read_from_file(file, os.path.join(root, file))
        elif os.path.exists(input):
            self.read_from_file(None, input)

    # specific design of KLEE "QF_AUFBV"
    def read_from_file(self, input, file=None):
        with open(input) as f:
            # if os.path.getsize(input) > 512 * 1024 or "klee" in input:
            if "klee" in input and "single_test" not in input:
                next = False
                start = False
                script = ""
                while(True):
                    try:
                        text_line = f.readline()
                        if text_line == "":
                            break
                    except:
                        continue
                    if "(set-logic QF_AUFBV )" in text_line:
                        start = True
                    if start:
                        script = script + text_line
                    if next == True:
                        self.script_list.append(script)
                        start = False
                        next = False
                        script = ""
                        if len(self.script_list) % 200 == 0:
                            print(len(self.script_list))
                    if "(exit)" in text_line:
                        next = True
            else:
                data = f.read()
                if data != "":
                    self.script_list.append(data)
                    # self.input_filename_list.append(file)

    def judge_json(self, data):
        try:
            json.loads(data)
            self.is_json = True
        except:
            pass

    # split dataset with program file name, the script of program with input file name would be the test_rl data
    def split_with_filename(self, test_filename=None):
        if not test_filename:
            random.shuffle(gnucore_file_list)
            test_filename = gnucore_file_list[:10]
        train_dataset = []
        test_dataset = []
        trt = 0
        tet = 0
        for qt in self.fs_list:
            if qt.filename in test_filename:
                test_dataset.append(qt)
                if qt.gettime() >= 100:
                    tet += 1
            else:
                train_dataset.append(qt)
                if qt.gettime() >= 100:
                    trt += 1
        if len(train_dataset) == 0:
            print("split training data has no data")
            # exit(0)
        if len(test_dataset) == 0:
            print("split testing data has no data")
            # exit(0)
        return train_dataset, test_dataset