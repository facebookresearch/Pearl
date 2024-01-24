import json
import math
import sys

import traceback

import numpy as np
import warnings

sys.setrecursionlimit(1000000)


class AST:
    def __init__(self, feature_tree, filename=None):
        self.feature = feature_tree.logic_tree
        self.origin_time = feature_tree.origin_time
        self.adjust_time = feature_tree.adjust_time
        try:
            self.filename = feature_tree.script_info.filename
        except:
            self.filename = filename
        self.feature_sum = feature_tree.feature

    def gettime(self, time_selection="original"):
        try:
            if time_selection == "original":
                return self.origin_time
            else:
                return self.adjust_time
        except:
            return self.timeout


class FV:
    def __init__(self, feature_vector, filename=None):
        if feature_vector == None:
            self.feature = None
            self.origin_time = None
            self.adjust_time = None
            self.feature_sum = None
            self.filename = None
            return
        self.feature = feature_vector.feature_list
        self.origin_time = feature_vector.origin_time
        self.adjust_time = feature_vector.adjust_time
        try:
            self.filename = feature_vector.script_info.filename
        except (KeyError,IndexError):
            self.filename = filename
        # self.feature = [math.log(x + 1) for x in feature_vector.feature]
        self.feature_sum = feature_vector.feature

    def from_json(self, feature, origin_time, adjust_time, filename):
        self.feature = feature
        self.origin_time = origin_time
        self.adjust_time = adjust_time
        self.filename = filename

    def gettime(self, time_selection="original"):
        try:
            if time_selection == "original":
                return self.origin_time
            else:
                return self.adjust_time
        except (KeyError,IndexError):
            return self.timeout


class FV2(FV):
    def __init__(self, feature_vector, filename=None):
        FV.__init__(self, feature_vector, filename)
        if feature_vector == None:
            return
        feature_list = self.feature
        if len(feature_list) == 2:
            self.feature = feature_list.flatten()
        else:
            warnings.warn("the feature vector sum up should be done during parsing", DeprecationWarning)
            self.feature = np.zeros(300)
            self.feature[:150] = np.sum(feature_list[:-1], axis=0)
            self.feature[150:] = feature_list[-1]
            self.feature = np.log10(self.feature+1)