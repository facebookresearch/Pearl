from preprocessing.abstract_tree_extraction import *
import numpy as np

class feature_vector_extraction(abstract_tree_extraction):
    # def __init__(self, script_info, time_selection="original"):
    #     query_tree.__init__(script_info, time_selection)

    def script_to_feature(self):
        data = self.script_info.script
        self.cal_training_label()
        assertions = self.handle_variable_defination(data)

        # for var_name in self.val_list:
        #     data = data.replace(var_name, self.val_dic[var_name])
        try:
            # parse assertion stack into expression trees
            self.assertions_to_feature_list(assertions)
            # summing up feature vectors of assertions
            self.standardlize()
        except (KeyError,IndexError) as e:
            # print(e)
            self.feature_list = np.zeros((self.feature_number_limit, 150))

    # the special take care of "QF_AUFBV" origins from the SMT generated from KLEE, all assertions are stuffed in one
    # command, so we parse it to get feature vectors for every assertion, so the processing time is much higher.
    def assertions_to_feature_list(self, assertions):
        limit = self.feature_number_limit
        if "QF_AUFBV" in assertions[0]:
            self.parse_klee_smt(assertions)
            self.feature = np.sum(self.feature_list, axis=0).tolist()
            return
        asserts = assertions[1:]
        if len(asserts) > limit:
            asserts[-limit] = "\n".join(asserts[:-limit + 1])
            asserts = asserts[-limit:]
        try:
            for assertion in asserts:
                feature = self.count_feature(assertion)
                self.feature_list.append(feature)
        except (KeyError,IndexError):
            traceback.print_exc()
            return

    # fix the length of feature vector, the length is determined by feature_number_limit,default length is 100 for tree,
    # 50 for rnn, 2 for KNN, then apply the logarithm to reduce the skewness
    def standardlize(self):
        limit = self.feature_number_limit
        if len(self.feature_list) == 0:
            self.feature_list = np.zeros((limit,150))
            return
        feature_list = np.array(self.feature_list)
        if len(feature_list) < limit:
            padding_num = limit - len(feature_list)
            feature_list = np.row_stack([feature_list, np.zeros([padding_num, 150])])
            self.feature_list = feature_list
        elif len(feature_list) > limit:
            feature_list[-limit] = np.sum(feature_list[:-limit + 1], axis=0)
            self.feature_list = feature_list[-limit:]
        self.feature = np.sum(feature_list, axis=0).tolist()
        if self.feature_number_limit == 2:
            self.feature_list = np.log10(np.array(self.feature_list)+1)
        else:
            self.feature_list = np.log(np.array(self.feature_list)+1)