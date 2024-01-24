import sys

from pysmt.operators import __OP_STR__

sys.setrecursionlimit(1000000)
# from Tree import varTree as Tree
from preprocessing.feature_extraction import *

class abstract_tree_extraction(feature_extractor):
    # def __init__(self, script_info, time_selection="original"):
    #     query_tree.__init__(script_info, time_selection)

    def script_to_feature(self):
        data = self.script_info.script
        self.cal_training_label()
        assertions = self.handle_variable_defination(data)
        # for var_name in self.val_list:
        #     data = data.replace(var_name, self.val_dic[var_name])
        try:
            # parse assertion stack into abstract trees
            self.assertions_to_feature_list(assertions)
            # merging sub tree: bottom_up_merging or accumulation
            self.accumulation()
            # self.bottom_up_merging()
            # truncate tree by depth. default 60
            self.cut_length()
            # collecting tree structure information
            self.feature[-4] = self.logic_tree.node
            self.feature[-2] = self.logic_tree.depth
        except (KeyError,IndexError) as e:
            # print(e)
            self.logic_tree = vartree('unknown', None, None, None)

    def assertions_to_feature_list(self, asserts):
        limit = self.feature_number_limit
        new_str = asserts[0]
        if "QF_AUFBV" in asserts[0]:
            self.parse_klee_smt(asserts)
            return
        asserts = asserts[1:]
        if len(asserts) > limit:
            asserts[-limit] = "\n".join(asserts[:-limit + 1])
            asserts = asserts[-limit:]
        asserts_bool = []
        for assertion in asserts:
            if "assert" not in assertion:
                continue
            if assertion.count("\n") > 20 or assertion.count("assert") > 1:
                asserts_bool.append(True)
            else:
                asserts_bool.append(False)
                new_str += assertion
        assertions = new_str
        ind = 0
        try:
            assertions = assertions.replace("bvurem_i", "bvurem")
            assertions = assertions.replace("bvudiv_i", "bvudiv")
            smt_parser = SmtLibParser()
            script = smt_parser.get_script(cStringIO(assertions))
        except (KeyError,IndexError,pysmt.exceptions.PysmtTypeError):
            # traceback.print_exc()
            return
        try:
            assert_list = script.commands
            command_ind = 0
            while(command_ind < len(assert_list) and assert_list[command_ind].name != "assert"):
                command_ind += 1
            for assert_ind in range(len(asserts)):
                if asserts_bool[assert_ind] == True:
                    new_tree = self.assertion_to_tree(None, asserts[assert_ind])
                else:
                    new_tree = self.assertion_to_tree(assert_list[command_ind], asserts[assert_ind])
                    command_ind += 1
                if new_tree != None:
                    self.feature_list.append(new_tree)
        except (KeyError,IndexError):
            traceback.print_exc()
            return

    def assertion_to_tree(self, command, assertion):
        if assertion.count("\n") < 20 and assertion.count("assert") == 1:
            root = self.fnode_to_tree(command.args[0], 10)
        else:
            val = list(map(lambda x: math.log(x + 1), self.count_feature(assertion)))
            root = vartree(val)
        return root

    def fnode_to_tree(self, fnode, depth=0):
        if depth == 0:
            root = Tree("")
            root.val = np.log(self.fnode_to_feature(fnode) + 1).tolist()
            return root
        transtable = list(__OP_STR__.values())
        # print(fnode)
        if fnode.is_symbol():
            if fnode.symbol_name() in self.val_list:
                root = vartree(self.val_dic[fnode.symbol_name()])
            else:
                root = vartree("constant")
        elif fnode.is_constant():
            root = vartree("constant")
        elif fnode.is_term():
            if fnode.is_and() and fnode.arg(1).is_true():
                root = self.fnode_to_tree(fnode.arg(0), depth - 1)
            else:
                subnode_list = []
                for subnode in fnode.args():
                    subnode_list.append(self.fnode_to_tree(subnode, depth - 1))
                subnode_list.extend([None, None, None])
                root = vartree(op[fnode.node_type()], subnode_list[0], subnode_list[1], subnode_list[2])
        else:
            root = vartree("unknown")
        return root

    def fnode_to_feature(self, fnode):
        features = np.zeros(150)
        if fnode.is_symbol():
            if fnode.symbol_name() in self.val_list:
                ind = min(int(self.val_dic[fnode.symbol_name()][3:]), 20)
                features[111 + ind] += 1
            else:
                features[133] += 1
        elif fnode.is_constant():
            features[21] += 1
        elif fnode.is_term():
            features[fnode.node_type()] += 1
            for subnode in fnode.args():
                features += self.fnode_to_feature(subnode)
        return features

    #
    def parse_klee_smt(self, asserts):
        self.feature_list = []
        try:
            smt_parser = SmtLibParser()
            script = smt_parser.get_script(cStringIO("".join(asserts)))
        except (KeyError,IndexError):
            return
        feature_list = []
        for assertion in script.commands:
            if assertion.name != "assert":
                continue
            fnode = assertion.args[0]
            while fnode.is_and():
                if self.treeforassert:
                    feature_list.append(self.fnode_to_tree(fnode.arg(1), 10))
                # elif self.feature_number_limit != 2:
                else:
                    # feature_list.append(self.fnode_to_feature(fnode.arg(1)))
                    feature_list.append(self.count_feature(fnode.arg(1).to_smtlib()))
                # else:
                #     pass
                fnode = fnode.arg(0)
            if self.treeforassert:
                feature_list.append(self.fnode_to_tree(fnode, 10))
            # elif self.feature_number_limit != 2:
            else:
                # feature_list.append(self.fnode_to_feature(fnode))
                feature_list.append(self.count_feature(fnode.to_smtlib()))
        if len(feature_list) != 0:
            # feature_list.reverse()
            self.feature_list.extend(feature_list)