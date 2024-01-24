import gc
# to do: use other serialize way to save the data. to allow KNN not use torch at all so that if you are intended not use
# tree structure, you may not use torch by remove torch dependency and usage in this file to run the only KNN model.
import torch as th

from .feature_extraction import Script_Info, feature_extractor
from .feature_structure import AST
from .abstract_tree_extraction import abstract_tree_extraction
from .Dataset import Dataset, handler

# input all kinds of scripts and return abstract tree
class Tree_Dataset(Dataset):

    def print_and_write(self, output_ind):
        if len(self.fs_list) % 500 == 0:
            print("processed script number for file" + str(output_ind) + ":" + str(len(self.fs_list)))
            gc.collect()
        if len(self.fs_list) == 5000:
            th.save(self.fs_list, self.save_address + str(output_ind))
            output_ind += 1
            del self.fs_list
            gc.collect()
            self.fs_list = []
        return output_ind

    def parse_data(self, script, time_selection):
        if not self.treeforassert and not self.klee:
            # my own parse for angr and smt-comp has been abandoned,to construct tree for asserts,please refer to pysmt
            featurestructure = feature_extractor(script, time_selection, self.feature_number_limit)
            featurestructure.treeforassert = self.treeforassert
        else:
            featurestructure = abstract_tree_extraction(script, time_selection, self.feature_number_limit)
            featurestructure.treeforassert = self.treeforassert
        featurestructure.script_to_feature()
        ast = AST(featurestructure)
        del featurestructure.logic_tree, featurestructure.feature_list
        del featurestructure
        del script
        return ast
