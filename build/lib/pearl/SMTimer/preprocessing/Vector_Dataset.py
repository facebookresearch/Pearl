from .feature_structure import FV, FV2
from .Dataset import Dataset, handler
from .feature_vector_extraction import *

class Vector_Dataset(Dataset):

    def print_and_write(self, output_ind):
        if len(self.fs_list) % 500 == 0:
            print("processed script number:" + str(len(self.fs_list)))

    def parse_data(self, script, time_selection):
        featurevectors = feature_vector_extraction(script, time_selection, self.feature_number_limit)
        featurevectors.script_to_feature()
        if self.feature_number_limit == 2:
            fv = FV2(featurevectors)
        else:
            fv = FV(featurevectors)
        return fv