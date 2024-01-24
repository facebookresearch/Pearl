import json

from preprocessing.feature_structure import FV2
from preprocessing import Vector_Dataset


def construct_data_from_json(train_file):
    with open (train_file, "r") as f:
        fvdata = json.load(f)
    dataset = Vector_Dataset(feature_number_limit=2)
    fs = []
    for ind in range(len(fvdata['x'])):
        new_fv = FV2(None)
        new_fv.from_json(fvdata['x'][ind], fvdata['original'][ind], fvdata['adjust'][ind], fvdata['filename'][ind])
        fs.append(new_fv)
    dataset.fs_list = fs
    return fs