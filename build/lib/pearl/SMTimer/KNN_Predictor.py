import os
import json

from .dgl_treelstm.KNN import KNN
import numpy as np
from .preprocessing import Vector_Dataset

class Predictor:
    model = None
    dataset = Vector_Dataset(feature_number_limit=2)
    filename = None
    load_file = None
    timeout_threshold = 200

    def __init__(self, filename, timeout_threshold=200, load_file="gnucore"):
        Predictor.load_file = load_file
        Predictor.filename = filename
        Predictor.timeout_threshold = timeout_threshold
        self.init_static()
        self.remove_name = False
        self.x = np.zeros((1, 300))

    @staticmethod
    def init_static():
        base_dir = os.path.dirname(os.path.abspath(__file__))
        try:
            with open(base_dir + "/KNN_training_data/" + Predictor.load_file, "r") as f:
                train_dataset = json.load(f)
        except (IOError, ValueError):
            with open(base_dir + "/KNN_training_data/gnucore.json", "r") as f:
                train_dataset = json.load(f)
        Predictor.model = KNN(k=3)
        y_train = np.array([1 if i > Predictor.timeout_threshold else 0 for i in train_dataset["adjust"]])
        x_train = np.array(train_dataset["x"])
        Predictor.model.fit(x_train, y_train)
        try:
            Predictor.model.filename = np.array(train_dataset["filename"])
            Predictor.model.remove_test(Predictor.filename)
        except (KeyError):
            pass

    def predict(self, script):
        if not Predictor.model:
            Predictor.init_static()
        if Predictor.filename != "" and self.remove_name == False:
            try:
                Predictor.model.remove_test(Predictor.filename)
            except (KeyError):
                pass
            self.remove_name = True
        model = Predictor.model
        try:
            dataset = Predictor.dataset.generate_feature_dataset([script], time_selection="z3")
        except (KeyError,IndexError) as e:
            print(e)
            return 0
        self.x = np.array(dataset[-1].feature).reshape(-1, 300)
        pred = model.predict(self.x)[0]
        return pred

    def increment_KNN_data(self, truth):
        Predictor.model.incremental(self.x, truth)
