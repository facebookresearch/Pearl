from __future__ import absolute_import

from collections import namedtuple
import networkx as nx

from dgl.graph import DGLGraph
from torch import nn

SMTBatch = namedtuple('SMTBatch', ['graph', 'wordid', 'label'])

class dgl_dataset(object):
    def __init__(self, data, embedding, vocab=None, task="regression", time_selection="original", time_threshold=200):
        self.trees = []
        self.task = task
        self.time_selection = time_selection
        self.filename_list = []
        self.vocab = vocab
        self.num_classes = 2
        self.time_threshold = time_threshold
        # self.embedding = nn.Embedding(133, 150)
        # self.embedding.weight.data.copy_(embedding)
        self._load(data)

    def _load(self, qt_list):
        # build trees
        for qt in qt_list:
            self.trees.append(self._build_tree(qt))
            try:
                self.filename_list.append(qt.filename)
            except:
                self.filename_list.append(None)

    def _build_tree(self, qt):
        root = qt.feature
        g = nx.DiGraph()
        def _rec_build(nid, root):
            for child in [root.left, root.mid, root.right]:
                if child:
                    cid = g.number_of_nodes()
                    try:
                        # word = self.vocab.labelToIdx[child.val]
                        word = self.featuretotensor(child.val)
                    except:
                        # print("unknown word", child.val)
                        word = [0] * 150
                        word[0] = 1
                    g.add_node(cid, x=word, y= 0)
                    g.add_edge(cid, nid)
                    _rec_build(cid, child)
        # add root
        solving_time = qt.gettime(self.time_selection)
        if self.task == "classification":
            if isinstance(solving_time, bool):
                result = 1 if solving_time else 0
            else:
                result = 1 if solving_time > self.time_threshold else 0
        else:
            result = solving_time
            if not result:
                result = 0.0
        if result == None:
            result = 0
        # g.add_node(0, x=self.vocab.labelToIdx[root.val], y=result)
        try:
            g.add_node(0, x=self.featuretotensor(root.val), y=result)
        except AttributeError:
            ret = [0.0] * 150
            ret[0] = 1.0
            g.add_node(0, x=ret, y=result)
        _rec_build(0, root)
        ret = DGLGraph()
        ret.from_networkx(g, node_attrs=['x', 'y'])
        return ret

    def featuretotensor(self, val):
        # return self.vocab.labelToIdx[val]
        if isinstance(val, list):
            return val
        elif val == None:
            ret = [0.0] * 150
            ret[0] = 1.0
        else:
            ind = self.vocab.labelToIdx[val]
            ret = [0.0] * 150
            ret[ind] = 1.0
            return ret

    def __getitem__(self, idx):
        return self.trees[idx], self.filename_list[idx]

    def __len__(self):
        return len(self.trees)

    @property
    def num_vocabs(self):
        return self.vocab.size()
