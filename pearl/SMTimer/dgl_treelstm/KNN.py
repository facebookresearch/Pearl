import numpy as np
import operator

class KNN(object):

    def __init__(self, k=3):
        self.k = k
        self.distance = []
        self.mask = False
        self.filename = None
        self.accept_error = False
        self.wait_append = {}

    def fit(self, x, y):
        self.x = np.array(x)
        self.y = np.array(y)
        self.construct_knowledge()
        # self.x = np.array(x)[self.distance_index]
        # self.y = np.array(y)[self.distance_index]
        if self.mask:
            self.x[:, 112:150] = 0
            self.x[:, 262:300] = 0

    def construct_knowledge(self):
        self.l = len(self.x)
        self.x_sum = np.sum(self.x, axis=1)
        self.x_sum_index = np.argsort(self.x_sum)
        self.x_sum = self.x_sum[self.x_sum_index]
        self.distance = np.linalg.norm(self.x, axis=1)
        self.distance_index = np.argsort(self.distance)
        self.distance = self.distance[self.distance_index]

    def remove_test(self, test_filename):
        if self.filename is not None:
            test_in_train = np.array(list(map(lambda x: x == test_filename, self.filename)))
            index = np.argwhere(test_in_train == False).reshape(-1)
            self.x = self.x[index]
            self.y = self.y[index]
            self.filename = self.filename[index]
            self.construct_knowledge()

    def _square_distance(self, x, index=None):
        if index:
            return np.sum(np.square(x - self.x[index]), axis=1)
        return np.sum(np.square(x - self.x), axis=1)

    def _vote(self, ys):
        votes_sum = sum(ys)
        ys_unique = np.unique(ys)
        vote_dict = {}
        for y in ys:
            if y not in vote_dict.keys():
                vote_dict[y] = 1
            else:
                vote_dict[y] += 1
        sorted_vote_dict = sorted(vote_dict.items(), key=operator.itemgetter(1), reverse=True)
        return sorted_vote_dict[0][0], votes_sum

    def predict(self, x, y=None):
        y_pred = []
        if isinstance(y, int):
            x, y = np.array([x]), np.array([y])
        for i in range(len(x)):
            dist_arr = self._square_distance(x[i])
            sorted_index = np.argsort(dist_arr)
            top_k_index = sorted_index[:self.k]
            y_vote, y_vote_sum = self._vote(ys=self.y[top_k_index])
            y_pred.append(y_vote)
        return np.array(y_pred)

    def score(self, y_true=None, y_pred=None):
        if y_true is None and y_pred is None:
            y_pred = self.predict(self.x)
            y_true = self.y
        score = 0.0
        positive = 0
        true = 0
        true_positive = 0
        for i in range(len(y_true)):
            if y_true[i] == y_pred[i]:
                score += 1
            if y_true[i] == 1:
                true += 1
            if y_pred[i] == 1:
                positive += 1
            if y_true[i] == 1 and y_pred[i] == 1:
                true_positive += 1
        acc = score / len(y_true)
        try:
            precision = true_positive / positive
        except ZeroDivisionError:
            precision = 0.0
        try:
            recall = true_positive / true
        except ZeroDivisionError:
            recall = 0.0
        try:
            f1 = 2 * precision * recall / (precision + recall)
        except ZeroDivisionError:
            f1 = 0.0
        return acc, precision, recall, f1

    def generate_suitable_list(self, x_sum, distance):
        a = list(filter(lambda j:abs(x_sum - self.x_sum[j]) < 1 or abs(distance - self.distance[j]) < 0.5, range(len(self.x))))
        if(len(a) < 100):
            a = list(filter(lambda j: abs(x_sum - self.x_sum[j]) < 2.5 or abs(distance - self.distance[j]) < 2,
                            range(len(self.x))))
        elif (len(a) > 6000):
            a = list(filter(lambda j: abs(x_sum - self.x_sum[j]) < 1 and abs(distance - self.distance[j]) < 0.5,
                            range(len(self.x))))
        # print(len(a))
        return a

    def fast_incremental_predict(self, x, y):
        y_pred = []
        if isinstance(y, int):
            x, y = [x], [y]
        for i in range(len(x)):
            x_sum = np.sum(x[i])
            distance = np.linalg.norm(x[i])
            ind = np.searchsorted(self.distance, distance)
            ind1 = np.searchsorted(self.x_sum, x_sum)
            ind_set = set()
            wide = 200
            ind_set.update(self.distance_index[list(range(max(0, ind - wide), min(ind + wide, self.l)))])
            ind_set.update(self.x_sum_index[list(range(max(0, ind1 - wide), min(ind1 + wide, self.l)))])
            index = list(ind_set) + list(range(self.l, len(self.x)))
            dist_arr = self._square_distance(x[i], index)
            # index = self.generate_suitable_list(x_sum, distance)
            # distance_arr = np.array([abs(distance - self.distance[j]) for j in range(len(self.x))])
            x_t = self.x[index]
            y_t = self.y[index]
            # dist_arr = np.sum(np.asarray(x[i] - x_t)**2, axis=1)
            # sorted_index = np.argsort(dist_arr)
            # top_k_index = sorted_index[:self.k]
            try:
                top_k_index = np.argpartition(dist_arr, self.k)[:self.k]
            except ValueError:
                top_k_index = np.array(range(len(dist_arr)))
            # dis = distance_arr[top_k_index]
            # print(dis)
            y_vote, y_vote_sum = self._vote(ys=y_t[top_k_index])
            y_pred.append(y_vote)
            if y_pred[-1] == 1:
                continue
            else:
                self.x = np.append(self.x, [x[i]], axis=0)
                self.x_sum = np.append(self.x_sum, x_sum)
                self.distance = np.append(self.distance, distance)
                self.y = np.append(self.y, y[i])
        return np.array(y_pred)

    def incremental_predict(self, x, y):
        y_pred = []
        if isinstance(y, int):
            x, y = np.array([x]), np.array([y])
        if self.mask:
            x[:, 112:150] = 0
            x[:, 262:300] = 0
        for i in range(len(x)):
            dist_arr = self._square_distance(x[i])
            # factor = np.square(x[i] - self.x)
            # factor_max = np.max(factor, axis=1)
            try:
                top_k_index = np.argpartition(dist_arr, self.k)[:self.k]
                # top_l_index = np.argsort(dist_arr)[:10]
            except ValueError:
                top_k_index = np.array(range(len(dist_arr)))
                # top_l_index = np.array(range(len(dist_arr)))
            # c = self.x[top_l_index].tolist()
            # a = c - x[i]
            # a1 = top_l_index.tolist()
            y_vote, y_vote_sum = self._vote(ys=self.y[top_k_index])
            y_pred.append(y_vote)
            if y_pred[-1] != y[i]:
                top_l_index = np.argsort(dist_arr)[:10]
                b = self.filename[top_l_index].tolist()
                b1 = self.y[top_l_index].tolist()
                c1 = dist_arr[top_l_index].tolist()
                a = 1
            if y_pred[-1] == 1:
                if not self.accept_error:
                    continue
                else:
                    self.x = np.append(self.x, [x[i]], axis=0)
                    self.y = np.append(self.y, y_pred[-1])
                    self.filename = np.append(self.filename, "")
            else:
                self.x = np.append(self.x, [x[i]], axis=0)
                self.y = np.append(self.y, y[i])
                self.filename = np.append(self.filename, "")
        return np.array(y_pred)

    def incremental(self, x, y, y_pred=None):
        if y_pred == 1:
            if not self.accept_error:
                return
            else:
                self.x = np.append(self.x, x, axis=0)
                self.y = np.append(self.y, y_pred)
                self.filename = np.append(self.filename, "")
        else:
            self.x = np.append(self.x, x, axis=0)
            self.y = np.append(self.y, y)
            self.filename = np.append(self.filename, "")

    def test(self, x):
        return self.predict(x)