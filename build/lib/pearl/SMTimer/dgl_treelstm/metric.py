from copy import deepcopy
import numpy as np
import torch


class Metrics():
    def __init__(self, num_classes=2):
        self.num_classes = num_classes

    def confusion_matrix(self, predictions, labels):
        confusion_matrix = torch.zeros(2, 2)
        for t, p in zip(labels.view(-1), predictions.view(-1)):
            confusion_matrix[t.long(), p.long()] += 1
        # print("confusion_matrix: ")
        # print(confusion_matrix)
        return confusion_matrix

    def f1_score(self, predictions, labels):
        confusion_matrix = torch.zeros(2, 2)
        # for i in range(len(predictions)):
        #     confusion[int(predictions[i])][int(labels[i])] += 1
        for t, p in zip(labels.view(-1), predictions.view(-1)):
            confusion_matrix[t.long(), p.long()] += 1
        precision = (confusion_matrix.diag() / confusion_matrix.sum(1))[1]
        recall = (confusion_matrix.diag() / confusion_matrix.sum(0))[1]
        f1 = 2 * precision * recall / (precision + recall)
        if torch.isnan(f1):
            f1 = 0

        return f1

    def precision(self, predictions, labels):
        confusion_matrix = torch.zeros(2, 2)
        for t, p in zip(labels.view(-1), predictions.view(-1)):
            confusion_matrix[t.long(), p.long()] += 1
        precision = (confusion_matrix.diag() / confusion_matrix.sum(1))[1]
        return precision

    def recall(self, predictions, labels):
        confusion_matrix = torch.zeros(2, 2)
        for t, p in zip(labels.view(-1), predictions.view(-1)):
            confusion_matrix[t.long(), p.long()] += 1
        recall = (confusion_matrix.diag() / confusion_matrix.sum(0))[1]
        return recall

    def accuracy(self, predictions, labels):
        x = deepcopy(predictions)
        y = deepcopy(labels)
        return torch.eq(x, y).sum().float().item() / x.shape[0]

    def right_num(self, predictions, labels):
        x = deepcopy(predictions)
        y = deepcopy(labels)
        return torch.eq(x, y).sum().float().item()

    def pearson(self, predictions, labels):
        x = deepcopy(predictions)
        y = deepcopy(labels)
        x = (x - x.mean()) / x.std()
        y = (y - y.mean()) / y.std()
        return torch.mean(torch.mul(x, y))

    def mse(self, predictions, labels):
        # x = deepcopy(predictions)
        # y = deepcopy(labels)
        return torch.mean((predictions - labels) ** 2)

    def msereducebysum(self, predictions, labels):
        predictions = predictions.reshape(-1)
        return torch.sum((predictions - labels) ** 2)

    def mae(self, predictions, labels):
        # x = deepcopy(predictions)
        # y = deepcopy(labels)
        return torch.mean(torch.abs(predictions - labels))

    def maereducebysum(self, predictions, labels):
        predictions = predictions.reshape(-1)
        return torch.sum((predictions - labels))