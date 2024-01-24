import time
import numpy as np

import torch as th

from dgl_treelstm.util import extract_root
import torch.nn.functional as F
import torch.nn.utils.rnn as rnn_utils

class Trainer(object):
    def __init__(self, args, model, criterion, optimizer, device, metric, metric_name):
        super(Trainer, self).__init__()
        self.args = args
        self.model = model
        self.criterion = criterion
        self.optimizer = optimizer
        self.device = device
        self.epoch = 0
        self.metric = metric
        self.metric_name = metric_name
        self.pred_tensor, self.label_tensor = None, None

    def cal_metrics_result(self, metric_result, dataset_len):
        if self.args.regression:
            result = metric_result / dataset_len
        else:
            confusion_matrix = metric_result
            precision = (confusion_matrix.diag() / confusion_matrix.sum(1))[1]
            recall = (confusion_matrix.diag() / confusion_matrix.sum(0))[1]
            result = 2 * precision * recall / (precision + recall)
            if th.isnan(result):
                result = 0
        return result

    # helper function for training
    def train(self, train_loader):
        self.model.train()
        total_loss = 0
        total_result = 0
        dur = []
        dataset_len = len(train_loader.dataset)
        for step, batch in enumerate(train_loader):
            g = batch.graph
            n = g.number_of_nodes()
            h = th.zeros((n, self.args.h_size)).to(self.device)
            c = th.zeros((n, self.args.h_size)).to(self.device)
            if step >= 3:
                t0 = time.time()  # tik
            logits = self.model(batch, h, c)
            batch_label, logits = extract_root(batch, self.device, logits)
            if self.args.regression:
                logits = logits.reshape(-1)
                loss = self.criterion(logits, batch_label)
                total_loss += loss * g.batch_size
                pred = logits
            else:
                loss = self.criterion(logits, batch_label)
                total_loss += loss
                pred = th.argmax(F.log_softmax(logits), 1)
            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()

            if step >= 3:
                dur.append(time.time() - t0)  # tok

            metric_result = self.metric(pred, batch_label)
            total_result += metric_result

            if step > 0 and step % self.args.log_every == 0:
                # if self.epoch % 10 == 9:
                #     print(th.transpose(th.cat((pred, batch_label)).reshape(2,-1), 0, 1))
                print("Epoch {:05d} | Step {:05d} | Loss {:.4f} | {:s} {:.4f} | Time(s) {:.4f}".format(
                    self.epoch, step, loss.item(), self.metric_name, self.cal_metrics_result(metric_result, g.batch_size), np.mean(dur)))
        self.epoch += 1
        total_result = self.cal_metrics_result(total_result, dataset_len)
        return total_result, total_loss

    # helper function for testing
    def test(self, test_loader):
        # eval on dev set
        pred_tensor = None
        label_tensor = None
        total_result = 0
        total_loss = 0
        self.model.eval()
        dataset_len = len(test_loader.dataset)
        for step, batch in enumerate(test_loader):
            g = batch.graph
            n = g.number_of_nodes()
            with th.no_grad():
                h = th.zeros((n, self.args.h_size)).to(self.device)
                c = th.zeros((n, self.args.h_size)).to(self.device)
                logits = self.model(batch, h, c)
            batch_label, logits = extract_root(batch, self.device, logits)
            if self.args.regression:
                logits = logits.reshape(-1)
                loss = self.criterion(logits, batch_label)
                total_loss += loss * g.batch_size
                pred = logits
            else:
                loss = self.criterion(logits, batch_label)
                total_loss += loss
                pred = th.argmax(F.log_softmax(logits), 1)
            metric_result = self.metric(pred, batch_label)
            total_result += metric_result
            if pred_tensor == None:
                pred_tensor = pred
                label_tensor = batch_label
            else:
                pred_tensor = th.cat([pred_tensor, pred], dim=-1)
                label_tensor = th.cat([label_tensor, batch_label], dim=-1)
            # if self.epoch % 10 == 0 and step == 0:
            #     print(th.transpose(th.cat((pred, batch_label)).reshape(2,-1), 0, 1))
        self.pred_tensor, self.label_tensor = pred_tensor, label_tensor
        total_result = self.cal_metrics_result(total_result, dataset_len)
        return total_result, total_loss, pred_tensor, label_tensor


class LSTM_Trainer(object):
    def __init__(self, args, model, criterion, optimizer, device, metric, metric_name):
        super(LSTM_Trainer, self).__init__()
        self.args = args
        self.model = model
        self.criterion = criterion
        self.optimizer = optimizer
        self.device = device
        self.epoch = 0
        self.metric = metric
        self.metric_name = metric_name
        self.pred_tensor, self.label_tensor = None, None

    def cal_metrics_result(self, metric_result, dataset_len):
        if self.args.regression or isinstance(metric_result, float):
            result = metric_result / dataset_len
        else:
            confusion_matrix = metric_result
            precision = (confusion_matrix.diag() / confusion_matrix.sum(1))[1]
            recall = (confusion_matrix.diag() / confusion_matrix.sum(0))[1]
            result = 2 * precision * recall / (precision + recall)
            if th.isnan(result):
                result = 0
        return result

    # helper function for training
    def train(self, train_loader):
        total_loss = 0
        total_result = 0
        self.model.train()
        dataset_len = len(train_loader.dataset)
        for step, batch in enumerate(train_loader):
            batch_feature = batch.feature.to(self.device)
            batch_label = batch.label.to(self.device)
            n = batch.feature.shape[0]
            if self.args.model == "lstm" or self.args.model == "rnn":
                batch_feature = rnn_utils.pack_padded_sequence(batch_feature, batch.data_len, enforce_sorted=False,
                                                           batch_first=True)
            if step >= 3:
                t0 = time.time()  # tik
            logits = self.model(batch_feature).to(self.device)
            if self.args.regression:
                logits = logits.reshape(-1)
                loss = self.criterion(logits, batch_label)
                total_loss += loss * n
                pred = logits
            else:
                loss = self.criterion(logits, batch_label)
                total_loss += loss
                pred = th.argmax(F.log_softmax(logits), 1)
            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()
            # max_time = th.Tensor([300.0]).to(self.device)
            # pred = pred.min(max_time)
            # batch_label = batch_label.min(max_time)
            metric_result = self.metric(pred, batch_label)
            total_result += metric_result

        self.epoch += 1
        total_result = self.cal_metrics_result(total_result, dataset_len)
        return total_result, total_loss

    # helper function for testing
    def test(self, test_loader):
        pred_tensor = None
        label_tensor = None
        total_result = 0
        total_loss = 0
        self.model.eval()
        dataset_len = len(test_loader.dataset)
        for step, batch in enumerate(test_loader):
            batch_feature = batch.feature.to(self.device)
            batch_label = batch.label.to(self.device)
            n = batch.feature.shape[0]
            if self.args.model == "lstm" or self.args.model == "rnn":
                batch_feature = rnn_utils.pack_padded_sequence(batch_feature, batch.data_len, enforce_sorted=False,
                                                           batch_first=True)
            with th.no_grad():
                logits = self.model(batch_feature).to(self.device)
            if self.args.regression:
                logits = logits.reshape(-1)
                loss = self.criterion(logits, batch_label)
                total_loss += loss * n
                pred = logits
            else:
                loss = self.criterion(logits, batch_label)
                total_loss += loss
                pred = th.argmax(F.log_softmax(logits), 1)
            # max_time = th.Tensor([300.0]).to(self.device)
            # pred = pred.min(max_time)
            # batch_label = batch_label.min(max_time)
            metric_result = self.metric(pred, batch_label)
            total_result += metric_result
            if pred_tensor == None:
                pred_tensor = pred
                label_tensor = batch_label
            else:
                pred_tensor = th.cat([pred_tensor, pred], dim=-1)
                label_tensor = th.cat([label_tensor, batch_label], dim=-1)
            # if self.epoch % 10 == 0 and step == 0:
            #     print(th.transpose(th.cat((pred, batch_label)).reshape(2,-1), 0, 1))
        self.pred_tensor, self.label_tensor = pred_tensor, label_tensor
        total_result = self.cal_metrics_result(total_result, dataset_len)
        return total_result, total_loss, pred_tensor, label_tensor