import argparse
import collections

import json
import os
import time
import random
import numpy as np
# from matplotlib import pyplot
from dataset_filename_seperation import get_dataset_seperation

np.set_printoptions(suppress=True)
import torch as th
import torch.nn.init as INIT
import torch.optim as optim
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader

import dgl
from check_time import process_data, z3fun, getlogger
from dgl_treelstm.trainer import Trainer,LSTM_Trainer
from dgl_treelstm.nn_models import TreeLSTM, LSTM, RNN, DNN
from dgl_treelstm.metric import Metrics
from dgl_treelstm.util import extract_root
from dgl_treelstm.dgl_dataset import dgl_dataset
from preprocessing import Tree_Dataset,Vocab,Constants,Vector_Dataset
import torch.nn.utils.rnn as rnn_utils

SSTBatch = collections.namedtuple('SSTBatch', ['graph', 'wordid', 'label', 'filename'])
FTBatch = collections.namedtuple('FTBatch', ['feature', 'label', 'filename', 'data_len'])
def batcher(device):
    def batcher_dev(batch):
        tree_batch = [x[0] for x in batch]
        try:
            batch_trees = dgl.batch(tree_batch, node_attrs=["y", "x"])
        except:
            for i in tree_batch:
                print(i.ndata['x'])

        return SSTBatch(graph=batch_trees,
                        wordid=batch_trees.ndata['x'].to(device),
                        label=batch_trees.ndata['y'].to(device),
                        filename=[x[1] for x in batch])
    return batcher_dev

def pad_feature_batcher(device, time_selection="original", task="regression", threshold=60):
    def batcher_dev(batch):
        # x = th.Tensor([item.feature for item in batch])
        x = [th.Tensor(item.feature) for item in batch]
        data_length = [len(sq) for sq in x]
        if time_selection == "original":
            y = th.Tensor([item.origin_time for item in batch])
        else:
            y = th.Tensor([item.adjust_time for item in batch])
        if task != "regression":
            y = th.LongTensor([1 if item > threshold else 0 for item in y])
        try:
            x = rnn_utils.pad_sequence(x, batch_first=True)
        except:
            print("error")
        return FTBatch(feature=x,
                        label=y,
                        filename=[item.filename for item in batch],
                       data_len=data_length)
    return batcher_dev

def feature_batcher(device, time_selection="original", task="regression", threshold=60):
    def batcher_dev(batch):
        x = th.Tensor([item.feature for item in batch])
        # x = [th.Tensor(item.feature) for item in batch]
        # data_length = [len(sq) for sq in x]
        if time_selection == "original":
            y = th.Tensor([item.origin_time for item in batch])
        else:
            y = th.Tensor([item.adjust_time for item in batch])
        if task != "regression":
            y = th.LongTensor([1 if item > threshold else 0 for item in y])
        # x = rnn_utils.pad_sequence(x, batch_first=True)
        return FTBatch(feature=x,
                        label=y,
                        filename=[item.filename for item in batch],
                       data_len=None)
    return batcher_dev

def main(args):
    np.random.seed(args.seed)
    th.manual_seed(args.seed)
    th.cuda.manual_seed(args.seed)

    best_epoch = -1

    device = th.device('cuda:{}'.format(args.gpu)) if th.cuda.is_available() else th.device('cpu')
    if device != th.device('cpu'):
        th.cuda.set_device(args.gpu)

    smt_vocab_file = 'smt.vocab'
    smt_vocab = Vocab(filename=smt_vocab_file,
                      data=[Constants.UNK_WORD])

    try:
        pretrained_emb = th.load('smt.pth')
    except:
        pretrained_emb = th.zeros(smt_vocab.size(), 150)
        for word in smt_vocab.labelToIdx.keys():
            pretrained_emb[smt_vocab.getIndex(word), smt_vocab.getIndex(word)] = 1
        th.save(pretrained_emb, './data/gnucore/smt.pth')
    if args.model == "lstm":
        model = LSTM(args.h_size, args.regression, args.attention)
    elif args.model == "rnn":
        model = RNN(args.h_size, args.regression)
    elif args.model == "dnn":
        model = DNN(args.regression)
    else:
        model = TreeLSTM(smt_vocab.size(),
                         150,
                         args.h_size,
                         args.num_classes,
                         args.dropout,
                         args.regression,
                         args.attention,
                         cell_type='childsum' if args.child_sum else 'childsum',
                         pretrained_emb = pretrained_emb)
    model.to(device)
    print(model)

    metrics = Metrics(args.num_classes)
    if args.regression:
        metric_name = "Mse"
        criterion = nn.MSELoss()
        metric = metrics.msereducebysum
        best_dev_metric = float("inf")
        task = "regression"
        metric_list = [metrics.mse, metrics.mae, metrics.pearson]
    else:
        metric_name = "Accuracy"
        criterion = nn.CrossEntropyLoss(reduction='sum')
        metric = metrics.confusion_matrix
        best_dev_metric = -1
        task = "classification"
        metric_list = [metrics.right_num, metrics.confusion_matrix, metrics.f1_score]

    optimizer = optim.Adagrad(filter(lambda p: p.requires_grad,
                                        model.parameters()), lr=args.lr, weight_decay=args.weight_decay)

    train_dataset, test_dataset = load_dataset(args)

    # test_rl
    if args.load_file is not None:
        checkpoint = th.load('checkpoints/{}.pkl'.format(args.load_file))
        model.load_state_dict(checkpoint['model'])
        optimizer = checkpoint['optim']

        qd = Tree_Dataset()
        qd.fs_list = test_dataset + train_dataset
        # test_filename = set([i.filename for i in qd.fs_list])
        test_filename = checkpoint["args"].test_filename
        loss_list = []
        if args.model == "lstm" or args.model == "rnn":
            trainer = LSTM_Trainer(args, model, criterion, optimizer, device, metric, metric_name)
        elif args.model == "tree-lstm":
            trainer = Trainer(args, model, criterion, optimizer, device, metric, metric_name)
        else:
            trainer = LSTM_Trainer(args, model, criterion, optimizer, device, metric, metric_name)

        if not args.single_program:
            _,test_dataset = qd.split_with_filename(test_filename)
            dataset, metric_dic, results, total_loss = evaluate_once(args, device, metric_list, metric_name,
                                                                     pretrained_emb, smt_vocab, task, test_dataset,
                                                                     trainer)
        else:
            from collections import defaultdict
            metric_dic = defaultdict(list)
            results = []
            for fn in test_filename:
                print(fn)
                _, test_dataset = qd.split_with_filename([fn])
                if len(test_dataset) == 0:
                    continue
                dataset, metric_single, result, total_loss = evaluate_once(args, device, metric_list, metric_name,
                                                                         pretrained_emb, smt_vocab, task, test_dataset,
                                                                         trainer)
                loss_list.append(total_loss / len(dataset))
                for key,value in metric_single.items():
                    metric_dic[key].append(value)
                results.append(result)
                print(result)

        checkpoint = {
            'model': model.state_dict(),
            'optim': optimizer,
            'metric': metric_name,
            'metric_value': metric_dic,
            'args': args,
            'result': results
        }
        print("------------------------------------------")
        for item in metric_dic.items():
            print(item)
        # print(checkpoint)
        dir = args.load_file[0]
        model_name = '_'.join([dir, 'evaluation', "r" if args.regression else "c"])
        if args.cross_index >= 0:
            model_name = '_'.join([model_name, str(args.cross_index)])
        th.save(checkpoint, 'checkpoints/{}.pkl'.format(model_name))
        return

    if args.model == "tree-lstm":
        trainer = Trainer(args, model, criterion, optimizer, device, metric, metric_name)
        random.shuffle(train_dataset)
        train_dataset = dgl_dataset(train_dataset, pretrained_emb, smt_vocab, task, args.time_selection, args.threshold)
        test_dataset = dgl_dataset(test_dataset, pretrained_emb, smt_vocab, task, args.time_selection, args.threshold)

        train_loader = DataLoader(dataset=train_dataset,
                                  batch_size=args.batch_size,
                                  collate_fn=batcher(device),
                                  shuffle=True,
                                  num_workers=0)
        test_loader = DataLoader(dataset=test_dataset,
                                 batch_size=100, collate_fn=batcher(device), shuffle=False, num_workers=0)
    elif args.model == "lstm" or args.model == "rnn":
        trainer = LSTM_Trainer(args, model, criterion, optimizer, device, metric, metric_name)

        train_loader = DataLoader(dataset=train_dataset,
                                  batch_size=args.batch_size,
                                  collate_fn=pad_feature_batcher(device, args.time_selection, task, args.threshold),
                                  shuffle=True,
                                  num_workers=0)
        test_loader = DataLoader(dataset=test_dataset,
                                 batch_size=100, collate_fn=pad_feature_batcher(device, args.time_selection, task, args.threshold), shuffle=False, num_workers=0)
    else:
        trainer = LSTM_Trainer(args, model, criterion, optimizer, device, metric, metric_name)
        train_loader = DataLoader(dataset=train_dataset,
                                  batch_size=args.batch_size,
                                  collate_fn=feature_batcher(device, args.time_selection, task, args.threshold),
                                  shuffle=True,
                                  num_workers=0)
        test_loader = DataLoader(dataset=test_dataset,
                                 batch_size=100, collate_fn=feature_batcher(device, args.time_selection, task, args.threshold), shuffle=False, num_workers=0)

    # training
    for epoch in range(args.epochs):
        t_epoch = time.time()

        total_result, total_loss = trainer.train(train_loader)

        print("==> Epoch {:05d} | Train Loss {:.4f} | {:s} {:.4f} | Time {:.4f}s".format(
            epoch, total_loss / len(train_dataset), metric_name, total_result, time.time() - t_epoch))


        total_result, total_loss, _, _ = trainer.test(test_loader)

        print("==> Epoch {:05d} | Dev Loss {:.4f} | {:s} {:.4f}".format(
            epoch, total_loss / len(test_dataset), metric_name, total_result))

        dev_metric = total_result

        if (args.regression and dev_metric < best_dev_metric) or (not args.regression and dev_metric > best_dev_metric):
            best_dev_metric = dev_metric
            best_epoch = epoch
            checkpoint = {
                'model': model.state_dict(),
                'optim': optimizer,
                'metric': metric_name,
                'metric_value': dev_metric,
                'args': args, 'epoch': epoch
            }
            checkpoint_name = args.input
            if "/" in args.input:
                name_list = args.input.split("/")
                dataset_name = name_list[-2][0]
                checkpoint_name = name_list[-1]
            mt = "r" if args.regression else "c"
            ts = str(args.threshold)
            model_name = '_'.join([dataset_name, checkpoint_name, args.model[0], args.time_selection[0], mt, ts])
            if args.cross_index >= 0:
                model_name = '_'.join([model_name, str(args.cross_index)])
            th.save(checkpoint, 'checkpoints/{}.pkl'.format(model_name))
        # else:
        #     if best_epoch <= epoch - 20:
        #         break
        #     pass

        # lr decay
        for param_group in optimizer.param_groups:
            # if (epoch + 1) % 10 == 0:
            #     param_group['lr'] = max(1e-5, param_group['lr'] * 0.8)  # 10
            # else:
            param_group['lr'] = max(1e-5, param_group['lr']*0.99) #10
            # print(param_group['lr'])

    print('------------------------------------------------------------------------------------')
    print("Epoch {:05d} | Test {:s} {:.4f}".format(
        best_epoch, metric_name, best_dev_metric))

    # total_result, total_loss = trainer.test_rl(test_loader)
    #
    # print("==> Epoch {:05d} | Test Loss {:.4f} | {:s} {:.4f}".format(
    #     epoch, total_loss / len(test_dataset), metric_name, total_result / len(test_dataset)))


def evaluate_once(args, device, metric_list, metric_name, pretrained_emb, smt_vocab, task, test_dataset, trainer):
    if args.model == "lstm" or args.model == "rnn":
        dataset = test_dataset
        test_loader = DataLoader(dataset=test_dataset,
                                 batch_size=100,
                                 collate_fn=pad_feature_batcher(device, args.time_selection, task, args.threshold),
                                 shuffle=False, num_workers=0)
    elif args.model == "tree-lstm":
        dataset = dgl_dataset(test_dataset, pretrained_emb, smt_vocab, task, args.time_selection, args.threshold)
        test_loader = DataLoader(dataset=dataset,
                                 batch_size=100, collate_fn=batcher(device), shuffle=False, num_workers=0)
    else:
        dataset = test_dataset
        test_loader = DataLoader(dataset=test_dataset,
                                 batch_size=100,
                                 collate_fn=feature_batcher(device, args.time_selection, task,
                                                            args.threshold),
                                 shuffle=False, num_workers=0)
    print("test_rl data:", len(test_dataset))
    total_result, total_loss, pred_tensor, label_tensor = trainer.test(test_loader)
    print("==> Test Loss {:.4f} | {:s} {:.4f}".format(
        total_loss / len(dataset), metric_name, total_result))
    metric_dic = {}
    # metric_list = [metrics.accuracy, metrics.confusion_matrix, metrics.f1_score]
    for m in metric_list:
        metric_dic[m.__name__] = m(pred_tensor, label_tensor)
    pred_tensor = [i.item() for i in pred_tensor]
    label_tensor = [i.item() for i in label_tensor]
    results = list(zip(pred_tensor, label_tensor))
    # print(results)
    return dataset, metric_dic, results, total_loss


# def load_file(args):
#     dataset = Tree_Dataset().generate_feature_dataset(args.data_source)
#     return dataset


# do feature extraction or load the saved dataset for training, return the list of data of corresponding feature
# structure that split with program name
def load_dataset(args):
    # choose feature dataset structure
    if args.model == "tree-lstm":
        dataset_type = Tree_Dataset
        feature_limit = 100
    else:
        dataset_type = Vector_Dataset
        feature_limit = 50
    #
    output_dir = args.input
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    train_file = os.path.join(output_dir, 'train')
    test_file = os.path.join(output_dir, 'test_rl')
    if args.cross_index < 0:
        fn_index = 0
    else:
        fn_index = args.cross_index
    test_filename = get_dataset_seperation(output_dir)[fn_index]

    dataset = []
    if os.path.isfile(train_file):
        train_dataset = th.load(train_file)
        if train_dataset == None:
            train_dataset = []
        try:
            ind = 0
            while(os.path.exists(train_file + str(ind))):
                train_dataset.extend(th.load(train_file + str(ind)))
                ind = ind + 1
        except IOError:
            pass
        if os.path.isfile(test_file):
            test_dataset = th.load(test_file)
        else:
            qd = dataset_type(feature_number_limit=feature_limit)
            qd.fs_list = train_dataset
            dataset = train_dataset
            # qd.fs_list = list(filter(lambda x:x.adjust_time > 1, qd.fs_list))
            if "smt-comp" in train_file:
                if args.random_test:
                    test_filename = list(set([x.filename for x in train_dataset]))
                    test_filename = list(set(x.split("_")[0] for x in test_filename))
                    l = int(len(test_filename) / 3)
                    test_filename = test_filename[:l]

                # if "smt-comp" in train_file and not args.load:
                #     qd.fs_list = list(filter(lambda x:x.adjust_time > 1, qd.fs_list))
                print("select program:", test_filename)
                test_filename1 = [x.filename for x in train_dataset]
                test_filename = list(filter(lambda x:x.split("_")[0] in test_filename, test_filename1))
            else:
                if args.random_test:
                    test_filename = list(set([x.filename for x in train_dataset]))
                    random.shuffle(test_filename)
                    l = int(len(test_filename) / 3)
                    test_filename = test_filename[:l]
                print("select program:", test_filename)
            train_dataset, test_dataset = qd.split_with_filename(test_filename)
            # train_dataset = train_dataset + test_dataset
    else:
        treeforassert = args.tree_for_assert
        qd = dataset_type(feature_number_limit=feature_limit, treeforassert=treeforassert, save_address=train_file)
        dataset = qd.generate_feature_dataset(args.data_source, args.time_selection)
        try:
            ind = 0
            while(os.path.exists(train_file + str(ind))):
                dataset.extend(th.load(train_file + str(ind)))
                ind = ind + 1
        except IOError:
            pass
        train_dataset, test_dataset = qd.split_with_filename(test_filename)

    if args.augment:
        qd = dataset_type(feature_number_limit=feature_limit)
        augment_path = os.path.join(args.augment_path, 'train')
        if os.path.isfile(augment_path):
            aug_dataset = th.load(augment_path)
            aug_dataset = list(filter(lambda x:x.adjust_time > 1, aug_dataset))
        else:
            print("augment data not found through the path")
            aug_dataset = []
        train_dataset = train_dataset + aug_dataset

    if not os.path.isfile(train_file):
        th.save(dataset, train_file)
    if len(train_dataset) == 0 or len(test_dataset) == 0:
        exit(0)
    print("train data:", len(train_dataset), "test_rl data:", len(test_dataset))
    args.test_filename = test_filename
    # del qd
    return train_dataset, test_dataset


def parse_arg():
    # global args
    parser = argparse.ArgumentParser()
    parser.add_argument('--gpu', type=int, default=0, help="the gpu number you want to use")
    parser.add_argument('--seed', type=int, default=41, help="random seed")
    parser.add_argument('--model', default='tree-lstm', help="model type, allow 'lstm', 'rnn', 'dnn', " 
                            "'tree-lstm', make sure match data when reuse processed data")
    parser.add_argument('--batch-size', type=int, default=64)
    parser.add_argument('--child-sum', action='store_true')
    parser.add_argument('--x-size', type=int, default=300)
    parser.add_argument('--h-size', type=int, default=150)
    parser.add_argument('--epochs', type=int, default=40)
    parser.add_argument('--log-every', type=int, default=40)
    parser.add_argument('--lr', type=float, default=0.05)
    parser.add_argument('--weight-decay', type=float, default=1e-3)
    parser.add_argument('--dropout', type=float, default=0.5)
    parser.add_argument('--num_classes', type=float, default=2)
    parser.add_argument('--data_source', default='data/gnucore/script_dataset/training', help="scripts saving directory")
    parser.add_argument('--input', default='data/gnucore/training', help="saving directory of feature after "
                            "extraction, avoid duplicate preprocess")
    parser.add_argument('--regression', action='store_true', help="used for time prediction(regression), "
                        "not use for timeout constraint classification(classification)")
    parser.add_argument('--attention', action='store_true')
    parser.add_argument('--single_program', action='store_true', help="evaluation model with single programs")
    parser.add_argument('--load_file', default=None, help="the path to model for evaluation")
    parser.add_argument('--single_test', action='store_true', help="test_rl for single script, not maintained")
    parser.add_argument('--time_selection', default='original', help="the time label you want to use, allow "
     "'original', 'z3', more type need data from different solvers e.g., 'msat', you may collect by your own")
    parser.add_argument('--tree_for_assert', action='store_true', help="true to use abstract trees as tree nodes"
                "in tree-LSTM model")
    parser.add_argument('--augment', action='store_true', help="make you own augment, not maintained")
    parser.add_argument('--augment_path', default='data/gnucore/augment/crosscombine')
    parser.add_argument('--random_test', action='store_true', help="random separation for program for test_rl")
    parser.add_argument('--threshold', type=int, default=200, help="the timeout threshold for solving, must less than 300")
    parser.add_argument('--cross_index', type=int, default=-1, help="cross-validation of data, must less than 3")
    args = parser.parse_args()
    print(args)
    return args


if __name__ == '__main__':
    args = parse_arg()
    main(args)