from __future__ import division
from __future__ import print_function

import math
import sys
import time

import torch.optim as optim
from tapnet.models import TapNet
from tapnet.utils import *
import torch.nn as nn
import torch.nn.functional as F



# x_train: 
def run_tapnet(args, x_train, y_train, x_test, y_test, logger, csa_version=0, ga_sigma=0.0, class_id_list=None):
    features, labels, idx_train, idx_val, idx_test, nclass = load_raw_ts(x_train, y_train, x_test, y_test)
    # update random permutation parameter
    if args.rp_params[0] < 0:
        dim = features.shape[1]
        args.rp_params = [3, math.floor(dim / (3 / 2))]
    else:
        dim = features.shape[1]
        args.rp_params[1] = math.floor(dim / args.rp_params[1])
    
    args.rp_params = [int(l) for l in args.rp_params]
    logger.info("rp_params:" + str(args.rp_params))

    # update dilation parameter
    if args.dilation == -1:
        args.dilation = math.floor(features.shape[2] / 64)

    logger.info("Data shape:" + str(features.size()))
    model = TapNet(nfeat=features.shape[1],
                   len_ts=features.shape[2],
                   layers=args.layers,
                   nclass=nclass,
                   dropout=args.dropout,
                   use_lstm=args.use_lstm,
                   use_cnn=args.use_cnn,
                   filters=args.filters,
                   dilation=args.dilation,
                   kernels=args.kernels,
                   use_metric=args.use_metric,
                   use_rp=args.use_rp,
                   rp_params=args.rp_params,
                   lstm_dim=args.lstm_dim,
                   csa_version=csa_version,
                   ga_sigma=ga_sigma
                   )


    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    logger.info("Device: " + str(device))

    if csa_version != 0:
        for i in range(len(class_id_list)):
            class_id_list[i] = class_id_list[i].to(device)
        model.csa_model.class_id_list = class_id_list
        model.csa_model.store_attn = model.csa_model.store_attn.to(device)

    # cuda
    model.to(device)
    features, labels, idx_train, idx_test = features.to(device), labels.to(device), idx_train.to(device), idx_test.to(device)
    input = (features, labels, idx_train, idx_val, idx_test)
    # print("feature:" + str(features.shape))
    # print("labels:" + str(labels.shape))
    # print("idx_train:" + str(idx_train))
    # print("idx_val:" + str(idx_val))
    # print("idx_test:" + str(idx_test))

    # init the optimizer
    optimizer = optim.Adam(model.parameters(),
                        lr=args.lr, weight_decay=args.wd)
    # Train model
    t_total = time.time()
    best_val_accuracy = train(args, model, optimizer, input, logger)
    # print("Optimization Finished!")
    # print("Total time elapsed: {:.4f}s".format(time.time() - t_total))
    logger.info("Optimization Finished!")
    train_time = time.time() - t_total
    logger.info("Total time elapsed: {:.4f}s".format(train_time))

    # Testing
    t_total = time.time()
    test_accuracy = test(args, model, input, logger)
    test_time = time.time() - t_total

    return best_val_accuracy, train_time, test_accuracy, test_time


# training function
def train(args, model, optimizer, input, logger):
    features, labels, idx_train, idx_val, idx_test = input
    loss_list = [sys.maxsize]
    test_best_possible, best_so_far = 0.0, sys.maxsize

    for epoch in range(args.epochs):
        t = time.time()
        model.train()
        optimizer.zero_grad()

        output, proto_dist = model(input, 1) # For this version, do not need to have epoch is 0 and csa_training False

        loss_train = F.cross_entropy(output[idx_train], torch.squeeze(labels[idx_train]))
        if args.use_metric:
            loss_train = loss_train + args.metric_param * proto_dist

        # if abs(loss_train.item() - loss_list[-1]) < args.stop_thres \
        #         or loss_train.item() > loss_list[-1]:
        #     break
        # else:
        #     loss_list.append(loss_train.item())
        loss_list.append(loss_train.item())

        acc_train = accuracy(output[idx_train], labels[idx_train])
        if model.csa_version != 0:
            model.csa_model.store_attn.detach_()
        loss_train.backward()
        optimizer.step()

        loss_val = F.cross_entropy(output[idx_val], torch.squeeze(labels[idx_val]))
        acc_val = accuracy(output[idx_val], labels[idx_val])

        print_str = ('Epoch: {:04d}'.format(epoch + 1),
              'loss_train: {:.8f}'.format(loss_train.item()),
              'acc_train: {:.4f}'.format(acc_train.item()),
              'loss_val: {:.4f}'.format(loss_val.item()),
              'acc_val: {:.4f}'.format(acc_val.item()),
              'time: {:.4f}s'.format(time.time() - t))
        # print(print_str)
        logger.info(print_str)

        if acc_val.item() > test_best_possible:
            test_best_possible = acc_val.item()
        if best_so_far > loss_train.item():
            best_so_far = loss_train.item()
            test_acc = acc_val.item()
    logger.info("test_acc: " + str(test_acc))
    logger.info("best possible: " + str(test_best_possible))
    return test_best_possible

# test function
def test(args, model, input, logger):
    features, labels, idx_train, idx_val, idx_test = input
    output, proto_dist = model(input)
    loss_test = F.cross_entropy(output[idx_test], torch.squeeze(labels[idx_test]))
    if args.use_metric:
        loss_test = loss_test - args.metric_param * proto_dist

    acc_test = accuracy(output[idx_test], labels[idx_test])
    print_str = (args.dataset, "Test set results:",
          "loss= {:.4f}".format(loss_test.item()),
          "accuracy= {:.4f}".format(acc_test.item()))
    logger.info(print_str)
    return acc_test


