import numpy as np
import time
from tapnet.run_tapnet import run_tapnet
from model.mlstm_fcn import run_mlstm_fcn
from model.data_load import torch_dataloader, train_tensorloader
from model.model_build import BuildModels
from model.utils import train, validation, load_datasets

import torch
import torch.optim as optim
from torchinfo import summary


def run_model(args, data_group, saved_path, logger):
    model_key = args.model_key
    # print(data_group.train_x_matrix.shape)
    # print(data_group.train_y_vector.shape)
    # print(model_key)
    if model_key == 'tapnet':
        train_y_shape = data_group.train_y_matrix.shape
        num_classes = train_y_shape[1]

        _, _, class_id_list = csa_preprocessing(data_group, num_classes)
        if args.attn_key == "group_attn":
            csa_version = 1
        elif args.attn_key == "group_attn_additional":
            csa_version = 2
        else:
            csa_version = 0
        return run_tapnet(args, data_group.train_x_matrix, data_group.train_y_vector, data_group.test_x_matrix, data_group.test_y_vector, logger, csa_version, args.ga_sigma, class_id_list)
    else:
        return run_model_with_batch(args, data_group, saved_path, logger)


def model_build(num_classes, in_shape, model_key, ga_sigma):
    model = BuildModels(num_classes=num_classes, in_shape=in_shape, model_key=model_key, ga_sigma=ga_sigma)
    # model = MLSTMfcn(num_classes=num_classes, max_seq_len=in_shape[0], num_features=in_shape[1])
    # if model_key == "fcn-mlstm":
    #     model = MLSTMfcn(num_classes=num_classes, max_seq_len=ts_length, num_features=var_number)
    # else:
    #     model = MLSTMfcn(num_classes=num_classes, max_seq_len=ts_length, num_features=var_number)
    return model


def model_train(args, model, train_loader, val_loader, train_x_tensor, train_y_tensor, class_id_list, saved_path, logger):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model.to(device)

    optimizer = optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.wd)
    criterion = torch.nn.NLLLoss()
    # criterion = torch.nn.CrossEntropyLoss()

    return train(logger, model, train_loader, val_loader, train_x_tensor, train_y_tensor,  class_id_list, criterion, optimizer, epochs=args.epochs, print_every=1, device=device, saved_path=saved_path)

def model_test(args, model, data_group):
    _, _, test_loader = torch_dataloader(data_group, args.batch_size)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    model.to(device)
    criterion = torch.nn.NLLLoss()
    # criterion = torch.nn.CrossEntropyLoss()

    test_loss, test_accuracy, validation_time = validation(model, test_loader, criterion, device)
    return test_loss, test_accuracy, validation_time


def run_model_with_batch(args, data_group, saved_path, logger):
    train_loader, val_loader, _ = torch_dataloader(data_group, args.batch_size)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    logger.info("Device: " + str(device))

    train_x_shape = data_group.train_x_matrix.shape
    train_y_shape = data_group.train_y_matrix.shape
    in_shape = train_x_shape[1:]
    num_classes = train_y_shape[1]

    train_x_tensor, train_y_tensor, class_id_list = csa_preprocessing(data_group, num_classes)

    model = BuildModels(num_classes=num_classes, in_shape=in_shape, model_key=args.model_key, attn_key=args.attn_key, ga_sigma=args.ga_sigma)

    logger.info(summary(model, input_size=(7,) + in_shape))
    # print(train_loader.shape)
    # print(val_loader.shape)
    # print(train_x_tensor.shape)
    # print(train_y_tensor.shape)
    # sdfdss
    best_val_accuracy, saved_path, train_time = model_train(args, model, train_loader, val_loader, train_x_tensor, train_y_tensor, class_id_list, saved_path, logger)
    test_loss, test_accuracy, test_time = model_test(args, model, data_group)
    logger.info("Model saved to: " + saved_path)
    return best_val_accuracy, train_time, test_accuracy, test_time


def csa_preprocessing(data_group, num_classes):
    train_x_tensor, train_y_tensor = train_tensorloader(data_group)
    # print(train_x_tensor.shape)
    # print(train_y_tensor.shape)
    class_id_list = []
    for i in range(num_classes):
        idx = (train_y_tensor.squeeze() == i).nonzero().squeeze(1)
        class_id_list.append(idx)
    # print(class_id_list)
    return train_x_tensor, train_y_tensor, class_id_list