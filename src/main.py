from logging import error
import sys
import time
import numpy as np

from parameter_proc import read_parameters

from fileio.data_io import train_test_file_reading, train_test_loading, data_group_processing, list_files, init_folder
from fileio.log_io import init_logging
from model.run_model import run_model
from args_process import ret_args
import os
import torch
torch.cuda.empty_cache()
# #import logging
# os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
# os.environ['TF_DETERMINISTIC_OPS'] = '1'
# os.environ['TF_CUDNN_DETERMINISTIC'] = 'true'
# # os.environ["CUDA_VISIBLE_DEVICES"] = "-1"
# os.environ["CUDA_VISIBLE_DEVICES"] = "0"
# os.environ['TF_FORCE_GPU_ALLOW_GROWTH'] = 'true'


# parameter_file: parameter file location
# file_keyword: file keyword to load the training and testing data
# function_key: the sub-folder name of the log and object files
def run_model_main(args, parameter_file, file_keyword):
    data_stru, model_setting = read_parameters(parameter_file, args.model_key+'_'+ args.attn_key + '_batch' + str(args.batch_control))

    data_pre_key = args.data_pre_key
    if data_pre_key == 'uts':
        data_stru.data_folder = data_stru.data_folder.replace('../data/', '../' + data_pre_key + "_data/")
        log_folder = model_setting.log_folder.replace("../log/", "../" + data_pre_key + "_log_pytorch_epoch" + str(args.epochs) + "/")
    else:
        log_folder = model_setting.log_folder.replace("../log/", "../log_pytorch_epoch" + str(args.epochs) + "/")

    if args.model_key == "cnn":
        # if data_stru.attr_len >= 200:
        #     exit()
        log_folder = log_folder.replace("log", "cnn_log")
    print(data_stru.data_folder)
    # print(model_setting.to_string())
    model_setting.read_setting()
    print("====")
    print(data_stru.to_string())
    print(model_setting.to_string())
    log_folder = init_folder(log_folder)
    model_setting.log_folder = log_folder
    out_obj_folder = init_folder(model_setting.out_obj_folder)
    out_model_folder = init_folder(model_setting.out_model_folder)

    file_list = list_files(data_stru.data_folder)
    file_count = 0

    class_column = 0
    header = True

    init_folder(out_obj_folder)
    init_folder(out_model_folder)

    result_obj_folder = model_setting.obj_folder + args.model_key + "_result_folder"
    result_obj_folder = init_folder(result_obj_folder)

    delimiter = ' '
    loop_count = -1

    for train_file in file_list:
        if file_keyword not in train_file:
            continue
        if 'y_train' in train_file:
            continue
        loop_count = loop_count + 1
        file_key = train_file.replace('.txt', '')
        saved_path = out_obj_folder + file_key + ".cpk"
        valid_file = data_stru.data_folder + train_file.replace('train', 'valid')
        if os.path.isfile(valid_file) is False:
            valid_file = ''

        test_file = data_stru.data_folder + train_file.replace('train', 'test')
        if os.path.isfile(test_file) is False:
            test_file = ''

        if data_pre_key == 'mts':
            data_group, attr_num = train_test_loading(data_stru.data_folder)
        else:
            data_group, attr_num = train_test_file_reading(
                data_stru.data_folder + train_file, test_file, valid_file,
                class_column, delimiter, header)
        data_group_processing(data_group, attr_num, args.model_key)

        data_group.data_check(data_stru.num_classes, data_stru.min_class)
        cnn_eval_key = model_setting.eval_method
        if cnn_eval_key == "f1":
            if data_stru.num_classes > 2:
                cnn_eval_key = "acc"

        log_file = log_folder + str(data_stru.data_key) + '_' + file_key + (
            '_') + args.model_key + "_" + args.attn_key + "_" + cnn_eval_key + "_bcgene" + str(args.batch_control) + "_gasigma" + str(args.ga_sigma) + '.log'
        print("log file: " + log_file)
        logger = init_logging(log_file)
        logger.info('\nlog file: ' + log_file)
        logger.info(train_file)
        logger.info('cnn setting:\n ' + model_setting.to_string())
        logger.info('method: ' + args.model_key)
        logger.info('ga_sigma: ' + str(args.ga_sigma))
        logger.info('learning rate: ' + str(args.lr))
        logger.info('============')
        if file_count == 0:
            logger.info('train matrix shape: ' +
                        str(data_group.train_x_matrix.shape))
            logger.info('train label shape: ' +
                        str(data_group.train_y_vector.shape))

        logger.info(data_group.train_x_matrix[0, 0:10, 0])
        print(model_setting, data_group, saved_path, logger)
        best_val_accuracy, train_time, test_accuracy, test_time = run_model(args, data_group, saved_path, logger)
        logger.info("best testing accuracy: {:.2f}".format(best_val_accuracy))
        logger.info("final testing accuracy: {:.2f}".format(test_accuracy))
        logger.info("training time: {:.2f}".format(train_time))
        logger.info("testing time: {:.2f}".format(test_time))

if __name__ == '__main__':
    args = ret_args()
    file_keyword = 'train'
    data_list = ["ges", "atn", "act"]
    model_dict = {1: "fcn", 2: 'cnn', 3: "mlstm", 4: "fcn-mlstm", 5: "tapnet", 6: "mlstm-trans", 7: "fcn-mlstm-trans", 8: "fcn-mlstm-trans"}
    attn_dict = {1: "group_attn", 2: "cross_attn", 3: "cross_group_attn", 4: "none", 5: "group_attn_base", 6: "cross_group_attn_base", 7:"group_attn_additional"} # "group_attn_additional" is only for tapnet

    try:
        model_key = int(args.model_key)
        args.model_key = model_dict[model_key]
    except Exception:
        if args.model_key not in model_dict.values():
            print ("Model Key is NOT supported!")
            exit()
    try:
        attn_key = int(args.attn_key)
        args.attn_key = attn_dict[attn_key]
    except Exception:
        if args.attn_key not in attn_dict.values():
            print ("Attention Key is NOT supported!")
            exit()
    data_key = args.data_key
    data_pre_key = args.data_pre_key
    print(args.data_key, args.model_key, args.attn_key, args.batch_control)
    if data_pre_key == 'uts' or data_pre_key == 'mts':
        parameter_file = '../' + data_pre_key + '_parameters/all_feature_classification_'
    else:
        parameter_file = '../parameters/all_feature_classification_'
    parameter_file += data_key + ".txt"
    print(parameter_file)
    run_model_main(args, parameter_file, file_keyword)
    
