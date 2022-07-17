
import argparse
import torch
import numpy as np
def boolean_string(s):
    if s not in {'False', 'True'}:
        raise ValueError('Not a valid boolean string')
    return s == 'True'

def ret_args():
    parser = argparse.ArgumentParser()

    # Original args:
    parser.add_argument('data_key', type=str, default="ges",
                        nargs='?')
    parser.add_argument('model_key', type=str, default="fcn",
                        nargs='?')
    parser.add_argument('attn_key', type=str, default="group_attn",
                        nargs='?')
    parser.add_argument('batch_control', type=str, default='false',
                        nargs='?')
    parser.add_argument('ga_sigma', type=float, default=1.0,
                        nargs='?')
    parser.add_argument('data_pre_key', type=str, default="",
                        nargs='?') # Can be "" or "mts" or "uts"

    # dataset settings
    parser.add_argument('--data_path', type=str, default="./data/",
                        help='the path of data.')
    parser.add_argument('--dataset', type=str, default="NATOPS", #NATOPS
                        help='time series dataset. Options: See the datasets list')

    # cuda settings
    parser.add_argument('--no-cuda', action='store_true', default=False,
                        help='Disables CUDA training.')
    parser.add_argument('--seed', type=int, default=-1, help='Random seed.')

    # Training parameter settings
    parser.add_argument("--batch_size", type=int, default=100)
    parser.add_argument('--epochs', type=int, default=4,
                        help='Number of epochs to train.')
    parser.add_argument('--lr', type=float, default=1e-4,
                        help='Initial learning rate. default:[0.00001]')
    parser.add_argument('--wd', type=float, default=1e-3,
                        help='Weight decay (L2 loss on parameters). default: 5e-3')
    parser.add_argument('--stop_thres', type=float, default=1e-9,
                        help='The stop threshold for the training error. If the difference between training losses '
                            'between epoches are less than the threshold, the training will be stopped. Default:1e-9')

    # Model parameters


    parser.add_argument('--use_cnn', type=boolean_string, default=True, 
                        help='whether to use CNN for feature extraction. Default:False')
    parser.add_argument('--use_lstm', type=boolean_string, default=True,
                        help='whether to use LSTM for feature extraction. Default:False')
    parser.add_argument('--use_rp', type=boolean_string, default=True,
                        help='Whether to use random projection')
    parser.add_argument('--rp_params', type=str, default='-1,3',
                        help='Parameters for random projection: number of random projection, '
                            'sub-dimension for each random projection')
    parser.add_argument('--use_metric', action='store_true', default=False,
                        help='whether to use the metric learning for class representation. Default:False')
    parser.add_argument('--metric_param', type=float, default=0.01,
                        help='Metric parameter for prototype distances between classes. Default:0.000001')
    parser.add_argument('--filters', type=str, default="256,256,128",
                        help='filters used for convolutional network. Default:256,256,128')
    parser.add_argument('--kernels', type=str, default="8,5,3",
                        help='kernels used for convolutional network. Default:8,5,3')
    parser.add_argument('--dilation', type=int, default=1,
                        help='the dilation used for the first convolutional layer. '
                            'If set to -1, use the automatic number. Default:-1')
    parser.add_argument('--layers', type=str, default="500,300",
                        help='layer settings of mapping function. [Default]: 500,300')
    parser.add_argument('--dropout', type=float, default=0,
                        help='Dropout rate (1 - keep probability). Default:0.5')
    parser.add_argument('--lstm_dim', type=int, default=128,
                        help='Dimension of LSTM Embedding.')
    args = parser.parse_args()
    args.cuda = not args.no_cuda and torch.cuda.is_available()

    if args.seed >= 0:
        sdfsd
        np.random.seed(args.seed)
        torch.manual_seed(args.seed)
        if args.cuda:
            torch.cuda.manual_seed(args.seed)
    args.sparse = True
    args.layers = [int(l) for l in args.layers.split(",")]
    args.kernels = [int(l) for l in args.kernels.split(",")]
    args.filters = [int(l) for l in args.filters.split(",")]
    args.rp_params = [float(l) for l in args.rp_params.split(",")]

    # if not args.use_lstm and not args.use_cnn:
    #     print("Must specify one encoding method: --use_lstm or --use_cnn")
    #     print("Program Exiting.")
    #     exit(-1)

    # print("\nParameters:")
    # for attr, value in sorted(args.__dict__.items()):
    #     print("\t{}={}".format(attr.upper(), value))


    # print("Loading dataset", args.dataset, "...")

    return args

if __name__ == '__main__':
    args = ret_args()
    print(args)
