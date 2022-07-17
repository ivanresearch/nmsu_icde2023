import torch
import torch.nn as nn
import torch.nn.functional as F
from tapnet.utils import euclidean_dist, normalize, output_conv_size, dump_embedding
from model.model_build import CSAModule
import numpy as np

class TapNet(nn.Module):
    def __init__(self, nfeat, len_ts, nclass, dropout, filters, kernels, dilation, layers, use_rp, rp_params,
                 use_att=True, use_metric=False, use_lstm=False, use_cnn=True, lstm_dim=128, csa_version=0, ga_sigma=0.0):
        super(TapNet, self).__init__()
        self.nclass = nclass
        self.dropout = dropout
        self.use_metric = use_metric
        self.use_lstm = use_lstm
        self.use_cnn = use_cnn
        self.csa_version = csa_version
        self.ga_sigma = ga_sigma

        # parameters for random projection
        self.use_rp = use_rp
        # self.use_rp = False
        self.rp_group, self.rp_dim = rp_params

        if True:
            # LSTM
            self.channel = nfeat
            self.ts_length = len_ts

            self.lstm_dim = lstm_dim

            # N * T * V
            # Channel : V
            # T: ts_length
            # N * T * V ->  N * Hidden * V
            # N * V * T -> N * Hidden * T

            self.lstm = nn.LSTM(self.channel, self.lstm_dim)
            # self.lstm = nn.LSTM(self.ts_length, self.lstm_dim)

            paddings = ['same', 'same', 'same']
            if self.use_rp:
                self.conv_1_models = nn.ModuleList()
                self.idx = []
                for i in range(self.rp_group):
                    self.conv_1_models.append(nn.Conv1d(self.rp_dim, filters[0], kernel_size=kernels[0], dilation=dilation, stride=1, padding=paddings[0]))
                    self.idx.append(np.random.permutation(nfeat)[0: self.rp_dim])
            else:
                self.conv_1 = nn.Conv1d(self.channel, filters[0], kernel_size=kernels[0], dilation=dilation, stride=1, padding=paddings[0])

            self.conv_bn_1 = nn.BatchNorm1d(filters[0])

            self.conv_2 = nn.Conv1d(filters[0], filters[1], kernel_size=kernels[1], stride=1, padding=paddings[1])

            self.conv_bn_2 = nn.BatchNorm1d(filters[1])

            self.conv_3 = nn.Conv1d(filters[1], filters[2], kernel_size=kernels[2], stride=1, padding=paddings[2])

            self.conv_bn_3 = nn.BatchNorm1d(filters[2])

            # compute the size of input for fully connected layers
            fc_input = 0
            if self.use_cnn:
                conv_size = len_ts
                # for i in range(len(filters)):
                #     conv_size = output_conv_size(conv_size, kernels[i], stride=1, padding=paddings[i])
                fc_input += filters[2]
                #* filters[-1]
            if self.use_lstm:
                fc_input += self.lstm_dim
            
            if self.use_rp:
                fc_input = self.rp_group * filters[2] + self.lstm_dim
        # print(fc_input)
        # sdfsd
        if self.csa_version == 1:
            self.fc_list = nn.ModuleList()
            for _ in range(self.nclass):
                fc = nn.Linear(fc_input, 1)
                self.fc_list.append(fc)
            # print(self.nclass, filters[-1], fc_input, ga_sigma)
            self.csa_model = CSAModule(self.nclass, fc_input, len_ts, ga_sigma)
        elif self.csa_version == 2:
            self.csa_model = CSAModule(self.nclass, fc_input, len_ts, ga_sigma)

        # Representation mapping function
        layers = [fc_input] + layers
        print("Layers", layers)
        # self.mapping = nn.Sequential()
        # for i in range(len(layers) - 2):
        #     self.mapping.add_module("fc_" + str(i), nn.Linear(layers[i], layers[i + 1]))
        #     self.mapping.add_module("bn_" + str(i), nn.BatchNorm1d(layers[i + 1]))
        #     self.mapping.add_module("relu_" + str(i), nn.LeakyReLU())

        # # add last layer
        # self.mapping.add_module("fc_" + str(len(layers) - 2), nn.Linear(layers[-2], layers[-1]))
        # if len(layers) == 2:  # if only one layer, add batch normalization
        #     self.mapping.add_module("bn_" + str(len(layers) - 2), nn.BatchNorm1d(layers[-1]))

        # Attention
        att_dim, semi_att_dim = 128, 128
        self.use_att = use_att
        if self.use_att:
            self.att_models = nn.ModuleList()
            if csa_version == 2:
                att_in_dim = fc_input
            else:
                # att_in_dim = layers[-1]
                att_in_dim = fc_input
            for _ in range(nclass):

                att_model = nn.Sequential(
                    nn.Linear(att_in_dim, att_dim),
                    nn.Tanh(),
                    nn.Linear(att_dim, 1)
                )
                self.att_models.append(att_model)

    # input[0]: N * C * L, where L is the time-series length
    def forward(self, input, iter_epoch=0):
        x, labels, idx_train, idx_val, idx_test = input  # x is N * L, where L is the time-series feature dimension

        if True:
            N = x.size(0)

            # LSTM
            if self.use_lstm:
                x_lstm = self.lstm(x.transpose(2, 1))[0]
                # x_lstm = self.lstm(x)[0]
                if self.csa_version == 0:
                    # N * V * Hidden
                    # Yifan: N * T * Hidden
                    # LSTM: N * V * Hidden
                    # CSA 1 for LSTM
                    # FNC: N * T * Hidden
                    # CSA 2 fpr the FCN
                    # N * T * (2 * Hidden)
                    # CSA
                    x_lstm = x_lstm.mean(1)
                    # N * Hidden
                    # N * Hidden
                    x_lstm = x_lstm.view(N, -1)
                # elif self.csa_version == 1: # Replace everything after lstm and cnn to be CSA
            
            if self.use_cnn:
                # Covolutional Network
                # input ts: # N * C * L
                if self.use_rp:
                    for i in range(len(self.conv_1_models)):
                        #x_conv = x
                        x_conv = self.conv_1_models[i](x[:, self.idx[i], :])
                        x_conv = self.conv_bn_1(x_conv)
                        x_conv = F.leaky_relu(x_conv)
                        # print("Conv1: " + str(x_conv.shape))

                        x_conv = self.conv_2(x_conv)
                        x_conv = self.conv_bn_2(x_conv)
                        x_conv = F.leaky_relu(x_conv)
                        # print("Conv2: " + str(x_conv.shape))

                        x_conv = self.conv_3(x_conv)
                        x_conv = self.conv_bn_3(x_conv)
                        x_conv = F.leaky_relu(x_conv)
                        # print("Conv3: " + str(x_conv.shape))
                        # x_conv = torch.mean(x_conv, 2)
                        # print("Mean: " + str(x_conv.shape))
                        # sdfsd

                        if i == 0:
                            x_conv_sum = x_conv
                        else:
                            x_conv_sum = torch.cat([x_conv_sum, x_conv], dim=1)
                        print(x_conv_sum.shape)
                        
                    x_conv = x_conv_sum
                else:
                    x_conv = x
                    # print("1: " + str(x_conv.shape))
                    x_conv = self.conv_1(x_conv)  # N * C * L
                    x_conv = self.conv_bn_1(x_conv)
                    x_conv = F.leaky_relu(x_conv)
                    # print("2: " + str(x_conv.shape))

                    x_conv = self.conv_2(x_conv)
                    x_conv = self.conv_bn_2(x_conv)
                    x_conv = F.leaky_relu(x_conv)
                    # print("3: " + str(x_conv.shape))

                    x_conv = self.conv_3(x_conv)
                    x_conv = self.conv_bn_3(x_conv)
                    x_conv = F.leaky_relu(x_conv)
                    # print("4: " + str(x_conv.shape))

                    # x_conv = x_conv.view(N, -1)

            # print("final Conv: " + str(x_conv.shape))
            # print("final LSTM: " + str(x_lstm.shape))
            if self.csa_version == 1 or self.csa_version == 2:
                x_lstm = x_lstm.transpose(2, 1)
            else:
                x_conv = torch.mean(x_conv, 2)
            # print(x_conv.shape)
            # print(x_lstm.shape)
            # sdfsd
            if self.use_lstm and self.use_cnn:
                x = torch.cat([x_conv, x_lstm], dim=1)
            elif self.use_lstm:
                x = x_lstm
            elif self.use_cnn:
                x = x_conv
            #
            # print("Combined: " + str(x.shape))

        if self.csa_version == 1:
            # csa_train_x = x[idx_train].cuda()
            # csa_test_x = x[idx_test].cuda()
            csa_train_x = x[idx_train]
            csa_test_x = x[idx_test]
            # if iter_epoch == 0:
            #     train_out = self.csa_model.forward(csa_train_x, iter_epoch, False)
            #     test_out = self.csa_model.forward(csa_test_x, iter_epoch, False)
            # else:
            #     train_out = self.csa_model.forward(csa_train_x, iter_epoch, True)
            #     test_out = self.csa_model.forward(csa_test_x, iter_epoch, False)
            train_out = self.csa_model.forward(csa_train_x, iter_epoch, True)
            test_out = self.csa_model.forward(csa_test_x, iter_epoch, False)
            # csa_out: B * C * 128, where C is the num_classes
            # print(train_out.shape)
            # print(test_out.shape)
            csa_out = torch.cat([train_out, test_out], dim=0)
            # print(csa_out.shape)
            for i in range(self.nclass):
                class_input = csa_out[:, 0, :]
                class_output = self.fc_list[i](class_input)
                if i == 0:
                    outputs = class_output
                else:
                    outputs = torch.cat((outputs, class_output), 1)
            outputs = F.log_softmax(outputs, dim=1)
            # outptus: B * num
            return outputs, None
        elif self.csa_version == 0: # default
            # linear mapping to low-dimensional space
            # x = self.mapping(x)

            # generate the class protocal with dimension C * D (nclass * dim)
            proto_list = []
            for i in range(self.nclass):
                idx = (labels[idx_train].squeeze() == i).nonzero().squeeze(1)
                if self.use_att:
                    A = self.att_models[i](x[idx_train][idx])  # N_k * 1
                    A = torch.transpose(A, 1, 0)  # 1 * N_k
                    A = F.softmax(A, dim=1)  # softmax over N_k

                    class_repr = torch.mm(A, x[idx_train][idx]) # 1 * L
                    class_repr = torch.transpose(class_repr, 1, 0)  # L * 1
                else:  # if do not use attention, simply use the mean of training samples with the same labels.
                    class_repr = x[idx_train][idx].mean(0)  # L * 1
                proto_list.append(class_repr.view(1, -1))
            x_proto = torch.cat(proto_list, dim=0)
            # print("x_proto " + str(x_proto.shape))
            # prototype distance
            proto_dists = euclidean_dist(x_proto, x_proto)
            proto_dists = torch.exp(-0.5*proto_dists)
            num_proto_pairs = int(self.nclass * (self.nclass - 1) / 2)
            # print("proto_dists: " + str(proto_dists.shape))
            proto_dist = torch.sum(proto_dists) / num_proto_pairs
            # print("proto_dists: " + str(proto_dists.shape))
            dists = euclidean_dist(x, x_proto)
            # print("dists: " + str(dists.shape))
            dump_embedding(x_proto, x, labels)
            return torch.exp(-0.5*dists), proto_dist
        elif self.csa_version == 2: # Additional CSA module
            # linear mapping to low-dimensional space

            csa_train_x = x[idx_train].cuda()
            csa_test_x = x[idx_test].cuda()
            # csa_train_x = x[idx_train]
            # csa_test_x = x[idx_test]
            
            # if iter_epoch == 0:
            #     train_out = self.csa_model.forward(csa_train_x, iter_epoch, False)
            #     test_out = self.csa_model.forward(csa_test_x, iter_epoch, False)
            # else:
            #     train_out = self.csa_model.forward(csa_train_x, iter_epoch, True)
            #     test_out = self.csa_model.forward(csa_test_x, iter_epoch, False)
            train_out = self.csa_model.forward(csa_train_x, iter_epoch, True)
            test_out = self.csa_model.forward(csa_test_x, iter_epoch, False)
            # csa_out: B * C * 128, where C is the num_classes
            csa_out = torch.cat([train_out, test_out], dim=0)

            # print(csa_out.shape) # 396 * 5 * 512

            # generate the class protocal with dimension C * D (nclass * dim)
            proto_list = []
            for i in range(self.nclass):
                idx = (labels[idx_train].squeeze() == i).nonzero().squeeze(1)
                class_train_out = train_out[:, i, :]
                if self.use_att:
                    A = self.att_models[i](class_train_out[idx])  # N_k * 1
                    A = torch.transpose(A, 1, 0)  # 1 * N_k
                    A = F.softmax(A, dim=1)  # softmax over N_k

                    class_repr = torch.mm(A, class_train_out[idx]) # 1 * L
                    class_repr = torch.transpose(class_repr, 1, 0)  # L * 1
                else:  # if do not use attention, simply use the mean of training samples with the same labels.
                    class_repr = class_train_out[idx].mean(0)  # L * 1
                proto_list.append(class_repr.view(1, -1))
            x_proto = torch.cat(proto_list, dim=0)

            # prototype distance
            proto_dists = euclidean_dist(x_proto, x_proto)
            proto_dists = torch.exp(-0.5*proto_dists)
            num_proto_pairs = int(self.nclass * (self.nclass - 1) / 2)
            proto_dist = torch.sum(proto_dists) / num_proto_pairs
            x = torch.mean(x, 2)
            # print(x.shape)
            # print(csa_out.shape)
            for i in range(self.nclass):
                class_dists = euclidean_dist(csa_out[:, i, :], x_proto[i:(i+1), :])
                # print("class: " + str(i))
                # print(csa_out[:, i, :].shape)
                # print(x_proto.shape)
                # print(class_dists.shape)
                if i == 0:
                    dists = class_dists
                else:
                    dists = torch.cat([dists, class_dists], dim=1)

            dump_embedding(x_proto, x, labels)
            return torch.exp(-0.5*dists), proto_dist
            



