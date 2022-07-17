import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import math


class SELayer(nn.Module):
    def __init__(self, channel, reduction=16):
        super(SELayer, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool1d(1)
        self.fc = nn.Sequential(
            nn.Linear(channel, channel // reduction, bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(channel // reduction, channel, bias=False),
            nn.Sigmoid()
        )

    def forward(self, x):
        b, c, _ = x.size()
        y = self.avg_pool(x).view(b, c)
        y = self.fc(y).view(b, c, 1)
        return x * y.expand_as(x)


class CSAModule(nn.Module):
    # num_f: number of channels/filters, 128 for fcn or 128 * 4 for tapnet
    # num_t: length of time series
    def __init__(self, num_classes, num_f, num_t, csa_sigma) -> None:
        super(CSAModule, self).__init__()
        self.num_classes = num_classes
        csa_dim = 128
        self.key_conv1 = nn.Conv1d(num_f, csa_dim, 1)
        self.query_conv1 = nn.Conv1d(num_f, csa_dim, 1)
        self.store_attn = torch.zeros(num_classes, num_t, num_t, requires_grad=False)
        self.class_id_list = None
        self.csa_version_only_for_base = 1
        if csa_sigma == 0:
            self.csa_sigma = nn.Parameter(torch.zeros(1))
        else:
            self.csa_sigma = nn.Parameter(torch.ones(1))

    # inputs: N * F * T
    # outputs: N * F * C
    # Here, T is the Time Dimension and F is the feature dimension
    # F can be the number of kernels/channels after a convolutional layer
    def forward(self, inputs, iter_epoch, csa_training):
        if iter_epoch == 0 and csa_training is False:
            inputs = inputs.mean(-1, keepdim=True)
            return inputs.repeat(1, 1, self.num_classes).transpose(2, 1)
        if csa_training == True:
            self.updata_storeQuery(inputs)
        
        for i in range(0, self.num_classes):
            class_out = torch.matmul(inputs, self.store_attn[i, :, :]).mean(-1, keepdim=True)
            if i == 0:
                outputs = class_out
            else:
                outputs = torch.cat((outputs, class_out), dim=2)
        outputs = outputs.transpose(2, 1)
        inputs = inputs.mean(-1, keepdim=True).repeat(1, 1, self.num_classes).transpose(2, 1)
        return inputs + self.csa_sigma * outputs

    def updata_storeQuery(self, train_x_tensor):
        key_tensor = self.key_conv1(train_x_tensor)
        query_tensor = self.query_conv1(train_x_tensor)
        num_classes = self.num_classes
        class_id_list = self.class_id_list
        class_query_matrix = None
        idx = class_id_list[0]
        class_query_matrix = torch.mean(query_tensor[idx], 0, keepdim=True)
        for i in range(1, num_classes):
            idx = class_id_list[i]
            class_query_matrix = torch.cat((class_query_matrix, torch.mean(query_tensor[idx], 0, keepdim=True)), 0)
        # print(class_query_matrix.shape)
        for i in range(num_classes):
            idx = class_id_list[i]
            class_key = torch.mean(key_tensor[idx], 0)
            # class_query = torch.mean(query_tensor[idx], 0)
            all_class_attn = torch.matmul(class_query_matrix.transpose(2, 1), class_key)
            class_attn = all_class_attn[i, :, :]
            if self.csa_version_only_for_base == 1: # proposed class differencet version
                class_attn = class_attn + abs(class_attn.unsqueeze(0) - all_class_attn).mean(0)
            self.store_attn[i, :, :] = class_attn
        self.store_attn = F.softmax(self.store_attn, dim=-1)

# in_shape: T * V
# functions ends with "_call" are intermediate functions
class BuildModels(nn.Module):
    def __init__(self, *,
                 num_classes,
                 in_shape,
                 model_key='fcn',
                 attn_key='none',
                 ga_sigma=0.0,
                 apply_se=True):
        super(BuildModels, self).__init__()
        self.num_classes = num_classes
        self.in_shape = in_shape
        self.model_key = model_key
        self.attn_key = attn_key
        self.ga_sigma = ga_sigma
        self.apply_se = apply_se

        self.t_dim = in_shape[0]
        self.v_dim = in_shape[1]
        # 
        if model_key == 'fcn':
            if attn_key == "none":
                self.fcn_init()
            elif attn_key == "group_attn":
                self.fcn_ga_init()
            elif attn_key == "group_attn_base":
                self.fcn_ga_init()
                self.csa_model.csa_version_only_for_base = -1
        elif model_key == 'cnn': # This is CNN-ATN model, not the simple CNN
            if attn_key == 'group_attn':
                self.ga_version = 1
                self.cnn_ga_init()
            elif attn_key == 'cross_attn':
                self.cnn_ca_init()
            elif attn_key == "cross_group_attn":
                self.cnn_ca_ga_init()
            elif attn_key == "cross_group_attn_base":
                self.ga_version = -1
                self.cnn_ca_ga_init()
        elif model_key == 'mlstm':
            if attn_key == 'none':
                self.mlstm_init()
            elif attn_key == 'group_attn':
                self.mlstm_ga_init()
        elif model_key == 'mlstm-trans':
            if attn_key == 'none':
                self.mlstm_init(True)
            elif attn_key == 'group_attn':
                self.mlstm_ga_init(True)
        elif model_key == 'fcn-mlstm':
            if attn_key == "none":
                self.fcn_mlstm_init()
            elif attn_key == "group_attn":
                self.fcn_mlstm_ga_init()
        elif model_key == 'fcn-mlstm-trans':
            if attn_key == "none":
                self.fcn_mlstm_init(True)
            elif attn_key == "group_attn":
                self.fcn_mlstm_ga_init(True)

    # 
    def forward(self, inputs, iter_epoch=0, csa_training=False):
        ''' input x should be in size [B * T * V], where 
            B = Batch size
            T = Time samples
            V = Variable number
        '''
        # self.hidden = self.init_hidden()
        model_key = self.model_key
        attn_key = self.attn_key
        if model_key == 'fcn':
            if attn_key == 'none':
                return self.fcn_forward(inputs)
            elif attn_key == "group_attn" or attn_key == "group_attn_base":
                return self.fcn_ga_forward(inputs, iter_epoch, csa_training)
        elif model_key == 'cnn':
            if attn_key == 'cross_attn':
                return self.cnn_ca_forward(inputs)
            elif attn_key == "group_attn":
                return None
            elif attn_key == "cross_group_attn" or attn_key == "cross_group_attn_base":
                return self.cnn_ca_ga_forward(inputs, iter_epoch, csa_training)
        elif model_key == 'mlstm':
            if attn_key == 'none':
                return self.mlstm_forward(inputs)
            elif attn_key == 'group_attn':
                return self.mlstm_ga_forward(inputs, iter_epoch, csa_training)
        elif model_key == 'mlstm-trans':
            if attn_key == 'none':
                return self.mlstm_forward(inputs, True)
            elif attn_key == 'group_attn':
                return self.mlstm_ga_forward(inputs, iter_epoch, csa_training)
        elif model_key == 'fcn-mlstm':
            if attn_key == "none":
                return self.fcn_mlstm_forward(inputs)
            elif attn_key == "group_attn":
                return self.fcn_mlstm_ga_forward(inputs, iter_epoch, csa_training)
        elif model_key == 'fcn-mlstm-trans':
            if attn_key == "none":
                return self.fcn_mlstm_forward(inputs, True)
            elif attn_key == "group_attn":
                return self.fcn_mlstm_ga_forward(inputs, iter_epoch, csa_training, True)
        return None

    def shared_linear_init(self):
        self.fc = nn.Linear(self.f_before_linear, self.num_classes)
    
    def csa_linear_init(self):
        self.fc_list = nn.ModuleList()
        for _ in range(self.num_classes):
            fc = nn.Linear(self.f_before_linear, 1)
            self.fc_list.append(fc)

    def csa_linear_forward(self, inputs):
        # print("inputs shape: " + str(inputs.shape))
        for i in range(self.num_classes):
            class_input = inputs[:, 0, :]
            class_output = self.fc_list[i](class_input)
            if i == 0:
                outputs = class_output
            else:
                outputs = torch.cat((outputs, class_output), 1)
        return outputs


    ##### 1. FCN Part
    def fcn_init(self):
        self.conv1d_init()
        self.f_before_linear = self.conv3_nf
        self.shared_linear_init()

    # Inputs: B * T * V, in GES: 100, 214, 18
    def fcn_forward(self, inputs):
        # print("inputs: " + str(inputs.shape))
        inputs = inputs.transpose(2, 1)
        # inputs: B * V * T, in GES: 100, 18, 214
        # print(inputs.shape)
        conv_out = self.relu(self.bn1(self.conv1(inputs)))
        # conv_out: B * 128 * T-8+1, in GES: 100, 128, 214
        conv_out = self.se1(conv_out)
        conv_out = self.relu(self.bn2(self.conv2(conv_out)))
        # conv_out: B * 256 * 207-5+1, in GES: 100, 256, 214
        conv_out = self.se2(conv_out)
        
        conv_out = self.relu(self.bn3(self.conv3(conv_out)))
        # conv_out: B * 128 * 203-3+1, in GES: 100, 128, 214

        pool_out = torch.mean(conv_out, 2)
        # pool_out: B * 128, in GES: 100, 128

        outputs = self.fc(pool_out)
        outputs = F.log_softmax(outputs, dim=1)
        # outptus: B * num_classes. in GES: 100, 5
        return outputs


    # Shared by Any FCN or CNN included models
    def conv_parameters_init(self):
        # number of features
        self.conv1_nf = 128
        self.conv2_nf = 256
        self.conv3_nf = 128

        # kernel length
        self.conv1_klen = 8
        self.conv2_klen = 5
        self.conv3_klen = 3
    
    def conv1d_init(self, padding_str='same'):
        self.conv_parameters_init()
        t_after_conv = self.t_dim
        if padding_str == 'same':
            self.conv1 = nn.Conv1d(self.v_dim, self.conv1_nf, self.conv1_klen, padding=padding_str)
            self.conv2 = nn.Conv1d(self.conv1_nf, self.conv2_nf, self.conv2_klen, padding=padding_str)
            self.conv3 = nn.Conv1d(self.conv2_nf, self.conv3_nf, self.conv3_klen, padding=padding_str)
            self.t_after_conv = t_after_conv
        else:
            self.conv1 = nn.Conv1d(self.v_dim, self.conv1_nf, self.conv1_klen)
            t_after_conv = t_after_conv - self.conv1_klen + 1
            self.conv2 = nn.Conv1d(self.conv1_nf, self.conv2_nf, self.conv2_klen)
            t_after_conv = t_after_conv - self.conv2_klen + 1
            self.conv3 = nn.Conv1d(self.conv2_nf, self.conv3_nf, self.conv3_klen)
            t_after_conv = t_after_conv - self.conv3_klen + 1
            self.t_after_conv = t_after_conv
        

        self.bn1 = nn.BatchNorm1d(self.conv1_nf, track_running_stats=False)
        self.bn2 = nn.BatchNorm1d(self.conv2_nf, track_running_stats=False)
        self.bn3 = nn.BatchNorm1d(self.conv3_nf, track_running_stats=False)

        self.se1 = SELayer(self.conv1_nf)  # ex 128
        self.se2 = SELayer(self.conv2_nf)  # ex 256

        self.relu = nn.ReLU()
    ##### End of FCN Part

    ##### 2. FCN-group_attn model
    def fcn_ga_init(self):
        self.conv1d_init()
        self.f_before_linear = self.conv3_nf
        # self.shared_linear_init()
        self.csa_linear_init()
        self.csa_model = CSAModule(self.num_classes, self.f_before_linear, self.t_after_conv, self.ga_sigma)
    
    # Inputs: B * T * V, in GES: 100, 214, 18
    def fcn_ga_forward(self, inputs, iter_epoch, csa_training):
        # print("inputs: " + str(inputs.shape))
        inputs = inputs.transpose(2, 1)
        # inputs: B * V * T, in GES: 100, 18, 214

        conv_out = self.relu(self.bn1(self.conv1(inputs)))

        # conv_out: B * 128 * T-8+1, in GES: 100, 128, 207
        conv_out = self.se1(conv_out)
        conv_out = self.relu(self.bn2(self.conv2(conv_out)))
        # conv_out: B * 256 * 207-5+1, in GES: 100, 256, 203
        conv_out = self.se2(conv_out)

        conv_out = self.relu(self.bn3(self.conv3(conv_out)))
        # conv_out: B * 128 * 203-3+1, in GES: 100, 128, 201

        csa_out = self.csa_model.forward(conv_out, iter_epoch, csa_training)
        # csa_out: B * C * 128, where C is the num_classes
        outputs = self.csa_linear_forward(csa_out)
        outputs = F.log_softmax(outputs, dim=1)
        # outptus: B * num_classes. in GES: 100, 5
        return outputs

    def ga_init(self):
        self.ga_models = nn.ModuleList()
        for _ in range(self.num_classes):
            return None
        return None
    ##### End of FCN-group_attn model


    ##### 3. MLSTM Part
    def mlstm_init(self):
        num_lstm_out = self.lstm_init()
        self.fc = nn.Linear(num_lstm_out, self.num_classes)

    # Inputs: B * T * V, in GES: 100, 214, 18
    def mlstm_forward(self, inputs):
        # if trans is True:
        #     inputs = inputs.transpose(2, 1)
        lstm_out = self.lstm_call(inputs)
        # B * F, in GES: 100 * 128

        outputs = self.fc(lstm_out)
        # B * num_classes, in GES: 100 * 5

        outputs = F.log_softmax(outputs, dim=1)
        # B * num_classes, in GES: 100 * 5
        return outputs
    
    def lstm_init(self):
        # lstm hard coded parameters
        num_lstm_out = 128
        num_lstm_layers = 1
        # if trans is True:
        #     input_dim = self.t_dim
        # else:
        #     input_dim = self.v_dim
        input_dim = self.v_dim
        self.lstm = nn.LSTM(input_size=input_dim, 
                            hidden_size=num_lstm_out,
                            num_layers=num_lstm_layers,
                            batch_first=True)
        return num_lstm_out

    # Call LSTM function and get directly output
    # Inputs: B * T * V, in GES: 100, 214, 18
    def lstm_call(self, inputs):
        lstm_out, (ht,ct) = self.lstm(inputs)
        # print("Lstm out 1: " + str(lstm_out.shape))
        # B * T * F, in GES: 100, 214, 128
        # lstm_out = lstm_out[:,-1,:]
        lstm_out = lstm_out.mean(1)
        # print("Lstm out 2: " + str(lstm_out.shape))
        # B * F, in GES: 100, 128

        return lstm_out
    ##### End of MLSTM Part

    ##### 4. MLSTM-group attn Part
    def mlstm_ga_init(self):
        self.f_before_linear = self.lstm_init()
        # if trans is False:
        #     self.dim_t_or_v = self.t_dim
        # else:
        #     self.dim_t_or_v = self.v_dim
        self.dim_t_or_v = self.t_dim
        self.csa_linear_init()
        self.csa_model = CSAModule(self.num_classes, self.f_before_linear, self.dim_t_or_v, self.ga_sigma)

    # Inputs: B * T * V, in GES: 100, 214, 18
    def mlstm_ga_forward(self, inputs, iter_epoch, csa_training):
        # print(inputs.shape)
        # if trans is True:
        #     inputs = inputs.transpose(2, 1)
        # print(inputs.shape)
        lstm_out, (ht,ct) = self.lstm(inputs)
        # B * T * F, in GES: 100 * 214 * 128

        lstm_out = lstm_out.transpose(2, 1)
        # B * F * T, in GES: 100 * 128 * 214

        csa_out = self.csa_model.forward(lstm_out, iter_epoch, csa_training)
        # csa_out: B * C * 128, where C is the num_classes
        
        outputs = self.csa_linear_forward(csa_out)
        outputs = F.log_softmax(outputs, dim=1)
        return outputs

    ##### 5. FCN-MLSTM Part
    def fcn_mlstm_init(self):
        num_lstm_out = self.lstm_init()
        self.conv1d_init()
        self.f_before_linear = self.conv3_nf + num_lstm_out
        self.shared_linear_init()

    def fcn_mlstm_forward(self, inputs):
        ''' input x should be in size [B * T * V], where 
            B = Batch size
            T = Time samples
            V = Variable number
        '''
        # if trans is True:
        #     inputs = inputs.transpose(2, 1)
        #     lstm_out, (ht,ct) = self.lstm(inputs)
        #     # lstm_out = lstm_out[:,-1,:]
        #     lstm_out = lstm_out.mean(1)
        # else:
        #     lstm_out, (ht,ct) = self.lstm(inputs)
        #     # lstm_out = lstm_out[:,-1,:]
        #     lstm_out = lstm_out.mean(1)
        #     inputs = inputs.transpose(2, 1)
        lstm_out, (ht,ct) = self.lstm(inputs)
        # lstm_out = lstm_out[:,-1,:]
        lstm_out = lstm_out.mean(1)
        inputs = inputs.transpose(2, 1)
        # inputs: B * V * T, in GES: 100, 18, 214
        conv_out = self.relu(self.bn1(self.conv1(inputs)))
        # conv_out: B * 128 * T-8+1, in GES: 100, 128, 207
        conv_out = self.se1(conv_out)

        conv_out = self.relu(self.bn2(self.conv2(conv_out)))
        # conv_out: B * 256 * 207-5+1, in GES: 100, 256, 203
        conv_out = self.se2(conv_out)

        conv_out = self.relu(self.bn3(self.conv3(conv_out)))
        conv_out = torch.mean(conv_out, 2)
        
        cat_all = torch.cat((lstm_out, conv_out),dim=1)
        outputs = self.fc(cat_all)
        outputs = F.log_softmax(outputs, dim=1)

        return outputs

    ##### 6. FCN-MLSTM-ga Part
    def fcn_mlstm_ga_init(self):
        num_lstm_out = self.lstm_init()
        self.conv1d_init('same')
        self.f_before_linear = self.conv3_nf + num_lstm_out
        # self.f_before_linear = self.conv3_nf
        self.csa_linear_init()
        self.csa_model = CSAModule(self.num_classes, self.f_before_linear, self.t_after_conv, self.ga_sigma)

    def fcn_mlstm_ga_forward(self, inputs, iter_epoch, csa_training):
        ''' input x should be in size [B * T * V], where 
            B = Batch size
            T = Time samples
            V = Variable number
        '''
        lstm_out, (ht,ct) = self.lstm(inputs)

        inputs = inputs.transpose(2, 1)
        # inputs: B * V * T, in GES: 100, 18, 214
        conv_out = self.relu(self.bn1(self.conv1(inputs)))
        # conv_out: B * 128 * T-8+1, in GES: 100, 128, 207
        conv_out = self.se1(conv_out)

        conv_out = self.relu(self.bn2(self.conv2(conv_out)))
        # conv_out: B * 256 * 207-5+1, in GES: 100, 256, 203
        conv_out = self.se2(conv_out)

        conv_out = self.relu(self.bn3(self.conv3(conv_out)))
        lstm_out = lstm_out.transpose(2, 1)
    
        cat_all = torch.cat((lstm_out, conv_out),dim=1)
        # cat_all = conv_out
        # cat_all: B * 256 * T, in GES: 100, 256, 214

        csa_out = self.csa_model.forward(cat_all, iter_epoch, csa_training)
        # csa_out: B * C * 128, where C is the num_classes
        
        outputs = self.csa_linear_forward(csa_out)
        outputs = F.log_softmax(outputs, dim=1)
        # outptus: B * num_classes. in GES: 100, 5
        return outputs


    ##### 7. CNN-ATN Part
    def cnn_pre_init(self):
        pre_pool_dim = 100
        total = self.t_dim * self.v_dim
        if total > pre_pool_dim:
            pre_pool_dim = math.ceil(float(total / pre_pool_dim))
            pre_out_dim = int(self.t_dim / pre_pool_dim)
            self.cnn_pre_pool = nn.AdaptiveAvgPool2d((self.v_dim, pre_out_dim))
            self.t_dim = pre_out_dim
        else:
            self.cnn_pre_pool = None

    def cnn_ca_init(self):
        self.conv2d_init()
        self.cnn_pre_init()
        self.f_before_linear = self.conv3_nf * self.v_dim
        self.shared_linear_init()
        self.cross_attn_init()

    # Inputs: B * T * V * 1, in GES: 100, 214, 18, 1
    def cnn_ca_forward(self, inputs):
        inputs = inputs.transpose(3, 1)
        conv_out = self.relu(self.bn1(self.conv1(inputs)))
        # conv_out: B * 128 * V * T, in GES: 100, 128, 18, 214
        # conv_out = self.se1(conv_out)
        conv_out = self.relu(self.bn2(self.conv2(conv_out)))
        # conv_out: B * 256 * V * T, in GES: 7, 256, 18, 214

        conv_out = self.relu(self.bn3(self.conv3(conv_out))) # conv_out: B * 128 * 203-3+1, in GES: 7, 128, 18, 214
        if self.cnn_pre_pool is not None:
            conv_out = self.cnn_pre_pool(conv_out)
        conv_out = self.cross_attn_call(conv_out) 
        # print(conv_out.shape)
        pool_out = torch.mean(conv_out, -1)
        # print(pool_out.shape)
        pool_out = torch.flatten(pool_out, start_dim=1)
        # pool_out: B * 128, in GES: 100, 128

        outputs = self.fc(pool_out)
        outputs = F.log_softmax(outputs, dim=1)
        # outptus: B * num_classes. in GES: 100, 5
        return outputs

    def conv2d_init(self, padding_str='same'):
        self.conv_parameters_init()
        t_after_conv = self.t_dim
        if padding_str == 'same':
            self.conv1 = nn.Conv2d(1, self.conv1_nf, kernel_size=(1, self.conv1_klen), padding=padding_str)
            self.conv2 = nn.Conv2d(self.conv1_nf, self.conv2_nf, kernel_size=(1, self.conv2_klen), padding=padding_str)
            self.conv3 = nn.Conv2d(self.conv2_nf, self.conv3_nf, kernel_size=(1, self.conv3_klen), padding=padding_str)
            self.t_after_conv = t_after_conv
        else:
            self.conv1 = nn.Conv2d(1, self.conv1_nf, kernel_size=(1, self.conv1_klen))
            t_after_conv = t_after_conv - self.conv1_klen + 1
            self.conv2 = nn.Conv2d(self.conv1_nf, self.conv2_nf, kernel_size=(1, self.conv2_klen))
            t_after_conv = t_after_conv - self.conv2_klen + 1
            self.conv3 = nn.Conv2d(self.conv2_nf, self.conv3_nf, kernel_size=(1, self.conv3_klen))
            t_after_conv = t_after_conv - self.conv3_klen + 1

        self.bn1 = nn.BatchNorm2d(self.conv1_nf, track_running_stats=False)
        self.bn2 = nn.BatchNorm2d(self.conv2_nf, track_running_stats=False)
        self.bn3 = nn.BatchNorm2d(self.conv3_nf, track_running_stats=False)

        self.relu = nn.ReLU()


    # Cross-Attention part
    # Starting point for cross-attention part
    # Let us use "ca" as short for cross-attention
    # cross_attn only works with conv2d output (n*T*v*128)
    # because cross_attn expect both V ant T as input
    # to calculate attention from V and T dimensions
    # output is same as input shape
    def cross_attn_init(self):
        if self.v_dim > 1:
            self.variable_attn_init()
        self.temporal_attn_init()

    def cross_attn_call(self, ca_inputs):
        tmp_out = self.temporal_attn_call(ca_inputs)
        if self.v_dim > 1:
            var_out = self.variable_attn_call(tmp_out)
            return var_out
        else:
            return tmp_out

    def variable_attn_init(self):
        in_chan = 128
        out_chan = in_chan//8
        if out_chan == 0:
            out_chan = 1
        self.var_query_conv = nn.Conv2d(in_chan, out_chan, kernel_size=(1, 1), padding="same")
        self.var_key_conv = nn.Conv2d(in_chan, out_chan, kernel_size=(1, 1), padding="same")
        self.var_sigma = nn.Parameter(torch.zeros(1))

    def variable_attn_call(self, ca_inputs):
        # ca_inputs: (B * 128 * V * T): [7, 128, 18, 214]
        var_query = self.var_query_conv(ca_inputs) # var_query: (B * 16 * V * T): [7, 16, 18, 214]
        var_query = var_query.permute(0, 3, 2, 1) # var_query: (B * T * V * 16): [7, 214, 18, 16]
        # print("var_query shape: " + str(var_query.shape))
        var_key = self.var_key_conv(ca_inputs) # var_key: (B * 16 * V * T): [7, 16, 18, 214]
        var_key = var_key.permute(0, 3, 1, 2) # var_key: (B * T * 16 * V): [7, 214, 16, 18]
        # print("var_key shape: " + str(var_key.shape))
        var_attn = torch.matmul(var_query, var_key) # var_attn: (B * T * V * V): [7, 214, 18, 18]
        var_attn = F.softmax(var_attn, dim=-1)
        var_value = ca_inputs.permute(0, 3, 2, 1) # var_value: (B * T * V * 128): [7, 214, 18, 128]
        output_value = torch.matmul(var_attn, var_value) # output_value: (B * T * V * 218): [7, 214, 18, 128]
        output_value = output_value.permute(0, 3, 2, 1) # output_value: (B * 128 * V * T): [7, 128, 18, 214]
        return ca_inputs + self.var_sigma * output_value


    def temporal_attn_init(self):
        in_chan = 128
        out_chan = in_chan//8
        if out_chan == 0:
            out_chan = 1
        self.tmp_query_conv = nn.Conv2d(in_chan, out_chan, kernel_size=(1, 1), padding="same")
        self.tmp_key_conv = nn.Conv2d(in_chan, out_chan, kernel_size=(1, 1), padding="same")
        self.tmp_sigma = nn.Parameter(torch.zeros(1))


    def temporal_attn_call(self, ca_inputs):
        # ca_inputs: (B * 218 * V * T): (7, 128, 18, 214)
        tmp_query = self.tmp_query_conv(ca_inputs)  # tmp_query shape: (B * 16 * V * T): [7, 16, 18, 214]
        tmp_query = tmp_query.permute(0, 2, 3, 1)  # (B * V * T * 16): [7, 18, 214, 16]
        tmp_key = self.tmp_key_conv(ca_inputs)  # temp_key shape: (B * T * V * 16): [7, 16, 18, 214]
        tmp_key = tmp_key.permute(0, 2, 1, 3)  # (B * V * 16 * T): [7, 18, 16, 214]
        tmp_attn = torch.matmul(tmp_query, tmp_key) # (B * V * T * T): (7, 18, 214, 214)
        tmp_attn = torch.tril(tmp_attn)
        tmp_attn = F.softmax(tmp_attn, dim=-1) # (B * V * T * T): (7, 18, 214, 214)
        tmp_value = ca_inputs.permute(0, 2, 3, 1)  # tmp_value: (B * V * T * 218): [7, 18, 214, 128]
        tmp_output = torch.matmul(tmp_attn, tmp_value) # tmp_output: (B * V * T * 218): [7, 18, 214, 128]
        tmp_output = tmp_output.permute(0, 3, 1, 2) # tmp_output: (B * 128 * V * T): [7, 128, 18, 214]
        return ca_inputs + self.tmp_sigma * tmp_output
    # End of Cross-Attention Part
    # End of CNN-ATN Part

    def cnn_ca_ga_init(self):
        # self.cnn_pre_init()
        self.conv2d_init()
        self.cnn_pre_init()
        self.f_before_linear = self.conv3_nf * self.v_dim
        self.cross_attn_init()
        self.csa_linear_init()
        self.csa_model = CSAModule(self.num_classes, self.f_before_linear, self.t_dim, self.ga_sigma)

    # Inputs: B * T * V * 1, in GES: 100, 214, 18, 1
    def cnn_ca_ga_forward(self, inputs, iter_epoch, csa_training):
        inputs = inputs.transpose(3, 1)
        # if self.cnn_pre_pool is not None:
        #     inputs = self.cnn_pre_pool(inputs)
        conv_out = self.relu(self.bn1(self.conv1(inputs)))
        # conv_out: B * 128 * V * T, in GES: 100, 128, 18, 214
        # conv_out = self.se1(conv_out)
        conv_out = self.relu(self.bn2(self.conv2(conv_out)))
        # conv_out: B * 256 * V * T, in GES: 7, 256, 18, 214

        conv_out = self.relu(self.bn3(self.conv3(conv_out))) # conv_out: B * 128 * 203-3+1, in GES: 7, 128, 18, 214
        if self.cnn_pre_pool is not None:
            conv_out = self.cnn_pre_pool(conv_out)
        conv_out = self.cross_attn_call(conv_out)
        pool_out = torch.flatten(conv_out, start_dim=1, end_dim=2)
        csa_out = self.csa_model.forward(pool_out, iter_epoch, csa_training)
        # csa_out: B * C * 128, where C is the num_classes
        outputs = self.csa_linear_forward(csa_out)
        outputs = F.log_softmax(outputs, dim=1)
        # outptus: B * num_classes. in GES: 100, 5
        return outputs
    # CNN-ATN-CSA Part
    # End of CNN-ATN-CSA Part