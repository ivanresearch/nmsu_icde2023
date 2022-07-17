import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim


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


class MLSTMfcn(nn.Module):
    def __init__(self, *, num_classes, max_seq_len, num_features,
                 num_lstm_out=128, num_lstm_layers=1, 
                 conv1_nf=128, conv2_nf=256, conv3_nf=128,
                 lstm_drop_p=0.8, fc_drop_p=0.3):
        super(MLSTMfcn, self).__init__()

        self.num_classes = num_classes
        self.max_seq_len = max_seq_len
        self.num_features = num_features

        self.num_lstm_out = num_lstm_out
        self.num_lstm_layers = num_lstm_layers

        self.conv1_nf = conv1_nf
        self.conv2_nf = conv2_nf
        self.conv3_nf = conv3_nf

        self.lstm_drop_p = lstm_drop_p
        self.fc_drop_p = fc_drop_p

        self.lstm = nn.LSTM(input_size=self.num_features, 
                            hidden_size=self.num_lstm_out,
                            num_layers=self.num_lstm_layers,
                            batch_first=True)
        
        self.conv1 = nn.Conv1d(self.num_features, self.conv1_nf, 8)
        self.conv2 = nn.Conv1d(self.conv1_nf, self.conv2_nf, 5)
        self.conv3 = nn.Conv1d(self.conv2_nf, self.conv3_nf, 3)

        self.bn1 = nn.BatchNorm1d(self.conv1_nf)
        self.bn2 = nn.BatchNorm1d(self.conv2_nf)
        self.bn3 = nn.BatchNorm1d(self.conv3_nf)

        self.se1 = SELayer(self.conv1_nf)  # ex 128
        self.se2 = SELayer(self.conv2_nf)  # ex 256

        self.relu = nn.ReLU()
        self.lstmDrop = nn.Dropout(self.lstm_drop_p)
        self.convDrop = nn.Dropout(self.fc_drop_p)

        self.fc = nn.Linear(self.conv3_nf+self.num_lstm_out, self.num_classes)
    
    def forward(self, x, seq_lens=None):
        ''' input x should be in size [B,T,F], where 
            B = Batch size
            T = Time samples
            F = features
        '''
        if seq_lens is None:
            x1, (ht,ct) = self.lstm(x)
        else:
            x1 = nn.utils.rnn.pack_padded_sequence(x, seq_lens, 
                                               batch_first=True, 
                                               enforce_sorted=False)
            x1, (ht,ct) = self.lstm(x1)
            x1, _ = nn.utils.rnn.pad_packed_sequence(x1, batch_first=True, 
                                                    padding_value=0.0)
        x1 = x1[:,-1,:]
        
        x2 = x.transpose(2,1)
        x2 = self.convDrop(self.relu(self.bn1(self.conv1(x2))))
        x2 = self.se1(x2)
        x2 = self.convDrop(self.relu(self.bn2(self.conv2(x2))))
        x2 = self.se2(x2)
        x2 = self.convDrop(self.relu(self.bn3(self.conv3(x2))))
        x2 = torch.mean(x2,2)
        
        x_all = torch.cat((x1,x2),dim=1)
        x_out = self.fc(x_all)
        x_out = F.log_softmax(x_out, dim=1)

        return x_out

# input: train_x_shape: N * T * V
def run_mlstm_fcn(args, model_setting, data_group, logger):
    model_key = model_setting.model_key
    attn_key = model_setting.attn_key
    batch_control = model_setting.batch_control
    train_x_shape = data_group.train_x_matrix.shape
    train_y_shape = data_group.train_y_matrix.shape
    print(train_x_shape)
    time_length = train_x_shape[1]
    variable_num = train_x_shape[2]
    # in_shape = train_x_shape[1:]
    num_classes = train_y_shape[1]
    # max_seq_len = train_x_shape[1]
    # num_features = train_x_shape[2]
    model = MLSTMfcn(num_classes, time_length, variable_num)