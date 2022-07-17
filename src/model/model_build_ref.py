import tensorflow as tf
from tensorflow.keras import layers
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import ModelCheckpoint, ReduceLROnPlateau, CSVLogger
from tensorflow.keras.layers import Input, Dense, LSTM, multiply, concatenate, Activation, Masking, Reshape
from tensorflow.keras.layers import BatchNormalization, GlobalAveragePooling1D, Dropout, Permute
from tensorflow.keras.regularizers import l2
from model.tapnet_model_build import TapNetBuild
import matplotlib.pyplot as plt
import numpy as np
from sklearn.preprocessing import LabelEncoder
from model.data_generator import MiBatchGenerator, class_input_process
import sys
import time


def pool_dim_update(in_dim, pool_rate):
    out_dim = (in_dim - pool_rate + 1)/pool_rate
    int_dim = int(out_dim)
    if out_dim == int_dim:
        out_dim = int_dim
    else:
        out_dim = int_dim + 1
    return out_dim


class ModelBuild(tf.keras.Model):
    def __init__(self,
                 num_classes,
                 in_shape,
                 model_key='fcn',
                 attn_key='none',
                 ga_sigma=0.0,
                 ga_version=1,
                 class_batch_size=20,
                 apply_se=True):
        super(ModelBuild, self).__init__()
        self.ga_sigma = ga_sigma
        self.model_key = model_key
        # Start of first pooling part
        # part used to do pooing if input matrix is too large
        # which means too big V or too big T
        in_shape = self.input_pool_init(in_shape)
        # End of first pooling part
        if self.model_key == 'cnn':
            apply_se = False
        else:
            apply_se = True
        self.ensure_shape = [None, in_shape[0], in_shape[1]]
        self.t_dim = in_shape[0]
        self.v_dim = in_shape[1]
        self.num_classes = num_classes
        self.class_batch_size = class_batch_size
        self.lstm_hidden = 128
        self.conv_last_dim = 128
        self.apply_se = apply_se
        self.kernel_initializer = tf.keras.initializers.he_normal()
        self.bias_initializer = "zero"
        self.attn_key = attn_key
        self.in_shape = in_shape
        self.ga_version = ga_version
        if model_key == 'fcn':
            if attn_key == "none":
                self.fcn_init()
            elif attn_key == "group_attn":
                self.ga_version = 1
                self.conv1d_init()
                self.conv_additional_init()
                self.fcn_group_attn_init()
            elif attn_key == "group_attn_base":
                self.ga_version = -1
                self.conv1d_init()
                self.conv_additional_init()
                self.fcn_group_attn_init()
        elif model_key == 'cnn':
            if attn_key == 'none':
                self.cnn_init()
            elif attn_key == 'group_attn':
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
                self.lstm_init()
                self.lstm_out_init()
            elif attn_key == 'group_attn':
                self.mlstm_ga_init()
        elif model_key == 'fcn-mlstm':
            if attn_key == "none":
                self.fcn_mlstm_init()
            elif attn_key == "group_attn":
                self.fcn_mlstm_ga_init()


    def call(self, inputs, training):
        if self.input_pool_bool:
            inputs = self.input_pool_layer(inputs)
        model_key = self.model_key
        attn_key = self.attn_key
        if model_key == 'fcn':
            if attn_key == 'none':
                return self.fcn_call(inputs)
            elif attn_key == "group_attn" or attn_key == "group_attn_base":
                output, after_csa, before_csa = self.fcn_ga_call(inputs, training)
                if training is True:
                    self.attn_out = [inputs, before_csa, after_csa]
                return output
        elif model_key == 'cnn':
            if attn_key == 'none':
                return self.cnn_call(inputs)
            elif attn_key == 'cross_attn':
                return self.cnn_ca_call(inputs)
            elif attn_key == "group_attn":
                return self.cnn_ga_call(inputs, training)
            elif attn_key == "cross_group_attn" or attn_key == "cross_group_attn_base":
                return self.cnn_ca_ga_call(inputs, training)
        elif model_key == 'mlstm':
            if attn_key == 'none':
                return self.mlstm_call(inputs)
            elif attn_key == 'group_attn':
                return self.mlstm_ga_call(inputs, training)
        elif model_key == 'fcn-mlstm':
            if attn_key == "none":
                return self.fcn_mlstm_call(inputs)
            elif attn_key == "group_attn":
                return self.fcn_mlstm_ga_call(inputs, training)

    def input_pool_init(self, in_shape):
        if self.model_key != 'cnn':
            self.input_pool_bool = False
            return in_shape
        input_pool_bool = False
        input_pool_var = 1
        v_dim = in_shape[1]
        v_limit = 100
        if v_dim > v_limit:
            input_pool_bool = True
            input_pool_var = v_dim//v_limit
            up_v_dim = pool_dim_update(v_dim, input_pool_var)
        else:
            up_v_dim = v_dim
        input_pool_time = 1
        t_dim = in_shape[0]
        t_limit = 20000
        if t_dim * up_v_dim > t_limit:
            input_pool_bool = True
            input_pool_time = 3
            up_t_dim = pool_dim_update(t_dim, input_pool_time)
        else:
            up_t_dim = t_dim
        if input_pool_bool:
            self.input_pool_layer = layers.AveragePooling2D(pool_size=(input_pool_time, input_pool_var), strides=(input_pool_time, input_pool_var))
        self.input_pool_bool = input_pool_bool
        return (up_t_dim, up_v_dim, 1)

    # 1. FCN model
    def fcn_init(self):
        self.conv1d_init()
        self.conv_additional_init()
        self.final_pool = layers.GlobalAveragePooling1D()
        self.final_dense = layers.Dense(self.num_classes, activation='softmax')

    def fcn_call(self, inputs):
        # inputs: B * T * V
        conv_out = self.conv_call(inputs)
        # conv_out: B * T * 128
        fcn_out = self.final_pool(conv_out)
        # fcn_out: B * 128
        outputs = self.final_dense(fcn_out)
        # outpus: B * num_classes
        return outputs
    # End of FCN model

    # 2. CNN (conv2d) model
    def cnn_init(self):
        self.conv2d_init()
        self.conv_additional_init()
        # self.final_pool = layers.GlobalAveragePooling2D()
        self.final_pool = layers.AveragePooling2D(pool_size=(self.t_dim, 1), strides=(self.t_dim, 1))
        self.final_flat = layers.Flatten()
        self.final_dense = layers.Dense(self.num_classes, activation='softmax')

    def cnn_call(self, inputs):
        # inputs: B * T * V * 1
        conv_out = self.conv_call(inputs)
        # tf.print("conv_out shape")
        # tf.print(conv_out.get_shape())
        # conv_out: B * T * V * 128
        # tf.print("conv_out shape")
        # tf.print(conv_out.get_shape())
        pool_out = self.final_pool(conv_out)
        # tf.print("pool_out shape")
        # tf.print(pool_out.get_shape())
        dense_input = self.final_flat(pool_out)
        # tf.print("dense_input shape")
        # tf.print(dense_input.get_shape())
        # dense_input: B * 128
        outputs = self.final_dense(dense_input)
        # outputs: B * num_classes
        return outputs
    # End of CNN model

    # 3. FCN-group_attn model
    def fcn_ga_init(self):
        self.conv1d_init()
        self.conv_additional_init()
        self.fcn_group_attn_init()

    def fcn_ga_call(self, inputs, training):
        # inputs: B * T * V
        conv_out = self.conv_call(inputs)
        # conv_out: B * T * 128
        outputs, attn_out = self.fcn_group_attn_call(conv_out, training)
        # outputs: B * num_classes
        return outputs, attn_out, conv_out
    # End of FCN-group_attn model

    # 4. CNN-cross_attn model
    def cnn_ca_init(self):
        self.cnn_init()
        self.cross_attn_init()

    def cnn_ca_call(self, inputs):
        # inputs: B * T * V * 1
        conv_out = self.conv_call(inputs)
        # conv_out: B * T * V * 128
        attn_out = self.cross_attn_call(conv_out)
        # attn_out: B * T * V * 128
        # tf.print("attn_out shape")
        # tf.print(attn_out.get_shape())
        pool_out = self.final_pool(attn_out)
        # tf.print("pool_out shape")
        # tf.print(pool_out.get_shape())
        dense_input = self.final_flat(pool_out)
        # tf.print("dense_input shape")
        # tf.print(dense_input.get_shape())
        # dense_input: B * 128
        outputs = self.final_dense(dense_input)
        # tf.print("final output")
        # tf.print(outputs.get_shape())
        # outputs: B * num_classes
        return outputs
    # End of CNN-cross_attn model

    # 5. CNN-cross_group_attn model
    def cnn_ca_ga_init(self):
        self.conv2d_init()
        self.conv_additional_init()
        self.cross_attn_init()
        self.ga_t_pool = layers.AveragePooling2D(pool_size=(self.t_dim, 1), strides=(self.t_dim, 1))
        self.fcn_group_attn_init()

    def cnn_ca_ga_call(self, inputs, training):
        # inputs: B * T * V * 1
        conv_out = self.conv_call(inputs)
        # conv_out: B * T * V * 128
        cnn_ca_out = self.cross_attn_call(conv_out)
        # tf.print("cnn_ca_out shape")
        # tf.print(cnn_ca_out.get_shape())
        ga_inputs = self.ga_t_pool(cnn_ca_out)
        ga_inputs = ga_inputs[:, 0, :, :]
        # ga_inputs = self.ga_v_pool(cnn_ca_out)
        # ga_inputs = ga_inputs[:, :, 0, :]
        # ga_inputs: B * 1 * V * 128
        # tf.print("ga_inputs shape")
        # tf.print(ga_inputs.get_shape())
        # ga_inputs: B * V * 128
        # tf.print("ga_inputs shape")
        # tf.print(ga_inputs.get_shape())
        outputs, attn_out = self.fcn_group_attn_call(ga_inputs, training)
        # tf.print("outputs shape")
        # tf.print(outputs.get_shape())
        return outputs
    # End of CNN-cross_group_attn model

    # 6. CNN-group_attn model
    def cnn_ga_init(self):
        self.conv2d_init()
        self.conv_additional_init()
        self.ga_t_pool = layers.AveragePooling2D(pool_size=(self.t_dim, 1), strides=(self.t_dim, 1))
        self.fcn_group_attn_init()

    def cnn_ga_call(self, inputs, training):
        # inputs: B * T * V * 1
        conv_out = self.conv_call(inputs)
        # conv_out: B * T * V * 128
        # tf.print("cnn_out shape")
        # tf.print(conv_out.get_shape())
        ga_inputs = self.ga_t_pool(conv_out)
        # tf.print("ga_inputs shape")
        # tf.print(ga_inputs.get_shape())
        ga_inputs = ga_inputs[:, 0, :, :]
        # tf.print("ga_inputs out shape")
        # tf.print(ga_inputs.get_shape())
        # sdfsd
        # ga_inputs = self.ga_v_pool(cnn_ca_out)
        # ga_inputs = ga_inputs[:, :, 0, :]
        # ga_inputs: B * 1 * V * 128
        # tf.print("ga_inputs shape")
        # tf.print(ga_inputs.get_shape())
        # ga_inputs: B * V * 128
        # tf.print("ga_inputs shape")
        # tf.print(ga_inputs.get_shape())
        outputs, attn_out = self.fcn_group_attn_call(ga_inputs, training)
        # tf.print("outputs shape")
        # tf.print(outputs.get_shape())
        return outputs
    # End of CNN-cross_group_attn model

    # 7. MLSTM part
    def lstm_init(self):
        self.lstm_mask = Masking()
        self.lstm = LSTM(self.lstm_hidden, return_sequences=True)
    
    def lstm_out_init(self):
        self.final_pool = GlobalAveragePooling1D()
        self.final_dense = Dense(self.num_classes, activation='softmax')
    
    ## input shape: n*T*v
    ## output shape: n*T*8
    def lstm_call(self, inputs):
        lstm_out = self.lstm_mask(inputs)
        lstm_out = self.lstm(lstm_out)
        return lstm_out
    
    def mlstm_call(self, inputs):
        #inputs = tf.where(tf.math.is_nan(inputs), 0., inputs)
        lstm_out = self.lstm_call(inputs)
        #tf.print(lstm_out[0, :, :], summarize=-1)
        lstm_out = self.final_pool(lstm_out)
        #tf.print(lstm_out[0, :], summarize=-1)
        outputs = self.final_dense(lstm_out)
        #tf.print(tf.reduce_max(tf.where(tf.math.is_nan(inputs), 0., 1)))
        #tf.print(training)
        #tf.print(inputs.get_shape())
        #tf.print(outputs[0, :], summarize=-1)
        #print("====")
        return outputs

    # 8. MLSTM-GA part
    def mlstm_ga_init(self):
        self.lstm_mask = Masking()
        self.lstm = LSTM(self.lstm_hidden, return_sequences=True)
        self.fcn_group_attn_init()

    def mlstm_ga_call(self, inputs, training):
        # inputs: B * T * V
        lstm_out = self.lstm_call(inputs)
        outputs, attn_out = self.fcn_group_attn_call(lstm_out, training)
        # outputs: B * num_classes
        return outputs

    # 9. FCN-MLSTM None part
    def fcn_mlstm_init(self):
        self.fcn_init()
        self.lstm_init()

    def fcn_mlstm_call(self, inputs):
        lstm_out = self.lstm_call(inputs)
        conv_out = self.conv_call(inputs)

        out = concatenate([lstm_out, conv_out])
        fcn_out = self.final_pool(out)
        outputs = self.final_dense(fcn_out)
        return outputs
    # End of  FCN-MLSTM None part

    # 10. FCN-MLSTM-GA None part
    def fcn_mlstm_ga_init(self):
        self.ga_version = 1
        self.conv1d_init()
        self.conv_additional_init()
        self.conv_last_dim = self.lstm_hidden + self.conv_last_dim
        self.fcn_group_attn_init()
        self.lstm_init()

    def fcn_mlstm_ga_call(self, inputs, training):
        lstm_out = self.lstm_call(inputs)
        conv_out = self.conv_call(inputs)
        ga_inputs = concatenate([lstm_out, conv_out])
        # ga_inputs = conv_out
        outputs, attn_out = self.fcn_group_attn_call(ga_inputs, training)
        return outputs
    # End of  FCN-MLSTM None part

    # 3. FCN-group_attn model
    def fcn_ga_init(self):
        self.conv1d_init()
        self.conv_additional_init()
        self.fcn_group_attn_init()

    def fcn_ga_call(self, inputs, training):
        # inputs: B * T * V
        conv_out = self.conv_call(inputs)
        # conv_out: B * T * 128
        outputs, attn_out = self.fcn_group_attn_call(conv_out, training)
        # outputs: B * num_classes
        return outputs, attn_out, conv_out

    # Starting point for CONV part
    # conv1d layers initilization for FCN
    def conv1d_init(self):
        self.conv1 = layers.Conv1D(128, 8, padding="same", bias_initializer=self.bias_initializer,
                                   input_shape=self.in_shape,
                                   kernel_initializer=self.kernel_initializer)
        self.conv2 = layers.Conv1D(256, 5, padding="same", bias_initializer=self.bias_initializer,
                                   kernel_initializer=self.kernel_initializer)
        self.conv3 = layers.Conv1D(128, 3, padding="same", bias_initializer=self.bias_initializer,
                                   kernel_initializer=self.kernel_initializer)

    # conv2d layers initilization for CNN
    def conv2d_init(self):
        self.conv1 = layers.Conv2D(128, (8, 1), bias_initializer=self.bias_initializer,
                                   padding="valid",
                                   input_shape=self.in_shape,
                                   kernel_initializer=self.kernel_initializer)
        self.conv2 = layers.Conv2D(256, (5, 1), bias_initializer=self.bias_initializer,
                                   padding="valid",
                                   kernel_initializer=self.kernel_initializer)
        self.conv3 = layers.Conv2D(128, (3, 1), bias_initializer=self.bias_initializer,
                                   padding="valid",
                                   kernel_initializer=self.kernel_initializer)
        self.t_dim = self.t_dim - 13

    # Conv Additional layers, batch_normamization, activation
    def conv_additional_init(self):
        apply_se = self.apply_se
        self.bn1 = BatchNormalization()
        self.act1 = Activation('relu')
        if apply_se is True:
            filters = 128
            self.se_fir_dense1 = layers.Dense(filters // 16,
                                              activation='relu',
                                              kernel_initializer=self.kernel_initializer,
                                              use_bias=False)
            self.se_sec_dense1 = layers.Dense(filters,
                                              activation='sigmoid',
                                              kernel_initializer=self.kernel_initializer,
                                              use_bias=False)

        self.bn2 = BatchNormalization()
        self.act2 = Activation('relu')
        if apply_se is True:
            filters = 256
            self.se_fir_dense2 = Dense(filters // 16,
                                       activation='relu',
                                       kernel_initializer=self.kernel_initializer,
                                       use_bias=False)
            self.se_sec_dense2 = Dense(filters,
                                       activation='sigmoid',
                                       kernel_initializer=self.kernel_initializer,
                                       use_bias=False)

        self.bn3 = BatchNormalization()
        self.act3 = Activation('relu')
        if apply_se is True:
            filters = self.conv_last_dim
            self.se_fir_dense3 = Dense(filters // 16,
                                       activation='relu',
                                       kernel_initializer=self.kernel_initializer,
                                       use_bias=False)
            self.se_sec_dense3 = Dense(filters,
                                       activation='sigmoid',
                                       kernel_initializer=self.kernel_initializer,
                                       use_bias=False)

    # input shape: n*T*v (conv1d) or n*T*v*1 (conv2d)
    # output shape: n*T*128 or n*T*v*128
    def conv_call(self, inputs):
        conv_out = self.conv1(inputs)
        conv_out = self.bn1(conv_out)
        conv_out = self.act1(conv_out)
        apply_se = self.apply_se
        if apply_se is True:
            # conv_out = self.squeeze_excite_block(conv_out)
            se_out = tf.reduce_mean(conv_out, axis=1, keepdims=True)
            se_out = self.se_fir_dense1(se_out)
            se_out = self.se_sec_dense1(se_out)
            conv_out = multiply([conv_out, se_out])

        # 2nd conv layer
        conv_out = self.conv2(conv_out)
        conv_out = self.bn2(conv_out)
        conv_out = self.act2(conv_out)
        if apply_se is True:
            se_out = tf.reduce_mean(conv_out, axis=1, keepdims=True)
            se_out = self.se_fir_dense2(se_out)
            se_out = self.se_sec_dense2(se_out)
            conv_out = multiply([conv_out, se_out])

        # 3rd conv layer
        conv_out = self.conv3(conv_out)
        conv_out = self.bn3(conv_out)
        conv_out = self.act3(conv_out)
        if apply_se is True:
            se_out = tf.reduce_mean(conv_out, axis=1, keepdims=True)
            se_out = self.se_fir_dense3(se_out)
            se_out = self.se_sec_dense3(se_out)
            conv_out = multiply([conv_out, se_out])
        return conv_out
    # Ending point for CONV part

    # Starting point for cross-attention part
    # Let us use "ca" as short for cross-attention
    # cross_attn only works with conv2d output (n*T*v*128)
    # because cross_attn expect both V ant T as input
    # to calculate attention from V and T dimensions
    # output is same as input shape
    def cross_attn_init(self):
        self.variable_attn_init()
        self.temporal_attn_init()

    def cross_attn_call(self, ca_inputs):
        tmp_out = self.temporal_attn_call(ca_inputs)
        var_out = self.variable_attn_call(tmp_out)
        return var_out

    def variable_attn_init(self):
        in_chan = self.conv_last_dim
        out_chan = in_chan//8
        if out_chan == 0:
            out_chan = 1
        self.var_query_conv = layers.Conv2D(out_chan, (1, 1), bias_initializer=self.bias_initializer,
                                            padding="same",
                                            kernel_initializer=self.kernel_initializer)
        self.var_key_conv = layers.Conv2D(out_chan, (1, 1), bias_initializer=self.bias_initializer,
                                          padding="same",
                                          kernel_initializer=self.kernel_initializer)
        # self.var_value_conv = layers.Conv2D(in_chan // 2, (1, 1), bias_initializer=self.bias_initializer,
        #                                    padding="same",
        #                                    kernel_initializer=self.kernel_initializer)
        self.var_softmax = layers.Activation('softmax')
        self.var_sigma = tf.Variable(initial_value=0.0, trainable=True, name="var_sigma", shape=())

    def variable_attn_call(self, ca_inputs):
        # tf.print("var_attn input shape: " + str(ca_inputs.get_shape()))
        var_query = self.var_query_conv(ca_inputs)
        # tf.print("var_query shape: " + str(var_query.get_shape()))
        var_key = self.var_key_conv(ca_inputs)
        # tf.print("var_key shape: " + str(var_key.get_shape()))
        var_attn = tf.matmul(var_query, var_key, transpose_b=True)
        # tf.print("var_attn shape: " + str(var_attn.get_shape()))
        var_attn = self.var_softmax(var_attn)
        var_value = ca_inputs
        # tf.print("var_value shape: " + str(var_value.get_shape()))
        # tf.print("var_value out shape: " + str((ca_inputs + self.var_sigma * tf.matmul(var_attn, var_value)).get_shape()))
        return ca_inputs + self.var_sigma * tf.matmul(var_attn, var_value)

    def temporal_attn_init(self):
        in_chan = self.conv_last_dim
        out_chan = in_chan//8
        if out_chan == 0:
            out_chan = 1
        self.tmp_permute = layers.Permute((2, 1, 3))
        self.tmp_query_conv = layers.Conv2D(out_chan, (1, 1), bias_initializer=self.bias_initializer,
                                            padding="same",
                                            kernel_initializer=self.kernel_initializer)
        self.tmp_key_conv = layers.Conv2D(out_chan, (1, 1), bias_initializer=self.bias_initializer,
                                          padding="same",
                                          kernel_initializer=self.kernel_initializer)
        # self.tmp_value_conv = layers.Conv2D(in_chan // 2, (1, 1), bias_initializer=self.bias_initializer,
        #                                    padding="same",
        #                                    kernel_initializer=self.kernel_initializer)
        self.tmp_softmax = layers.Activation('softmax')
        self.tmp_sigma = tf.Variable(initial_value=0.0, name="tmp_sigma", shape=())

    def temporal_attn_call(self, ca_inputs):
        # tf.print("tempral attn start")
        # tf.print("input shape: " + str(ca_inputs.get_shape()))
        # function input: (B * T * V * C)
        tmp_query = self.tmp_query_conv(ca_inputs)  # tmp_query shape: (B * T * V * 16)
        tmp_query = self.tmp_permute(tmp_query)  # tmp_query shape: (B * V * T * 16)
        tmp_key = self.tmp_key_conv(ca_inputs)  # temp_key shape: (B * T * V * 16)
        tmp_key = self.tmp_permute(tmp_key)  # temp_key shape: (B * V * T * 16)
        tmp_attn = tf.matmul(tmp_query, tmp_key, transpose_b=True)  # tmp_attn shape: (B * V * T * T)
        tmp_attn = tf.linalg.LinearOperatorLowerTriangular(tmp_attn).to_dense()
        tmp_attn = self.tmp_softmax(tmp_attn)  # tmp_attn shape: (B * V * T * T)
        tmp_value = self.tmp_permute(ca_inputs)  # tmp_value shape: (B * V * T * C)
        tmp_attn_out = tf.matmul(tmp_attn, tmp_value)  # tmp_attn_out shape: (B * V * T * C)
        # Output shape: (B * T * V * C)
        return ca_inputs + self.tmp_sigma * self.tmp_permute(tmp_attn_out)

    # Ending point for cross-attention part

    # Starting point for group-attention (class-attention) part

    # Let us use "ga" as short for group-attention
    # Please Note: ga needs have special generators for training batches
    # class_attn works for both conv1d (FCN) and conv2d (CNN) output
    # the output fron conv2d (n*T*v*128) should apply pooling on v
    # the input should be n*T*128 or n*T*v*128

    # group_attn init function for FCN model
    def fcn_group_attn_init(self):
        self.ga_repeat = tf.constant([1, self.num_classes, 1, 1], tf.int32)
        self.ga_pool_init()
        # if self.num_classes > 2:
        #     self.ga_sigma_initial = 0.1
        # else:
        #     self.ga_sigma_initial = 0.0
        self.ga_sigma_initial = self.ga_sigma
        # The sigma parameter to merge class_attn back to input
        self.ga_sigma = tf.Variable(initial_value=self.ga_sigma_initial, trainable=True, name="ga_sigma", shape=())

        # if len(self.in_shape) == 3:  # CNN based model
        #     ga_sec_dim = self.conv_last_dim * self.v_dim
        #     k_q_dim = ga_sec_dim // 8
        #     if k_q_dim < 128:
        #         k_q_dim = 128
        # else:
        ga_sec_dim = self.conv_last_dim
        k_q_dim = ga_sec_dim
        self.store_ga_query = tf.Variable(tf.zeros([self.num_classes, self.ga_fir_dim, ga_sec_dim]), trainable=False)
        self.ga_divide = tf.Variable(0.0, trainable=False)

        self.query_conv = layers.Conv1D(k_q_dim, 1, padding='same', kernel_initializer=self.kernel_initializer)
        self.key_conv = layers.Conv1D(k_q_dim, 1, padding='same', kernel_initializer=self.kernel_initializer)

        # ga final pooling layer, average on T dimension
        # self.ga_final_pool = layers.AveragePooling2D(pool_size=(1, self.ga_fir_dim), strides=(1, self.ga_fir_dim))
        if self.model_key == 'cnn':
            reshape_dim = self.ga_fir_dim * ga_sec_dim
            self.ga_final_reshape = layers.Reshape((self.num_classes, 1, reshape_dim))
        else:
            reshape_dim = ga_sec_dim
            self.ga_final_reshape = layers.AveragePooling2D(pool_size=(1, self.ga_fir_dim), strides=(1, self.ga_fir_dim))
        # ga dense layer after group-attention calculation
        w_shape = [self.num_classes, reshape_dim, 1]
        w_init = self.kernel_initializer
        self.dense_w = tf.Variable(initial_value=w_init(shape=w_shape,
                                                        dtype='float32'),
                                   trainable=True)
        b_shape = [self.num_classes]
        b_init = tf.zeros_initializer()
        self.dense_b = tf.Variable(initial_value=b_init(shape=b_shape,
                                                        dtype='float32'),
                                   trainable=True)
        self.out_act = layers.Activation('softmax')

    # group_attn call function for FCN model
    # ga_inputs (n*F*128) as input of group attn
    # F = T for FCN model and F = T for CNN model
    # C = 128 for FCN model and C = 128 for CNN model
    # training: boolean variable
    def fcn_group_attn_call(self, ga_inputs, training):
        # print(ga_inputs.shape)
        # ga_inputs: B * F * 128
        # tf.print("ga_inputs shape")
        # tf.print(ga_inputs.get_shape())
        ga_pool_out = self.ga_pool_call(ga_inputs)
        # print(ga_pool_out.shape)
        # tf.print("ga_pool_out shape")
        # tf.print(ga_pool_out.get_shape())
        # fcn_outputs: B * F' * 128 (F' <= F and 128 may smaller as well)
        # If F is too large, this pool will reduce F and make sure F < 200
        # Speed and efficiency consideration
        if training is True:
            self.ga_query_update(ga_pool_out)
        attn_out = self.fcn_ga_config(ga_pool_out)
        # print(attn_out.shape)
        # tf.print("attn_out shape")
        # tf.print(attn_out.get_shape())
        # attn_out: B * num_classes * F' * 128, as same as fcn_outpus
        reshape_out = self.ga_final_reshape(attn_out)
        # tf.print("reshape_out shape")
        # tf.print(reshape_out.get_shape())
        outputs = tf.matmul(reshape_out, self.dense_w)
        # outputs: B * num_classes * 1 * 1
        # tf.print("outputs shape")
        # tf.print(outputs.get_shape())
        outputs = outputs[:, :, 0, 0]
        # print(outputs.shape)
        # outputs: B * num_classes
        # final outputs: B * num_classes
        return self.out_act(outputs + self.dense_b), reshape_out

    # fcn_ga_config
    # main function to calculate the group-attn outputs
    # fcn_outputs is the inputs tensor, which is the output of previous fcn layers
    def fcn_ga_config(self, fcn_outputs):
        # self.store_ga_query only stores the sum of all previous inputs
        # before using it as query tensor
        # divide the sum by the number of instance and get a overall average
        store_ga_query = self.store_ga_query / (self.ga_divide * self.class_batch_size)
        query_tensor = self.query_conv(store_ga_query)
        key_tensor = self.key_conv(fcn_outputs)
        key_tensor = tf.expand_dims(key_tensor, axis=1)
        repeat = self.ga_repeat
        key_tensor = tf.tile(key_tensor, repeat)

        ga_tensor = tf.expand_dims(fcn_outputs, axis=1)
        ga_tensor = tf.tile(ga_tensor, repeat)

        # value_tensor = self.val_conv(fcn_outputs)
        value_tensor = fcn_outputs
        value_tensor = tf.expand_dims(value_tensor, axis=1)
        value_tensor = tf.tile(value_tensor, repeat)
        fcn_ga_outputs = self.group_attn_config(key_tensor, query_tensor, value_tensor)
        # if self.sigma < 0:
        #     self.sigma = 0
        # sigma = tf.math.abs(self.sigma)
        return ga_tensor + self.ga_sigma * fcn_ga_outputs
        # return self.ga_sigma * fcn_ga_outputs

    # function used to control the way of attn_tensor back to value_tensor
    # if self.ga_version == -1: it similarly multuply back to value_tensor
    # if self.ga_version == 1: What we proposed, we get the absolute difference
    # between groups/classes, then multiply back to value tensor
    def group_attn_config(self, key_tensor, query_tensor, value_tensor):
        attn_tensor = tf.matmul(query_tensor, key_tensor, transpose_b=True)
        if self.ga_version == -1:  # traditional way
            attn_tensor = tf.nn.softmax(attn_tensor, axis=-1)
            return tf.matmul(attn_tensor, value_tensor)
        else:  # proposed way, get the differences between groups
            attn_sum_tensor = tf.reduce_sum(attn_tensor, axis=1, keepdims=True)
            attn_sum_tensor = tf.tile(attn_sum_tensor, self.ga_repeat)
            attn_other_tensor = attn_sum_tensor - attn_tensor
            attn_other_tensor = attn_other_tensor/(self.num_classes-1)
            attn_final_tensor = tf.math.abs(attn_other_tensor - attn_tensor)
            attn_tensor = tf.nn.softmax(attn_final_tensor, axis=-1)
            return tf.matmul(attn_tensor, value_tensor)

    # set the limitation of total number of features
    # calculation can be super slow if thGoere are too many features
    def ga_pool_init(self):
        ga_fir_dim = self.in_shape[-2]
        # ga_fir_dim = self.t_dim
        self.ga_fir_bool = False
        if ga_fir_dim >= 200:
            fir_pool_rate = int(ga_fir_dim/100)
            self.ga_fir_bool = True
            self.ga_fir_pool = layers.AveragePooling1D(pool_size=fir_pool_rate, strides=fir_pool_rate)
            ga_fir_dim = (ga_fir_dim - fir_pool_rate + 1)/fir_pool_rate
            int_dim = int(ga_fir_dim)
            if ga_fir_dim == int_dim:
                ga_fir_dim = int_dim
            else:
                ga_fir_dim = int_dim + 1
        ga_total_feature = self.num_classes * ga_fir_dim
        limit = 2000
        self.ga_sec_bool = False
        if ga_total_feature > limit:
            self.ga_sec_bool = True
            sec_pool_rate = ga_total_feature/limit
            int_rate = int(sec_pool_rate)
            if sec_pool_rate == int_rate:
                sec_pool_rate = int_rate
            else:
                sec_pool_rate = int_rate + 1

            self.ga_sec_bool = True
            self.ga_sec_pool = layers.AveragePooling1D(pool_size=sec_pool_rate, strides=sec_pool_rate)
            ga_fir_dim = (ga_fir_dim - sec_pool_rate + 1)/sec_pool_rate
            int_dim = int(ga_fir_dim)
            if ga_fir_dim == int_dim:
                ga_fir_dim = int_dim
            else:
                ga_fir_dim = int_dim + 1
        self.ga_fir_dim = ga_fir_dim

    def ga_pool_call(self, ga_inputs):
        if self.ga_fir_bool is True:
            ga_inputs = self.ga_fir_pool(ga_inputs)
        if self.ga_sec_bool is True:
            ga_inputs = self.ga_sec_pool(ga_inputs)
        return ga_inputs

    def ga_query_update(self, ga_inputs):
        num_classes = self.num_classes
        class_batch_size = self.class_batch_size
        for i in range(num_classes):
            start = 0 + i * class_batch_size
            end = start + class_batch_size
            class_tensor = ga_inputs[start:end, :, :]
            class_tensor = tf.reduce_sum(class_tensor, axis=0, keepdims=True)
            if i == 0:
                query_input = class_tensor
            else:
                query_input = tf.concat([query_input, class_tensor], axis=0)
        store_ga_query = self.store_ga_query + query_input
        self.ga_divide.assign(self.ga_divide + 1)
        # store_ga_query = store_ga_query / (self.ga_divide * class_batch_size)
        self.store_ga_query.assign(store_ga_query)
    # Ending point for group-attention part


def run_model(model_setting, data_group, saved_path, logger):
    model_key = model_setting.model_key
    attn_key = model_setting.attn_key
    batch_control = model_setting.batch_control
    train_x_shape = data_group.train_x_matrix.shape
    train_y_shape = data_group.train_y_matrix.shape
    in_shape = train_x_shape[1:]
    num_classes = train_y_shape[1]
    training_generator = None
    class_batch_size = 20
    if batch_control:
        logger.info("batch control confirm: " + str(batch_control))
        train_x_matrix = data_group.train_x_matrix
        train_y_vector = data_group.train_y_vector
        train_y_matrix = to_categorical(train_y_vector, len(np.unique(train_y_vector)))
        class_batch_size = ret_class_batch_size(model_setting.batch_size, data_group.train_y_vector)
        # if class_batch_size < 5:
        #     class_batch_size = 5
        logger.info("new batch size: " + str(class_batch_size))

        batch_size = class_batch_size * num_classes
        model_setting.batch_size = batch_size
        x_train_list, y_train_list = class_input_process(train_x_matrix, train_y_vector, train_y_matrix, num_classes)
        training_generator = MiBatchGenerator(x_train_list, y_train_list, class_batch_size, True)
    print("2: " + str(data_group.train_x_matrix.shape))

    # apply_se = True
    if model_key == 'tapnet':
        model = TapNetBuild(num_classes, in_shape, attn_key, model_setting.ga_sigma, class_batch_size)
    else:
        model = ModelBuild(num_classes, in_shape, model_key, attn_key, model_setting.ga_sigma, class_batch_size)
    model.build((None,) + in_shape)
    stringlist = []
    model.summary(print_fn=lambda x: stringlist.append(x))
    short_model_summary = "\n".join(stringlist)
    logger.info(short_model_summary)
    return model_training(model, data_group, saved_path, logger, model_setting, training_generator)


def model_training(model, data_group, saved_path, logger, cnn_setting=None, generator=None):
    epochs = 50
    batch_size = 128
    learning_rate = 1e-3
    monitor = 'loss'
    optimization_mode = 'auto'
    if cnn_setting is not None:
        epochs = cnn_setting.max_iter
        batch_size = cnn_setting.batch_size
        learning_rate = cnn_setting.learning_rate
    train_x_matrix = data_group.train_x_matrix
    train_y_vector = data_group.train_y_vector
    test_x_matrix = data_group.test_x_matrix
    test_y_vector = data_group.test_y_vector

    classes = np.unique(train_y_vector)
    le = LabelEncoder()
    y_ind = le.fit_transform(train_y_vector.ravel())
    recip_freq = len(train_y_vector) / (len(le.classes_) * np.bincount(y_ind).astype(np.float64))
    class_weight = recip_freq[le.transform(classes)]
    logger.info("Class weights : " + str(class_weight))
    print("Class weights : ", class_weight)
    class_w_dict = {}
    class_index = 0
    for item in class_weight:
        class_w_dict[class_index] = item
        class_index = class_index + 1
    train_y_matrix = to_categorical(train_y_vector, len(np.unique(train_y_vector)))
    test_y_matrix = to_categorical(test_y_vector, len(np.unique(test_y_vector)))

    factor = 1. / np.cbrt(2)

    model_checkpoint = ModelCheckpoint(saved_path, verbose=1, mode=optimization_mode, monitor=monitor, save_best_only=True, save_weights_only=True)
    reduce_lr = ReduceLROnPlateau(monitor=monitor, patience=2000, mode=optimization_mode, factor=factor, cooldown=0, min_lr=1e-4, verbose=2)
    callback_list = [model_checkpoint, reduce_lr]
    # callback_list = [model_checkpoint]

    optm = Adam(lr=learning_rate)

    model.compile(optimizer=optm, loss='categorical_crossentropy', metrics=['accuracy'])
    if cnn_setting.model_key == 'tapnet' and model.attn_type=='tap_attn':
        model.add_loss(lambda: 0.01 * model.loss_term)
    if generator is None:
        start_time = time.time()
        log_history = model.fit(train_x_matrix, train_y_matrix, batch_size=batch_size, epochs=epochs, callbacks=callback_list, class_weight=class_w_dict, verbose=2, validation_data=(test_x_matrix, test_y_matrix))
        training_time = time.time() - start_time
    else:
        start_time = time.time()
        # log_history = model.fit_generator(generator=generator, epochs=epochs, callbacks=callback_list, class_weight=class_w_dict, verbose=2, validation_data=(test_x_matrix, test_y_matrix))
        log_history = model.fit_generator(generator=generator, epochs=epochs, callbacks=callback_list, verbose=2, validation_data=(test_x_matrix, test_y_matrix))
        training_time = time.time() - start_time
    return log_history, model, training_time


def ret_class_batch_size(batch_size, train_y_vector, version=0):
    unique, counts = np.unique(train_y_vector, return_counts=True)
    train_size = len(train_y_vector)
    num_iter = train_size/float(batch_size)
    if num_iter == int(num_iter):
        num_iter = int(num_iter)
    else:
        num_iter = int(num_iter) + 1
    max_class_len = max(counts)
    class_batch_size = float(max_class_len)/num_iter
    if class_batch_size != int(class_batch_size):
        class_batch_size = int(class_batch_size) + 1
    else:
        class_batch_size = int(class_batch_size)
    return class_batch_size

