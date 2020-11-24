'''
    ST-ResNet: Deep Spatio-temporal Residual Networks
'''

from __future__ import print_function
from keras.layers import (
    Input,
    Activation,
    merge,
    Dense,
    Reshape
)
from keras.layers.convolutional import Convolution2D
from keras.layers.normalization import BatchNormalization
from keras.models import Model
import numpy as np
import tensorflow as tf
#from keras.utils.visualize_util import plot


def _shortcut(input, residual):
    # return merge([input, residual], mode='sum')
    return merge.add([input, residual])


def _bn_relu_conv(nb_filter, nb_row, nb_col, subsample=(1, 1), bn=False):
    def f(input):
        if bn:
            input = BatchNormalization(mode=0, axis=1)(input)
        activation = Activation('relu')(input)
        # return Convolution2D(filters=nb_filter, kernel_size=(nb_row, nb_col), subsample=subsample, border_mode="same")(activation)
        # return Convolution2D(filters=nb_filter, kernel_size=(nb_row, nb_col), strides=subsample, padding='same', data_format='channels_first')(activation)
        return Convolution2D(filters=nb_filter, kernel_size=(nb_row, nb_col), strides=subsample, padding='same')(activation)
    return f


def _residual_unit(nb_filter, init_subsample=(1, 1)):
    def f(input):
        residual = _bn_relu_conv(nb_filter, 3, 3)(input)
        residual = _bn_relu_conv(nb_filter, 3, 3)(residual)
        return _shortcut(input, residual)
    return f


def ResUnits(residual_unit, nb_filter, repetations=1):
    def f(input):
        for i in range(repetations):
            init_subsample = (1, 1)
            input = residual_unit(nb_filter=nb_filter,
                                  init_subsample=init_subsample)(input)
        return input
    return f


def stresnet(c_conf=(3, 2, 32, 32), p_conf=(3, 2, 32, 32), t_conf=(3, 2, 32, 32), external_dim=8, nb_residual_unit=3):
    '''
    C - Temporal Closeness
    P - Period
    T - Trend
    conf = (len_seq, nb_flow, map_height, map_width)
    external_dim
    '''

    # main input
    main_inputs = []
    outputs = []
    for conf in [c_conf, p_conf, t_conf]:
        if conf is not None:
            len_seq, nb_flow, map_height, map_width = conf
            input = Input(shape=(map_height, map_width, nb_flow * len_seq))
            # input = Input(shape=(nb_flow * len_seq, map_height, map_width))
            main_inputs.append(input)
            # Conv1
            # conv1 = Convolution2D(filters=64, kernel_size=3, border_mode="same")(input)
            # conv1 = Convolution2D(filters=64, kernel_size=(3, 3), padding='same', data_format='channels_first')(input)
            conv1 = Convolution2D(filters=64, kernel_size=(3, 3), padding='same')(input)
            # [nb_residual_unit] Residual Units
            residual_output = ResUnits(_residual_unit, nb_filter=64,
                              repetations=nb_residual_unit)(conv1)
            # Conv2
            activation = Activation('relu')(residual_output)
            # conv2 = Convolution2D(filters=nb_flow, kernel_size=3, border_mode="same")(activation)
            # conv2 = Convolution2D(filters=nb_flow, kernel_size=(3, 3), padding='same', data_format='channels_first')(activation)
            conv2 = Convolution2D(filters=nb_flow, kernel_size=(3, 3), padding='same')(activation)
            outputs.append(conv2)

    # parameter-matrix-based fusion
    if len(outputs) == 1:
        main_output = outputs[0]
    else:
        from .iLayer import iLayer
        # new_outputs = []
        # for output in outputs:
        #     new_outputs.append(iLayer()(output))
        # main_output = merge(new_outputs, mode='sum')

        new_outputs = []
        for output in outputs:
            output = Reshape((map_height, map_width, nb_flow))(output)
            new_outputs.append(iLayer()(output))
        main_output = merge.add(new_outputs)
        
        # closeness_output = iLayer()(outputs[0])
        # period_output = iLayer()(outputs[1])
        # trend_output = iLayer()(outputs[2])


        # output_c = tf.reshape(closeness_output, [closeness_output.shape[1], closeness_output.shape[2], closeness_output.shape[3]])
        # output_p = tf.reshape(period_output, [period_output.shape[1], period_output.shape[2], period_output.shape[3]])
        # output_t = tf.reshape(trend_output, [trend_output.shape[1], trend_output.shape[2], trend_output.shape[3]])

        # out = merge.add([output_c[:][0:2], output_c[:][2:4]])
        # out = merge.add([out, output_c[:][4:6]])
        # out = merge.add([merge.add([out, output_p]), output_t])
        # main_output = tf.expand_dims(out, 0)


    # fusing with external component
    if external_dim != None and external_dim > 0:
        # external input
        external_input = Input(shape=(external_dim,))
        main_inputs.append(external_input)
        # embedding = Dense(output_dim=10)(external_input)
        embedding = Dense(10)(external_input)
        embedding = Activation('relu')(embedding)
        # h1 = Dense(output_dim=nb_flow * map_height * map_width)(embedding)
        h1 = Dense(nb_flow * map_height * map_width)(embedding)
        activation = Activation('relu')(h1)
        external_output = Reshape((map_height, map_width, nb_flow))(activation)
        main_output = merge.add([main_output, external_output])
    else:
        print('external_dim:', external_dim)

    main_output = Activation('tanh')(main_output)
    model = Model(inputs=main_inputs, outputs=main_output)

    return model

if __name__ == '__main__':
    model = stresnet(external_dim=28, nb_residual_unit=12)
    #plot(model, to_file='ST-ResNet.png', show_shapes=True)
    model.summary()
