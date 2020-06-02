# Copyright University College London 2019, 2020
# Author: Alexander Whitehead, Institute of Nuclear Medicine, UCL
# For internal research only.


import math
import tensorflow as tf
import keras as k


# https://github.com/zizhaozhang/unet-tensorflow-keras/blob/master/model.py
def get_crop_shape(target, refer):
    # depth, the 5th dimension
    cd = int(target.get_shape()[4]) - int(refer.get_shape()[4])

    if cd % 2 != 0:
        cd1, cd2 = int(cd / 2), int(cd / 2) + 1
    else:
        cd1, cd2 = int(cd / 2), int(cd / 2)

    # width, the 4rd dimension
    cw = int(target.get_shape()[3]) - int(refer.get_shape()[3])

    if cw % 2 != 0:
        cw1, cw2 = int(cw / 2), int(cw / 2) + 1
    else:
        cw1, cw2 = int(cw / 2), int(cw / 2)

    # height, the 3nd dimension
    ch = int(target.get_shape()[2]) - int(refer.get_shape()[2])

    if ch % 2 != 0:
        ch1, ch2 = int(ch / 2), int(ch / 2) + 1
    else:
        ch1, ch2 = int(ch / 2), int(ch / 2)

    # time, the 2nd dimension
    ct = int(target.get_shape()[1]) - int(refer.get_shape()[1])

    if ct % 2 != 0:
        ct1, ct2 = int(ct / 2), int(ct / 2) + 1
    else:
        ct1, ct2 = int(ct / 2), int(ct / 2)

    return (ct1, ct2), (ch1, ch2), (cw1, cw2), (cd1, cd2)


def crop(source, target):
    ct, ch, cw, cd = get_crop_shape(source, target)
    x = k.layers.TimeDistributed(k.layers.Cropping3D(cropping=(ch, cw, cd)))(source)

    return x


def dense_dynamic_signal_extractor(x, stride, layers, layers_layers, input_size, lone, ltwo, rnn_layers, rnn_units,
                                   dropout,
                                   output_size, output_shape):
    x = k.layers.TimeDistributed(k.layers.Flatten())(x)

    # DOWNSAMPLE
    for i in range(stride, (layers * stride) + stride, stride):
        for j in range(layers_layers):
            units = int(math.ceil(input_size / i))

            x = k.layers.TimeDistributed(k.layers.Dense(units=units,
                                                        kernel_initializer=k.initializers.Constant(0.1),
                                                        bias_initializer=k.initializers.Constant(0.1),
                                                        kernel_regularizer=k.regularizers.l1_l2(l1=lone, l2=ltwo),
                                                        kernel_constraint=k.constraints.UnitNorm()))(x)
            x = k.layers.TimeDistributed(k.layers.BatchNormalization())(x)
            x = k.layers.TimeDistributed(k.layers.Activation("selu"))(x)

    # RNN
    x_1 = x

    x_1 = k.layers.TimeDistributed(k.layers.Flatten())(x_1)

    for i in range(rnn_layers - 1):
        x_1 = k.layers.LSTM(units=rnn_units,
                            activation="tanh",
                            recurrent_activation="sigmoid",
                            dropout=dropout,
                            recurrent_dropout=dropout,
                            return_sequences=True,
                            kernel_initializer="glorot_normal",
                            bias_initializer=k.initializers.Constant(0.1),
                            recurrent_initializer="glorot_uniform",
                            kernel_regularizer=k.regularizers.l1_l2(l1=lone, l2=ltwo),
                            kernel_constraint=k.constraints.UnitNorm())(x_1)

    if rnn_layers > 0:
        x_1 = k.layers.LSTM(units=rnn_units,
                            activation="tanh",
                            recurrent_activation="sigmoid",
                            dropout=dropout,
                            recurrent_dropout=dropout,
                            return_sequences=False,
                            kernel_initializer="glorot_normal",
                            bias_initializer=k.initializers.Constant(0.1),
                            recurrent_initializer="glorot_uniform",
                            kernel_regularizer=k.regularizers.l1_l2(l1=lone, l2=ltwo),
                            kernel_constraint=k.constraints.UnitNorm())(x_1)

        x_1 = k.layers.Reshape((x_1.shape.as_list()[1],))(x_1)
    else:
        x_1 = k.layers.Flatten()(x_1)

    x_1 = k.layers.Dense(units=output_size,
                         kernel_initializer=k.initializers.Constant(0.1),
                         bias_initializer=k.initializers.Constant(0.1))(x_1)
    x_1 = k.layers.Activation("linear", name="output_1")(x_1)

    # UPSAMPLE
    x_2 = x

    for i in reversed(range(stride, (layers * stride), stride)):
        for j in range(layers_layers):
            units = int(math.ceil(input_size / i))

            x_2 = k.layers.TimeDistributed(k.layers.Dense(units=units,
                                                          kernel_initializer=k.initializers.Constant(0.1),
                                                          bias_initializer=k.initializers.Constant(0.1),
                                                          kernel_regularizer=k.regularizers.l1_l2(l1=lone, l2=ltwo),
                                                          kernel_constraint=k.constraints.UnitNorm()))(x_2)
            x_2 = k.layers.TimeDistributed(k.layers.BatchNormalization())(x_2)
            x_2 = k.layers.TimeDistributed(k.layers.Activation("selu"))(x_2)

    x_2 = k.layers.TimeDistributed(k.layers.Dense(units=input_size,
                                                  kernel_initializer=k.initializers.Constant(0.1),
                                                  bias_initializer=k.initializers.Constant(0.1)))(x_2)
    x_2 = k.layers.TimeDistributed(k.layers.Activation("linear"))(x_2)

    x_2 = k.layers.TimeDistributed(k.layers.Reshape(output_shape), name="output_2")(x_2)

    return x_1, x_2


def crnn_dynamic_signal_extractor(x, cnn_start_units, cnn_layers, cnn_increase_layer_density_bool, cnn_layer_layers,
                                  lone, ltwo, cnn_pool_bool, cnn_max_pool_bool, cnn_deconvolution_bool, rnn_layers,
                                  rnn_units, dropout, output_size):
    cnn_start_units_log = int(math.floor(math.log(cnn_start_units, 2)))
    cnn_end_units_log = cnn_start_units_log + cnn_layers

    x_skip = []
    filters = 0

    # DOWNSAMPLE
    x = k.layers.TimeDistributed(k.layers.Convolution3D(filters=1,
                                                        kernel_size=(3, 3, 3),
                                                        strides=(1, 1, 1),
                                                        padding="same",
                                                        kernel_initializer="glorot_normal",
                                                        bias_initializer=k.initializers.Constant(0.1)))(x)
    x = k.layers.TimeDistributed(k.layers.Activation("tanh"))(x)

    for i in range(cnn_start_units_log, cnn_end_units_log):
        x_skip.append(x)

        if cnn_increase_layer_density_bool:
            filters = int(math.floor(math.pow(2, i)))
        else:
            filters = int(math.floor(math.pow(2, cnn_start_units_log)))

        for j in range(cnn_layer_layers):
            x = k.layers.TimeDistributed(k.layers.Convolution3D(filters=filters,
                                                                kernel_size=(3, 3, 3),
                                                                strides=(1, 1, 1),
                                                                padding="same",
                                                                kernel_initializer="lecun_normal",
                                                                bias_initializer=k.initializers.Constant(0.1),
                                                                kernel_regularizer=k.regularizers.l1_l2(l1=lone,
                                                                                                        l2=ltwo),
                                                                kernel_constraint=k.constraints.UnitNorm()))(x)
            x = k.layers.TimeDistributed(k.layers.BatchNormalization())(x)
            x = k.layers.TimeDistributed(k.layers.Activation("selu"))(x)

        if cnn_pool_bool:
            if cnn_max_pool_bool:
                x = k.layers.TimeDistributed(k.layers.MaxPooling3D(padding="same"))(x)
            else:
                x = k.layers.TimeDistributed(k.layers.AveragePooling3D(padding="same"))(x)
        else:
            x = k.layers.TimeDistributed(k.layers.Convolution3D(filters=filters,
                                                                kernel_size=(3, 3, 3),
                                                                strides=(2, 2, 2),
                                                                padding="same",
                                                                kernel_initializer="lecun_normal",
                                                                bias_initializer=k.initializers.Constant(0.1),
                                                                kernel_regularizer=k.regularizers.l1_l2(l1=lone,
                                                                                                        l2=ltwo),
                                                                kernel_constraint=k.constraints.UnitNorm()))(x)
            x = k.layers.TimeDistributed(k.layers.BatchNormalization())(x)
            x = k.layers.TimeDistributed(k.layers.Activation("selu"))(x)

    for j in range(cnn_layer_layers):
        x = k.layers.TimeDistributed(k.layers.Convolution3D(filters=filters,
                                                            kernel_size=(3, 3, 3),
                                                            strides=(1, 1, 1),
                                                            padding="same",
                                                            kernel_initializer="lecun_normal",
                                                            bias_initializer=k.initializers.Constant(0.1),
                                                            kernel_regularizer=k.regularizers.l1_l2(l1=lone,
                                                                                                    l2=ltwo),
                                                            kernel_constraint=k.constraints.UnitNorm()))(x)
        x = k.layers.TimeDistributed(k.layers.BatchNormalization())(x)
        x = k.layers.TimeDistributed(k.layers.Activation("selu"))(x)

    # RNN
    x_1 = x

    x_1 = k.layers.TimeDistributed(k.layers.Flatten())(x_1)

    for i in range(rnn_layers - 1):
        x_1 = k.layers.LSTM(units=rnn_units,
                            activation="tanh",
                            recurrent_activation="sigmoid",
                            dropout=dropout,
                            recurrent_dropout=dropout,
                            return_sequences=True,
                            kernel_initializer="glorot_normal",
                            bias_initializer=k.initializers.Constant(0.1),
                            recurrent_initializer="glorot_uniform",
                            kernel_regularizer=k.regularizers.l1_l2(l1=lone, l2=ltwo),
                            kernel_constraint=k.constraints.UnitNorm())(x_1)

    if rnn_layers > 0:
        x_1 = k.layers.LSTM(units=rnn_units,
                            activation="tanh",
                            recurrent_activation="sigmoid",
                            dropout=dropout,
                            recurrent_dropout=dropout,
                            return_sequences=False,
                            kernel_initializer="glorot_normal",
                            bias_initializer=k.initializers.Constant(0.1),
                            recurrent_initializer="glorot_uniform",
                            kernel_regularizer=k.regularizers.l1_l2(l1=lone, l2=ltwo),
                            kernel_constraint=k.constraints.UnitNorm())(x_1)

        x_1 = k.layers.Reshape((x_1.shape.as_list()[1],))(x_1)
    else:
        x_1 = k.layers.Flatten()(x_1)

    x_1 = k.layers.Dense(units=output_size,
                         kernel_initializer="glorot_normal",
                         bias_initializer=k.initializers.Constant(0.1))(x_1)
    x_1 = k.layers.Activation("tanh", name="output_1")(x_1)

    # UPSAMPLE
    decrement = len(x_skip) - 1

    x_2 = x

    for j in range(cnn_layer_layers):
        x_2 = k.layers.TimeDistributed(k.layers.Convolution3D(filters=filters,
                                                              kernel_size=(3, 3, 3),
                                                              strides=(1, 1, 1),
                                                              padding="same",
                                                              kernel_initializer="lecun_normal",
                                                              bias_initializer=k.initializers.Constant(0.1),
                                                              kernel_regularizer=k.regularizers.l1_l2(l1=lone,
                                                                                                      l2=ltwo),
                                                              kernel_constraint=k.constraints.UnitNorm()))(x_2)
        x_2 = k.layers.TimeDistributed(k.layers.BatchNormalization())(x_2)
        x_2 = k.layers.TimeDistributed(k.layers.Activation("selu"))(x_2)

    for i in reversed(range(cnn_start_units_log, cnn_end_units_log)):
        if cnn_increase_layer_density_bool:
            filters = int(math.floor(math.pow(2, i)))
        else:
            filters = int(math.floor(math.pow(2, cnn_start_units_log)))

        if cnn_deconvolution_bool:
            x_2_shape = x_2.get_shape()

            x_2 = k.layers.TimeDistributed(k.layers.Deconvolution3D(filters=filters,
                                                                    kernel_size=(3, 3, 3),
                                                                    strides=(2, 2, 2),
                                                                    padding="same",
                                                                    kernel_initializer="lecun_normal",
                                                                    bias_initializer=k.initializers.Constant(0.1),
                                                                    kernel_regularizer=k.regularizers.l1_l2(l1=lone,
                                                                                                            l2=ltwo),
                                                                    kernel_constraint=k.constraints.UnitNorm()))(x_2)
            x_2 = k.layers.TimeDistributed(k.layers.BatchNormalization())(x_2)
            x_2 = k.layers.TimeDistributed(k.layers.Activation("selu"))(x_2)
            x_2 = k.layers.Reshape((int(x_2_shape[1]), int(x_2_shape[2] * 2), int(x_2_shape[3] * 2),
                                    int(x_2_shape[4] * 2), int(filters),))(x_2)
        else:
            x_2 = k.layers.TimeDistributed(k.layers.UpSampling3D())(x_2)

        x_2 = crop(x_2, x_skip[decrement])

        decrement = decrement - 1

        for j in range(cnn_layer_layers):
            x_2 = k.layers.TimeDistributed(k.layers.Convolution3D(filters=filters,
                                                                  kernel_size=(3, 3, 3),
                                                                  strides=(1, 1, 1),
                                                                  padding="same",
                                                                  kernel_initializer="lecun_normal",
                                                                  bias_initializer=k.initializers.Constant(0.1),
                                                                  kernel_regularizer=k.regularizers.l1_l2(l1=lone,
                                                                                                          l2=ltwo),
                                                                  kernel_constraint=k.constraints.UnitNorm()))(x_2)
            x_2 = k.layers.TimeDistributed(k.layers.BatchNormalization())(x_2)
            x_2 = k.layers.TimeDistributed(k.layers.Activation("selu"))(x_2)

    x_2 = k.layers.TimeDistributed(k.layers.Convolution3D(filters=1,
                                                          kernel_size=(3, 3, 3),
                                                          strides=(1, 1, 1),
                                                          padding="same",
                                                          kernel_initializer="glorot_normal",
                                                          bias_initializer=k.initializers.Constant(0.1)))(x_2)
    x_2 = k.layers.TimeDistributed(k.layers.Activation("tanh"), name="output_2")(x_2)

    return x_1, x_2


def tof_crnn_dynamic_signal_extractor(x, cnn_start_units, cnn_layers, cnn_increase_layer_density_bool, cnn_layer_layers,
                                      lone, ltwo, rnn_layers, rnn_units, dropout, output_size):
    cnn_start_units_log = int(math.floor(math.log(cnn_start_units, 2)))
    cnn_end_units_log = cnn_start_units_log + cnn_layers

    x_2 = x

    # DOWNSAMPLE
    x = k.layers.TimeDistributed(Conv(filters=1,
                                      kernel_size=(1, 1, 1, 1),
                                      strides=(1, 1, 1, 1),
                                      padding="same",
                                      kernel_initializer=k.initializers.Constant(0.1),
                                      bias_initializer=k.initializers.Constant(0.1)))(x)
    x = k.layers.TimeDistributed(k.layers.Activation("linear"))(x)

    for i in range(cnn_start_units_log, cnn_end_units_log):
        if cnn_increase_layer_density_bool:
            filters = int(math.floor(math.pow(2, i)))
        else:
            filters = int(math.floor(math.pow(2, cnn_start_units_log)))

        for j in range(cnn_layer_layers):
            x = k.layers.TimeDistributed(Conv(filters=filters,
                                              kernel_size=(3, 3, 3, 3),
                                              strides=(1, 1, 1, 1),
                                              padding="same",
                                              kernel_initializer="lecun_normal",
                                              bias_initializer=k.initializers.Constant(0.1),
                                              kernel_regularizer=k.regularizers.l1_l2(l1=lone,
                                                                                      l2=ltwo),
                                              kernel_constraint=k.constraints.UnitNorm()))(x)
            x = k.layers.TimeDistributed(k.layers.BatchNormalization())(x)
            x = k.layers.TimeDistributed(k.layers.Activation("selu"))(x)

        x = k.layers.TimeDistributed(Conv(filters=filters,
                                          kernel_size=(3, 3, 3, 3),
                                          strides=(2, 2, 2, 2),
                                          padding="same",
                                          kernel_initializer="lecun_normal",
                                          bias_initializer=k.initializers.Constant(0.1),
                                          kernel_regularizer=k.regularizers.l1_l2(l1=lone,
                                                                                  l2=ltwo),
                                          kernel_constraint=k.constraints.UnitNorm()))(x)
        x = k.layers.TimeDistributed(k.layers.BatchNormalization())(x)
        x = k.layers.TimeDistributed(k.layers.Activation("selu"))(x)

    # RNN
    x_1 = x

    x_1 = k.layers.TimeDistributed(k.layers.Flatten())(x_1)

    for i in range(rnn_layers - 1):
        x_1 = k.layers.LSTM(units=rnn_units,
                            activation="tanh",
                            recurrent_activation="sigmoid",
                            dropout=dropout,
                            recurrent_dropout=dropout,
                            return_sequences=True,
                            kernel_initializer="glorot_normal",
                            bias_initializer=k.initializers.Constant(0.1),
                            recurrent_initializer="glorot_uniform",
                            kernel_regularizer=k.regularizers.l1_l2(l1=lone, l2=ltwo),
                            kernel_constraint=k.constraints.UnitNorm())(x_1)

    if rnn_layers > 0:
        x_1 = k.layers.LSTM(units=rnn_units,
                            activation="tanh",
                            recurrent_activation="sigmoid",
                            dropout=dropout,
                            recurrent_dropout=dropout,
                            return_sequences=False,
                            kernel_initializer="glorot_normal",
                            bias_initializer=k.initializers.Constant(0.1),
                            recurrent_initializer="glorot_uniform",
                            kernel_regularizer=k.regularizers.l1_l2(l1=lone, l2=ltwo),
                            kernel_constraint=k.constraints.UnitNorm())(x_1)

        x_1 = k.layers.Reshape((x_1.shape.as_list()[1],))(x_1)
    else:
        x_1 = k.layers.Flatten()(x_1)

    x_1 = k.layers.Dense(units=output_size,
                         kernel_initializer=k.initializers.Constant(0.1),
                         bias_initializer=k.initializers.Constant(0.1))(x_1)
    x_1 = k.layers.Activation("linear", name="output_1")(x_1)

    return x_1, x_2


class Conv(tf.keras.layers.Layer):
    def __init__(self, filters, kernel_size, padding="VALID", **kwargs):
        self.filters = filters
        self.kernel_size = kernel_size  # must be a tuple!!!!
        self.padding = padding

        super(Conv, self).__init__(**kwargs)

    # using channels last!!!
    def build(self, input_shape):
        spatial_dims = len(self.kernel_size)
        all_dims = len(input_shape)
        assert all_dims == spatial_dims + 2  # spatial dimensions + batch size + channels

        kernel_shape = self.kernel_size + (input_shape[-1], self.filters)
        # (spatial1, spatial2,...., spatialN, input_channels, output_channels)

        bias_shape = tuple(1 for _ in range(all_dims - 1)) + (self.filters,)

        self.kernel = self.add_weight(name="kernel",
                                      shape=kernel_shape,
                                      initializer="uniform",
                                      trainable=True)
        self.bias = self.add_weight(name="bias",
                                    shape=bias_shape,
                                    initializer="zeros",
                                    trainable=True)
        self.built = True

    def call(self, inputs, **kwargs):
        results = tf.nn.convolution(inputs, self.kernel, padding=self.padding)
        return results + self.bias

    def compute_output_shape(self, input_shape):
        sizes = input_shape[1:-1]

        if self.padding == "VALID" or self.padding == "valid":
            sizes = [s - kSize + 1 for s, kSize in zip(sizes, self.kernel_size)]

        return input_shape[:1] + sizes + (self.filters,)
