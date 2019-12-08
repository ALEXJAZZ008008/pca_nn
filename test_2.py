import keras as k


def test_activation(x, batch_normalisation_bool, activation):
    if batch_normalisation_bool:
        x = k.layers.BatchNormalization()(x)

    if activation == "prelu":
        x = k.layers.PReLU()(x)
    else:
        x = k.layers.Activation(activation)(x)

    return x


def test_module_in(x, activation, regularisation, filters, initializer):
    if activation == "selu":
        if regularisation:
            x = k.layers.TimeDistributed(k.layers.Convolution3D(filters=filters, kernel_size=(3, 3, 3),
                                                                strides=(1, 1, 1),
                                                                padding='same',
                                                                kernel_initializer=initializer,
                                                                bias_initializer=k.initializers.Constant(0.1),
                                                                kernel_regularizer=k.regularizers.l2(0.01),
                                                                bias_regularizer=k.regularizers.l2(0.01),
                                                                kernel_constraint=k.constraints.max_norm(3),
                                                                bias_constraint=k.constraints.max_norm(3)))(x)
            x = test_activation(x, True, "relu")
            x = k.layers.AlphaDropout(0.5)(x)
        else:
            x = k.layers.TimeDistributed(k.layers.Convolution3D(filters=filters, kernel_size=(3, 3, 3),
                                                                strides=(1, 1, 1),
                                                                padding='same',
                                                                kernel_initializer=initializer,
                                                                bias_initializer=k.initializers.Constant(0.1)))(x)
            x = test_activation(x, True, "relu")

    return x


def test_module_down(x, regularisation, filters, initializer, batch_normalisation_bool, activation):
    if regularisation:
        x = k.layers.TimeDistributed(k.layers.Convolution3D(filters=filters,
                                                            kernel_size=(3, 3, 3),
                                                            strides=(2, 2, 2),
                                                            padding="same",
                                                            kernel_initializer=initializer,
                                                            bias_initializer=k.initializers.Constant(0.1),
                                                            kernel_regularizer=k.regularizers.l2(0.01),
                                                            bias_regularizer=k.regularizers.l2(0.01),
                                                            kernel_constraint=k.constraints.max_norm(3),
                                                            bias_constraint=k.constraints.max_norm(3)))(x)
        x = test_activation(x, batch_normalisation_bool, activation)
        x = k.layers.AlphaDropout(0.5)(x)
    else:
        x = k.layers.TimeDistributed(k.layers.Convolution3D(filters=filters,
                                                            kernel_size=(3, 3, 3),
                                                            strides=(2, 2, 2),
                                                            padding="same",
                                                            kernel_initializer=initializer,
                                                            bias_initializer=k.initializers.Constant(0.1)))(x)
        x = test_activation(x, False, activation)

    return x


def test_down(x, layers, regularisation, filters, initializer, batch_normalisation_bool, activation):
    x_skip = []

    for _ in range(layers):
        x_skip.append(x)

        x = test_module_down(x, regularisation, filters, initializer, batch_normalisation_bool, activation)

    return x, x_skip


def test_module_out(x):
    x = k.layers.Flatten()(x)

    return x


def test_in_down_out(x, activation, regularisation, filters, initializer, layers, batch_normalisation_bool):
    x = test_module_in(x, activation, regularisation, filters, initializer)

    x, x_skip = test_down(x, layers, regularisation, filters, initializer, batch_normalisation_bool, activation)

    x = test_module_out(x)

    return x


def test_module_rnn(x, regularisation, units, activation, return_sequences, initializer, unroll):
    if regularisation:
        x = k.layers.SimpleRNN(units=units,
                               activation=activation,
                               return_sequences=return_sequences,
                               dropout=0.25,
                               recurrent_dropout=0.25,
                               kernel_initializer=initializer,
                               bias_initializer=k.initializers.Constant(0.1),
                               recurrent_initializer=initializer,
                               kernel_regularizer=k.regularizers.l2(0.01),
                               bias_regularizer=k.regularizers.l2(0.01),
                               recurrent_regularizer=k.regularizers.l2(0.01),
                               kernel_constraint=k.constraints.max_norm(3),
                               bias_constraint=k.constraints.max_norm(3),
                               recurrent_constraint=k.constraints.max_norm(3),
                               unroll=unroll)(x)
    else:
        x = k.layers.SimpleRNN(units=units,
                               activation=activation,
                               return_sequences=return_sequences,
                               kernel_initializer=initializer,
                               bias_initializer=k.initializers.Constant(0.1),
                               recurrent_initializer=initializer,
                               unroll=unroll)(x)

    return x


def test_module_lstm(x, regularisation, units, activation, return_sequences, initializer, unroll):
    if regularisation:
        x = k.layers.LSTM(units=units,
                          activation=activation,
                          recurrent_activation=activation,
                          return_sequences=return_sequences,
                          dropout=0.25,
                          recurrent_dropout=0.25,
                          kernel_initializer=initializer,
                          bias_initializer=k.initializers.Constant(0.1),
                          recurrent_initializer=initializer,
                          kernel_regularizer=k.regularizers.l2(0.01),
                          bias_regularizer=k.regularizers.l2(0.01),
                          recurrent_regularizer=k.regularizers.l2(0.01),
                          kernel_constraint=k.constraints.max_norm(3),
                          bias_constraint=k.constraints.max_norm(3),
                          recurrent_constraint=k.constraints.max_norm(3),
                          unroll=unroll)(x)
    else:
        x = k.layers.LSTM(units=units,
                          activation=activation,
                          recurrent_activation=activation,
                          return_sequences=return_sequences,
                          kernel_initializer=initializer,
                          bias_initializer=k.initializers.Constant(0.1),
                          recurrent_initializer=initializer,
                          unroll=unroll)(x)

    return x


def test_module_gru(x, regularisation, units, activation, return_sequences, initializer, unroll):
    if regularisation:
        x = k.layers.GRU(units=units,
                         activation=activation,
                         recurrent_activation=activation,
                         return_sequences=return_sequences,
                         dropout=0.25,
                         recurrent_dropout=0.25,
                         kernel_initializer=initializer,
                         bias_initializer=k.initializers.Constant(0.1),
                         recurrent_initializer=initializer,
                         kernel_regularizer=k.regularizers.l2(0.01),
                         bias_regularizer=k.regularizers.l2(0.01),
                         recurrent_regularizer=k.regularizers.l2(0.01),
                         kernel_constraint=k.constraints.max_norm(3),
                         bias_constraint=k.constraints.max_norm(3),
                         recurrent_constraint=k.constraints.max_norm(3),
                         unroll=unroll)(x)
    else:
        x = k.layers.GRU(units=units,
                         activation=activation,
                         recurrent_activation=activation,
                         return_sequences=return_sequences,
                         kernel_initializer=initializer,
                         bias_initializer=k.initializers.Constant(0.1),
                         recurrent_initializer=initializer,
                         unroll=unroll)(x)

    return x


def test_module_rnn_out(x, layers, rnn_type, regularisation, units, activation, initializer, unroll):
    if layers > 1:
        x = k.layers.TimeDistributed(k.layers.Flatten())(x)

        for _ in range(layers - 1):
            if rnn_type == "rnn":
                x = test_module_rnn(x, regularisation, units, activation, True, initializer, unroll)
            else:
                if rnn_type == "lstm":
                    x = test_module_lstm(x, regularisation, units, activation, True, initializer, unroll)
                else:
                    if rnn_type == "gru":
                        x = test_module_gru(x, regularisation, units, activation, True, initializer, unroll)

    return x


def test_rnn_out(x, layers, rnn_type, regularisation, units, activation, initializer, unroll):
    x = test_module_rnn_out(x, layers, rnn_type, regularisation, units, activation, initializer, unroll)

    return x


def test_in_down_rnn_out(x, activation, regularisation, filters, initializer, down_layers, batch_normalisation_bool,
                         out_layers, rnn_type, units, rnn_activation, rnn_initializer, unroll):
    x = test_module_in(x, activation, regularisation, filters, initializer)

    x, x_skip = test_down(x, down_layers, regularisation, filters, initializer, batch_normalisation_bool, activation)

    x = test_module_rnn_out(x, out_layers, rnn_type, regularisation, units, rnn_activation, rnn_initializer, unroll)

    return x


def test_module_up(x, regularisation, filters, initializer, batch_normalisation_bool, activation):
    x_shape = x.get_shape()

    if regularisation:
        x = k.layers.TimeDistributed(k.layers.Deconvolution3D(filters=filters,
                                                              kernel_size=(3, 3, 3),
                                                              strides=(2, 2, 2),
                                                              padding="same",
                                                              kernel_initializer=initializer,
                                                              bias_initializer=k.initializers.Constant(0.1),
                                                              kernel_regularizer=k.regularizers.l2(0.01),
                                                              bias_regularizer=k.regularizers.l2(0.01),
                                                              kernel_constraint=k.constraints.max_norm(3),
                                                              bias_constraint=k.constraints.max_norm(3)))(x)
        x = test_activation(x, batch_normalisation_bool, activation)
        x = k.layers.AlphaDropout(0.5)(x)
    else:
        x = k.layers.TimeDistributed(k.layers.Deconvolution3D(filters=filters,
                                                              kernel_size=(3, 3, 3),
                                                              strides=(2, 2, 2),
                                                              padding="same",
                                                              kernel_initializer=initializer,
                                                              bias_initializer=k.initializers.Constant(0.1)))(x)
        x = test_activation(x, False, activation)

    x = k.layers.Reshape((int(x_shape[1]), int(x_shape[2] * 2), int(x_shape[3] * 2), int(x_shape[4] * 2), filters))(x)

    return x


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


def test_crop(source, target):
    ct, ch, cw, cd = get_crop_shape(source, target)
    x = k.layers.TimeDistributed(k.layers.Cropping3D(cropping=(ch, cw, cd)))(source)

    return x


def test_up(x, x_skip, regularisation, filters, initializer, batch_normalisation_bool, activation):
    for i in range(len(x_skip)):
        x = test_module_up(x, regularisation, filters, initializer, batch_normalisation_bool, activation)

        x = test_crop(x, x_skip[(len(x_skip) - 1) - i])

    return x


def test_multi_out(x, activation, regularisation, filters, initializer, layers, batch_normalisation_bool):
    x = test_module_in(x, activation, regularisation, filters, initializer)

    x, x_skip = test_down(x, layers, regularisation, filters, initializer, batch_normalisation_bool, activation)

    x_2 = test_up(x, x_skip, regularisation, filters, initializer, batch_normalisation_bool, activation)

    x_1 = test_module_out(x)

    return x_1, x_2


def test_multi_rnn_out(x, activation, regularisation, filters, initializer, down_layers, batch_normalisation_bool,
                       tof_bool, out_layers, rnn_type, units, rnn_activation, rnn_initializer, unroll):
    x = test_module_in(x, activation, regularisation, filters, initializer)

    x, x_skip = test_down(x, down_layers, regularisation, filters, initializer, batch_normalisation_bool, activation)

    x_2 = test_up(x, x_skip, regularisation, filters, initializer, batch_normalisation_bool, activation)

    x_1 = test_module_rnn_out(x, out_layers, rnn_type, regularisation, units, rnn_activation, rnn_initializer, unroll)

    return x_1, x_2
