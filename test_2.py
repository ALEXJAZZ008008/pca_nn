import keras as k


def test_activation(x, batch_normalisation_bool, activation, regularisation, dropout):
    if batch_normalisation_bool:
        x = k.layers.BatchNormalization()(x)

    if activation == "lrelu":
        x = k.layers.LeakyReLU()(x)
    else:
        if activation == "prelu":
            x = k.layers.PReLU()(x)
        else:
            x = k.layers.Activation(activation)(x)

    if regularisation:
        x = k.layers.Dropout(dropout)(x)

    return x


def test_in(x, initializer, batch_normalisation_bool, activation):
    x = k.layers.TimeDistributed(k.layers.Convolution3D(filters=1,
                                                        kernel_size=(3, 3, 3),
                                                        strides=(1, 1, 1),
                                                        padding="same",
                                                        kernel_initializer=initializer,
                                                        bias_initializer=k.initializers.Constant(0.1)))(x)

    x = test_activation(x, batch_normalisation_bool, activation, False, 0.0)

    return x


def test_module_module_down(x, filters, kernal_size, initializer, batch_normalisation_bool, activation, regularisation,
                            ltwo, dropout, feature_downsample_bool):
    feature_downsample_filters = int(filters / 2)

    if regularisation:
        x = k.layers.TimeDistributed(k.layers.Convolution3D(filters=filters,
                                                            kernel_size=(kernal_size, kernal_size, kernal_size),
                                                            strides=(2, 2, 2),
                                                            padding="same",
                                                            kernel_initializer=initializer,
                                                            bias_initializer=k.initializers.Constant(0.1),
                                                            kernel_regularizer=k.regularizers.l2(ltwo),
                                                            kernel_constraint=k.constraints.UnitNorm()))(x)

        x = test_activation(x, batch_normalisation_bool, activation, regularisation, dropout)

        if feature_downsample_bool:
            x = k.layers.TimeDistributed(k.layers.Convolution3D(filters=feature_downsample_filters,
                                                                kernel_size=(1, 1, 1),
                                                                strides=(1, 1, 1),
                                                                padding="same",
                                                                kernel_initializer=initializer,
                                                                bias_initializer=k.initializers.Constant(0.1),
                                                                kernel_regularizer=k.regularizers.l2(ltwo),
                                                                kernel_constraint=k.constraints.UnitNorm()))(x)

            x = test_activation(x, batch_normalisation_bool, activation, regularisation, dropout)
    else:
        x = k.layers.TimeDistributed(k.layers.Convolution3D(filters=filters,
                                                            kernel_size=(kernal_size, kernal_size, kernal_size),
                                                            strides=(2, 2, 2),
                                                            padding="same",
                                                            kernel_initializer=initializer,
                                                            bias_initializer=k.initializers.Constant(0.1)))(x)


        x = test_activation(x, batch_normalisation_bool, activation, regularisation, dropout)

        if feature_downsample_bool:
            x = k.layers.TimeDistributed(k.layers.Convolution3D(filters=feature_downsample_filters,
                                                                kernel_size=(1, 1, 1),
                                                                strides=(1, 1, 1),
                                                                padding="same",
                                                                kernel_initializer=initializer,
                                                                bias_initializer=k.initializers.Constant(0.1)))(x)

            x = test_activation(x, batch_normalisation_bool, activation, regularisation, dropout)

    return x


def test_module_down(x, filters, initializer, batch_normalisation_bool, activation, regularisation, ltwo, dropout,
                     feature_downsample_bool, concat_bool):
    x_1 = test_module_module_down(x, filters, 3, initializer, batch_normalisation_bool, activation, regularisation,
                                  ltwo, dropout, feature_downsample_bool)

    if concat_bool:
        x_2 = test_module_module_down(x, filters, 5, initializer, batch_normalisation_bool, activation, regularisation,
                                      ltwo, dropout, feature_downsample_bool)

        x_3 = test_module_module_down(x, filters, 7, initializer, batch_normalisation_bool, activation, regularisation,
                                      ltwo, dropout, feature_downsample_bool)

        x = k.layers.Concatenate(axis=4)([x_1, x_2, x_3])
    else:
        x = x_1

    return x


def test_down(x, x_skip, layers, filters, initializer, batch_normalisation_bool, activation, regularisation, ltwo,
              dropout, feature_downsample_bool, concat_bool):
    for _ in range(layers):
        x_skip.append(x)

        x = test_module_down(x, filters, initializer, batch_normalisation_bool, activation, regularisation, ltwo,
                             dropout, feature_downsample_bool, concat_bool)

    return x, x_skip


def test_module_out(x):
    x = k.layers.TimeDistributed(k.layers.Flatten())(x)

    return x


def test_in_down_out(x, x_skip, activation, regularisation, filters, initializer, layers, batch_normalisation_bool,
                     ltwo, dropout, feature_downsample_bool, concat_bool):
    x = test_in(x, initializer, batch_normalisation_bool, activation)

    x, x_skip = test_down(x, x_skip, layers, filters, initializer, batch_normalisation_bool, activation, regularisation,
                          ltwo, dropout, feature_downsample_bool, concat_bool)

    x = test_module_out(x)

    return x


def test_module_rnn_internal(x, regularisation, units, activation, ltwo, dropout, return_sequences, initializer,
                             recurrent_initializer, unroll):
    if regularisation:
        x = k.layers.SimpleRNN(units=units,
                               activation=activation,
                               dropout=dropout,
                               return_sequences=return_sequences,
                               kernel_initializer=initializer,
                               bias_initializer=k.initializers.Constant(0.1),
                               recurrent_initializer=recurrent_initializer,
                               kernel_regularizer=k.regularizers.l2(ltwo),
                               kernel_constraint=k.constraints.UnitNorm(),
                               unroll=unroll)(x)
    else:
        x = k.layers.SimpleRNN(units=units,
                               activation=activation,
                               return_sequences=return_sequences,
                               kernel_initializer=initializer,
                               bias_initializer=k.initializers.Constant(0.1),
                               recurrent_initializer=recurrent_initializer,
                               unroll=unroll)(x)

    return x


def test_module_lstm_internal(x, regularisation, units, activation, return_activation, ltwo, dropout, return_sequences,
                              initializer, recurrent_initializer, unroll):
    if regularisation:
        x = k.layers.LSTM(units=units,
                          activation=activation,
                          recurrent_activation=return_activation,
                          dropout=dropout,
                          return_sequences=return_sequences,
                          kernel_initializer=initializer,
                          bias_initializer=k.initializers.Constant(0.1),
                          recurrent_initializer=recurrent_initializer,
                          kernel_regularizer=k.regularizers.l2(ltwo),
                          kernel_constraint=k.constraints.UnitNorm(),
                          unroll=unroll)(x)
    else:
        x = k.layers.LSTM(units=units,
                          activation=activation,
                          recurrent_activation=return_activation,
                          return_sequences=return_sequences,
                          kernel_initializer=initializer,
                          bias_initializer=k.initializers.Constant(0.1),
                          recurrent_initializer=recurrent_initializer,
                          unroll=unroll)(x)

    return x


def test_module_gru_internal(x, regularisation, units, activation, return_activation, ltwo, dropout, return_sequences,
                             initializer, recurrent_initializer, unroll):
    if regularisation:
        x = k.layers.GRU(units=units,
                         activation=activation,
                         recurrent_activation=return_activation,
                         dropout=dropout,
                         return_sequences=return_sequences,
                         kernel_initializer=initializer,
                         bias_initializer=k.initializers.Constant(0.1),
                         recurrent_initializer=recurrent_initializer,
                         kernel_regularizer=k.regularizers.l2(ltwo),
                         kernel_constraint=k.constraints.UnitNorm(),
                         unroll=unroll)(x)
    else:
        x = k.layers.GRU(units=units,
                         activation=activation,
                         recurrent_activation=return_activation,
                         return_sequences=return_sequences,
                         kernel_initializer=initializer,
                         bias_initializer=k.initializers.Constant(0.1),
                         recurrent_initializer=recurrent_initializer,
                         unroll=unroll)(x)

    return x


def test_module_rnn(x, units, return_sequences, initializer, recurrent_initializer, unroll, batch_normalisation_bool,
                    activation, regularisation, ltwo, dropout):
    if regularisation:
        x = k.layers.SimpleRNN(units=units,
                               return_sequences=return_sequences,
                               kernel_initializer=initializer,
                               bias_initializer=k.initializers.Constant(0.1),
                               recurrent_initializer=recurrent_initializer,
                               kernel_regularizer=k.regularizers.l2(ltwo),
                               kernel_constraint=k.constraints.UnitNorm(),
                               unroll=unroll)(x)
    else:
        x = k.layers.SimpleRNN(units=units,
                               return_sequences=return_sequences,
                               kernel_initializer=initializer,
                               bias_initializer=k.initializers.Constant(0.1),
                               recurrent_initializer=recurrent_initializer,
                               unroll=unroll)(x)

    x = test_activation(x, batch_normalisation_bool, activation, regularisation, dropout)

    return x


def test_module_lstm(x, units, return_activation, return_sequences, initializer, recurrent_initializer, unroll,
                     batch_normalisation_bool, activation, regularisation, ltwo, dropout):
    if regularisation:
        x = k.layers.LSTM(units=units,
                          recurrent_activation=return_activation,
                          return_sequences=return_sequences,
                          kernel_initializer=initializer,
                          bias_initializer=k.initializers.Constant(0.1),
                          recurrent_initializer=recurrent_initializer,
                          kernel_regularizer=k.regularizers.l2(ltwo),
                          kernel_constraint=k.constraints.UnitNorm(),
                          unroll=unroll)(x)
    else:
        x = k.layers.LSTM(units=units,
                          recurrent_activation=return_activation,
                          return_sequences=return_sequences,
                          kernel_initializer=initializer,
                          bias_initializer=k.initializers.Constant(0.1),
                          recurrent_initializer=recurrent_initializer,
                          unroll=unroll)(x)

    x = test_activation(x, batch_normalisation_bool, activation, regularisation, dropout)

    return x


def test_module_gru(x, units, return_activation, return_sequences, initializer, recurrent_initializer, unroll,
                    batch_normalisation_bool, activation, regularisation, ltwo, dropout):
    if regularisation:
        x = k.layers.GRU(units=units,
                         recurrent_activation=return_activation,
                         return_sequences=return_sequences,
                         kernel_initializer=initializer,
                         bias_initializer=k.initializers.Constant(0.1),
                         recurrent_initializer=recurrent_initializer,
                         kernel_regularizer=k.regularizers.l2(ltwo),
                         kernel_constraint=k.constraints.UnitNorm(),
                         unroll=unroll)(x)
    else:
        x = k.layers.GRU(units=units,
                         recurrent_activation=return_activation,
                         return_sequences=return_sequences,
                         kernel_initializer=initializer,
                         bias_initializer=k.initializers.Constant(0.1),
                         recurrent_initializer=recurrent_initializer,
                         unroll=unroll)(x)

    x = test_activation(x, batch_normalisation_bool, activation, regularisation, dropout)

    return x


def test_module_module_rnn_out(x, rnn_type, internal_bool, units, return_activation, initializer,
                               recurrent_initializer, unroll, batch_normalisation_bool, activation, ltwo, dropout,
                               regularisation, return_sequences):
    if internal_bool:
        if rnn_type == "rnn":
            x = test_module_rnn_internal(x, regularisation, units, activation, ltwo, dropout, return_sequences,
                                         recurrent_initializer,
                                         initializer, unroll)
        else:
            if rnn_type == "lstm":
                x = test_module_lstm_internal(x, regularisation, units, activation, return_activation, ltwo,
                                              dropout, return_sequences, recurrent_initializer, initializer, unroll)
            else:
                if rnn_type == "gru":
                    x = test_module_gru_internal(x, regularisation, units, activation, return_activation, ltwo,
                                                 dropout, return_sequences, recurrent_initializer, initializer, unroll)
    else:
        if rnn_type == "rnn":
            x = test_module_rnn(x, units, return_sequences, recurrent_initializer, initializer, unroll,
                                batch_normalisation_bool, activation, regularisation, ltwo, dropout)
        else:
            if rnn_type == "lstm":
                x = test_module_lstm(x, units, return_activation, return_sequences, recurrent_initializer, initializer,
                                     unroll, batch_normalisation_bool, activation, regularisation, ltwo, dropout)
            else:
                if rnn_type == "gru":
                    x = test_module_gru(x, units, return_activation, return_sequences, recurrent_initializer,
                                        initializer, unroll, batch_normalisation_bool, activation, regularisation, ltwo,
                                        dropout)

    return x


def test_module_rnn_out(x, layers, rnn_type, internal_bool, units, return_activation, initializer,
                        recurrent_initializer, unroll, batch_normalisation_bool, activation, ltwo, dropout,
                        regularisation, return_sequences):
    x = k.layers.TimeDistributed(k.layers.Flatten())(x)

    for _ in range(1, layers):
        x = test_module_module_rnn_out(x, rnn_type, internal_bool, units, return_activation, initializer,
                                       recurrent_initializer, unroll, batch_normalisation_bool, activation, ltwo,
                                       dropout, regularisation, True)

    x = test_module_module_rnn_out(x, rnn_type, internal_bool, units, return_activation, initializer,
                                   recurrent_initializer, unroll, batch_normalisation_bool, activation, ltwo, dropout,
                                   regularisation, return_sequences)

    return x


def test_rnn_out(x, layers, rnn_type, internal_bool, units, return_activation, initializer, recurrent_initializer,
                 unroll, batch_normalisation_bool, activation, ltwo, dropout, regularisation, return_sequences):
    x = test_module_rnn_out(x, layers, rnn_type, internal_bool, units, return_activation, recurrent_initializer,
                            initializer, unroll, batch_normalisation_bool, activation, ltwo, dropout, regularisation,
                            return_sequences)

    return x


def test_in_down_rnn_out(x, x_skip, activation, regularisation, ltwo, dropout, filters, initializer, down_layers,
                         batch_normalisation_bool, out_layers, rnn_type, internal_bool, units, rnn_return_activation,
                         unroll, rnn_batch_normalisation_bool, rnn_initializer, rnn_recurrent_initializer,
                         rnn_activation, rnn_ltwo, rnn_dropout, rnn_regularisation, rnn_return_sequences,
                         feature_downsample_bool, concat_bool):
    x = test_in(x, initializer, batch_normalisation_bool, activation)

    x, x_skip = test_down(x, x_skip, down_layers, filters, initializer, batch_normalisation_bool, activation,
                          regularisation, ltwo, dropout, feature_downsample_bool, concat_bool)

    x = test_module_rnn_out(x, out_layers, rnn_type, internal_bool, units, rnn_return_activation, rnn_initializer,
                            rnn_recurrent_initializer, unroll, rnn_batch_normalisation_bool, rnn_activation, rnn_ltwo,
                            rnn_dropout, rnn_regularisation, rnn_return_sequences)

    return x


def test_module_up(x, regularisation, dropout, filters, initializer, batch_normalisation_bool, activation):
    x_shape = x.get_shape()

    x = k.layers.TimeDistributed(k.layers.Deconvolution3D(filters=filters,
                                                          kernel_size=(3, 3, 3),
                                                          strides=(2, 2, 2),
                                                          padding="same",
                                                          kernel_initializer=initializer,
                                                          bias_initializer=k.initializers.Constant(0.1)))(x)

    x = test_activation(x, batch_normalisation_bool, activation, regularisation, dropout)

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


def test_up(x, x_skip, regularisation, dropout, filters, initializer, batch_normalisation_bool, activation):
    for i in range(len(x_skip)):
        x = test_module_up(x, regularisation, dropout, filters, initializer, batch_normalisation_bool, activation)

        x = test_crop(x, x_skip[(len(x_skip) - 1) - i])

    return x


def test_multi_out(x, x_skip, activation, regularisation, ltwo, dropout, filters, initializer, layers,
                   batch_normalisation_bool, up_filters, feature_downsample_bool, concat_bool):
    x = test_in(x, initializer, batch_normalisation_bool, activation)

    x, x_skip = test_down(x, x_skip, layers, filters, initializer, batch_normalisation_bool, activation, regularisation,
                          ltwo, dropout, feature_downsample_bool, concat_bool)

    x_2 = test_up(x, x_skip, False, 0.0, up_filters, initializer, batch_normalisation_bool, activation)

    x_1 = test_module_out(x)

    return x, x_skip, x_1, x_2


def test_multi_rnn_out(x, x_skip, activation, regularisation, ltwo, dropout, filters, initializer, down_layers,
                       batch_normalisation_bool, up_filters, out_layers, rnn_type, internal_bool, units,
                       rnn_return_activation, rnn_initializer, rnn_recurrent_initializer, unroll,
                       rnn_batch_normalisation_bool, rnn_activation, rnn_ltwo, rnn_dropout, rnn_regularisation,
                       rnn_return_sequences, feature_downsample_bool, concat_bool):
    #x = test_in(x, initializer, batch_normalisation_bool, activation)

    x, x_skip = test_down(x, x_skip, down_layers, filters, initializer, batch_normalisation_bool, activation,
                          regularisation, ltwo, dropout, feature_downsample_bool, concat_bool)

    x_2 = test_up(x, x_skip, False, 0.0, up_filters, initializer, batch_normalisation_bool, activation)

    x_1 = test_module_rnn_out(x, out_layers, rnn_type, internal_bool, units, rnn_return_activation, rnn_initializer,
                              rnn_recurrent_initializer, unroll, rnn_batch_normalisation_bool, rnn_activation, rnn_ltwo,
                              rnn_dropout, rnn_regularisation, rnn_return_sequences)

    return x, x_skip, x_1, x_2


def output_module_1(x, internal_bool, rnn_type, units, rnn_activation, rnn_initializer, rnn_recurrent_initializer,
                    unroll, rnn_recurrent_activation, initializer, activation, return_sequence,
                    batch_normalisation_bool, name):
    #x = output_module_1_module(x, internal_bool, rnn_type, units, rnn_activation, rnn_initializer,
    #                           rnn_recurrent_initializer, unroll, rnn_recurrent_activation, return_sequence,
    #                           batch_normalisation_bool)

    x = k.layers.TimeDistributed(k.layers.Dense(units=1,
                                                kernel_initializer=initializer,
                                                bias_initializer=k.initializers.Constant(0.1)))(x)

    if batch_normalisation_bool:
        x = k.layers.BatchNormalization()(x)

    x = k.layers.Activation(activation)(x)

    x = k.layers.Reshape(tuple([t for t in x.shape.as_list() if t != 1 and t is not None]), name=name)(x)

    return x


def output_module_1_module(x, internal_bool, rnn_type, units, activation, initializer, recurrent_initializer, unroll,
                           recurrent_activation, return_sequence, batch_normalisation_bool):
    if internal_bool:
        if rnn_type == "rnn":
            x = k.layers.SimpleRNN(units=units,
                                   activation=activation,
                                   return_sequences=return_sequence,
                                   kernel_initializer=initializer,
                                   bias_initializer=k.initializers.Constant(0.1),
                                   recurrent_initializer=recurrent_initializer,
                                   unroll=unroll)(x)
        else:
            if rnn_type == "lstm":
                x = k.layers.LSTM(units=units,
                                  activation=activation,
                                  recurrent_activation=recurrent_activation,
                                  return_sequences=return_sequence,
                                  kernel_initializer=initializer,
                                  bias_initializer=k.initializers.Constant(0.1),
                                  recurrent_initializer=recurrent_initializer,
                                  unroll=unroll)(x)
            else:
                if rnn_type == "gru":
                    x = k.layers.GRU(units=units,
                                     activation=activation,
                                     recurrent_activation=recurrent_activation,
                                     return_sequences=return_sequence,
                                     kernel_initializer=initializer,
                                     bias_initializer=k.initializers.Constant(0.1),
                                     recurrent_initializer=recurrent_initializer,
                                     unroll=unroll)(x)
    else:
        if rnn_type == "rnn":
            x = k.layers.SimpleRNN(units=units,
                                   return_sequences=return_sequence,
                                   kernel_initializer=initializer,
                                   bias_initializer=k.initializers.Constant(0.1),
                                   recurrent_initializer=recurrent_initializer,
                                   unroll=unroll)(x)
        else:
            if rnn_type == "lstm":
                x = k.layers.LSTM(units=units,
                                  recurrent_activation=recurrent_activation,
                                  return_sequences=return_sequence,
                                  kernel_initializer=initializer,
                                  bias_initializer=k.initializers.Constant(0.1),
                                  recurrent_initializer=recurrent_initializer,
                                  unroll=unroll)(x)
            else:
                if rnn_type == "gru":
                    x = k.layers.GRU(units=units,
                                     recurrent_activation=recurrent_activation,
                                     return_sequences=return_sequence,
                                     kernel_initializer=initializer,
                                     bias_initializer=k.initializers.Constant(0.1),
                                     recurrent_initializer=recurrent_initializer,
                                     unroll=unroll)(x)

        if batch_normalisation_bool:
            x = k.layers.BatchNormalization()(x)

        x = k.layers.Activation(activation)(x)

    return x


def output_module_2(x, initializer, activation, batch_normalisation_bool, name):
    x = k.layers.TimeDistributed(k.layers.Convolution3D(filters=1,
                                                        kernel_size=(3, 3, 3),
                                                        strides=(1, 1, 1),
                                                        padding='same',
                                                        kernel_initializer=initializer,
                                                        bias_initializer=k.initializers.Constant(0.1)))(x)

    if batch_normalisation_bool:
        x = k.layers.BatchNormalization()(x)

    x = k.layers.Activation(activation, name=name)(x)

    return x
