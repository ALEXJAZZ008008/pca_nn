import keras as k


def test_activation(x, batch_normalisation_bool, activation):
    if batch_normalisation_bool:
        x = k.layers.BatchNormalization()(x)

    if activation == "prelu":
        x = k.layers.PReLU()(x)
    else:
        x = k.layers.Activation(activation)(x)

    return x


def test_module_in(x, filters, initializer):
    x = k.layers.Convolution3D(filters=filters, kernel_size=(1, 1, 1), strides=(1, 1, 1), padding='same',
                               kernel_initializer=initializer, bias_initializer=k.initializers.Constant(0.1))(x)
    x = test_activation(x, True, "tanh")
    x = k.layers.AlphaDropout(0.5)(x)

    return x


def test_module_out(x, filters, initializer, batch_normalisation_bool, activation):
    x = k.layers.Convolution3D(filters=filters, kernel_size=(1, 1, 1), strides=(1, 1, 1), padding='same',
                               kernel_initializer=initializer, bias_initializer=k.initializers.Constant(0.1))(x)
    x = test_activation(x, batch_normalisation_bool, activation)
    x = k.layers.AlphaDropout(0.5)(x)

    x = k.layers.Flatten()(x)

    return x


def test_module_down_skip(x, filters, kernal_size, strides, initializer, batch_normalisation_bool, activation):
    x = k.layers.Convolution3D(filters=filters,
                               kernel_size=(kernal_size, kernal_size, kernal_size),
                               strides=(strides, strides, strides),
                               padding="same",
                               kernel_initializer=initializer,
                               bias_initializer=k.initializers.Constant(0.1))(x)
    x = test_activation(x, batch_normalisation_bool, activation)

    return x


def test_module_down(x, filters, initializer, batch_normalisation_bool, activation):
    x_1 = k.layers.Convolution3D(filters=filters,
                                 kernel_size=(3, 3, 3),
                                 strides=(2, 2, 2),
                                 padding="same",
                                 kernel_initializer=initializer,
                                 bias_initializer=k.initializers.Constant(0.1))(x)
    x_1 = test_activation(x_1, batch_normalisation_bool, activation)
    x_1 = k.layers.AlphaDropout(0.5)(x_1)

    x_2 = k.layers.Convolution3D(filters=filters, kernel_size=(5, 5, 5), strides=(2, 2, 2), padding="same",
                                 kernel_initializer=initializer, bias_initializer=k.initializers.Constant(0.1))(x)
    x_2 = test_activation(x_2, batch_normalisation_bool, activation)
    x_2 = k.layers.AlphaDropout(0.5)(x_2)

    x_3 = k.layers.Convolution3D(filters=filters, kernel_size=(7, 7, 7), strides=(2, 2, 2), padding="same",
                                 kernel_initializer=initializer, bias_initializer=k.initializers.Constant(0.1))(x)
    x_3 = test_activation(x_3, batch_normalisation_bool, activation)
    x_3 = k.layers.AlphaDropout(0.5)(x_3)

    x_4 = k.layers.Convolution3D(filters=filters, kernel_size=(9, 9, 9), strides=(2, 2, 2), padding="same",
                                 kernel_initializer=initializer, bias_initializer=k.initializers.Constant(0.1))(x)
    x_4 = test_activation(x_4, batch_normalisation_bool, activation)
    x_4 = k.layers.AlphaDropout(0.5)(x_4)

    x = k.layers.Concatenate(axis=4)([x_1, x_2, x_3, x_4])

    return x


def test_down(x, initializer, batch_normalisation_bool, activation):
    x_shortcut_1 = x
    x_skip_1_1 = test_module_down_skip(x, 1, 2, 2, initializer, batch_normalisation_bool, activation)
    x_skip_1_2 = test_module_down_skip(x, 2, 4, 4, initializer, batch_normalisation_bool, activation)
    x_skip_1_3 = test_module_down_skip(x, 3, 8, 8, initializer, batch_normalisation_bool, activation)
    x_skip_1_4 = test_module_down_skip(x, 5, 16, 16, initializer, batch_normalisation_bool, activation)
    x_skip_1_5 = test_module_down_skip(x, 8, 32, 32, initializer, batch_normalisation_bool, activation)
    x = test_module_down(x, 1, initializer, batch_normalisation_bool, activation)
    x = k.layers.Concatenate(axis=4)([x, x_skip_1_1])

    x_shortcut_2 = x
    x_skip_2_2 = test_module_down_skip(x, 2, 2, 2, initializer, batch_normalisation_bool, activation)
    x_skip_2_3 = test_module_down_skip(x, 3, 4, 4, initializer, batch_normalisation_bool, activation)
    x_skip_2_4 = test_module_down_skip(x, 5, 8, 8, initializer, batch_normalisation_bool, activation)
    x_skip_2_5 = test_module_down_skip(x, 8, 16, 16, initializer, batch_normalisation_bool, activation)
    x_skip_2_6 = test_module_down_skip(x, 13, 32, 32, initializer, batch_normalisation_bool, activation)
    x = test_module_down(x, 2, initializer, batch_normalisation_bool, activation)
    x = k.layers.Concatenate(axis=4)([x_skip_1_2, x_skip_2_2, x])

    x_shortcut_3 = x
    x_skip_3_3 = test_module_down_skip(x, 3, 2, 2, initializer, batch_normalisation_bool, activation)
    x_skip_3_4 = test_module_down_skip(x, 5, 4, 4, initializer, batch_normalisation_bool, activation)
    x_skip_3_5 = test_module_down_skip(x, 8, 8, 8, initializer, batch_normalisation_bool, activation)
    x_skip_3_6 = test_module_down_skip(x, 13, 16, 16, initializer, batch_normalisation_bool, activation)
    x_skip_3_7 = test_module_down_skip(x, 21, 32, 32, initializer, batch_normalisation_bool, activation)
    x = test_module_down(x, 3, initializer, batch_normalisation_bool, activation)
    x = k.layers.Concatenate(axis=4)([x_skip_1_3, x_skip_2_3, x_skip_3_3, x])

    x_shortcut_4 = x
    x_skip_4_4 = test_module_down_skip(x, 5, 2, 2, initializer, batch_normalisation_bool, activation)
    x_skip_4_5 = test_module_down_skip(x, 8, 4, 4, initializer, batch_normalisation_bool, activation)
    x_skip_4_6 = test_module_down_skip(x, 13, 8, 8, initializer, batch_normalisation_bool, activation)
    x_skip_4_7 = test_module_down_skip(x, 21, 16, 16, initializer, batch_normalisation_bool, activation)
    x = test_module_down(x, 5, initializer, batch_normalisation_bool, activation)
    x = k.layers.Concatenate(axis=4)([x_skip_1_4, x_skip_2_4, x_skip_3_4, x_skip_4_4, x])

    x_shortcut_5 = x
    x_skip_5_5 = test_module_down_skip(x, 8, 2, 2, initializer, batch_normalisation_bool, activation)
    x_skip_5_6 = test_module_down_skip(x, 13, 4, 4, initializer, batch_normalisation_bool, activation)
    x_skip_5_7 = test_module_down_skip(x, 21, 8, 8, initializer, batch_normalisation_bool, activation)
    x = test_module_down(x, 8, initializer, batch_normalisation_bool, activation)
    x = k.layers.Concatenate(axis=4)([x_skip_1_5, x_skip_2_5, x_skip_3_5, x_skip_4_5, x_skip_5_5, x])

    x_shortcut_6 = x
    x_skip_6_6 = test_module_down_skip(x, 13, 2, 2, initializer, batch_normalisation_bool, activation)
    x_skip_6_7 = test_module_down_skip(x, 21, 4, 4, initializer, batch_normalisation_bool, activation)
    x = test_module_down(x, 13, initializer, batch_normalisation_bool, activation)
    x = k.layers.Concatenate(axis=4)([x_skip_2_6, x_skip_3_6, x_skip_4_6, x_skip_5_6, x_skip_6_6, x])

    x_shortcut_7 = x
    x_skip_7_7 = test_module_down_skip(x, 21, 2, 2, initializer, batch_normalisation_bool, activation)
    x = test_module_down(x, 21, initializer, batch_normalisation_bool, activation)
    x = k.layers.Concatenate(axis=4)([x_skip_3_7, x_skip_4_7, x_skip_5_7, x_skip_6_7, x_skip_7_7, x])

    return x, x_shortcut_1, x_shortcut_2, x_shortcut_3, x_shortcut_4, x_shortcut_5, x_shortcut_6, x_shortcut_7


def test_in_down_out(x, filters, initializer, batch_normalisation_bool, activation):
    x = test_module_in(x, filters, initializer)

    x, x_shortcut_1, x_shortcut_2, x_shortcut_3, x_shortcut_4, x_shortcut_5, x_shortcut_6, x_shortcut_7 = test_down(x,
                                                                                                                    initializer,
                                                                                                                    batch_normalisation_bool,
                                                                                                                    activation)

    x = test_module_out(x, filters, initializer, batch_normalisation_bool, activation)

    return x


def test_module_rnn(x, units, activation, return_sequences, initializer, unroll):
    x = k.layers.SimpleRNN(units=units,
                           activation=activation,
                           recurrent_activation=activation,
                           return_sequences=return_sequences,
                           dropout=0.5,
                           recurrent_dropout=0.5,
                           kernel_initializer=initializer,
                           bias_initializer=k.initializers.Constant(0.0),
                           recurrent_initializer=initializer,
                           unroll=unroll)(x)

    return x


def test_module_lstm(x, units, activation, return_sequences, initializer, unroll):
    x = k.layers.LSTM(units=units,
                      activation=activation,
                      recurrent_activation=activation,
                      return_sequences=return_sequences,
                      dropout=0.5,
                      recurrent_dropout=0.5,
                      kernel_initializer=initializer,
                      bias_initializer=k.initializers.Constant(0.0),
                      recurrent_initializer=initializer,
                      unroll=unroll)(x)

    return x


def test_module_gru(x, units, activation, return_sequences, initializer, unroll):
    x = k.layers.GRU(units=units,
                     activation=activation,
                     recurrent_activation=activation,
                     return_sequences=return_sequences,
                     dropout=0.5,
                     recurrent_dropout=0.5,
                     kernel_initializer=initializer,
                     bias_initializer=k.initializers.Constant(0.0),
                     recurrent_initializer=initializer,
                     unroll=unroll)(x)

    return x


def test_rnn(x, tof_bool, layers, rnn_type, units, activation, initializer, unroll):
    if tof_bool:
        x_shape = [x.shape.as_list()[1], x.shape.as_list()[2], x.shape.as_list()[3], x.shape.as_list()[4],
                   x.shape.as_list()[5]]

        x = k.layers.Reshape((x.shape.as_list()[1] * x.shape.as_list()[2] * x.shape.as_list()[3] * x.shape.as_list()[4],
                              x.shape.as_list()[5]))(x)
    else:
        x_shape = [x.shape.as_list()[1], x.shape.as_list()[2], x.shape.as_list()[3], x.shape.as_list()[4]]

        x = k.layers.Reshape(
            (x.shape.as_list()[1] * x.shape.as_list()[2] * x.shape.as_list()[3], x.shape.as_list()[4]))(x)

    for _ in range(layers):
        if rnn_type == "rnn":
            x = test_module_rnn(x, units, activation, True, initializer, unroll)
        else:
            if rnn_type == "lstm":
                x = test_module_lstm(x, units, activation, True, initializer, unroll)
            else:
                if rnn_type == "gru":
                    x = test_module_gru(x, units, activation, True, initializer, unroll)

    if tof_bool:
        x = k.layers.Reshape((x_shape[0], x_shape[1], x_shape[2], x_shape[3], x_shape[4]))(x)
    else:
        x = k.layers.Reshape((x_shape[0], x_shape[1], x_shape[2], x_shape[3]))(x)

    return x


def test_module_rnn_out(x, tof_bool, rnn_type, units, activation, initializer, unroll):
    if tof_bool:
        x = k.layers.Reshape((x.shape.as_list()[1] * x.shape.as_list()[2] * x.shape.as_list()[3] * x.shape.as_list()[4],
                              x.shape.as_list()[5]))(x)
    else:
        x = k.layers.Reshape(
            (x.shape.as_list()[1] * x.shape.as_list()[2] * x.shape.as_list()[3], x.shape.as_list()[4]))(x)

    if rnn_type == "rnn":
        x = test_module_rnn(x, units, activation, False, initializer, unroll)
    else:
        if rnn_type == "lstm":
            x = test_module_lstm(x, units, activation, False, initializer, unroll)
        else:
            if rnn_type == "gru":
                x = test_module_gru(x, units, activation, False, initializer, unroll)

    return x


def test_rnn_out(x, tof_bool, layers, rnn_type, units, activation, initializer, unroll):
    if layers > 0:
        x = test_rnn(x, tof_bool, layers, rnn_type, units, activation, initializer, unroll)

    x = test_module_rnn_out(x, tof_bool, rnn_type, units, activation, initializer, unroll)

    return x


def test_washer(x, filters, initializer, batch_normalisation_bool, activation):
    x = k.layers.Convolution3D(filters=filters, kernel_size=(1, 1, 1), strides=(1, 1, 1), padding='same',
                               kernel_initializer=initializer, bias_initializer=k.initializers.Constant(0.1))(x)
    x = test_activation(x, batch_normalisation_bool, activation)
    x = k.layers.AlphaDropout(0.5)(x)

    return x


def test_in_down_rnn_out(x, filters, initializer, batch_normalisation_bool, activation, tof_bool, layers, rnn_type,
                         units, rnn_activation, rnn_initializer, unroll):
    x = test_module_in(x, filters, initializer)

    x, x_shortcut_1, x_shortcut_2, x_shortcut_3, x_shortcut_4, x_shortcut_5, x_shortcut_6, x_shortcut_7 = test_down(x,
                                                                                                                    initializer,
                                                                                                                    batch_normalisation_bool,
                                                                                                                    activation)

    x = test_washer(x, filters, initializer, batch_normalisation_bool, activation)

    x = test_rnn(x, tof_bool, layers, rnn_type, units, rnn_activation, rnn_initializer, unroll)

    x = test_module_out(x, filters, initializer, batch_normalisation_bool, activation)

    return x


def test_in_rnn_down_rnn_out(x, filters, initializer, tof_bool, layers, rnn_type, units, rnn_activation,
                             rnn_initializer, batch_normalisation_bool,
                             activation, unroll):
    x = test_module_in(x, filters, initializer)

    x = test_rnn(x, tof_bool, layers, rnn_type, units, rnn_activation, rnn_initializer, unroll)

    x = test_washer(x, filters, batch_normalisation_bool, initializer, activation)

    x, x_shortcut_1, x_shortcut_2, x_shortcut_3, x_shortcut_4, x_shortcut_5, x_shortcut_6, x_shortcut_7 = test_down(x,
                                                                                                                    initializer,
                                                                                                                    batch_normalisation_bool,
                                                                                                                    activation)

    x = test_washer(x, filters, batch_normalisation_bool, initializer, activation)

    x = test_rnn(x, tof_bool, layers, rnn_type, units, rnn_activation, rnn_initializer, unroll)

    x = test_module_out(x, filters, initializer, batch_normalisation_bool, activation)

    return x


def test_module_up_skip(x, filters, kernal_size, strides, initializer, batch_normalisation_bool, activation):
    x_shape = x.get_shape()

    x = k.layers.Deconvolution3D(filters=filters,
                                 kernel_size=(kernal_size, kernal_size, kernal_size),
                                 strides=(strides, strides, strides),
                                 padding="same",
                                 kernel_initializer=initializer,
                                 bias_initializer=k.initializers.Constant(0.1))(x)
    x = test_activation(x, batch_normalisation_bool, activation)
    x = k.layers.AlphaDropout(0.5)(x)
    x = k.layers.Reshape((int(x_shape[1] * strides), int(x_shape[2] * strides), int(x_shape[3] * strides), filters))(x)

    return x


def test_module_up(x, filters, initializer, batch_normalisation_bool, activation):
    x_1 = k.layers.Deconvolution3D(filters=filters,
                                   kernel_size=(3, 3, 3),
                                   strides=(2, 2, 2),
                                   padding="same",
                                   kernel_initializer=initializer,
                                   bias_initializer=k.initializers.Constant(0.1))(x)
    x_1 = test_activation(x_1, batch_normalisation_bool, activation)
    x_1 = k.layers.AlphaDropout(0.5)(x_1)
    x_1 = k.layers.Reshape((int(x.get_shape()[1] * 2), int(x.get_shape()[2] * 2), int(x.get_shape()[3] * 2), filters))(
        x_1)

    x_2 = k.layers.Deconvolution3D(filters=filters,
                                   kernel_size=(5, 5, 5),
                                   strides=(2, 2, 2),
                                   padding="same",
                                   kernel_initializer=initializer,
                                   bias_initializer=k.initializers.Constant(0.1))(x)
    x_2 = test_activation(x_2, batch_normalisation_bool, activation)
    x_2 = k.layers.AlphaDropout(0.5)(x_2)
    x_2 = k.layers.Reshape((int(x.get_shape()[1] * 2), int(x.get_shape()[2] * 2), int(x.get_shape()[3] * 2), filters))(
        x_2)

    x_3 = k.layers.Deconvolution3D(filters=filters,
                                   kernel_size=(7, 7, 7),
                                   strides=(2, 2, 2),
                                   padding="same",
                                   kernel_initializer=initializer,
                                   bias_initializer=k.initializers.Constant(0.1))(x)
    x_3 = test_activation(x_3, batch_normalisation_bool, activation)
    x_3 = k.layers.AlphaDropout(0.5)(x_3)
    x_3 = k.layers.Reshape((int(x.get_shape()[1] * 2), int(x.get_shape()[2] * 2), int(x.get_shape()[3] * 2), filters))(
        x_3)

    x_4 = k.layers.Deconvolution3D(filters=filters,
                                   kernel_size=(7, 7, 7),
                                   strides=(2, 2, 2),
                                   padding="same",
                                   kernel_initializer=initializer,
                                   bias_initializer=k.initializers.Constant(0.1))(x)
    x_4 = test_activation(x_4, batch_normalisation_bool, activation)
    x_4 = k.layers.AlphaDropout(0.5)(x_4)
    x_4 = k.layers.Reshape((int(x.get_shape()[1] * 2), int(x.get_shape()[2] * 2), int(x.get_shape()[3] * 2), filters))(
        x_4)

    x = k.layers.Concatenate(axis=4)([x_1, x_2, x_3, x_4])

    return x


# https://github.com/zizhaozhang/unet-tensorflow-keras/blob/master/model.py
def get_crop_shape(target, refer):
    # depth, the 4th dimension
    cd = int(target.get_shape()[3]) - int(refer.get_shape()[3])

    if cd % 2 != 0:
        cd1, cd2 = int(cd / 2), int(cd / 2) + 1
    else:
        cd1, cd2 = int(cd / 2), int(cd / 2)

    # width, the 3rd dimension
    cw = int(target.get_shape()[2]) - int(refer.get_shape()[2])

    if cw % 2 != 0:
        cw1, cw2 = int(cw / 2), int(cw / 2) + 1
    else:
        cw1, cw2 = int(cw / 2), int(cw / 2)

    # height, the 2nd dimension
    ch = int(target.get_shape()[1]) - int(refer.get_shape()[1])

    if ch % 2 != 0:
        ch1, ch2 = int(ch / 2), int(ch / 2) + 1
    else:
        ch1, ch2 = int(ch / 2), int(ch / 2)

    return (ch1, ch2), (cw1, cw2), (cd1, cd2)


def test_crop(source, target):
    ch, cw, cd = get_crop_shape(source, target)
    x = k.layers.Cropping3D(cropping=(ch, cw, cd))(source)

    return x


def test_up(x, initializer, batch_normalisation_bool, activation, x_shortcut_7, x_shortcut_6, x_shortcut_5,
            x_shortcut_4, x_shortcut_3, x_shortcut_2, x_shortcut_1):
    x_skip_7_7 = test_module_up_skip(x, 21, 2, 2, initializer, batch_normalisation_bool, activation)
    x_skip_7_6 = test_module_up_skip(x, 13, 4, 4, initializer, batch_normalisation_bool, activation)
    x_skip_7_5 = test_module_up_skip(x, 8, 8, 8, initializer, batch_normalisation_bool, activation)
    x_skip_7_4 = test_module_up_skip(x, 5, 16, 16, initializer, batch_normalisation_bool, activation)
    x_skip_7_3 = test_module_up_skip(x, 3, 32, 32, initializer, batch_normalisation_bool, activation)
    x = test_module_up(x, 21, initializer, batch_normalisation_bool, activation)

    x = test_crop(x_skip_7_7, x)
    x = k.layers.Concatenate(axis=4)([x_skip_7_7, x])

    x = test_crop(x, x_shortcut_7)
    x = k.layers.Concatenate(axis=4)([x, x_shortcut_7])

    x_skip_6_6 = test_module_up_skip(x, 13, 2, 2, initializer, batch_normalisation_bool, activation)
    x_skip_6_5 = test_module_up_skip(x, 8, 4, 4, initializer, batch_normalisation_bool, activation)
    x_skip_6_4 = test_module_up_skip(x, 5, 8, 8, initializer, batch_normalisation_bool, activation)
    x_skip_6_3 = test_module_up_skip(x, 3, 16, 16, initializer, batch_normalisation_bool, activation)
    x_skip_6_2 = test_module_up_skip(x, 2, 32, 32, initializer, batch_normalisation_bool, activation)
    x = test_module_up(x, 13, initializer, batch_normalisation_bool, activation)

    x_skip_7_6 = test_crop(x_skip_7_6, x)
    x_skip_6_6 = test_crop(x_skip_6_6, x)
    x = k.layers.Concatenate(axis=4)([x_skip_7_6, x_skip_6_6, x])

    x = test_crop(x, x_shortcut_6)
    x = k.layers.Concatenate(axis=4)([x, x_shortcut_6])

    x_skip_5_5 = test_module_up_skip(x, 8, 2, 2, initializer, batch_normalisation_bool, activation)
    x_skip_5_4 = test_module_up_skip(x, 5, 4, 4, initializer, batch_normalisation_bool, activation)
    x_skip_5_3 = test_module_up_skip(x, 3, 8, 8, initializer, batch_normalisation_bool, activation)
    x_skip_5_2 = test_module_up_skip(x, 2, 16, 16, initializer, batch_normalisation_bool, activation)
    x_skip_5_1 = test_module_up_skip(x, 1, 32, 32, initializer, batch_normalisation_bool, activation)
    x = test_module_up(x, 8, initializer, batch_normalisation_bool, activation)

    x_skip_7_5 = test_crop(x_skip_7_5, x)
    x_skip_6_5 = test_crop(x_skip_6_5, x)
    x_skip_5_5 = test_crop(x_skip_5_5, x)
    x = k.layers.Concatenate(axis=4)([x_skip_7_5, x_skip_6_5, x_skip_5_5, x])

    x = test_crop(x, x_shortcut_5)
    x = k.layers.Concatenate(axis=4)([x, x_shortcut_5])

    x_skip_4_4 = test_module_up_skip(x, 5, 2, 2, initializer, batch_normalisation_bool, activation)
    x_skip_4_3 = test_module_up_skip(x, 3, 4, 4, initializer, batch_normalisation_bool, activation)
    x_skip_4_2 = test_module_up_skip(x, 2, 8, 8, initializer, batch_normalisation_bool, activation)
    x_skip_4_1 = test_module_up_skip(x, 1, 16, 16, initializer, batch_normalisation_bool, activation)
    x = test_module_up(x, 5, initializer, batch_normalisation_bool, activation)

    x_skip_7_4 = test_crop(x_skip_7_4, x)
    x_skip_6_4 = test_crop(x_skip_6_4, x)
    x_skip_5_4 = test_crop(x_skip_5_4, x)
    x_skip_4_4 = test_crop(x_skip_4_4, x)
    x = k.layers.Concatenate(axis=4)([x_skip_7_4, x_skip_6_4, x_skip_5_4, x_skip_4_4, x])

    x = test_crop(x, x_shortcut_4)
    x = k.layers.Concatenate(axis=4)([x, x_shortcut_4])

    x_skip_3_3 = test_module_up_skip(x, 3, 2, 2, initializer, batch_normalisation_bool, activation)
    x_skip_3_2 = test_module_up_skip(x, 2, 4, 4, initializer, batch_normalisation_bool, activation)
    x_skip_3_1 = test_module_up_skip(x, 1, 8, 8, initializer, batch_normalisation_bool, activation)
    x = test_module_up(x, 3, initializer, batch_normalisation_bool, activation)

    x_skip_7_3 = test_crop(x_skip_7_3, x)
    x_skip_6_3 = test_crop(x_skip_6_3, x)
    x_skip_5_3 = test_crop(x_skip_5_3, x)
    x_skip_4_3 = test_crop(x_skip_4_3, x)
    x_skip_3_3 = test_crop(x_skip_3_3, x)
    x = k.layers.Concatenate(axis=4)([x_skip_7_3, x_skip_6_3, x_skip_5_3, x_skip_4_3, x_skip_3_3, x])

    x = test_crop(x, x_shortcut_3)
    x = k.layers.Concatenate(axis=4)([x, x_shortcut_3])

    x_skip_2_2 = test_module_up_skip(x, 2, 2, 2, initializer, batch_normalisation_bool, activation)
    x_skip_2_1 = test_module_up_skip(x, 1, 4, 4, initializer, batch_normalisation_bool, activation)
    x = test_module_up(x, 2, initializer, batch_normalisation_bool, activation)

    x_skip_6_2 = test_crop(x_skip_6_2, x)
    x_skip_5_2 = test_crop(x_skip_5_2, x)
    x_skip_4_2 = test_crop(x_skip_4_2, x)
    x_skip_3_2 = test_crop(x_skip_3_2, x)
    x_skip_2_2 = test_crop(x_skip_2_2, x)
    x = k.layers.Concatenate(axis=4)([x_skip_6_2, x_skip_5_2, x_skip_4_2, x_skip_3_2, x_skip_2_2, x])

    x = test_crop(x, x_shortcut_2)
    x = k.layers.Concatenate(axis=4)([x, x_shortcut_2])

    x_skip_1_1 = test_module_up_skip(x, 1, 2, 2, initializer, batch_normalisation_bool, activation)
    x = test_module_up(x, 1, initializer, batch_normalisation_bool, activation)

    x_skip_5_1 = test_crop(x_skip_5_1, x)
    x_skip_4_1 = test_crop(x_skip_4_1, x)
    x_skip_3_1 = test_crop(x_skip_3_1, x)
    x_skip_2_1 = test_crop(x_skip_2_1, x)
    x_skip_1_1 = test_crop(x_skip_1_1, x)
    x = k.layers.Concatenate(axis=4)([x_skip_5_1, x_skip_4_1, x_skip_3_1, x_skip_2_1, x_skip_1_1, x])

    x = test_crop(x, x_shortcut_1)
    x = k.layers.Concatenate(axis=4)([x, x_shortcut_1])

    return x


def test_in_down_up_out(x, filters, initializer, batch_normalisation_bool, activation):
    x = test_module_in(x, filters, initializer)

    x, x_shortcut_1, x_shortcut_2, x_shortcut_3, x_shortcut_4, x_shortcut_5, x_shortcut_6, x_shortcut_7 = test_down(x,
                                                                                                                    initializer,
                                                                                                                    batch_normalisation_bool,
                                                                                                                    activation)

    x = test_washer(x, filters, initializer, batch_normalisation_bool, activation)
    x_shortcut_1 = test_washer(x_shortcut_1, filters, initializer, batch_normalisation_bool, activation)
    x_shortcut_2 = test_washer(x_shortcut_2, filters, initializer, batch_normalisation_bool, activation)
    x_shortcut_3 = test_washer(x_shortcut_3, filters, initializer, batch_normalisation_bool, activation)
    x_shortcut_4 = test_washer(x_shortcut_4, filters, initializer, batch_normalisation_bool, activation)
    x_shortcut_5 = test_washer(x_shortcut_5, filters, initializer, batch_normalisation_bool, activation)
    x_shortcut_6 = test_washer(x_shortcut_6, filters, initializer, batch_normalisation_bool, activation)
    x_shortcut_7 = test_washer(x_shortcut_7, filters, initializer, batch_normalisation_bool, activation)

    x = test_up(x, initializer, batch_normalisation_bool, activation, x_shortcut_7, x_shortcut_6, x_shortcut_5,
                x_shortcut_4, x_shortcut_3, x_shortcut_2, x_shortcut_1)

    x = test_module_out(x, filters, initializer, batch_normalisation_bool, activation)

    return x


def test_in_down_up_rnn_out(x, filters, initializer, batch_normalisation_bool, activation, tof_bool, layers, rnn_type,
                            units, rnn_activation, rnn_initializer, unroll):
    x = test_module_in(x, filters, initializer)

    x, x_shortcut_1, x_shortcut_2, x_shortcut_3, x_shortcut_4, x_shortcut_5, x_shortcut_6, x_shortcut_7 = test_down(x,
                                                                                                                    initializer,
                                                                                                                    batch_normalisation_bool,
                                                                                                                    activation)

    x = test_washer(x, filters, initializer, batch_normalisation_bool, activation)
    x_shortcut_1 = test_washer(x_shortcut_1, filters, initializer, batch_normalisation_bool, activation)
    x_shortcut_2 = test_washer(x_shortcut_2, filters, initializer, batch_normalisation_bool, activation)
    x_shortcut_3 = test_washer(x_shortcut_3, filters, initializer, batch_normalisation_bool, activation)
    x_shortcut_4 = test_washer(x_shortcut_4, filters, initializer, batch_normalisation_bool, activation)
    x_shortcut_5 = test_washer(x_shortcut_5, filters, initializer, batch_normalisation_bool, activation)
    x_shortcut_6 = test_washer(x_shortcut_6, filters, initializer, batch_normalisation_bool, activation)
    x_shortcut_7 = test_washer(x_shortcut_7, filters, initializer, batch_normalisation_bool, activation)

    x = test_up(x, initializer, batch_normalisation_bool, activation, x_shortcut_7, x_shortcut_6, x_shortcut_5,
                x_shortcut_4, x_shortcut_3, x_shortcut_2, x_shortcut_1)

    x = test_washer(x, filters, initializer, batch_normalisation_bool, activation)

    x = test_rnn(x, tof_bool, layers, rnn_type, units, rnn_activation, rnn_initializer, unroll)

    x = test_module_out(x, filters, initializer, batch_normalisation_bool, activation)

    return x


def test_in_rnn_down_up_rnn_out(x, filters, initializer, tof_bool, layers, rnn_type, units, rnn_activation,
                                rnn_initializer, batch_normalisation_bool,
                                activation, unroll):
    x = test_module_in(x, filters, initializer)

    x = test_rnn(x, tof_bool, layers, rnn_type, units, rnn_activation, rnn_initializer, unroll)

    x = test_washer(x, filters, initializer, batch_normalisation_bool, activation)

    x, x_shortcut_1, x_shortcut_2, x_shortcut_3, x_shortcut_4, x_shortcut_5, x_shortcut_6, x_shortcut_7 = test_down(x,
                                                                                                                    initializer,
                                                                                                                    batch_normalisation_bool,
                                                                                                                    activation)

    x = test_washer(x, filters, initializer, batch_normalisation_bool, activation)
    x_shortcut_1 = test_washer(x_shortcut_1, filters, initializer, batch_normalisation_bool, activation)
    x_shortcut_2 = test_washer(x_shortcut_2, filters, initializer, batch_normalisation_bool, activation)
    x_shortcut_3 = test_washer(x_shortcut_3, filters, initializer, batch_normalisation_bool, activation)
    x_shortcut_4 = test_washer(x_shortcut_4, filters, initializer, batch_normalisation_bool, activation)
    x_shortcut_5 = test_washer(x_shortcut_5, filters, initializer, batch_normalisation_bool, activation)
    x_shortcut_6 = test_washer(x_shortcut_6, filters, initializer, batch_normalisation_bool, activation)
    x_shortcut_7 = test_washer(x_shortcut_7, filters, initializer, batch_normalisation_bool, activation)

    x = test_up(x, initializer, batch_normalisation_bool, activation, x_shortcut_7, x_shortcut_6, x_shortcut_5,
                x_shortcut_4, x_shortcut_3, x_shortcut_2, x_shortcut_1)

    x = test_washer(x, filters, initializer, batch_normalisation_bool, activation)

    x = test_rnn(x, tof_bool, layers, rnn_type, units, rnn_activation, rnn_initializer, unroll)

    x = test_module_out(x, filters, initializer, batch_normalisation_bool, activation)

    return x


def test_in_down_up_down_out(x, filters, initializer, batch_normalisation_bool, activation):
    x = test_module_in(x, filters, initializer)

    x, x_shortcut_1, x_shortcut_2, x_shortcut_3, x_shortcut_4, x_shortcut_5, x_shortcut_6, x_shortcut_7 = test_down(x,
                                                                                                                    initializer,
                                                                                                                    batch_normalisation_bool,
                                                                                                                    activation)

    x = test_washer(x, filters, initializer, batch_normalisation_bool, activation)
    x_shortcut_1 = test_washer(x_shortcut_1, filters, initializer, batch_normalisation_bool, activation)
    x_shortcut_2 = test_washer(x_shortcut_2, filters, initializer, batch_normalisation_bool, activation)
    x_shortcut_3 = test_washer(x_shortcut_3, filters, initializer, batch_normalisation_bool, activation)
    x_shortcut_4 = test_washer(x_shortcut_4, filters, initializer, batch_normalisation_bool, activation)
    x_shortcut_5 = test_washer(x_shortcut_5, filters, initializer, batch_normalisation_bool, activation)
    x_shortcut_6 = test_washer(x_shortcut_6, filters, initializer, batch_normalisation_bool, activation)
    x_shortcut_7 = test_washer(x_shortcut_7, filters, initializer, batch_normalisation_bool, activation)

    x = test_up(x, initializer, batch_normalisation_bool, activation, x_shortcut_7, x_shortcut_6, x_shortcut_5,
                x_shortcut_4, x_shortcut_3, x_shortcut_2, x_shortcut_1)

    x = test_washer(x, filters, initializer, batch_normalisation_bool, activation)

    x, x_shortcut_1, x_shortcut_2, x_shortcut_3, x_shortcut_4, x_shortcut_5, x_shortcut_6, x_shortcut_7 = test_down(x,
                                                                                                                    initializer,
                                                                                                                    batch_normalisation_bool,
                                                                                                                    activation)

    x = test_module_out(x, filters, initializer, batch_normalisation_bool, activation)

    return x


def test_in_down_up_down_rnn_out(x, filters, initializer, batch_normalisation_bool, activation, tof_bool, layers,
                                 rnn_type, units, rnn_activation, rnn_initializer, unroll):
    x = test_module_in(x, filters, initializer)

    x, x_shortcut_1, x_shortcut_2, x_shortcut_3, x_shortcut_4, x_shortcut_5, x_shortcut_6, x_shortcut_7 = test_down(x,
                                                                                                                    initializer,
                                                                                                                    batch_normalisation_bool,
                                                                                                                    activation)

    x = test_washer(x, filters, initializer, batch_normalisation_bool, activation)
    x_shortcut_1 = test_washer(x_shortcut_1, filters, initializer, batch_normalisation_bool, activation)
    x_shortcut_2 = test_washer(x_shortcut_2, filters, initializer, batch_normalisation_bool, activation)
    x_shortcut_3 = test_washer(x_shortcut_3, filters, initializer, batch_normalisation_bool, activation)
    x_shortcut_4 = test_washer(x_shortcut_4, filters, initializer, batch_normalisation_bool, activation)
    x_shortcut_5 = test_washer(x_shortcut_5, filters, initializer, batch_normalisation_bool, activation)
    x_shortcut_6 = test_washer(x_shortcut_6, filters, initializer, batch_normalisation_bool, activation)
    x_shortcut_7 = test_washer(x_shortcut_7, filters, initializer, batch_normalisation_bool, activation)

    x = test_up(x, initializer, batch_normalisation_bool, activation, x_shortcut_7, x_shortcut_6, x_shortcut_5,
                x_shortcut_4, x_shortcut_3, x_shortcut_2, x_shortcut_1)

    x = test_washer(x, filters, initializer, batch_normalisation_bool, activation)

    x, x_shortcut_1, x_shortcut_2, x_shortcut_3, x_shortcut_4, x_shortcut_5, x_shortcut_6, x_shortcut_7 = test_down(x,
                                                                                                                    initializer,
                                                                                                                    batch_normalisation_bool,
                                                                                                                    activation)

    x = test_washer(x, filters, initializer, batch_normalisation_bool, activation)

    x = test_rnn(x, tof_bool, layers, rnn_type, units, rnn_activation, rnn_initializer, unroll)

    x = test_module_out(x, filters, initializer, batch_normalisation_bool, activation)

    return x


def test_in_rnn_down_up_down_rnn_out(x, filters, initializer, tof_bool, layers, rnn_type, units, rnn_activation,
                                     rnn_initializer,
                                     batch_normalisation_bool, activation, unroll):
    x = test_module_in(x, filters, initializer)

    x = test_rnn(x, tof_bool, layers, rnn_type, units, rnn_activation, rnn_initializer, unroll)

    x = test_washer(x, filters, initializer, batch_normalisation_bool, activation)

    x, x_shortcut_1, x_shortcut_2, x_shortcut_3, x_shortcut_4, x_shortcut_5, x_shortcut_6, x_shortcut_7 = test_down(x,
                                                                                                                    initializer,
                                                                                                                    batch_normalisation_bool,
                                                                                                                    activation)

    x = test_washer(x, filters, initializer, batch_normalisation_bool, activation)
    x_shortcut_1 = test_washer(x_shortcut_1, filters, initializer, batch_normalisation_bool, activation)
    x_shortcut_2 = test_washer(x_shortcut_2, filters, initializer, batch_normalisation_bool, activation)
    x_shortcut_3 = test_washer(x_shortcut_3, filters, initializer, batch_normalisation_bool, activation)
    x_shortcut_4 = test_washer(x_shortcut_4, filters, initializer, batch_normalisation_bool, activation)
    x_shortcut_5 = test_washer(x_shortcut_5, filters, initializer, batch_normalisation_bool, activation)
    x_shortcut_6 = test_washer(x_shortcut_6, filters, initializer, batch_normalisation_bool, activation)
    x_shortcut_7 = test_washer(x_shortcut_7, filters, initializer, batch_normalisation_bool, activation)

    x = test_up(x, initializer, batch_normalisation_bool, activation, x_shortcut_7, x_shortcut_6, x_shortcut_5,
                x_shortcut_4, x_shortcut_3, x_shortcut_2, x_shortcut_1)

    x = test_washer(x, filters, initializer, batch_normalisation_bool, activation)

    x, x_shortcut_1, x_shortcut_2, x_shortcut_3, x_shortcut_4, x_shortcut_5, x_shortcut_6, x_shortcut_7 = test_down(x,
                                                                                                                    initializer,
                                                                                                                    batch_normalisation_bool,
                                                                                                                    activation)

    x = test_washer(x, filters, initializer, batch_normalisation_bool, activation)

    x = test_rnn(x, tof_bool, layers, rnn_type, units, rnn_activation, rnn_initializer, unroll)

    x = test_module_out(x, filters, initializer, batch_normalisation_bool, activation)

    return x
