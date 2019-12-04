import keras as k


def test_activation(x, batch_normalisation_bool, activation):
    if batch_normalisation_bool:
        x = k.layers.BatchNormalization()(x)

    if activation == "prelu":
        x = k.layers.PReLU()(x)
    else:
        x = k.layers.Activation(activation)(x)

    return x


def test_module_in(x, activation, filters, initializer):
    if activation == "selu":
        x = k.layers.Convolution3D(filters=filters, kernel_size=(3, 3, 3), strides=(1, 1, 1), padding='same',
                                   kernel_initializer=initializer, bias_initializer=k.initializers.Constant(0.1))(x)
        x = test_activation(x, True, "hard_sigmoid")
        x = k.layers.AlphaDropout(0.5)(x)

    return x


def test_module_down(x, filters, initializer, batch_normalisation_bool, activation):
    x = k.layers.Convolution3D(filters=filters,
                               kernel_size=(3, 3, 3),
                               strides=(1, 1, 1),
                               padding="same",
                               kernel_initializer=initializer,
                               bias_initializer=k.initializers.Constant(0.1))(x)
    x = test_activation(x, batch_normalisation_bool, activation)
    x = k.layers.AlphaDropout(0.5)(x)

    x = k.layers.Convolution3D(filters=filters,
                               kernel_size=(3, 3, 3),
                               strides=(3, 3, 3),
                               padding="same",
                               kernel_initializer=initializer,
                               bias_initializer=k.initializers.Constant(0.1))(x)
    x = test_activation(x, batch_normalisation_bool, activation)
    x = k.layers.AlphaDropout(0.5)(x)

    return x


def test_down(x, layers, initializer, batch_normalisation_bool, activation):
    for _ in range(layers):
        x = test_module_down(x, 40, initializer, batch_normalisation_bool, activation)

    return x


def test_module_out(x):
    x = k.layers.Flatten()(x)

    return x


def test_in_down_out(x, activation, filters, initializer, layers, batch_normalisation_bool):
    x = test_module_in(x, activation, filters, initializer)

    x = test_down(x, layers, initializer, batch_normalisation_bool, activation)

    x = test_module_out(x)

    return x


def test_module_rnn(x, units, activation, return_sequences, initializer, unroll):
    x = k.layers.SimpleRNN(units=units,
                           activation=activation,
                           recurrent_activation=activation,
                           return_sequences=return_sequences,
                           dropout=0.25,
                           recurrent_dropout=0.25,
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
                      dropout=0.25,
                      recurrent_dropout=0.25,
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
                     dropout=0.25,
                     recurrent_dropout=0.25,
                     kernel_initializer=initializer,
                     bias_initializer=k.initializers.Constant(0.0),
                     recurrent_initializer=initializer,
                     unroll=unroll)(x)

    return x


def test_rnn(x, tof_bool, layers, rnn_type, units, activation, initializer, unroll):
    if layers > 0:
        if tof_bool:
            x_shape = [x.shape.as_list()[1], x.shape.as_list()[2], x.shape.as_list()[3], x.shape.as_list()[4],
                       x.shape.as_list()[5]]

            x = k.layers.Reshape(
                (x.shape.as_list()[1] * x.shape.as_list()[2] * x.shape.as_list()[3] * x.shape.as_list()[4],
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
    x = test_rnn(x, tof_bool, layers, rnn_type, units, activation, initializer, unroll)

    x = test_module_rnn_out(x, tof_bool, rnn_type, units, activation, initializer, unroll)

    return x


def test_in_down_rnn_out(x, activation, filters, initializer, down_layers, batch_normalisation_bool, tof_bool, out_layers, rnn_type,
                         units, rnn_activation, rnn_initializer, unroll):
    x = test_module_in(x, activation, filters, initializer)

    x = test_down(x, down_layers, initializer, batch_normalisation_bool, activation)

    x = test_rnn(x, tof_bool, out_layers, rnn_type, units, rnn_activation, rnn_initializer, unroll)

    x = test_module_rnn_out(x, tof_bool, rnn_type, units, activation, initializer, unroll)

    return x
