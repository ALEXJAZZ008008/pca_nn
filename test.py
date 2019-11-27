import keras as k


def test_in(x, filters):
    x = k.layers.Convolution3D(filters=filters, kernel_size=(1, 1, 1), strides=(1, 1, 1), padding='same', kernel_initializer="lecun_normal", bias_initializer="lecun_normal")(x)
    x = k.layers.BatchNormalization()(x)
    x = k.layers.Activation("tanh")(x)
    x = k.layers.AlphaDropout(0.5)(x)

    return x


def test_activation(x, batch_normalisation_bool, activation):
    if batch_normalisation_bool:
        x = k.layers.BatchNormalization()(x)

    if activation == "prelu":
        x = k.layers.PReLU()(x)
    else:
        x = k.layers.Activation(activation)(x)

    return x


def test_fully_connected(x, layers, filters, batch_normalisation_bool, activation):
    x_shape = [x.shape.as_list()[1], x.shape.as_list()[2], x.shape.as_list()[3], x.shape.as_list()[4]]

    x = k.layers.Reshape((x.shape.as_list()[1] * x.shape.as_list()[2] * x.shape.as_list()[3], x.shape.as_list()[4]))(x)

    for _ in range(layers):
        x = k.layers.Convolution1D(filters=filters, kernel_size=x.shape.as_list()[1], strides=1, padding='same', kernel_initializer="lecun_normal", bias_initializer="lecun_normal")(x)
        x = test_activation(x, batch_normalisation_bool, activation)
        x = k.layers.AlphaDropout(0.5)(x)

    x = k.layers.Reshape((x_shape[0], x_shape[1], x_shape[2], x_shape[3]))(x)

    return x

def test_out(x, filters, batch_normalisation_bool, activation):
    x = k.layers.Convolution3D(filters=filters, kernel_size=(1, 1, 1), strides=(1, 1, 1), padding='same', kernel_initializer="lecun_normal", bias_initializer="lecun_normal")(x)
    x = test_activation(x, batch_normalisation_bool, activation)
    x = k.layers.AlphaDropout(0.5)(x)

    x = k.layers.Flatten()(x)

    return x


def test_fully_connected_out(x, layers, filters, batch_normalisation_bool, activation):
    x = test_in(x, filters)

    x = test_fully_connected(x, layers, filters, batch_normalisation_bool, activation)

    x = test_out(x, filters, batch_normalisation_bool, activation)

    return x


def test_module_down_skip(x, filters, batch_normalisation_bool, activation):
    x = k.layers.Convolution3D(filters=filters, kernel_size=(4, 4, 4), strides=(4, 4, 4), padding="same", kernel_initializer="lecun_normal", bias_initializer="lecun_normal")(x)
    x = test_activation(x, batch_normalisation_bool, activation)
    x = k.layers.AlphaDropout(0.5)(x)

    return x


def test_module_down(x, filters, batch_normalisation_bool, activation):
    x_1 = k.layers.Convolution3D(filters=filters, kernel_size=(3, 3, 3), strides=(2, 2, 2), padding="same", kernel_initializer="lecun_normal", bias_initializer="lecun_normal")(x)
    x_1 = test_activation(x_1, batch_normalisation_bool, activation)
    x_1 = k.layers.AlphaDropout(0.5)(x_1)

    x_2 = k.layers.Convolution3D(filters=filters, kernel_size=(5, 5, 5), strides=(2, 2, 2), padding="same", kernel_initializer="lecun_normal", bias_initializer="lecun_normal")(x)
    x_2 = test_activation(x_2, batch_normalisation_bool, activation)
    x_2 = k.layers.AlphaDropout(0.5)(x_2)

    x_3 = k.layers.Convolution3D(filters=filters, kernel_size=(7, 7, 7), strides=(2, 2, 2), padding="same", kernel_initializer="lecun_normal", bias_initializer="lecun_normal")(x)
    x_3 = test_activation(x_3, batch_normalisation_bool, activation)
    x_3 = k.layers.AlphaDropout(0.5)(x_3)

    x_4 = k.layers.Convolution3D(filters=filters, kernel_size=(9, 9, 9), strides=(2, 2, 2), padding="same", kernel_initializer="lecun_normal", bias_initializer="lecun_normal")(x)
    x_4 = test_activation(x_4, batch_normalisation_bool, activation)
    x_4 = k.layers.AlphaDropout(0.5)(x_4)

    x = k.layers.Concatenate(axis=4)([x_1, x_2, x_3, x_4])

    return x


def test_down(x, batch_normalisation_bool, activation):
    x_shortcut_1 = x
    x_skip_1 = test_module_down_skip(x, 1, batch_normalisation_bool, activation)
    x = test_module_down(x, 1, batch_normalisation_bool, activation)

    x_shortcut_2 = x
    x_skip_2 = test_module_down_skip(x, 1, batch_normalisation_bool, activation)
    x = test_module_down(x, 2, batch_normalisation_bool, activation)
    x = k.layers.Add()([x_skip_1, x])

    x_shortcut_3 = x
    x_skip_1 = test_module_down_skip(x, 1, batch_normalisation_bool, activation)
    x = test_module_down(x, 4, batch_normalisation_bool, activation)
    x = k.layers.Add()([x_skip_2, x])

    x_shortcut_4 = x
    x_skip_2 = test_module_down_skip(x, 1, batch_normalisation_bool, activation)
    x = test_module_down(x, 8, batch_normalisation_bool, activation)
    x = k.layers.Add()([x_skip_1, x])

    x_shortcut_5 = x
    x_skip_1 = test_module_down_skip(x, 1, batch_normalisation_bool, activation)
    x = test_module_down(x, 16, batch_normalisation_bool, activation)
    x = k.layers.Add()([x_skip_2, x])

    x_shortcut_6 = x
    x_skip_2 = test_module_down_skip(x, 1, batch_normalisation_bool, activation)
    x = test_module_down(x, 32, batch_normalisation_bool, activation)
    x = k.layers.Add()([x_skip_1, x])

    x_shortcut_7 = x
    x = test_module_down(x, 64, batch_normalisation_bool, activation)
    x = k.layers.Add()([x_skip_2, x])

    return x, x_shortcut_1, x_shortcut_2, x_shortcut_3, x_shortcut_4, x_shortcut_5, x_shortcut_6, x_shortcut_7


def test_down_out(x, filters, batch_normalisation_bool, activation):
    x = test_in(x, filters)

    x, x_shortcut_1, x_shortcut_2, x_shortcut_3, x_shortcut_4, x_shortcut_5, x_shortcut_6, x_shortcut_7 = test_down(x, batch_normalisation_bool, activation)

    x = test_out(x, filters, batch_normalisation_bool, activation)

    return x


def test_rnn_module(x, units, batch_normalisation_bool, activation):
    x = k.layers.LSTM(units=units, return_sequences=True, kernel_initializer="lecun_normal", bias_initializer="lecun_normal", recurrent_initializer="lecun_normal")(x)
    x = test_activation(x, batch_normalisation_bool, activation)
    x = k.layers.AlphaDropout(0.5)(x)

    return x


def test_rnn(x, layers, units, batch_normalisation_bool, activation):
    x_shape = [x.shape.as_list()[1], x.shape.as_list()[2], x.shape.as_list()[3], x.shape.as_list()[4]]

    x = k.layers.Reshape((x.shape.as_list()[1] * x.shape.as_list()[2] * x.shape.as_list()[3], x.shape.as_list()[4]))(x)

    for _ in range(layers):
        x = test_rnn_module(x, units, batch_normalisation_bool, activation)

    x = k.layers.Reshape((x_shape[0], x_shape[1], x_shape[2], x_shape[3]))(x)

    return x


def test_rnn_out(x, filters, layers, units, batch_normalisation_bool, activation):
    x = test_in(x, filters)

    x = test_rnn(x, layers, units, batch_normalisation_bool, activation)

    x = test_out(x, filters, batch_normalisation_bool, activation)

    return x


def test_washer(x, filters, batch_normalisation_bool, activation):
    x = k.layers.Convolution3D(filters=filters, kernel_size=(1, 1, 1), strides=(1, 1, 1), padding='same', kernel_initializer="lecun_normal", bias_initializer="lecun_normal")(x)
    x = test_activation(x, batch_normalisation_bool, activation)
    x = k.layers.AlphaDropout(0.5)(x)

    return x


def test_down_rnn_out(x, filters, batch_normalisation_bool, activation, layers, units):
    x = test_in(x, filters)

    x, x_shortcut_1, x_shortcut_2, x_shortcut_3, x_shortcut_4, x_shortcut_5, x_shortcut_6, x_shortcut_7 = test_down(x, batch_normalisation_bool, activation)

    x = test_washer(x, filters, batch_normalisation_bool, activation)

    x = test_rnn(x, layers, units, batch_normalisation_bool, activation)

    x = test_out(x, filters, batch_normalisation_bool, activation)

    return x


def test_fully_connected_down_rnn_out(x, filters, layers, batch_normalisation_bool, activation, units):
    x = test_in(x, filters)

    x = test_fully_connected(x, layers, filters, batch_normalisation_bool, activation)

    x = test_washer(x, filters, batch_normalisation_bool, activation)

    x, x_shortcut_1, x_shortcut_2, x_shortcut_3, x_shortcut_4, x_shortcut_5, x_shortcut_6, x_shortcut_7 = test_down(x, batch_normalisation_bool, activation)

    x = test_washer(x, filters, batch_normalisation_bool, activation)

    x = test_rnn(x, layers, units, batch_normalisation_bool, activation)

    x = test_out(x, filters, batch_normalisation_bool, activation)

    return x


def test_rnn_down_rnn_out(x, filters, layers, units, batch_normalisation_bool, activation):
    x = test_in(x, filters)

    x = test_rnn(x, layers, units, batch_normalisation_bool, activation)

    x = test_washer(x, filters, batch_normalisation_bool, activation)

    x, x_shortcut_1, x_shortcut_2, x_shortcut_3, x_shortcut_4, x_shortcut_5, x_shortcut_6, x_shortcut_7 = test_down(x, batch_normalisation_bool, activation)

    x = test_washer(x, filters, batch_normalisation_bool, activation)

    x = test_rnn(x, layers, units, batch_normalisation_bool, activation)

    x = test_out(x, filters, batch_normalisation_bool, activation)

    return x


def test_module_up_skip(x, filters, batch_normalisation_bool, activation):
    x_shape = x.get_shape()

    x = k.layers.Deconvolution3D(filters=filters, kernel_size=(2, 2, 2), strides=(2, 2, 2), padding="same", kernel_initializer="lecun_normal", bias_initializer="lecun_normal")(x)
    x = test_activation(x, batch_normalisation_bool, activation)
    x = k.layers.AlphaDropout(0.5)(x)
    x = k.layers.Reshape((int(x_shape[1] * 2), int(x_shape[2] * 2), int(x_shape[3] * 2), filters))(x)

    return x


def test_module_up(x, filters, batch_normalisation_bool, activation):
    x_1 = k.layers.Deconvolution3D(filters=filters, kernel_size=(3, 3, 3), strides=(2, 2, 2), padding="same", kernel_initializer="lecun_normal", bias_initializer="lecun_normal")(x)
    x_1 = test_activation(x_1, batch_normalisation_bool, activation)
    x_1 = k.layers.AlphaDropout(0.5)(x_1)
    x_1 = k.layers.Reshape((int(x.get_shape()[1] * 2), int(x.get_shape()[2] * 2), int(x.get_shape()[3] * 2), filters))(x_1)

    x_2 = k.layers.Deconvolution3D(filters=filters, kernel_size=(5, 5, 5), strides=(2, 2, 2), padding="same", kernel_initializer="lecun_normal", bias_initializer="lecun_normal")(x)
    x_2 = test_activation(x_2, batch_normalisation_bool, activation)
    x_2 = k.layers.AlphaDropout(0.5)(x_2)
    x_2 = k.layers.Reshape((int(x.get_shape()[1] * 2), int(x.get_shape()[2] * 2), int(x.get_shape()[3] * 2), filters))(x_2)

    x_3 = k.layers.Deconvolution3D(filters=filters, kernel_size=(7, 7, 7), strides=(2, 2, 2), padding="same", kernel_initializer="lecun_normal", bias_initializer="lecun_normal")(x)
    x_3 = test_activation(x_3, batch_normalisation_bool, activation)
    x_3 = k.layers.AlphaDropout(0.5)(x_3)
    x_3 = k.layers.Reshape((int(x.get_shape()[1] * 2), int(x.get_shape()[2] * 2), int(x.get_shape()[3] * 2), filters))(x_3)

    x_4 = k.layers.Deconvolution3D(filters=filters, kernel_size=(7, 7, 7), strides=(2, 2, 2), padding="same", kernel_initializer="lecun_normal", bias_initializer="lecun_normal")(x)
    x_4 = test_activation(x_4, batch_normalisation_bool, activation)
    x_4 = k.layers.AlphaDropout(0.5)(x_4)
    x_4 = k.layers.Reshape((int(x.get_shape()[1] * 2), int(x.get_shape()[2] * 2), int(x.get_shape()[3] * 2), filters))(x_4)

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


def test_up(x, batch_normalisation_bool, activation, x_shortcut_7, x_shortcut_6, x_shortcut_5, x_shortcut_4, x_shortcut_3, x_shortcut_2, x_shortcut_1):
    x_shortcut = test_module_up_skip(x, 1, batch_normalisation_bool, activation)
    x = test_module_up(x, 64, batch_normalisation_bool, activation)
    x = k.layers.Add()([x_shortcut, x])

    ch, cw, cd = get_crop_shape(x, x_shortcut_7)
    x = k.layers.Cropping3D(cropping=(ch, cw, cd))(x)
    x = k.layers.Concatenate(axis=4)([x, x_shortcut_7])

    x_shortcut = test_module_up_skip(x, 1, batch_normalisation_bool, activation)
    x = test_module_up(x, 32, batch_normalisation_bool, activation)
    x = k.layers.Add()([x_shortcut, x])

    ch, cw, cd = get_crop_shape(x, x_shortcut_6)
    x = k.layers.Cropping3D(cropping=(ch, cw, cd))(x)
    x = k.layers.Concatenate(axis=4)([x, x_shortcut_6])

    x_shortcut = test_module_up_skip(x, 1, batch_normalisation_bool, activation)
    x = test_module_up(x, 16, batch_normalisation_bool, activation)
    x = k.layers.Add()([x_shortcut, x])

    ch, cw, cd = get_crop_shape(x, x_shortcut_5)
    x = k.layers.Cropping3D(cropping=(ch, cw, cd))(x)
    x = k.layers.Concatenate(axis=4)([x, x_shortcut_5])

    x_shortcut = test_module_up_skip(x, 1, batch_normalisation_bool, activation)
    x = test_module_up(x, 8, batch_normalisation_bool, activation)
    x = k.layers.Add()([x_shortcut, x])

    ch, cw, cd = get_crop_shape(x, x_shortcut_4)
    x = k.layers.Cropping3D(cropping=(ch, cw, cd))(x)
    x = k.layers.Concatenate(axis=4)([x, x_shortcut_4])

    x_shortcut = test_module_up_skip(x, 1, batch_normalisation_bool, activation)
    x = test_module_up(x, 4, batch_normalisation_bool, activation)
    x = k.layers.Add()([x_shortcut, x])

    ch, cw, cd = get_crop_shape(x, x_shortcut_3)
    x = k.layers.Cropping3D(cropping=(ch, cw, cd))(x)
    x = k.layers.Concatenate(axis=4)([x, x_shortcut_3])

    x_shortcut = test_module_up_skip(x, 1, batch_normalisation_bool, activation)
    x = test_module_up(x, 2, batch_normalisation_bool, activation)
    x = k.layers.Add()([x_shortcut, x])

    ch, cw, cd = get_crop_shape(x, x_shortcut_2)
    x = k.layers.Cropping3D(cropping=(ch, cw, cd))(x)
    x = k.layers.Concatenate(axis=4)([x, x_shortcut_2])

    x_shortcut = test_module_up_skip(x, 1, batch_normalisation_bool, activation)
    x = test_module_up(x, 1, batch_normalisation_bool, activation)
    x = k.layers.Add()([x_shortcut, x])

    ch, cw, cd = get_crop_shape(x, x_shortcut_1)
    x = k.layers.Cropping3D(cropping=(ch, cw, cd))(x)
    x = k.layers.Concatenate(axis=4)([x, x_shortcut_1])

    return x


def test_down_up_out(x, filters, batch_normalisation_bool, activation):
    x = test_in(x, filters)

    x, x_shortcut_1, x_shortcut_2, x_shortcut_3, x_shortcut_4, x_shortcut_5, x_shortcut_6, x_shortcut_7 = test_down(x, batch_normalisation_bool, activation)

    x = test_washer(x, filters, batch_normalisation_bool, activation)
    x_shortcut_1 = test_washer(x_shortcut_1, filters, batch_normalisation_bool, activation)
    x_shortcut_2 = test_washer(x_shortcut_2, filters, batch_normalisation_bool, activation)
    x_shortcut_3 = test_washer(x_shortcut_3, filters, batch_normalisation_bool, activation)
    x_shortcut_4 = test_washer(x_shortcut_4, filters, batch_normalisation_bool, activation)
    x_shortcut_5 = test_washer(x_shortcut_5, filters, batch_normalisation_bool, activation)
    x_shortcut_6 = test_washer(x_shortcut_6, filters, batch_normalisation_bool, activation)
    x_shortcut_7 = test_washer(x_shortcut_7, filters, batch_normalisation_bool, activation)

    x = test_up(x, batch_normalisation_bool, activation, x_shortcut_7, x_shortcut_6, x_shortcut_5, x_shortcut_4, x_shortcut_3, x_shortcut_2, x_shortcut_1)

    x = test_out(x, filters, batch_normalisation_bool, activation)

    return x

def test_fully_connected_down_up_rnn_out(x, filters, layers, batch_normalisation_bool, activation, units):
    x = test_in(x, filters)

    x = test_fully_connected(x, layers, filters, batch_normalisation_bool, activation)

    x = test_washer(x, filters, batch_normalisation_bool, activation)

    x, x_shortcut_1, x_shortcut_2, x_shortcut_3, x_shortcut_4, x_shortcut_5, x_shortcut_6, x_shortcut_7 = test_down(x, batch_normalisation_bool, activation)

    x = test_washer(x, filters, batch_normalisation_bool, activation)
    x_shortcut_1 = test_washer(x_shortcut_1, filters, batch_normalisation_bool, activation)
    x_shortcut_2 = test_washer(x_shortcut_2, filters, batch_normalisation_bool, activation)
    x_shortcut_3 = test_washer(x_shortcut_3, filters, batch_normalisation_bool, activation)
    x_shortcut_4 = test_washer(x_shortcut_4, filters, batch_normalisation_bool, activation)
    x_shortcut_5 = test_washer(x_shortcut_5, filters, batch_normalisation_bool, activation)
    x_shortcut_6 = test_washer(x_shortcut_6, filters, batch_normalisation_bool, activation)
    x_shortcut_7 = test_washer(x_shortcut_7, filters, batch_normalisation_bool, activation)

    x = test_up(x, batch_normalisation_bool, activation, x_shortcut_7, x_shortcut_6, x_shortcut_5, x_shortcut_4, x_shortcut_3, x_shortcut_2, x_shortcut_1)

    x = test_washer(x, filters, batch_normalisation_bool, activation)

    x = test_rnn(x, layers, units, batch_normalisation_bool, activation)

    x = test_out(x, filters, batch_normalisation_bool, activation)

    return x


def test_rnn_down_up_rnn_out(x, filters, layers, units, batch_normalisation_bool, activation):
    x = test_in(x, filters)

    x = test_rnn(x, layers, units, batch_normalisation_bool, activation)

    x = test_washer(x, filters, batch_normalisation_bool, activation)

    x, x_shortcut_1, x_shortcut_2, x_shortcut_3, x_shortcut_4, x_shortcut_5, x_shortcut_6, x_shortcut_7 = test_down(x, batch_normalisation_bool, activation)

    x = test_washer(x, filters, batch_normalisation_bool, activation)
    x_shortcut_1 = test_washer(x_shortcut_1, filters, batch_normalisation_bool, activation)
    x_shortcut_2 = test_washer(x_shortcut_2, filters, batch_normalisation_bool, activation)
    x_shortcut_3 = test_washer(x_shortcut_3, filters, batch_normalisation_bool, activation)
    x_shortcut_4 = test_washer(x_shortcut_4, filters, batch_normalisation_bool, activation)
    x_shortcut_5 = test_washer(x_shortcut_5, filters, batch_normalisation_bool, activation)
    x_shortcut_6 = test_washer(x_shortcut_6, filters, batch_normalisation_bool, activation)
    x_shortcut_7 = test_washer(x_shortcut_7, filters, batch_normalisation_bool, activation)

    x = test_up(x, batch_normalisation_bool, activation, x_shortcut_7, x_shortcut_6, x_shortcut_5, x_shortcut_4, x_shortcut_3, x_shortcut_2, x_shortcut_1)

    x = test_washer(x, filters, batch_normalisation_bool, activation)

    x = test_rnn(x, layers, units, batch_normalisation_bool, activation)

    x = test_out(x, filters, batch_normalisation_bool, activation)

    return x


def test_down_up_down_out(x, filters, batch_normalisation_bool, activation):
    x = test_in(x, filters)

    x, x_shortcut_1, x_shortcut_2, x_shortcut_3, x_shortcut_4, x_shortcut_5, x_shortcut_6, x_shortcut_7 = test_down(x, batch_normalisation_bool, activation)

    x = test_washer(x, filters, batch_normalisation_bool, activation)
    x_shortcut_1 = test_washer(x_shortcut_1, filters, batch_normalisation_bool, activation)
    x_shortcut_2 = test_washer(x_shortcut_2, filters, batch_normalisation_bool, activation)
    x_shortcut_3 = test_washer(x_shortcut_3, filters, batch_normalisation_bool, activation)
    x_shortcut_4 = test_washer(x_shortcut_4, filters, batch_normalisation_bool, activation)
    x_shortcut_5 = test_washer(x_shortcut_5, filters, batch_normalisation_bool, activation)
    x_shortcut_6 = test_washer(x_shortcut_6, filters, batch_normalisation_bool, activation)
    x_shortcut_7 = test_washer(x_shortcut_7, filters, batch_normalisation_bool, activation)

    x = test_up(x, batch_normalisation_bool, activation, x_shortcut_7, x_shortcut_6, x_shortcut_5, x_shortcut_4, x_shortcut_3, x_shortcut_2, x_shortcut_1)

    x = test_washer(x, filters, batch_normalisation_bool, activation)

    x, x_shortcut_1, x_shortcut_2, x_shortcut_3, x_shortcut_4, x_shortcut_5, x_shortcut_6, x_shortcut_7 = test_down(x, batch_normalisation_bool, activation)

    x = test_out(x, filters, batch_normalisation_bool, activation)

    return x

def test_fully_connected_down_up_down_rnn_out(x, filters, layers, batch_normalisation_bool, activation, units):
    x = test_in(x, filters)

    x = test_fully_connected(x, layers, filters, batch_normalisation_bool, activation)

    x = test_washer(x, filters, batch_normalisation_bool, activation)

    x, x_shortcut_1, x_shortcut_2, x_shortcut_3, x_shortcut_4, x_shortcut_5, x_shortcut_6, x_shortcut_7 = test_down(x, batch_normalisation_bool, activation)

    x = test_washer(x, filters, batch_normalisation_bool, activation)
    x_shortcut_1 = test_washer(x_shortcut_1, filters, batch_normalisation_bool, activation)
    x_shortcut_2 = test_washer(x_shortcut_2, filters, batch_normalisation_bool, activation)
    x_shortcut_3 = test_washer(x_shortcut_3, filters, batch_normalisation_bool, activation)
    x_shortcut_4 = test_washer(x_shortcut_4, filters, batch_normalisation_bool, activation)
    x_shortcut_5 = test_washer(x_shortcut_5, filters, batch_normalisation_bool, activation)
    x_shortcut_6 = test_washer(x_shortcut_6, filters, batch_normalisation_bool, activation)
    x_shortcut_7 = test_washer(x_shortcut_7, filters, batch_normalisation_bool, activation)

    x = test_up(x, batch_normalisation_bool, activation, x_shortcut_7, x_shortcut_6, x_shortcut_5, x_shortcut_4, x_shortcut_3, x_shortcut_2, x_shortcut_1)

    x = test_washer(x, filters, batch_normalisation_bool, activation)

    x, x_shortcut_1, x_shortcut_2, x_shortcut_3, x_shortcut_4, x_shortcut_5, x_shortcut_6, x_shortcut_7 = test_down(x, batch_normalisation_bool, activation)

    x = test_washer(x, filters, batch_normalisation_bool, activation)

    x = test_rnn(x, layers, units, batch_normalisation_bool, activation)

    x = test_out(x, filters, batch_normalisation_bool, activation)

    return x


def test_rnn_down_up_down_rnn_out(x, filters, layers, units, batch_normalisation_bool, activation):
    x = test_in(x, filters)

    x = test_rnn(x, layers, units, batch_normalisation_bool, activation)

    x = test_washer(x, filters, batch_normalisation_bool, activation)

    x, x_shortcut_1, x_shortcut_2, x_shortcut_3, x_shortcut_4, x_shortcut_5, x_shortcut_6, x_shortcut_7 = test_down(x, batch_normalisation_bool, activation)

    x = test_washer(x, filters, batch_normalisation_bool, activation)
    x_shortcut_1 = test_washer(x_shortcut_1, filters, batch_normalisation_bool, activation)
    x_shortcut_2 = test_washer(x_shortcut_2, filters, batch_normalisation_bool, activation)
    x_shortcut_3 = test_washer(x_shortcut_3, filters, batch_normalisation_bool, activation)
    x_shortcut_4 = test_washer(x_shortcut_4, filters, batch_normalisation_bool, activation)
    x_shortcut_5 = test_washer(x_shortcut_5, filters, batch_normalisation_bool, activation)
    x_shortcut_6 = test_washer(x_shortcut_6, filters, batch_normalisation_bool, activation)
    x_shortcut_7 = test_washer(x_shortcut_7, filters, batch_normalisation_bool, activation)

    x = test_up(x, batch_normalisation_bool, activation, x_shortcut_7, x_shortcut_6, x_shortcut_5, x_shortcut_4, x_shortcut_3, x_shortcut_2, x_shortcut_1)

    x = test_washer(x, filters, batch_normalisation_bool, activation)

    x, x_shortcut_1, x_shortcut_2, x_shortcut_3, x_shortcut_4, x_shortcut_5, x_shortcut_6, x_shortcut_7 = test_down(x, batch_normalisation_bool, activation)

    x = test_washer(x, filters, batch_normalisation_bool, activation)

    x = test_rnn(x, layers, units, batch_normalisation_bool, activation)

    x = test_out(x, filters, batch_normalisation_bool, activation)

    return x
