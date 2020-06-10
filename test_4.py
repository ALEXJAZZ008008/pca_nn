# Copyright University College London 2019, 2020
# Author: Alexander Whitehead, Institute of Nuclear Medicine, UCL
# For internal research only.


import math
import tensorflow.keras as k


def encoder(x, cnn_start_units_log, cnn_end_units_log, cnn_increase_layer_density_bool, cnn_layer_layers, lone, ltwo,
            cnn_pool_bool, cnn_max_pool_bool):
    x_skip = []
    filters = 0

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

    return x, x_skip, filters


def encoder_autoencoder_latenet_space(x, filters):
    x = k.layers.TimeDistributed(k.layers.Convolution3D(filters=filters,
                                                        kernel_size=(3, 3, 3),
                                                        strides=(1, 1, 1),
                                                        padding="same",
                                                        kernel_initializer="glorot_normal",
                                                        bias_initializer=k.initializers.Constant(0.1)))(x)
    x = k.layers.TimeDistributed(k.layers.Activation("linear"))(x)

    x_latent = k.layers.TimeDistributed(k.layers.Flatten())(x)

    return x, x_latent


# https://www.machinecurve.com/index.php/2019/12/30/how-to-create-a-variational-autoencoder-with-keras/
# Define sampling with reparameterization trick
def sample_z(args):
    mu, sigma = args
    batch = k.backend.shape(mu)[0]
    dim = k.backend.int_shape(mu)[1]
    eps = k.backend.random_normal(shape=(batch, dim))
    return mu + k.backend.exp(sigma / 2) * eps


def encoder_variational_autoencoder_latenet_space(x, dense_units, lone, ltwo, variational_latent_dimentions):
    conv_shape = k.backend.int_shape(x)

    x = k.layers.Flatten()(x)

    x = k.layers.Dense(units=dense_units,
                       kernel_initializer="lecun_normal",
                       bias_initializer=k.initializers.Constant(0.1),
                       kernel_regularizer=k.regularizers.l1_l2(l1=lone, l2=ltwo),
                       kernel_constraint=k.constraints.UnitNorm())(x)
    x = k.layers.BatchNormalization()(x)
    x = k.layers.Activation("selu")(x)

    mu = k.layers.Dense(units=variational_latent_dimentions,
                        kernel_initializer="lecun_normal",
                        bias_initializer=k.initializers.Constant(0.1),
                        name='latent_mu')(x)
    sigma = k.layers.Dense(units=variational_latent_dimentions,
                           kernel_initializer="lecun_normal",
                           bias_initializer=k.initializers.Constant(0.1),
                           name='latent_sigma')(x)

    # Use reparameterization trick to ensure correct gradient
    z = k.layers.Lambda(sample_z, output_shape=(variational_latent_dimentions,), name='z')([mu, sigma])

    x_latent = z

    conv_shape_units = int(math.ceil(conv_shape[1] * conv_shape[2] * conv_shape[3] * conv_shape[4] * conv_shape[5]))

    x = k.layers.Dense(units=conv_shape_units,
                       kernel_initializer="lecun_normal",
                       bias_initializer=k.initializers.Constant(0.1),
                       kernel_regularizer=k.regularizers.l1_l2(l1=lone, l2=ltwo),
                       kernel_constraint=k.constraints.UnitNorm())(x)
    x = k.layers.BatchNormalization()(x)
    x = k.layers.Activation("selu")(x)

    x = k.layers.Reshape(
        (int(conv_shape[1]), int(conv_shape[2]), int(conv_shape[3]), int(conv_shape[4]), int(conv_shape[5]),))(x)

    return x, x_latent, mu, sigma


# https://github.com/zizhaozhang/unet-tensorflow-keras/blob/master/model.py
def get_crop_shape(target, refer):
    # depth, the 5th dimension
    cd = int(target.get_shape()[4]) - int(refer.get_shape()[4])

    if cd % 2 != 0:
        cd1, cd2 = int(cd / 2), int(cd / 2) + 1
    else:
        cd1, cd2 = int(cd / 2), int(cd / 2)

    # width, the 4th dimension
    cw = int(target.get_shape()[3]) - int(refer.get_shape()[3])

    if cw % 2 != 0:
        cw1, cw2 = int(cw / 2), int(cw / 2) + 1
    else:
        cw1, cw2 = int(cw / 2), int(cw / 2)

    # height, the 3rd dimension
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


def decoder(x, cnn_start_units_log, cnn_end_units_log, x_skip, filters, cnn_increase_layer_density_bool,
            cnn_layer_layers, lone, ltwo, cnn_deconvolution_bool):
    decrement = len(x_skip) - 1

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

    for i in reversed(range(cnn_start_units_log, cnn_end_units_log)):
        if cnn_increase_layer_density_bool:
            filters = int(math.floor(math.pow(2, i)))
        else:
            filters = int(math.floor(math.pow(2, cnn_start_units_log)))

        if cnn_deconvolution_bool:
            x_shape = x.get_shape()

            x = k.layers.TimeDistributed(k.layers.Convolution3DTranspose(filters=filters,
                                                                         kernel_size=(3, 3, 3),
                                                                         strides=(2, 2, 2),
                                                                         padding="same",
                                                                         kernel_initializer="lecun_normal",
                                                                         bias_initializer=k.initializers.Constant(
                                                                             0.1),
                                                                         kernel_regularizer=k.regularizers.l1_l2(
                                                                             l1=lone,
                                                                             l2=ltwo),
                                                                         kernel_constraint=k.constraints.UnitNorm()))(
                x)
            x = k.layers.TimeDistributed(k.layers.BatchNormalization())(x)
            x = k.layers.TimeDistributed(k.layers.Activation("selu"))(x)
            x = k.layers.Reshape((int(x_shape[1]), int(x_shape[2] * 2), int(x_shape[3] * 2),
                                  int(x_shape[4] * 2), int(filters),))(x)
        else:
            x = k.layers.TimeDistributed(k.layers.UpSampling3D())(x)

        x = crop(x, x_skip[decrement])

        decrement = decrement - 1

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

    x = k.layers.TimeDistributed(k.layers.Convolution3D(filters=1,
                                                        kernel_size=(3, 3, 3),
                                                        strides=(1, 1, 1),
                                                        padding="same",
                                                        kernel_initializer="glorot_normal",
                                                        bias_initializer=k.initializers.Constant(0.1)))(x)
    x = k.layers.TimeDistributed(k.layers.Activation("linear"), name="output_2")(x)

    return x


def dense_dynamic_signal_extractor(x, lone, ltwo, dense_layers, dropout, output_size):
    for i in reversed(range(dense_layers)):
        units = output_size * (i + 1)

        x = k.layers.Dense(units=units,
                           kernel_initializer="lecun_normal",
                           bias_initializer=k.initializers.Constant(0.1),
                           kernel_regularizer=k.regularizers.l1_l2(l1=lone, l2=ltwo),
                           kernel_constraint=k.constraints.UnitNorm())(x)
        x = k.layers.BatchNormalization()(x)
        x = k.layers.Activation("selu")(x)
        x = k.layers.AlphaDropout(dropout)(x)

    x = k.layers.Dense(units=output_size,
                       kernel_initializer="glorot_normal",
                       bias_initializer=k.initializers.Constant(0.1))(x)
    x = k.layers.Activation("linear", name="output_1")(x)

    return x


def rnn_dynamic_signal_extractor(x, lone, ltwo, rnn_layers, rnn_units, dropout, output_size):
    for i in range(rnn_layers - 1):
        x = k.layers.LSTM(units=rnn_units,
                          activation="tanh",
                          recurrent_activation="sigmoid",
                          dropout=dropout,
                          recurrent_dropout=dropout,
                          return_sequences=True,
                          kernel_initializer="glorot_normal",
                          bias_initializer=k.initializers.Constant(0.1),
                          recurrent_initializer="glorot_uniform",
                          kernel_regularizer=k.regularizers.l1_l2(l1=lone, l2=ltwo),
                          kernel_constraint=k.constraints.UnitNorm())(x)

    if rnn_layers > 0:
        x = k.layers.LSTM(units=rnn_units,
                          activation="tanh",
                          recurrent_activation="sigmoid",
                          dropout=dropout,
                          recurrent_dropout=dropout,
                          return_sequences=False,
                          kernel_initializer="glorot_normal",
                          bias_initializer=k.initializers.Constant(0.1),
                          recurrent_initializer="glorot_uniform",
                          kernel_regularizer=k.regularizers.l1_l2(l1=lone, l2=ltwo),
                          kernel_constraint=k.constraints.UnitNorm())(x)

        x = k.layers.Reshape((x.shape.as_list()[1],))(x)
    else:
        x = k.layers.Flatten()(x)

    x = k.layers.Dense(units=output_size,
                       kernel_initializer="glorot_normal",
                       bias_initializer=k.initializers.Constant(0.1))(x)
    x = k.layers.Activation("linear", name="output_1")(x)

    return x


def autoencoder_dynamic_signal_extractor(x, cnn_start_units, cnn_layers, cnn_increase_layer_density_bool,
                                         cnn_layer_layers, lone, ltwo, cnn_pool_bool, cnn_max_pool_bool, variational_bool,
                                         dense_units, variational_latent_dimentions, dense_layers,
                                         cnn_deconvolution_bool, rnn_layers, rnn_units, dropout, output_size):
    cnn_start_units_log = int(math.floor(math.log(cnn_start_units, 2)))
    cnn_end_units_log = cnn_start_units_log + cnn_layers

    # DOWNSAMPLE
    x, x_skip, filters = encoder(x, cnn_start_units_log, cnn_end_units_log, cnn_increase_layer_density_bool,
                                 cnn_layer_layers, lone, ltwo, cnn_pool_bool, cnn_max_pool_bool)

    mu = None
    sigma = None

    if variational_bool:
        x, x_latent, mu, sigma = encoder_variational_autoencoder_latenet_space(x, dense_units, lone, ltwo, variational_latent_dimentions)

        # DENSE
        x_1 = dense_dynamic_signal_extractor(x_latent, lone, ltwo, dense_layers, dropout, output_size)
    else:
        x, x_latent = encoder_autoencoder_latenet_space(x, filters)

        # RNN
        x_1 = rnn_dynamic_signal_extractor(x_latent, lone, ltwo, rnn_layers, rnn_units, dropout, output_size)

    # UPSAMPLE
    x_2 = decoder(x, cnn_start_units_log, cnn_end_units_log, x_skip, filters, cnn_increase_layer_density_bool,
                  cnn_layer_layers, lone, ltwo, cnn_deconvolution_bool)

    return x_1, x_2, mu, sigma
