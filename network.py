import keras as k


def output_module(x, size, activation):
    x = k.layers.Dense(units=size)(x)
    x = k.layers.Activation(activation)(x)

    return x


def flatten(x):
    x = k.layers.Flatten()(x)

    return x


def perceptron(x, activation):
    x = k.layers.Flatten()(x)

    x = k.layers.Dense(units=1)(x)
    x = k.layers.Activation(activation)(x)

    return x


def fully_connected(x, units, batch_normalisation_bool, activation):
    x = k.layers.Flatten()(x)

    x = k.layers.Dense(units=units)(x)

    if batch_normalisation_bool:
        x = k.layers.BatchNormalization()(x)

    x = k.layers.Activation(activation)(x)
    x = k.layers.Dropout(0.5)(x)

    return x


def deep_fully_connected(x, layers, units, batch_normalisation_bool, activation):
    x = k.layers.Flatten()(x)

    for _ in range(layers):
        x = k.layers.Dense(units=units)(x)

        if batch_normalisation_bool:
            x = k.layers.BatchNormalization()(x)

        x = k.layers.Activation(activation)(x)
        x = k.layers.Dropout(0.5)(x)

    return x


def conv(x, filters, batch_normalisation_bool, activation):
    x = k.layers.Convolution3D(filters=filters, kernel_size=(3, 3, 3), strides=(2, 2, 2), padding="same")(x)

    if batch_normalisation_bool:
        x = k.layers.BatchNormalization()(x)

    x = k.layers.Activation(activation)(x)
    x = k.layers.Dropout(0.5)(x)

    x = k.layers.Flatten()(x)

    return x


def deep_conv(x, layers, filters, batch_normalisation_bool, activation):
    for _ in range(layers):
        x = k.layers.Convolution3D(filters=filters, kernel_size=(3, 3, 3), strides=(2, 2, 2), padding="same")(x)

        if batch_normalisation_bool:
            x = k.layers.BatchNormalization()(x)

        x = k.layers.Activation(activation)(x)
        x = k.layers.Dropout(0.5)(x)

    x = k.layers.Flatten()(x)

    return x


def deep_conv_fully_connected(x, conv_layers, filters, batch_normalisation_bool, activation, fully_connected_size, units):
    for _ in range(conv_layers):
        x = k.layers.Convolution3D(filters=filters, kernel_size=(3, 3, 3), strides=(2, 2, 2), padding="same")(x)

        if batch_normalisation_bool:
            x = k.layers.BatchNormalization()(x)

        x = k.layers.Activation(activation)(x)
        x = k.layers.Dropout(0.5)(x)

    x = deep_fully_connected(x, fully_connected_size, batch_normalisation_bool, activation, units)

    return x


def rnn(x, units, batch_normalisation_bool, activation):
    x = k.layers.Reshape((x.shape.as_list()[1] * x.shape.as_list()[2] * x.shape.as_list()[3], x.shape.as_list()[4]))(x)

    x = k.layers.LSTM(units=units, return_sequences=True)(x)

    if batch_normalisation_bool:
        x = k.layers.BatchNormalization()(x)

    x = k.layers.Activation(activation)(x)
    x = k.layers.Dropout(0.5)(x)

    return x


def deep_rnn(x, layers, units, batch_normalisation_bool, activation):
    x = k.layers.Reshape((x.shape.as_list()[1] * x.shape.as_list()[2] * x.shape.as_list()[3], x.shape.as_list()[4]))(x)

    for _ in range(layers):
        x = k.layers.LSTM(units, return_sequences=True)(x)

        if batch_normalisation_bool:
            x = k.layers.BatchNormalization()(x)

        x = k.layers.Activation(activation)(x)
        x = k.layers.Dropout(0.5)(x)

    return x


def lenet(x):
    x = k.layers.Convolution3D(filters=20, kernel_size=(5, 5, 5), strides=(1, 1, 1), padding='same')(x)
    x = k.layers.BatchNormalization()(x)
    x = k.layers.Activation("sigmoid")(x)
    x = k.layers.Dropout(0.5)(x)

    x = k.layers.MaxPooling3D(pool_size=(2, 2, 2), strides=(2, 2, 2), padding="valid")(x)

    x = k.layers.Convolution3D(filters=50, kernel_size=(5, 5, 5), strides=(1, 1, 1), padding='same')(x)
    x = k.layers.BatchNormalization()(x)
    x = k.layers.Activation("sigmoid")(x)
    x = k.layers.Dropout(0.5)(x)

    x = k.layers.MaxPooling3D(pool_size=(2, 2, 2), strides=(2, 2, 2), padding="valid")(x)

    x = k.layers.Flatten()(x)

    x = k.layers.Dense(units=500)(x)
    x = k.layers.BatchNormalization()(x)
    x = k.layers.Activation("sigmoid")(x)
    x = k.layers.Dropout(0.5)(x)

    return x


def alexnet_module(x):
    x = k.layers.Convolution3D(filters=48, kernel_size=(5, 5, 5), strides=(1, 1, 1), padding="valid")(x)
    x = k.layers.BatchNormalization()(x)
    x = k.layers.Activation("sigmoid")(x)
    x = k.layers.Dropout(0.5)(x)

    x = k.layers.MaxPooling3D(pool_size=(2, 2, 2), strides=(2, 2, 2), padding="valid")(x)

    x = k.layers.Convolution3D(filters=128, kernel_size=(3, 3, 3), strides=(1, 1, 1), padding="valid")(x)
    x = k.layers.BatchNormalization()(x)
    x = k.layers.Activation("sigmoid")(x)
    x = k.layers.Dropout(0.5)(x)

    x = k.layers.MaxPooling3D(pool_size=(2, 2, 2), strides=(2, 2, 2), padding="valid")(x)

    x = k.layers.Convolution3D(filters=192, kernel_size=(3, 3, 3), strides=(1, 1, 1), padding="valid")(x)
    x = k.layers.BatchNormalization()(x)
    x = k.layers.Activation("sigmoid")(x)
    x = k.layers.Dropout(0.5)(x)

    x = k.layers.Convolution3D(filters=192, kernel_size=(3, 3, 3), strides=(1, 1, 1), padding="valid")(x)
    x = k.layers.BatchNormalization()(x)
    x = k.layers.Activation("sigmoid")(x)
    x = k.layers.Dropout(0.5)(x)

    x = k.layers.Convolution3D(filters=128, kernel_size=(3, 3, 3), strides=(1, 1, 1), padding="valid")(x)
    x = k.layers.BatchNormalization()(x)
    x = k.layers.Activation("sigmoid")(x)
    x = k.layers.Dropout(0.5)(x)

    x = k.layers.MaxPooling3D(pool_size=(2, 2, 2), strides=(2, 2, 2), padding="valid")(x)

    return x


def alexnet_output_module(x):
    for _ in range(2):
        x = k.layers.Dense(units=4096)(x)
        x = k.layers.BatchNormalization()(x)
        x = k.layers.Activation("sigmoid")(x)
        x = k.layers.Dropout(0.5)(x)

    x = k.layers.Dense(units=1000)(x)
    x = k.layers.BatchNormalization()(x)
    x = k.layers.Activation("sigmoid")(x)
    x = k.layers.Dropout(0.5)(x)

    return x


def alexnet(x):
    x = k.layers.Convolution3D(filters=3, kernel_size=(11, 11, 11), strides=(4, 4, 4), padding="valid")(x)
    x = k.layers.BatchNormalization()(x)
    x = k.layers.Activation("sigmoid")(x)
    x = k.layers.Dropout(0.5)(x)

    x_1 = alexnet_module(x)
    x_2 = alexnet_module(x)

    x = k.layers.Add()([x_1, x_2])
    x = k.layers.BatchNormalization()(x)
    x = k.layers.Activation("sigmoid")(x)
    x = k.layers.Dropout(0.5)(x)

    x = k.layers.Flatten()(x)

    x = alexnet_output_module(x)

    return x


def vggnet_module(x, layers, filters):
    for _ in range(layers):
        x = k.layers.Convolution3D(filters=filters, kernel_size=(3, 3, 3), strides=(1, 1, 1), padding="valid")(x)
        x = k.layers.BatchNormalization()(x)
        x = k.layers.Activation("sigmoid")(x)
        x = k.layers.Dropout(0.5)(x)

    x = k.layers.MaxPooling3D(pool_size=(2, 2, 2), strides=(2, 2, 2), padding="valid")(x)

    return x


def vgg16net(x):
    x = vggnet_module(x, 2, 64)
    x = vggnet_module(x, 2, 128)
    x = vggnet_module(x, 2, 256)
    x = vggnet_module(x, 3, 512)
    x = vggnet_module(x, 3, 512)

    x = k.layers.Flatten()(x)

    x = alexnet_output_module(x)

    return x


def vgg19net(x):
    x = vggnet_module(x, 2, 64)
    x = vggnet_module(x, 2, 128)
    x = vggnet_module(x, 2, 256)
    x = vggnet_module(x, 4, 512)
    x = vggnet_module(x, 4, 512)

    x = k.layers.Flatten()(x)

    x = alexnet_output_module(x)

    return x


def googlenet_module(x, filters_1, filters_2, filters_3, filters_4, filters_5, filters_6):
    x_1 = k.layers.Convolution3D(filters=filters_1, kernel_size=(1, 1, 1), strides=(1, 1, 1), padding="same")(x)
    x_1 = k.layers.BatchNormalization()(x_1)
    x_1 = k.layers.Activation("relu")(x_1)
    x_1 = k.layers.Dropout(0.5)(x_1)

    x_2 = k.layers.Convolution3D(filters=filters_2, kernel_size=(1, 1, 1), strides=(1, 1, 1), padding="same")(x)
    x_2 = k.layers.BatchNormalization()(x_2)
    x_2 = k.layers.Activation("relu")(x_2)
    x_2 = k.layers.Dropout(0.5)(x_2)
    x_2 = k.layers.Convolution3D(filters=filters_3, kernel_size=(3, 3, 3), strides=(1, 1, 1), padding="same")(x_2)
    x_2 = k.layers.BatchNormalization()(x_2)
    x_2 = k.layers.Activation("relu")(x_2)
    x_2 = k.layers.Dropout(0.5)(x_2)

    x_3 = k.layers.Convolution3D(filters=filters_4, kernel_size=(1, 1, 1), strides=(1, 1, 1), padding="same")(x)
    x_3 = k.layers.BatchNormalization()(x_3)
    x_3 = k.layers.Activation("relu")(x_3)
    x_3 = k.layers.Dropout(0.5)(x_3)
    x_3 = k.layers.Convolution3D(filters=filters_5, kernel_size=(5, 5, 5), strides=(1, 1, 1), padding="same")(x_3)
    x_3 = k.layers.BatchNormalization()(x_3)
    x_3 = k.layers.Activation("relu")(x_3)
    x_3 = k.layers.Dropout(0.5)(x_3)

    x_4 = k.layers.MaxPooling3D(pool_size=(3, 3, 3), strides=(1, 1, 1), padding="same")(x)
    x_4 = k.layers.Convolution3D(filters=filters_6, kernel_size=(1, 1, 1), strides=(1, 1, 1), padding="same")(x_4)
    x_4 = k.layers.BatchNormalization()(x_4)
    x_4 = k.layers.Activation("relu")(x_4)
    x_4 = k.layers.Dropout(0.5)(x_4)

    x = k.layers.Concatenate(axis=4)([x_1, x_2, x_3, x_4])

    return x


def googlenet_output_module(x):
    x = k.layers.AveragePooling3D(pool_size=(5, 5, 5), strides=(3, 3, 3))(x)

    x = k.layers.Convolution3D(filters=128, kernel_size=(1, 1, 1), strides=(1, 1, 1), padding="same")(x)
    x = k.layers.BatchNormalization()(x)
    x = k.layers.Activation("relu")(x)
    x = k.layers.Dropout(0.5)(x)

    x = k.layers.Flatten()(x)

    x = k.layers.Dense(units=1024)(x)
    x = k.layers.BatchNormalization()(x)
    x = k.layers.Activation("relu")(x)
    x = k.layers.Dropout(0.5)(x)

    return x


def googlenet_input(x):
    x = k.layers.Convolution3D(filters=64, kernel_size=(7, 7, 7), strides=(2, 2, 2), padding="same")(x)
    x = k.layers.BatchNormalization()(x)
    x = k.layers.Activation("relu")(x)
    x = k.layers.Dropout(0.5)(x)

    x = k.layers.MaxPooling3D(pool_size=(3, 3, 3), strides=(2, 2, 2), padding="valid")(x)

    x = k.layers.Convolution3D(filters=64, kernel_size=(1, 1, 1), strides=(1, 1, 1), padding="same")(x)
    x = k.layers.BatchNormalization()(x)
    x = k.layers.Activation("relu")(x)
    x = k.layers.Dropout(0.5)(x)

    x = k.layers.Convolution3D(filters=192, kernel_size=(3, 3, 3), strides=(1, 1, 1), padding="same")(x)
    x = k.layers.BatchNormalization()(x)
    x = k.layers.Activation("relu")(x)
    x = k.layers.Dropout(0.5)(x)

    x = k.layers.MaxPooling3D(pool_size=(3, 3, 3), strides=(2, 2, 2), padding="valid")(x)

    x = googlenet_module(x, 64, 96, 128, 16, 32, 32)
    x = googlenet_module(x, 128, 128, 192, 32, 96, 64)

    x = k.layers.MaxPooling3D(pool_size=(3, 3, 3), strides=(2, 2, 2), padding="valid")(x)

    x = googlenet_module(x, 192, 96, 208, 16, 48, 64)

    return x


def shallow_googlenet(x):
    x = googlenet_input(x)

    return x


def googlenet(x):
    x = googlenet_input(x)

    x_1 = googlenet_output_module(x)

    x = googlenet_module(x, 160, 112, 224, 24, 64, 64)
    x = googlenet_module(x, 128, 128, 256, 24, 64, 64)
    x = googlenet_module(x, 112, 144, 288, 32, 64, 64)

    x_2 = googlenet_output_module(x)

    x = googlenet_module(x, 256, 160, 320, 32, 128, 128)

    x = k.layers.MaxPooling3D(pool_size=(3, 3, 3), strides=(2, 2, 2), padding="valid")(x)

    x = googlenet_module(x, 256, 160, 320, 32, 128, 128)
    x = googlenet_module(x, 384, 192, 384, 48, 128, 128)

    x = googlenet_output_module(x)

    return x, x_1, x_2


def resnet_module(x, filters_1, filters_2, kernal_size):
    x = k.layers.Convolution3D(filters=filters_1, kernel_size=(kernal_size, kernal_size), strides=(1, 1, 1), padding="same")(x)
    x = k.layers.BatchNormalization()(x)
    x = k.layers.Activation("relu")(x)
    x = k.layers.Dropout(0.5)(x)

    x = k.layers.Convolution3D(filters=filters_2, kernel_size=(1, 1, 1), strides=(1, 1, 1), padding="valid")(x)
    x = k.layers.BatchNormalization()(x)
    x = k.layers.Activation("relu")(x)
    x = k.layers.Dropout(0.5)(x)

    return x


def resnet_conv_module(x, filters_1, filters_2, filters_3, kernal_size, strides):
    x_shortcut = x

    x = k.layers.Convolution3D(filters=filters_1, kernel_size=(1, 1, 1), strides=(strides, strides),
                               padding="valid")(x)
    x = k.layers.BatchNormalization()(x)
    x = k.layers.Activation("relu")(x)
    x = k.layers.Dropout(0.5)(x)

    x = resnet_module(x, filters_2, filters_3, kernal_size)

    x_shortcut = k.layers.Convolution3D(filters=filters_3, kernel_size=(1, 1, 1), strides=(strides, strides), padding="valid")(x_shortcut)

    x = k.layers.Add()([x, x_shortcut])
    x = k.layers.BatchNormalization()(x)
    x = k.layers.Activation("relu")(x)
    x = k.layers.Dropout(0.5)(x)

    return x


def resnet_identity_module(x, filters_1, filters_2, filters_3, kernal_size):
    x_shortcut = x

    x = k.layers.Convolution3D(filters=filters_1, kernel_size=(1, 1, 1), strides=(1, 1, 1), padding="valid", )(x)
    x = k.layers.BatchNormalization()(x)
    x = k.layers.Activation("relu")(x)
    x = k.layers.Dropout(0.5)(x)

    x = resnet_module(x, filters_2, filters_3, kernal_size)

    x = k.layers.Add()([x, x_shortcut])
    x = k.layers.Activation("relu")(x)

    return x


def resnet(x):
    x = k.layers.Convolution3D(filters=64, kernel_size=(7, 7, 7), strides=(2, 2, 2), padding="same")(x)
    x = k.layers.BatchNormalization()(x)
    x = k.layers.Activation("relu")(x)
    x = k.layers.Dropout(0.5)(x)

    x = k.layers.MaxPooling3D(pool_size=(3, 3, 3), strides=(2, 2, 2), padding="same")(x)

    x = resnet_conv_module(x, 64, 64, 256, 3, 1)

    for _ in range(2):
        x = resnet_identity_module(x, 64, 64, 256, 3)

    x = resnet_conv_module(x, 128, 128, 512, 3, 2)

    for _ in range(3):
        x = resnet_identity_module(x, 128, 128, 512, 3)

    x = resnet_conv_module(x, 256, 256, 1024, 3, 2)

    for _ in range(5):
        x = resnet_identity_module(x, 256, 256, 1024, 3)

    x = resnet_conv_module(x, 512, 512, 2048, 3, 2)

    for _ in range(2):
        x = resnet_identity_module(x, 512, 512, 2048, 3)

    x = k.layers.AveragePooling3D(pool_size=(2, 2, 2), strides=(1, 1, 1), padding="same")(x)

    x = k.layers.Flatten()(x)

    x = k.layers.Dense(units=1000)(x)
    x = k.layers.BatchNormalization()(x)
    x = k.layers.Activation("relu")(x)
    x = k.layers.Dropout(0.5)(x)

    return x


# https://github.com/zizhaozhang/unet-tensorflow-keras/blob/master/model.py
def get_crop_shape(target, refer):
    # width, the 3rd dimension
    cw = target.get_shape()[2] - refer.get_shape()[2]

    if cw % 2 != 0:
        cw1, cw2 = int(cw / 2), int(cw / 2) + 1
    else:
        cw1, cw2 = int(cw / 2), int(cw / 2)

    # height, the 2nd dimension
    ch = target.get_shape()[1] - refer.get_shape()[1]

    if ch % 2 != 0:
        ch1, ch2 = int(ch / 2), int(ch / 2) + 1
    else:
        ch1, ch2 = int(ch / 2), int(ch / 2)

    return (ch1, ch2), (cw1, cw2)


def unet(x):
    x = k.layers.Convolution3D(filters=1, kernel_size=(3, 3, 3), strides=(1, 1, 1), padding='same')(x)
    x = k.layers.BatchNormalization()(x)
    x = k.layers.Activation("relu")(x)
    x = k.layers.Dropout(0.5)(x)

    x = k.layers.Convolution3D(filters=64, kernel_size=(3, 3, 3), strides=(1, 1, 1), padding='same')(x)
    x = k.layers.BatchNormalization()(x)
    x = k.layers.Activation("relu")(x)
    x = k.layers.Dropout(0.5)(x)

    x = k.layers.Convolution3D(filters=64, kernel_size=(3, 3, 3), strides=(1, 1, 1), padding='same')(x)
    x = k.layers.BatchNormalization()(x)
    x = k.layers.Activation("relu")(x)
    x_shortcut_1 = k.layers.Dropout(0.5)(x)

    x = k.layers.MaxPooling3D(pool_size=(2, 2, 2), strides=(2, 2, 2), padding="same")(x_shortcut_1)

    x = k.layers.Convolution3D(filters=64, kernel_size=(3, 3, 3), strides=(1, 1, 1), padding='same')(x)
    x = k.layers.BatchNormalization()(x)
    x = k.layers.Activation("relu")(x)
    x = k.layers.Dropout(0.5)(x)

    x = k.layers.Convolution3D(filters=128, kernel_size=(3, 3, 3), strides=(1, 1, 1), padding='same')(x)
    x = k.layers.BatchNormalization()(x)
    x = k.layers.Activation("relu")(x)
    x = k.layers.Dropout(0.5)(x)

    x = k.layers.Convolution3D(filters=128, kernel_size=(3, 3, 3), strides=(1, 1, 1), padding='same')(x)
    x = k.layers.BatchNormalization()(x)
    x = k.layers.Activation("relu")(x)
    x_shortcut_2 = k.layers.Dropout(0.5)(x)

    x = k.layers.MaxPooling3D(pool_size=(2, 2, 2), strides=(2, 2, 2), padding="same")(x_shortcut_2)

    x = k.layers.Convolution3D(filters=128, kernel_size=(3, 3, 3), strides=(1, 1, 1), padding='same')(x)
    x = k.layers.BatchNormalization()(x)
    x = k.layers.Activation("relu")(x)
    x = k.layers.Dropout(0.5)(x)

    x = k.layers.Convolution3D(filters=256, kernel_size=(3, 3, 3), strides=(1, 1, 1), padding='same')(x)
    x = k.layers.BatchNormalization()(x)
    x = k.layers.Activation("relu")(x)
    x = k.layers.Dropout(0.5)(x)

    x = k.layers.Convolution3D(filters=256, kernel_size=(3, 3, 3), strides=(1, 1, 1), padding='same')(x)
    x = k.layers.BatchNormalization()(x)
    x = k.layers.Activation("relu")(x)
    x_shortcut_3 = k.layers.Dropout(0.5)(x)

    x = k.layers.MaxPooling3D(pool_size=(2, 2, 2), strides=(2, 2, 2), padding="same")(x_shortcut_3)

    x = k.layers.Convolution3D(filters=256, kernel_size=(3, 3, 3), strides=(1, 1, 1), padding='same')(x)
    x = k.layers.BatchNormalization()(x)
    x = k.layers.Activation("relu")(x)
    x = k.layers.Dropout(0.5)(x)

    x = k.layers.Convolution3D(filters=512, kernel_size=(3, 3, 3), strides=(1, 1, 1), padding='same')(x)
    x = k.layers.BatchNormalization()(x)
    x = k.layers.Activation("relu")(x)
    x = k.layers.Dropout(0.5)(x)

    x = k.layers.Convolution3D(filters=512, kernel_size=(3, 3, 3), strides=(1, 1, 1), padding='same')(x)
    x = k.layers.BatchNormalization()(x)
    x = k.layers.Activation("relu")(x)
    x_shortcut_4 = k.layers.Dropout(0.5)(x)

    x = k.layers.MaxPooling3D(pool_size=(2, 2, 2), strides=(2, 2, 2), padding="same")(x_shortcut_4)

    x = k.layers.Convolution3D(filters=512, kernel_size=(3, 3, 3), strides=(1, 1, 1), padding='same')(x)
    x = k.layers.BatchNormalization()(x)
    x = k.layers.Activation("relu")(x)
    x = k.layers.Dropout(0.5)(x)

    for _ in range(2):
        x = k.layers.Convolution3D(filters=1024, kernel_size=(3, 3, 3), strides=(1, 1, 1), padding='same')(x)
        x = k.layers.BatchNormalization()(x)
        x = k.layers.Activation("relu")(x)
        x = k.layers.Dropout(0.5)(x)

    x = k.layers.UpSampling3D(size=(2, 2, 2))(x)

    ch, cw = get_crop_shape(x, x_shortcut_4)
    x = k.layers.Cropping3D(cropping=(ch, cw))(x)
    x = k.layers.Concatenate(axis=4)([x, x_shortcut_4])

    x = k.layers.Convolution3D(filters=1024, kernel_size=(3, 3, 3), strides=(1, 1, 1), padding='same')(x)
    x = k.layers.BatchNormalization()(x)
    x = k.layers.Activation("relu")(x)
    x = k.layers.Dropout(0.5)(x)

    for _ in range(2):
        x = k.layers.Convolution3D(filters=512, kernel_size=(3, 3, 3), strides=(1, 1, 1), padding='same')(x)
        x = k.layers.BatchNormalization()(x)
        x = k.layers.Activation("relu")(x)
        x = k.layers.Dropout(0.5)(x)

    x = k.layers.UpSampling3D(size=(2, 2, 2))(x)

    ch, cw = get_crop_shape(x, x_shortcut_3)
    x = k.layers.Cropping3D(cropping=(ch, cw))(x)
    x = k.layers.Concatenate(axis=4)([x, x_shortcut_3])

    x = k.layers.Convolution3D(filters=512, kernel_size=(3, 3, 3), strides=(1, 1, 1), padding='same')(x)
    x = k.layers.BatchNormalization()(x)
    x = k.layers.Activation("relu")(x)
    x = k.layers.Dropout(0.5)(x)

    for _ in range(2):
        x = k.layers.Convolution3D(filters=256, kernel_size=(3, 3, 3), strides=(1, 1, 1), padding='same')(x)
        x = k.layers.BatchNormalization()(x)
        x = k.layers.Activation("relu")(x)
        x = k.layers.Dropout(0.5)(x)

    x = k.layers.UpSampling3D(size=(2, 2, 2))(x)

    ch, cw = get_crop_shape(x, x_shortcut_2)
    x = k.layers.Cropping3D(cropping=(ch, cw))(x)
    x = k.layers.Concatenate(axis=4)([x, x_shortcut_2])

    x = k.layers.Convolution3D(filters=256, kernel_size=(3, 3, 3), strides=(1, 1, 1), padding='same')(x)
    x = k.layers.BatchNormalization()(x)
    x = k.layers.Activation("relu")(x)
    x = k.layers.Dropout(0.5)(x)

    for _ in range(2):
        x = k.layers.Convolution3D(filters=128, kernel_size=(3, 3, 3), strides=(1, 1, 1), padding='same')(x)
        x = k.layers.BatchNormalization()(x)
        x = k.layers.Activation("relu")(x)
        x = k.layers.Dropout(0.5)(x)

    x = k.layers.UpSampling3D(size=(2, 2, 2))(x)

    ch, cw = get_crop_shape(x, x_shortcut_1)
    x = k.layers.Cropping3D(cropping=(ch, cw))(x)
    x = k.layers.Concatenate(axis=4)([x, x_shortcut_1])

    x = k.layers.Convolution3D(filters=128, kernel_size=(3, 3, 3), strides=(1, 1, 1), padding='same')(x)
    x = k.layers.BatchNormalization()(x)
    x = k.layers.Activation("relu")(x)
    x = k.layers.Dropout(0.5)(x)

    for _ in range(2):
        x = k.layers.Convolution3D(filters=64, kernel_size=(3, 3, 3), strides=(1, 1, 1), padding='same')(x)
        x = k.layers.BatchNormalization()(x)
        x = k.layers.Activation("relu")(x)
        x = k.layers.Dropout(0.5)(x)

    x = k.layers.Convolution3D(filters=2, kernel_size=(3, 3, 3), strides=(1, 1, 1), padding='same')(x)
    x = k.layers.BatchNormalization()(x)
    x = k.layers.Activation("relu")(x)
    x = k.layers.Dropout(0.5)(x)

    x = k.layers.Flatten()(x)

    return x


def voxelmorph(x):
    x = k.layers.Convolution3D(filters=16, kernel_size=(3, 3, 3), strides=(1, 1, 1), padding='same')(x)
    x = k.layers.BatchNormalization()(x)
    x = k.layers.Activation("relu")(x)
    x_shortcut_1 = k.layers.Dropout(0.5)(x)

    x = k.layers.Convolution3D(filters=32, kernel_size=(3, 3, 3), strides=(2, 2, 2), padding='same')(x_shortcut_1)
    x = k.layers.BatchNormalization()(x)
    x = k.layers.Activation("relu")(x)
    x_shortcut_2 = k.layers.Dropout(0.5)(x)

    x = k.layers.Convolution3D(filters=32, kernel_size=(3, 3, 3), strides=(2, 2, 2), padding='same')(x_shortcut_2)
    x = k.layers.BatchNormalization()(x)
    x = k.layers.Activation("relu")(x)
    x_shortcut_3 = k.layers.Dropout(0.5)(x)

    x = k.layers.Convolution3D(filters=32, kernel_size=(3, 3, 3), strides=(2, 2, 2), padding='same')(x_shortcut_3)
    x = k.layers.BatchNormalization()(x)
    x = k.layers.Activation("relu")(x)
    x_shortcut_4 = k.layers.Dropout(0.5)(x)

    x = k.layers.Convolution3D(filters=32, kernel_size=(3, 3, 3), strides=(2, 2, 2), padding='same')(x_shortcut_4)
    x = k.layers.BatchNormalization()(x)
    x = k.layers.Activation("relu")(x)
    x = k.layers.Dropout(0.5)(x)

    x = k.layers.UpSampling3D(size=(2, 2, 2))(x)

    x = k.layers.Convolution3D(filters=32, kernel_size=(3, 3, 3), strides=(1, 1, 1), padding='same')(x)
    x = k.layers.BatchNormalization()(x)
    x = k.layers.Activation("relu")(x)
    x = k.layers.Dropout(0.5)(x)

    ch, cw = get_crop_shape(x, x_shortcut_4)
    x = k.layers.Cropping3D(cropping=(ch, cw))(x)
    x = k.layers.Concatenate(axis=4)([x, x_shortcut_4])

    x = k.layers.UpSampling3D(size=(2, 2, 2))(x)

    x = k.layers.Convolution3D(filters=32, kernel_size=(3, 3, 3), strides=(1, 1, 1), padding='same')(x)
    x = k.layers.BatchNormalization()(x)
    x = k.layers.Activation("relu")(x)
    x = k.layers.Dropout(0.5)(x)

    ch, cw = get_crop_shape(x, x_shortcut_3)
    x = k.layers.Cropping3D(cropping=(ch, cw))(x)
    x = k.layers.Concatenate(axis=4)([x, x_shortcut_3])

    x = k.layers.UpSampling3D(size=(2, 2, 2))(x)

    x = k.layers.Convolution3D(filters=32, kernel_size=(3, 3, 3), strides=(1, 1, 1), padding='same')(x)
    x = k.layers.BatchNormalization()(x)
    x = k.layers.Activation("relu")(x)
    x = k.layers.Dropout(0.5)(x)

    ch, cw = get_crop_shape(x, x_shortcut_2)
    x = k.layers.Cropping3D(cropping=(ch, cw))(x)
    x = k.layers.Concatenate(axis=4)([x, x_shortcut_2])

    x = k.layers.UpSampling3D(size=(2, 2, 2))(x)

    x = k.layers.Convolution3D(filters=32, kernel_size=(3, 3, 3), strides=(1, 1, 1), padding='same')(x)
    x = k.layers.BatchNormalization()(x)
    x = k.layers.Activation("relu")(x)
    x = k.layers.Dropout(0.5)(x)

    ch, cw = get_crop_shape(x, x_shortcut_1)
    x = k.layers.Cropping3D(cropping=(ch, cw))(x)
    x = k.layers.Concatenate(axis=4)([x, x_shortcut_1])
    x = k.layers.Activation("relu")(x)

    x = k.layers.Convolution3D(filters=16, kernel_size=(3, 3, 3), strides=(1, 1, 1), padding='same')(x)
    x = k.layers.BatchNormalization()(x)
    x = k.layers.Activation("relu")(x)
    x = k.layers.Dropout(0.5)(x)

    x = k.layers.Convolution3D(filters=16, kernel_size=(3, 3, 3), strides=(1, 1, 1), padding='same')(x)
    x = k.layers.BatchNormalization()(x)
    x = k.layers.Activation("relu")(x)
    x = k.layers.Dropout(0.5)(x)

    x = k.layers.Convolution3D(filters=3, kernel_size=(3, 3, 3), strides=(1, 1, 1), padding='same')(x)
    x = k.layers.BatchNormalization()(x)
    x = k.layers.Activation("relu")(x)
    x = k.layers.Dropout(0.5)(x)

    x = k.layers.Flatten()(x)

    return x
