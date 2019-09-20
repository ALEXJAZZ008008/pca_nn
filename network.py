from tensorflow import keras as k


def output_module(x, size):
    x = k.layers.Dense(units=size)(x)
    x = k.layers.Activation("tanh")(x)

    return x


def perceptron(x):
    x = k.layers.Flatten()(x)

    return x


def fully_connected(x):
    x = k.layers.Flatten()(x)

    x = k.layers.Dense(units=256)(x)
    x = k.layers.BatchNormalization()(x)
    x = k.layers.PReLU()(x)
    x = k.layers.Dropout(0.5)(x)

    return x


def deep_fully_connected(x, fully_connected_bool, size, output_size):
    x = k.layers.Flatten()(x)

    for _ in range(size):
        x = k.layers.Dense(units=256)(x)
        x = k.layers.BatchNormalization()(x)
        x = k.layers.PReLU()(x)
        x = k.layers.Dropout(0.5)(x)

    if fully_connected_bool:
        x = k.layers.Dense(units=output_size * 10)(x)
        x = k.layers.BatchNormalization()(x)
        x = k.layers.PReLU()(x)
        x = k.layers.Dropout(0.5)(x)

        x = k.layers.Dense(units=output_size)(x)
        x = k.layers.BatchNormalization()(x)
        x = k.layers.PReLU()(x)
        x = k.layers.Dropout(0.5)(x)

    return x


def conv(x):
    x = k.layers.Conv3D(filters=64, kernel_size=(3, 3, 3), strides=(1, 1, 1), padding="same")(x)
    x = k.layers.BatchNormalization()(x)
    x = k.layers.PReLU()(x)
    x = k.layers.Dropout(0.5)(x)

    x = k.layers.AveragePooling3D(pool_size=(2, 2, 2), strides=(1, 1, 1), padding="same")(x)

    x = k.layers.BatchNormalization()(x)
    x = k.layers.PReLU()(x)
    x = k.layers.Dropout(0.5)(x)

    x = k.layers.Flatten()(x)

    return x


def deep_conv(x):
    for _ in range(2):
        x = k.layers.Conv3D(filters=64, kernel_size=(3, 3, 3), strides=(1, 1, 1), padding="same")(x)
        x = k.layers.BatchNormalization()(x)
        x = k.layers.PReLU()(x)
        x = k.layers.Dropout(0.5)(x)

        x = k.layers.AveragePooling3D(pool_size=(2, 2, 2), strides=(1, 1, 1), padding="same")(x)

    x = k.layers.BatchNormalization()(x)
    x = k.layers.PReLU()(x)
    x = k.layers.Dropout(0.5)(x)

    x = k.layers.Flatten()(x)

    return x


def deep_conv_fully_connected(x, output_size):
    for _ in range(2):
        x = k.layers.Conv3D(filters=64, kernel_size=(3, 3, 3), strides=(1, 1, 1), padding="same")(x)
        x = k.layers.BatchNormalization()(x)
        x = k.layers.PReLU()(x)
        x = k.layers.Dropout(0.5)(x)

        x = k.layers.AveragePooling3D(pool_size=(2, 2, 2), strides=(1, 1, 1), padding="same")(x)

    x = k.layers.Conv3D(filters=1, kernel_size=(3, 3, 3), strides=(1, 1, 1), padding="same")(x)
    x = k.layers.BatchNormalization()(x)
    x = k.layers.PReLU()(x)
    x = k.layers.Dropout(0.5)(x)

    x = deep_fully_connected(x, True, 2, output_size)

    return x


def papernet(x):
    for _ in range(4):
        x = k.layers.Conv3D(filters=64, kernel_size=(3, 3, 3), strides=(1, 1, 1), padding="same")(x)
        x = k.layers.BatchNormalization()(x)
        x = k.layers.PReLU()(x)
        x = k.layers.Dropout(0.5)(x)

        x = k.layers.AveragePooling3D(pool_size=(2, 2, 2), strides=(1, 1, 1), padding="same")(x)

    for _ in range(3):
        x = k.layers.Conv3D(filters=64, kernel_size=(1, 1, 1), strides=(1, 1, 1), padding="same")(x)
        x = k.layers.BatchNormalization()(x)
        x = k.layers.PReLU()(x)
        x = k.layers.Dropout(0.5)(x)

    x = k.layers.Flatten()(x)

    return x


def lenet(x):
    x = k.layers.Conv3D(filters=20, kernel_size=(5, 5, 5), strides=(1, 1, 1), padding='same')(x)
    x = k.layers.BatchNormalization()(x)
    x = k.layers.PReLU()(x)
    x = k.layers.Dropout(0.5)(x)

    x = k.layers.MaxPool3D(pool_size=(2, 2, 2), strides=(2, 2, 2), padding="valid")(x)

    x = k.layers.Conv3D(filters=50, kernel_size=(5, 5, 5), strides=(1, 1, 1), padding='same')(x)
    x = k.layers.BatchNormalization()(x)
    x = k.layers.PReLU()(x)
    x = k.layers.Dropout(0.5)(x)

    x = k.layers.MaxPool3D(pool_size=(2, 2, 2), strides=(2, 2, 2), padding="valid")(x)

    x = k.layers.Flatten()(x)

    x = k.layers.Dense(units=500)(x)
    x = k.layers.BatchNormalization()(x)
    x = k.layers.PReLU()(x)
    x = k.layers.Dropout(0.5)(x)

    return x


def alexnet_module(x):
    x = k.layers.Conv3D(filters=48, kernel_size=(11, 11), strides=(4, 4, 4), padding="valid")(x)
    x = k.layers.BatchNormalization()(x)
    x = k.layers.PReLU()(x)
    x = k.layers.Dropout(0.5)(x)

    x = k.layers.MaxPooling3D(pool_size=(2, 2, 2), strides=(2, 2, 2), padding="valid")(x)

    x = k.layers.Conv3D(filters=128, kernel_size=(11, 11), strides=(1, 1, 1), padding="valid")(x)
    x = k.layers.BatchNormalization()(x)
    x = k.layers.PReLU()(x)
    x = k.layers.Dropout(0.5)(x)

    x = k.layers.MaxPooling3D(pool_size=(2, 2, 2), strides=(2, 2, 2), padding="valid")(x)

    x = k.layers.Conv3D(filters=192, kernel_size=(3, 3, 3), strides=(1, 1, 1), padding="valid")(x)
    x = k.layers.BatchNormalization()(x)
    x = k.layers.PReLU()(x)
    x = k.layers.Dropout(0.5)(x)

    x = k.layers.Conv3D(filters=192, kernel_size=(3, 3, 3), strides=(1, 1, 1), padding="valid")(x)
    x = k.layers.BatchNormalization()(x)
    x = k.layers.PReLU()(x)
    x = k.layers.Dropout(0.5)(x)

    x = k.layers.Conv3D(filters=128, kernel_size=(3, 3, 3), strides=(1, 1, 1), padding="valid")(x)
    x = k.layers.BatchNormalization()(x)
    x = k.layers.PReLU()(x)
    x = k.layers.Dropout(0.5)(x)

    x = k.layers.MaxPooling3D(pool_size=(2, 2, 2), strides=(2, 2, 2), padding="valid")(x)

    return x


def alexnet_output_module(x):
    for _ in range(2):
        x = k.layers.Dense(units=4096)(x)
        x = k.layers.BatchNormalization()(x)
        x = k.layers.PReLU()(x)
        x = k.layers.Dropout(0.5)(x)

    x = k.layers.Dense(units=1000)(x)
    x = k.layers.BatchNormalization()(x)
    x = k.layers.PReLU()(x)
    x = k.layers.Dropout(0.5)(x)

    return x


def alexnet(x):
    x_1 = alexnet_module(x)
    x_2 = alexnet_module(x)

    x = k.layers.Add()([x_1, x_2])
    x = k.layers.BatchNormalization()(x)
    x = k.layers.PReLU()(x)
    x = k.layers.Dropout(0.5)(x)

    x = k.layers.Flatten()(x)

    x = alexnet_output_module(x)

    return x


def vggnet_module(x, conv_filter, iterations):
    for _ in range(iterations):
        x = k.layers.Conv3D(filters=conv_filter, kernel_size=(3, 3, 3), strides=(1, 1, 1), padding="valid")(x)
        x = k.layers.BatchNormalization()(x)
        x = k.layers.PReLU()(x)
        x = k.layers.Dropout(0.5)(x)

    x = k.layers.MaxPooling3D(pool_size=(2, 2, 2), strides=(2, 2, 2), padding="valid")(x)

    return x


def vgg16net(x):
    x = vggnet_module(x, 64, 2)
    x = vggnet_module(x, 128, 2)
    x = vggnet_module(x, 256, 2)
    x = vggnet_module(x, 512, 3)
    x = vggnet_module(x, 512, 3)

    x = k.layers.Flatten()(x)

    x = alexnet_output_module(x)

    return x


def vgg19net(x):
    x = vggnet_module(x, 64, 2)
    x = vggnet_module(x, 128, 2)
    x = vggnet_module(x, 256, 2)
    x = vggnet_module(x, 512, 4)
    x = vggnet_module(x, 512, 4)

    x = k.layers.Flatten()(x)

    x = alexnet_output_module(x)

    return x


def googlenet_module(x, conv_filter_1, conv_filter_2, conv_filter_3, conv_filter_4, conv_filter_5, conv_filter_6):
    x_1 = k.layers.Conv3D(filters=conv_filter_1, kernel_size=(1, 1, 1), strides=(1, 1, 1), padding="same")(x)
    x_1 = k.layers.BatchNormalization()(x_1)
    x_1 = k.layers.PReLU()(x_1)
    x_1 = k.layers.Dropout(0.5)(x_1)

    x_2 = k.layers.Conv3D(filters=conv_filter_2, kernel_size=(1, 1, 1), strides=(1, 1, 1), padding="same")(x)
    x_2 = k.layers.BatchNormalization()(x_2)
    x_2 = k.layers.PReLU()(x_2)
    x_2 = k.layers.Dropout(0.5)(x_2)
    x_2 = k.layers.Conv3D(filters=conv_filter_3, kernel_size=(3, 3, 3), strides=(1, 1, 1), padding="same")(x_2)
    x_2 = k.layers.BatchNormalization()(x_2)
    x_2 = k.layers.PReLU()(x_2)
    x_2 = k.layers.Dropout(0.5)(x_2)

    x_3 = k.layers.Conv3D(filters=conv_filter_4, kernel_size=(1, 1, 1), strides=(1, 1, 1), padding="same")(x)
    x_3 = k.layers.BatchNormalization()(x_3)
    x_3 = k.layers.PReLU()(x_3)
    x_3 = k.layers.Dropout(0.5)(x_3)
    x_3 = k.layers.Conv3D(filters=conv_filter_5, kernel_size=(5, 5, 5), strides=(1, 1, 1), padding="same")(x_3)
    x_3 = k.layers.BatchNormalization()(x_3)
    x_3 = k.layers.PReLU()(x_3)
    x_3 = k.layers.Dropout(0.5)(x_3)

    x_4 = k.layers.MaxPooling3D(pool_size=(3, 3, 3), strides=(1, 1, 1), padding="same")(x)
    x_4 = k.layers.Conv3D(filters=conv_filter_6, kernel_size=(1, 1, 1), strides=(1, 1, 1), padding="same")(x_4)
    x_4 = k.layers.BatchNormalization()(x_4)
    x_4 = k.layers.PReLU()(x_4)
    x_4 = k.layers.Dropout(0.5)(x_4)

    x = k.layers.Concatenate([x_1, x_2, x_3, x_4], axis=4)

    return x


def googlenet_output_module(x):
    x = k.layers.AveragePooling3D(pool_size=(5, 5, 5), strides=(3, 3, 3))(x)

    x = k.layers.Conv3D(filters=128, kernel_size=(1, 1, 1), strides=(1, 1, 1), padding="same")(x)
    x = k.layers.BatchNormalization()(x)
    x = k.layers.PReLU()(x)
    x = k.layers.Dropout(0.5)(x)

    x = k.layers.Flatten()(x)

    x = k.layers.Dense(units=1024)(x)
    x = k.layers.BatchNormalization()(x)
    x = k.layers.PReLU()(x)
    x = k.layers.Dropout(0.5)(x)

    return x


def googlenet_input(x):
    x = k.layers.Conv3D(filters=64, kernel_size=(7, 7, 7), strides=(2, 2, 2), padding="same")(x)
    x = k.layers.BatchNormalization()(x)
    x = k.layers.PReLU()(x)
    x = k.layers.Dropout(0.5)(x)

    x = k.layers.MaxPooling3D(pool_size=(3, 3, 3), strides=(2, 2, 2), padding="valid")(x)

    x = k.layers.Conv3D(filters=64, kernel_size=(1, 1, 1), strides=(1, 1, 1), padding="same")(x)
    x = k.layers.BatchNormalization()(x)
    x = k.layers.PReLU()(x)
    x = k.layers.Dropout(0.5)(x)

    x = k.layers.Conv3D(filters=192, kernel_size=(3, 3, 3), strides=(1, 1, 1), padding="same")(x)
    x = k.layers.BatchNormalization()(x)
    x = k.layers.PReLU()(x)
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


def resnet_module(x, conv_filter_1, conv_filter_2, conv_kernal):
    x = k.layers.Conv3D(filters=conv_filter_1,
                        kernel_size=(conv_kernal, conv_kernal),
                        strides=(1, 1, 1),
                        padding="same")(x)
    x = k.layers.BatchNormalization()(x)
    x = k.layers.PReLU()(x)
    x = k.layers.Dropout(0.5)(x)

    x = k.layers.Conv3D(filters=conv_filter_2, kernel_size=(1, 1, 1), strides=(1, 1, 1), padding="valid")(x)
    x = k.layers.BatchNormalization()(x)
    x = k.layers.PReLU()(x)
    x = k.layers.Dropout(0.5)(x)

    return x


def resnet_conv_module(x, conv_filter_1, conv_filter_2, conv_filter_3, conv_kernal, conv_stride):
    x_shortcut = x

    x = k.layers.Conv3D(filters=conv_filter_1,
                        kernel_size=(1, 1, 1),
                        strides=(conv_stride, conv_stride),
                        padding="valid")(x)
    x = k.layers.BatchNormalization()(x)
    x = k.layers.PReLU()(x)
    x = k.layers.Dropout(0.5)(x)

    x = resnet_module(x, conv_filter_2, conv_filter_3, conv_kernal)

    x_shortcut = k.layers.Conv3D(filters=conv_filter_3,
                                 kernel_size=(1, 1, 1),
                                 strides=(conv_stride, conv_stride),
                                 padding="valid")(x_shortcut)

    x = k.layers.Add()([x, x_shortcut])
    x = k.layers.BatchNormalization()(x)
    x = k.layers.PReLU()(x)
    x = k.layers.Dropout(0.5)(x)

    return x


def resnet_identity_module(x, conv_filter_1, conv_filter_2, conv_filter_3, conv_kernal):
    x_shortcut = x

    x = k.layers.Conv3D(filters=conv_filter_1, kernel_size=(1, 1, 1), strides=(1, 1, 1), padding="valid", )(x)
    x = k.layers.BatchNormalization()(x)
    x = k.layers.PReLU()(x)
    x = k.layers.Dropout(0.5)(x)

    x = resnet_module(x, conv_filter_2, conv_filter_3, conv_kernal)

    x = k.layers.Add()([x, x_shortcut])
    x = k.layers.PReLU()(x)

    return x


def resnet(x):
    x = k.layers.Conv3D(filters=64, kernel_size=(7, 7, 7), strides=(2, 2, 2), padding="same")(x)
    x = k.layers.BatchNormalization()(x)
    x = k.layers.PReLU()(x)
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
    x = k.layers.PReLU()(x)
    x = k.layers.Dropout(0.5)(x)

    return x


def simple_resnet(x, output_size):
    x = k.layers.Conv3D(filters=output_size, kernel_size=(1, 1, 1), strides=(1, 1, 1), padding='same')(x)
    x = k.layers.BatchNormalization()(x)
    x = k.layers.PReLU()(x)
    x_shortcut = k.layers.Dropout(0.5)(x)

    x = k.layers.Conv3D(filters=output_size, kernel_size=(3, 3, 3), strides=(1, 1, 1), padding='same')(x_shortcut)
    x = k.layers.BatchNormalization()(x)
    x = k.layers.PReLU()(x)
    x = k.layers.Dropout(0.5)(x)

    x = k.layers.Add()([x, x_shortcut])
    x = k.layers.PReLU()(x)

    x = k.layers.AveragePooling3D(pool_size=(2, 2, 2), strides=(2, 2, 2), padding="same")(x)

    x = k.layers.Conv3D(filters=output_size * 2, kernel_size=(1, 1, 1), strides=(1, 1, 1), padding='same')(x)
    x = k.layers.BatchNormalization()(x)
    x = k.layers.PReLU()(x)
    x_shortcut = k.layers.Dropout(0.5)(x)

    x = k.layers.Conv3D(filters=output_size * 2, kernel_size=(3, 3, 3), strides=(1, 1, 1), padding='same')(x_shortcut)
    x = k.layers.BatchNormalization()(x)
    x = k.layers.PReLU()(x)
    x = k.layers.Dropout(0.5)(x)

    x = k.layers.Add()([x, x_shortcut])
    x = k.layers.PReLU()(x)

    x = k.layers.AveragePooling3D(pool_size=(2, 2, 2), strides=(2, 2, 2), padding="same")(x)

    x = k.layers.Conv3D(filters=output_size * 3, kernel_size=(1, 1, 1), strides=(1, 1, 1), padding='same')(x)
    x = k.layers.BatchNormalization()(x)
    x = k.layers.PReLU()(x)
    x_shortcut = k.layers.Dropout(0.5)(x)

    x = k.layers.Conv3D(filters=output_size * 3, kernel_size=(3, 3, 3), strides=(1, 1, 1), padding='same')(x_shortcut)
    x = k.layers.BatchNormalization()(x)
    x = k.layers.PReLU()(x)
    x = k.layers.Dropout(0.5)(x)

    x = k.layers.Add()([x, x_shortcut])
    x = k.layers.PReLU()(x)

    x = k.layers.AveragePooling3D(pool_size=(2, 2, 2), strides=(2, 2, 2), padding="same")(x)

    x = k.layers.Conv3D(filters=output_size * 4, kernel_size=(1, 1, 1), strides=(1, 1, 1), padding='same')(x)
    x = k.layers.BatchNormalization()(x)
    x = k.layers.PReLU()(x)
    x_shortcut = k.layers.Dropout(0.5)(x)

    x = k.layers.Conv3D(filters=output_size * 4, kernel_size=(3, 3, 3), strides=(1, 1, 1), padding='same')(x_shortcut)
    x = k.layers.BatchNormalization()(x)
    x = k.layers.PReLU()(x)
    x = k.layers.Dropout(0.5)(x)

    x = k.layers.Add()([x, x_shortcut])
    x = k.layers.PReLU()(x)

    x = k.layers.AveragePooling3D(pool_size=(2, 2, 2), strides=(2, 2, 2), padding="same")(x)

    x = k.layers.Conv3D(filters=output_size, kernel_size=(1, 1, 1), strides=(1, 1, 1), padding='same')(x)
    x = k.layers.BatchNormalization()(x)
    x = k.layers.PReLU()(x)
    x = k.layers.Dropout(0.5)(x)

    x = deep_fully_connected(x, True, 2, output_size)

    return x


def alexinceptionresnet_module_module_bottleneck(x, conv_filter_bottleneck):
    x = k.layers.Conv3D(filters=conv_filter_bottleneck, kernel_size=(1, 1, 1), strides=(1, 1, 1), padding="same")(x)
    x = k.layers.BatchNormalization()(x)
    x = k.layers.PReLU()(x)
    x = k.layers.Dropout(0.5)(x)

    return x


def alexinceptionresnet_module_module_module(x, conv_filter, conv_filter_bottleneck, iterations):
    x_shortcut = x

    x = k.layers.Conv3D(filters=int(conv_filter / 2), kernel_size=(1, 1, 1), strides=(1, 1, 1), padding="same")(x)
    x = k.layers.BatchNormalization()(x)
    x = k.layers.PReLU()(x)
    x = k.layers.Dropout(0.5)(x)

    for _ in range(iterations):
        x = k.layers.Conv3D(filters=conv_filter, kernel_size=(3, 3, 3), strides=(1, 1, 1), padding="same")(x)
        x = k.layers.BatchNormalization()(x)
        x = k.layers.PReLU()(x)
        x = k.layers.Dropout(0.5)(x)

    x = k.layers.Conv3D(filters=int(conv_filter / 2), kernel_size=(1, 1, 1), strides=(1, 1, 1), padding="same")(x)
    x = k.layers.BatchNormalization()(x)
    x = k.layers.PReLU()(x)
    x = k.layers.Dropout(0.5)(x)

    x = alexinceptionresnet_module_module_bottleneck(x, conv_filter_bottleneck)

    x = k.layers.Add()([x, x_shortcut])
    x = k.layers.PReLU()(x)

    return x


def alexinceptionresnet_module_module(x,
                                      conv_filter_1,
                                      conv_filter_2,
                                      conv_filter_3,
                                      conv_filter_bottleneck):
    x_shortcut = x

    x_1 = alexinceptionresnet_module_module_module(x, conv_filter_1, conv_filter_bottleneck, 0)
    x_2 = alexinceptionresnet_module_module_module(x, conv_filter_1, conv_filter_bottleneck, 1)
    x_3 = alexinceptionresnet_module_module_module(x, conv_filter_2, conv_filter_bottleneck, 2)
    x_4 = alexinceptionresnet_module_module_module(x, conv_filter_3, conv_filter_bottleneck, 3)

    x_5 = alexinceptionresnet_module_module_bottleneck(x, conv_filter_bottleneck)
    x_5 = k.layers.AveragePooling3D(pool_size=(3, 3, 3), strides=(1, 1, 1), padding="same")(x_5)
    x_5 = alexinceptionresnet_module_module_bottleneck(x_5, conv_filter_bottleneck)

    x = k.layers.Concatenate([x_1, x_2, x_3, x_4, x_5], axis=4)

    x = alexinceptionresnet_module_module_bottleneck(x, conv_filter_bottleneck)

    x = k.layers.Add()([x, x_shortcut])
    x = k.layers.PReLU()(x)

    return x


def alexinceptionresnet_module(x):
    for _ in range(1):
        x = alexinceptionresnet_module_module(x, 4, 8, 16, 64)

    for _ in range(1):
        x = alexinceptionresnet_module_module(x, 8, 16, 32, 64)

    for _ in range(1):
        x = alexinceptionresnet_module_module(x, 16, 32, 64, 64)

    for _ in range(1):
        x = alexinceptionresnet_module_module(x, 32, 64, 128, 64)

    for _ in range(0):
        x = alexinceptionresnet_module_module(x, 64, 128, 256, 64)

    return x


def alexinceptionresnet_conv_module(x):
    for _ in range(3):
        x = k.layers.Conv3D(filters=64, kernel_size=(3, 3, 3), strides=(2, 2, 2), padding="same")(x)
        x = k.layers.Activation("relu")(x)

    x = k.layers.AveragePooling3D(pool_size=(3, 3, 3), strides=(1, 1, 1), padding="same")(x)

    x_1 = alexinceptionresnet_module(x)
    x_2 = alexinceptionresnet_module(x)

    x = k.layers.Add()([x_1, x_2])
    x = k.layers.BatchNormalization()(x)
    x = k.layers.PReLU()(x)
    x = k.layers.Dropout(0.5)(x)

    x = k.layers.AveragePooling3D(pool_size=(3, 3, 3), strides=(1, 1, 1), padding="same")(x)

    return x


def alexinceptionresnet_conv(x):
    x = alexinceptionresnet_conv_module(x)

    x = k.layers.Flatten()(x)

    return x


def alexinceptionresnet_fully_connected(x, output_size):
    x = alexinceptionresnet_conv_module(x)

    x = k.layers.Conv3D(filters=1, kernel_size=(3, 3, 3), strides=(1, 1, 1), padding="same")(x)
    x = k.layers.BatchNormalization()(x)
    x = k.layers.PReLU()(x)
    x = k.layers.Dropout(0.5)(x)

    x = deep_fully_connected(x, True, 2, output_size)

    return x


# https://github.com/zizhaozhang/unet-tensorflow-keras/blob/master/model.py
def get_crop_shape(target, refer):
    # depth, the 4rd dimension
    cd = (target.get_shape()[3] - refer.get_shape()[3]).value

    if cd % 2 != 0:
        cd1, cd2 = int(cd / 2), int(cd / 2) + 1
    else:
        cd1, cd2 = int(cd / 2), int(cd / 2)

    # width, the 3rd dimension
    cw = (target.get_shape()[2] - refer.get_shape()[2]).value

    if cw % 2 != 0:
        cw1, cw2 = int(cw / 2), int(cw / 2) + 1
    else:
        cw1, cw2 = int(cw / 2), int(cw / 2)

    # height, the 2nd dimension
    ch = (target.get_shape()[1] - refer.get_shape()[1]).value

    if ch % 2 != 0:
        ch1, ch2 = int(ch / 2), int(ch / 2) + 1
    else:
        ch1, ch2 = int(ch / 2), int(ch / 2)

    return (ch1, ch2), (cw1, cw2), (cd1, cd2)


def unet(x):
    x = k.layers.Conv3D(filters=1, kernel_size=(3, 3, 3), strides=(1, 1, 1), padding='same')(x)
    x = k.layers.BatchNormalization()(x)
    x = k.layers.PReLU()(x)
    x = k.layers.Dropout(0.5)(x)

    x = k.layers.Conv3D(filters=64, kernel_size=(3, 3, 3), strides=(1, 1, 1), padding='same')(x)
    x = k.layers.BatchNormalization()(x)
    x = k.layers.PReLU()(x)
    x = k.layers.Dropout(0.5)(x)

    x = k.layers.Conv3D(filters=64, kernel_size=(3, 3, 3), strides=(1, 1, 1), padding='same')(x)
    x = k.layers.BatchNormalization()(x)
    x = k.layers.PReLU()(x)
    x_shortcut_1 = k.layers.Dropout(0.5)(x)

    x = k.layers.MaxPooling3D(pool_size=(2, 2, 2), strides=(2, 2, 2), padding="same")(x_shortcut_1)

    x = k.layers.Conv3D(filters=64, kernel_size=(3, 3, 3), strides=(1, 1, 1), padding='same')(x)
    x = k.layers.BatchNormalization()(x)
    x = k.layers.PReLU()(x)
    x = k.layers.Dropout(0.5)(x)

    x = k.layers.Conv3D(filters=128, kernel_size=(3, 3, 3), strides=(1, 1, 1), padding='same')(x)
    x = k.layers.BatchNormalization()(x)
    x = k.layers.PReLU()(x)
    x = k.layers.Dropout(0.5)(x)

    x = k.layers.Conv3D(filters=128, kernel_size=(3, 3, 3), strides=(1, 1, 1), padding='same')(x)
    x = k.layers.BatchNormalization()(x)
    x = k.layers.PReLU()(x)
    x_shortcut_2 = k.layers.Dropout(0.5)(x)

    x = k.layers.MaxPooling3D(pool_size=(2, 2, 2), strides=(2, 2, 2), padding="same")(x_shortcut_2)

    x = k.layers.Conv3D(filters=128, kernel_size=(3, 3, 3), strides=(1, 1, 1), padding='same')(x)
    x = k.layers.BatchNormalization()(x)
    x = k.layers.PReLU()(x)
    x = k.layers.Dropout(0.5)(x)

    x = k.layers.Conv3D(filters=256, kernel_size=(3, 3, 3), strides=(1, 1, 1), padding='same')(x)
    x = k.layers.BatchNormalization()(x)
    x = k.layers.PReLU()(x)
    x = k.layers.Dropout(0.5)(x)

    x = k.layers.Conv3D(filters=256, kernel_size=(3, 3, 3), strides=(1, 1, 1), padding='same')(x)
    x = k.layers.BatchNormalization()(x)
    x = k.layers.PReLU()(x)
    x_shortcut_3 = k.layers.Dropout(0.5)(x)

    x = k.layers.MaxPooling3D(pool_size=(2, 2, 2), strides=(2, 2, 2), padding="same")(x_shortcut_3)

    x = k.layers.Conv3D(filters=256, kernel_size=(3, 3, 3), strides=(1, 1, 1), padding='same')(x)
    x = k.layers.BatchNormalization()(x)
    x = k.layers.PReLU()(x)
    x = k.layers.Dropout(0.5)(x)

    x = k.layers.Conv3D(filters=512, kernel_size=(3, 3, 3), strides=(1, 1, 1), padding='same')(x)
    x = k.layers.BatchNormalization()(x)
    x = k.layers.PReLU()(x)
    x = k.layers.Dropout(0.5)(x)

    x = k.layers.Conv3D(filters=512, kernel_size=(3, 3, 3), strides=(1, 1, 1), padding='same')(x)
    x = k.layers.BatchNormalization()(x)
    x = k.layers.PReLU()(x)
    x_shortcut_4 = k.layers.Dropout(0.5)(x)

    x = k.layers.MaxPooling3D(pool_size=(2, 2, 2), strides=(2, 2, 2), padding="same")(x_shortcut_4)

    x = k.layers.Conv3D(filters=512, kernel_size=(3, 3, 3), strides=(1, 1, 1), padding='same')(x)
    x = k.layers.BatchNormalization()(x)
    x = k.layers.PReLU()(x)
    x = k.layers.Dropout(0.5)(x)

    for _ in range(2):
        x = k.layers.Conv3D(filters=1024, kernel_size=(3, 3, 3), strides=(1, 1, 1), padding='same')(x)
        x = k.layers.BatchNormalization()(x)
        x = k.layers.PReLU()(x)
        x = k.layers.Dropout(0.5)(x)

    x = k.layers.UpSampling3D(size=(2, 2, 2))(x)

    ch, cw, cd = get_crop_shape(x, x_shortcut_4)
    x = k.layers.Cropping3D(cropping=(ch, cw, cd))(x)
    x = k.layers.concatenate([x, x_shortcut_4], axis=4)

    x = k.layers.Conv3D(filters=1024, kernel_size=(3, 3, 3), strides=(1, 1, 1), padding='same')(x)
    x = k.layers.BatchNormalization()(x)
    x = k.layers.PReLU()(x)
    x = k.layers.Dropout(0.5)(x)

    for _ in range(2):
        x = k.layers.Conv3D(filters=512, kernel_size=(3, 3, 3), strides=(1, 1, 1), padding='same')(x)
        x = k.layers.BatchNormalization()(x)
        x = k.layers.PReLU()(x)
        x = k.layers.Dropout(0.5)(x)

    x = k.layers.UpSampling3D(size=(2, 2, 2))(x)

    ch, cw, cd = get_crop_shape(x, x_shortcut_3)
    x = k.layers.Cropping3D(cropping=(ch, cw, cd))(x)
    x = k.layers.concatenate([x, x_shortcut_3], axis=4)

    x = k.layers.Conv3D(filters=512, kernel_size=(3, 3, 3), strides=(1, 1, 1), padding='same')(x)
    x = k.layers.BatchNormalization()(x)
    x = k.layers.PReLU()(x)
    x = k.layers.Dropout(0.5)(x)

    for _ in range(2):
        x = k.layers.Conv3D(filters=256, kernel_size=(3, 3, 3), strides=(1, 1, 1), padding='same')(x)
        x = k.layers.BatchNormalization()(x)
        x = k.layers.PReLU()(x)
        x = k.layers.Dropout(0.5)(x)

    x = k.layers.UpSampling3D(size=(2, 2, 2))(x)

    ch, cw, cd = get_crop_shape(x, x_shortcut_2)
    x = k.layers.Cropping3D(cropping=(ch, cw, cd))(x)
    x = k.layers.concatenate([x, x_shortcut_2], axis=4)

    x = k.layers.Conv3D(filters=256, kernel_size=(3, 3, 3), strides=(1, 1, 1), padding='same')(x)
    x = k.layers.BatchNormalization()(x)
    x = k.layers.PReLU()(x)
    x = k.layers.Dropout(0.5)(x)

    for _ in range(2):
        x = k.layers.Conv3D(filters=128, kernel_size=(3, 3, 3), strides=(1, 1, 1), padding='same')(x)
        x = k.layers.BatchNormalization()(x)
        x = k.layers.PReLU()(x)
        x = k.layers.Dropout(0.5)(x)

    x = k.layers.UpSampling3D(size=(2, 2, 2))(x)

    ch, cw, cd = get_crop_shape(x, x_shortcut_1)
    x = k.layers.Cropping3D(cropping=(ch, cw, cd))(x)
    x = k.layers.concatenate([x, x_shortcut_1], axis=4)

    x = k.layers.Conv3D(filters=128, kernel_size=(3, 3, 3), strides=(1, 1, 1), padding='same')(x)
    x = k.layers.BatchNormalization()(x)
    x = k.layers.PReLU()(x)
    x = k.layers.Dropout(0.5)(x)

    for _ in range(2):
        x = k.layers.Conv3D(filters=64, kernel_size=(3, 3, 3), strides=(1, 1, 1), padding='same')(x)
        x = k.layers.BatchNormalization()(x)
        x = k.layers.PReLU()(x)
        x = k.layers.Dropout(0.5)(x)

    x = k.layers.Conv3D(filters=2, kernel_size=(3, 3, 3), strides=(1, 1, 1), padding='same')(x)
    x = k.layers.BatchNormalization()(x)
    x = k.layers.PReLU()(x)
    x = k.layers.Dropout(0.5)(x)

    x = k.layers.Flatten()(x)

    return x


def voxelmorph(x):
    x = k.layers.Conv3D(filters=16, kernel_size=(3, 3, 3), strides=(1, 1, 1), padding='same')(x)
    x = k.layers.BatchNormalization()(x)
    x = k.layers.PReLU()(x)
    x_shortcut_1 = k.layers.Dropout(0.5)(x)

    x = k.layers.Conv3D(filters=32, kernel_size=(3, 3, 3), strides=(2, 2, 2), padding='same')(x_shortcut_1)
    x = k.layers.BatchNormalization()(x)
    x = k.layers.PReLU()(x)
    x_shortcut_2 = k.layers.Dropout(0.5)(x)

    x = k.layers.Conv3D(filters=32, kernel_size=(3, 3, 3), strides=(2, 2, 2), padding='same')(x_shortcut_2)
    x = k.layers.BatchNormalization()(x)
    x = k.layers.PReLU()(x)
    x_shortcut_3 = k.layers.Dropout(0.5)(x)

    x = k.layers.Conv3D(filters=32, kernel_size=(3, 3, 3), strides=(2, 2, 2), padding='same')(x_shortcut_3)
    x = k.layers.BatchNormalization()(x)
    x = k.layers.PReLU()(x)
    x_shortcut_4 = k.layers.Dropout(0.5)(x)

    x = k.layers.Conv3D(filters=32, kernel_size=(3, 3, 3), strides=(2, 2, 2), padding='same')(x_shortcut_4)
    x = k.layers.BatchNormalization()(x)
    x = k.layers.PReLU()(x)
    x = k.layers.Dropout(0.5)(x)

    x = k.layers.UpSampling3D(size=(2, 2, 2))(x)

    x = k.layers.Conv3D(filters=32, kernel_size=(3, 3, 3), strides=(1, 1, 1), padding='same')(x)
    x = k.layers.BatchNormalization()(x)
    x = k.layers.PReLU()(x)
    x = k.layers.Dropout(0.5)(x)

    ch, cw, cd = get_crop_shape(x, x_shortcut_4)
    x = k.layers.Cropping3D(cropping=(ch, cw, cd))(x)
    x = k.layers.concatenate([x, x_shortcut_4], axis=4)

    x = k.layers.UpSampling3D(size=(2, 2, 2))(x)

    x = k.layers.Conv3D(filters=32, kernel_size=(3, 3, 3), strides=(1, 1, 1), padding='same')(x)
    x = k.layers.BatchNormalization()(x)
    x = k.layers.PReLU()(x)
    x = k.layers.Dropout(0.5)(x)

    ch, cw, cd = get_crop_shape(x, x_shortcut_3)
    x = k.layers.Cropping3D(cropping=(ch, cw, cd))(x)
    x = k.layers.concatenate([x, x_shortcut_3], axis=4)

    x = k.layers.UpSampling3D(size=(2, 2, 2))(x)

    x = k.layers.Conv3D(filters=32, kernel_size=(3, 3, 3), strides=(1, 1, 1), padding='same')(x)
    x = k.layers.BatchNormalization()(x)
    x = k.layers.PReLU()(x)
    x = k.layers.Dropout(0.5)(x)

    ch, cw, cd = get_crop_shape(x, x_shortcut_2)
    x = k.layers.Cropping3D(cropping=(ch, cw, cd))(x)
    x = k.layers.concatenate([x, x_shortcut_2], axis=4)

    x = k.layers.UpSampling3D(size=(2, 2, 2))(x)

    x = k.layers.Conv3D(filters=32, kernel_size=(3, 3, 3), strides=(1, 1, 1), padding='same')(x)
    x = k.layers.BatchNormalization()(x)
    x = k.layers.PReLU()(x)
    x = k.layers.Dropout(0.5)(x)

    ch, cw, cd = get_crop_shape(x, x_shortcut_1)
    x = k.layers.Cropping3D(cropping=(ch, cw, cd))(x)
    x = k.layers.concatenate([x, x_shortcut_1], axis=4)
    x = k.layers.PReLU()(x)

    x = k.layers.Conv3D(filters=16, kernel_size=(3, 3, 3), strides=(1, 1, 1), padding='same')(x)
    x = k.layers.BatchNormalization()(x)
    x = k.layers.PReLU()(x)
    x = k.layers.Dropout(0.5)(x)

    x = k.layers.Conv3D(filters=16, kernel_size=(3, 3, 3), strides=(1, 1, 1), padding='same')(x)
    x = k.layers.BatchNormalization()(x)
    x = k.layers.PReLU()(x)
    x = k.layers.Dropout(0.5)(x)

    x = k.layers.Conv3D(filters=3, kernel_size=(3, 3, 3), strides=(1, 1, 1), padding='same')(x)
    x = k.layers.BatchNormalization()(x)
    x = k.layers.PReLU()(x)
    x = k.layers.Dropout(0.5)(x)

    x = k.layers.Flatten()(x)

    return x


def resvoxelmorph(x, output_size):
    x = k.layers.Conv3D(filters=1, kernel_size=(1, 1, 1), strides=(1, 1, 1), padding='same')(x)
    x = k.layers.BatchNormalization()(x)
    x = k.layers.PReLU()(x)
    x = k.layers.Dropout(0.5)(x)

    x = k.layers.Conv3D(filters=8, kernel_size=(3, 3, 3), strides=(1, 1, 1), padding='same')(x)
    x = k.layers.BatchNormalization()(x)
    x = k.layers.PReLU()(x)
    x_shortcut_1 = k.layers.Dropout(0.5)(x)

    x = k.layers.AveragePooling3D(pool_size=(2, 2, 2), strides=(2, 2, 2), padding="same")(x_shortcut_1)

    x = k.layers.Conv3D(filters=16, kernel_size=(3, 3, 3), strides=(1, 1, 1), padding='same')(x)
    x = k.layers.BatchNormalization()(x)
    x = k.layers.PReLU()(x)
    x_shortcut_2 = k.layers.Dropout(0.5)(x)

    x = k.layers.AveragePooling3D(pool_size=(2, 2, 2), strides=(2, 2, 2), padding="same")(x_shortcut_2)

    x = k.layers.Conv3D(filters=32, kernel_size=(3, 3, 3), strides=(1, 1, 1), padding='same')(x)
    x = k.layers.BatchNormalization()(x)
    x = k.layers.PReLU()(x)
    x_shortcut_3 = k.layers.Dropout(0.5)(x)

    x = k.layers.AveragePooling3D(pool_size=(2, 2, 2), strides=(2, 2, 2), padding="same")(x_shortcut_3)

    x = k.layers.Conv3D(filters=64, kernel_size=(3, 3, 3), strides=(1, 1, 1), padding='same')(x)
    x = k.layers.BatchNormalization()(x)
    x = k.layers.PReLU()(x)
    x_shortcut_4 = k.layers.Dropout(0.5)(x)

    x = k.layers.AveragePooling3D(pool_size=(2, 2, 2), strides=(2, 2, 2), padding="same")(x_shortcut_4)

    x = k.layers.Conv3D(filters=64, kernel_size=(3, 3, 3), strides=(1, 1, 1), padding='same')(x)
    x = k.layers.BatchNormalization()(x)
    x = k.layers.PReLU()(x)
    x = k.layers.Dropout(0.5)(x)

    x = k.layers.UpSampling3D(size=(2, 2, 2))(x)

    x = k.layers.Conv3D(filters=64, kernel_size=(3, 3, 3), strides=(1, 1, 1), padding='same')(x)
    x = k.layers.BatchNormalization()(x)
    x = k.layers.PReLU()(x)
    x = k.layers.Dropout(0.5)(x)

    ch, cw, cd = get_crop_shape(x, x_shortcut_4)
    x = k.layers.Cropping3D(cropping=(ch, cw, cd))(x)
    x = k.layers.concatenate([x, x_shortcut_4], axis=4)
    x = k.layers.PReLU()(x)

    x = k.layers.UpSampling3D(size=(2, 2, 2))(x)

    x = k.layers.Conv3D(filters=32, kernel_size=(3, 3, 3), strides=(1, 1, 1), padding='same')(x)
    x = k.layers.BatchNormalization()(x)
    x = k.layers.PReLU()(x)
    x = k.layers.Dropout(0.5)(x)

    ch, cw, cd = get_crop_shape(x, x_shortcut_3)
    x = k.layers.Cropping3D(cropping=(ch, cw, cd))(x)
    x = k.layers.concatenate([x, x_shortcut_3], axis=4)
    x = k.layers.PReLU()(x)

    x = k.layers.UpSampling3D(size=(2, 2, 2))(x)

    x = k.layers.Conv3D(filters=16, kernel_size=(3, 3, 3), strides=(1, 1, 1), padding='same')(x)
    x = k.layers.BatchNormalization()(x)
    x = k.layers.PReLU()(x)
    x = k.layers.Dropout(0.5)(x)

    ch, cw, cd = get_crop_shape(x, x_shortcut_2)
    x = k.layers.Cropping3D(cropping=(ch, cw, cd))(x)
    x = k.layers.concatenate([x, x_shortcut_2], axis=4)
    x = k.layers.PReLU()(x)

    x = k.layers.UpSampling3D(size=(2, 2, 2))(x)

    x = k.layers.Conv3D(filters=8, kernel_size=(3, 3, 3), strides=(1, 1, 1), padding='same')(x)
    x = k.layers.BatchNormalization()(x)
    x = k.layers.PReLU()(x)
    x = k.layers.Dropout(0.5)(x)

    ch, cw, cd = get_crop_shape(x, x_shortcut_1)
    x = k.layers.Cropping3D(cropping=(ch, cw, cd))(x)
    x = k.layers.concatenate([x, x_shortcut_1], axis=4)
    x = k.layers.PReLU()(x)

    x = k.layers.Conv3D(filters=8, kernel_size=(1, 1, 1), strides=(1, 1, 1), padding='same')(x)
    x = k.layers.BatchNormalization()(x)
    x = k.layers.PReLU()(x)
    x_shortcut = k.layers.Dropout(0.5)(x)

    x = k.layers.Conv3D(filters=8, kernel_size=(3, 3, 3), strides=(1, 1, 1), padding='same')(x_shortcut)
    x = k.layers.BatchNormalization()(x)
    x = k.layers.PReLU()(x)
    x = k.layers.Dropout(0.5)(x)

    x = k.layers.Add()([x, x_shortcut])
    x = k.layers.PReLU()(x)

    x = k.layers.AveragePooling3D(pool_size=(2, 2, 2), strides=(2, 2, 2), padding="same")(x)

    x = k.layers.Conv3D(filters=16, kernel_size=(1, 1, 1), strides=(1, 1, 1), padding='same')(x)
    x = k.layers.BatchNormalization()(x)
    x = k.layers.PReLU()(x)
    x_shortcut = k.layers.Dropout(0.5)(x)

    x = k.layers.Conv3D(filters=16, kernel_size=(3, 3, 3), strides=(1, 1, 1), padding='same')(x_shortcut)
    x = k.layers.BatchNormalization()(x)
    x = k.layers.PReLU()(x)
    x = k.layers.Dropout(0.5)(x)

    x = k.layers.Add()([x, x_shortcut])
    x = k.layers.PReLU()(x)

    x = k.layers.AveragePooling3D(pool_size=(2, 2, 2), strides=(2, 2, 2), padding="same")(x)

    x = k.layers.Conv3D(filters=32, kernel_size=(1, 1, 1), strides=(1, 1, 1), padding='same')(x)
    x = k.layers.BatchNormalization()(x)
    x = k.layers.PReLU()(x)
    x_shortcut = k.layers.Dropout(0.5)(x)

    x = k.layers.Conv3D(filters=32, kernel_size=(3, 3, 3), strides=(1, 1, 1), padding='same')(x_shortcut)
    x = k.layers.BatchNormalization()(x)
    x = k.layers.PReLU()(x)
    x = k.layers.Dropout(0.5)(x)

    x = k.layers.Add()([x, x_shortcut])
    x = k.layers.PReLU()(x)

    x = k.layers.AveragePooling3D(pool_size=(2, 2, 2), strides=(2, 2, 2), padding="same")(x)

    x = k.layers.Conv3D(filters=64, kernel_size=(1, 1, 1), strides=(1, 1, 1), padding='same')(x)
    x = k.layers.BatchNormalization()(x)
    x = k.layers.PReLU()(x)
    x_shortcut = k.layers.Dropout(0.5)(x)

    x = k.layers.Conv3D(filters=64, kernel_size=(3, 3, 3), strides=(1, 1, 1), padding='same')(x_shortcut)
    x = k.layers.BatchNormalization()(x)
    x = k.layers.PReLU()(x)
    x = k.layers.Dropout(0.5)(x)

    x = k.layers.Add()([x, x_shortcut])
    x = k.layers.PReLU()(x)

    x = k.layers.AveragePooling3D(pool_size=(2, 2, 2), strides=(2, 2, 2), padding="same")(x)

    x = k.layers.Conv3D(filters=64, kernel_size=(3, 3, 3), strides=(1, 1, 1), padding='same')(x)
    x = k.layers.BatchNormalization()(x)
    x = k.layers.PReLU()(x)
    x = k.layers.Dropout(0.5)(x)

    x = k.layers.Conv3D(filters=1, kernel_size=(1, 1, 1), strides=(1, 1, 1), padding='same')(x)
    x = k.layers.BatchNormalization()(x)
    x = k.layers.PReLU()(x)
    x = k.layers.Dropout(0.5)(x)

    x = deep_fully_connected(x, True, 2, output_size)

    return x


def rnn(x, output_size):
    x = k.layers.Reshape((x.shape.as_list()[1] * x.shape.as_list()[2] * x.shape.as_list()[3], x.shape.as_list()[4]))(x)

    x = k.layers.LSTM(output_size, return_sequences=True)(x)
    x = k.layers.BatchNormalization()(x)
    x = k.layers.PReLU()(x)
    x = k.layers.Dropout(0.5)(x)

    x = k.layers.Flatten()(x)

    return x


def deep_rnn(x, rnn_size, output_size):
    x = k.layers.Reshape((x.shape.as_list()[1] * x.shape.as_list()[2] * x.shape.as_list()[3], x.shape.as_list()[4]))(x)

    for _ in range(rnn_size):
        x = k.layers.LSTM(output_size, return_sequences=True)(x)
        x = k.layers.BatchNormalization()(x)
        x = k.layers.PReLU()(x)
        x = k.layers.Dropout(0.5)(x)

    x = k.layers.Flatten()(x)

    return x


def deep_fully_connected_rnn(x, rnn_size, fully_connected_bool, size, output_size):
    x = k.layers.Reshape((x.shape.as_list()[1] * x.shape.as_list()[2] * x.shape.as_list()[3], x.shape.as_list()[4]))(x)

    for _ in range(rnn_size):
        x = k.layers.LSTM(output_size, return_sequences=True)(x)
        x = k.layers.BatchNormalization()(x)
        x = k.layers.PReLU()(x)
        x = k.layers.Dropout(0.5)(x)

    x = k.layers.Flatten()(x)

    x = deep_fully_connected(x, fully_connected_bool, size, output_size)

    return x


def conv_fully_connected_rnn(x, output_size):
    x = k.layers.Conv3D(filters=64, kernel_size=(3, 3, 3), strides=(1, 1, 1), padding='same')(x)
    x = k.layers.BatchNormalization()(x)
    x = k.layers.PReLU()(x)
    x = k.layers.Dropout(0.5)(x)

    x = deep_fully_connected_rnn(x, 2, True, 0, output_size)

    return x


def deep_conv_fully_connected_rnn(x, output_size):
    for _ in range(2):
        x = k.layers.Conv3D(filters=64, kernel_size=(3, 3, 3), strides=(1, 1, 1), padding='same')(x)
        x = k.layers.BatchNormalization()(x)
        x = k.layers.PReLU()(x)
        x = k.layers.Dropout(0.5)(x)

    x = deep_fully_connected_rnn(x, 2, True, 0, output_size)

    return x


def papernet_rnn(x, output_size):
    x = k.layers.Conv3D(filters=output_size, kernel_size=(1, 1, 1), strides=(1, 1, 1), padding='same')(x)
    x = k.layers.BatchNormalization()(x)
    x = k.layers.PReLU()(x)
    x = k.layers.Dropout(0.5)(x)

    x = k.layers.Conv3D(filters=output_size, kernel_size=(3, 3, 3), strides=(1, 1, 1), padding='same')(x)
    x = k.layers.BatchNormalization()(x)
    x = k.layers.PReLU()(x)
    x = k.layers.Dropout(0.5)(x)

    x = k.layers.AveragePooling3D(pool_size=(2, 2, 2), strides=(2, 2, 2), padding="same")(x)

    x = k.layers.Conv3D(filters=output_size * 2, kernel_size=(3, 3, 3), strides=(1, 1, 1), padding='same')(x)
    x = k.layers.BatchNormalization()(x)
    x = k.layers.PReLU()(x)
    x = k.layers.Dropout(0.5)(x)

    x = k.layers.AveragePooling3D(pool_size=(2, 2, 2), strides=(2, 2, 2), padding="same")(x)

    x = k.layers.Conv3D(filters=output_size, kernel_size=(1, 1, 1), strides=(1, 1, 1), padding='same')(x)
    x = k.layers.BatchNormalization()(x)
    x = k.layers.PReLU()(x)
    x = k.layers.Dropout(0.5)(x)

    x = deep_fully_connected_rnn(x, 2, True, 0, output_size)

    return x


def resnet_rnn(x, output_size):
    x = k.layers.Conv3D(filters=output_size, kernel_size=(1, 1, 1), strides=(1, 1, 1), padding='same')(x)
    x = k.layers.BatchNormalization()(x)
    x = k.layers.PReLU()(x)
    x_shortcut = k.layers.Dropout(0.5)(x)

    x = k.layers.Conv3D(filters=output_size, kernel_size=(3, 3, 3), strides=(1, 1, 1), padding='same')(x_shortcut)
    x = k.layers.BatchNormalization()(x)
    x = k.layers.PReLU()(x)
    x = k.layers.Dropout(0.5)(x)

    x = k.layers.Add()([x, x_shortcut])
    x = k.layers.PReLU()(x)

    x = k.layers.AveragePooling3D(pool_size=(2, 2, 2), strides=(2, 2, 2), padding="same")(x)

    x = k.layers.Conv3D(filters=output_size * 2, kernel_size=(1, 1, 1), strides=(1, 1, 1), padding='same')(x)
    x = k.layers.BatchNormalization()(x)
    x = k.layers.PReLU()(x)
    x_shortcut = k.layers.Dropout(0.5)(x)

    x = k.layers.Conv3D(filters=output_size * 2, kernel_size=(3, 3, 3), strides=(1, 1, 1), padding='same')(x_shortcut)
    x = k.layers.BatchNormalization()(x)
    x = k.layers.PReLU()(x)
    x = k.layers.Dropout(0.5)(x)

    x = k.layers.Add()([x, x_shortcut])
    x = k.layers.PReLU()(x)

    x = k.layers.AveragePooling3D(pool_size=(2, 2, 2), strides=(2, 2, 2), padding="same")(x)

    x = k.layers.Conv3D(filters=output_size * 3, kernel_size=(1, 1, 1), strides=(1, 1, 1), padding='same')(x)
    x = k.layers.BatchNormalization()(x)
    x = k.layers.PReLU()(x)
    x_shortcut = k.layers.Dropout(0.5)(x)

    x = k.layers.Conv3D(filters=output_size * 3, kernel_size=(3, 3, 3), strides=(1, 1, 1), padding='same')(x_shortcut)
    x = k.layers.BatchNormalization()(x)
    x = k.layers.PReLU()(x)
    x = k.layers.Dropout(0.5)(x)

    x = k.layers.Add()([x, x_shortcut])
    x = k.layers.PReLU()(x)

    x = k.layers.AveragePooling3D(pool_size=(2, 2, 2), strides=(2, 2, 2), padding="same")(x)

    x = k.layers.Conv3D(filters=output_size * 4, kernel_size=(1, 1, 1), strides=(1, 1, 1), padding='same')(x)
    x = k.layers.BatchNormalization()(x)
    x = k.layers.PReLU()(x)
    x_shortcut = k.layers.Dropout(0.5)(x)

    x = k.layers.Conv3D(filters=output_size * 4, kernel_size=(3, 3, 3), strides=(1, 1, 1), padding='same')(x_shortcut)
    x = k.layers.BatchNormalization()(x)
    x = k.layers.PReLU()(x)
    x = k.layers.Dropout(0.5)(x)

    x = k.layers.Add()([x, x_shortcut])
    x = k.layers.PReLU()(x)

    x = k.layers.AveragePooling3D(pool_size=(2, 2, 2), strides=(2, 2, 2), padding="same")(x)

    x = k.layers.Conv3D(filters=output_size, kernel_size=(1, 1, 1), strides=(1, 1, 1), padding='same')(x)
    x = k.layers.BatchNormalization()(x)
    x = k.layers.PReLU()(x)
    x = k.layers.Dropout(0.5)(x)

    x = deep_fully_connected_rnn(x, 2, True, 0, output_size)

    return x


def resvoxelmorph_rnn(x, output_size):
    x = k.layers.Conv3D(filters=output_size, kernel_size=(1, 1, 1), strides=(1, 1, 1), padding='same')(x)
    x = k.layers.BatchNormalization()(x)
    x = k.layers.PReLU()(x)
    x = k.layers.Dropout(0.5)(x)

    x = k.layers.Conv3D(filters=output_size, kernel_size=(3, 3, 3), strides=(1, 1, 1), padding='same')(x)
    x = k.layers.BatchNormalization()(x)
    x = k.layers.PReLU()(x)
    x_shortcut_1 = k.layers.Dropout(0.5)(x)

    x = k.layers.AveragePooling3D(pool_size=(2, 2, 2), strides=(2, 2, 2), padding="same")(x_shortcut_1)

    x = k.layers.Conv3D(filters=output_size * 2, kernel_size=(3, 3, 3), strides=(1, 1, 1), padding='same')(x)
    x = k.layers.BatchNormalization()(x)
    x = k.layers.PReLU()(x)
    x_shortcut_2 = k.layers.Dropout(0.5)(x)

    x = k.layers.AveragePooling3D(pool_size=(2, 2, 2), strides=(2, 2, 2), padding="same")(x_shortcut_2)

    x = k.layers.Conv3D(filters=output_size * 3, kernel_size=(3, 3, 3), strides=(1, 1, 1), padding='same')(x)
    x = k.layers.BatchNormalization()(x)
    x = k.layers.PReLU()(x)
    x_shortcut_3 = k.layers.Dropout(0.5)(x)

    x = k.layers.AveragePooling3D(pool_size=(2, 2, 2), strides=(2, 2, 2), padding="same")(x_shortcut_3)

    x = k.layers.Conv3D(filters=output_size * 4, kernel_size=(3, 3, 3), strides=(1, 1, 1), padding='same')(x)
    x = k.layers.BatchNormalization()(x)
    x = k.layers.PReLU()(x)
    x_shortcut_4 = k.layers.Dropout(0.5)(x)

    x = k.layers.AveragePooling3D(pool_size=(2, 2, 2), strides=(2, 2, 2), padding="same")(x_shortcut_4)

    x = k.layers.Conv3D(filters=output_size * 4, kernel_size=(3, 3, 3), strides=(1, 1, 1), padding='same')(x)
    x = k.layers.BatchNormalization()(x)
    x = k.layers.PReLU()(x)
    x = k.layers.Dropout(0.5)(x)

    x = k.layers.UpSampling3D(size=(2, 2, 2))(x)

    x = k.layers.Conv3D(filters=output_size * 4, kernel_size=(3, 3, 3), strides=(1, 1, 1), padding='same')(x)
    x = k.layers.BatchNormalization()(x)
    x = k.layers.PReLU()(x)
    x = k.layers.Dropout(0.5)(x)

    ch, cw, cd = get_crop_shape(x, x_shortcut_4)
    x = k.layers.Cropping3D(cropping=(ch, cw, cd))(x)
    x = k.layers.concatenate([x, x_shortcut_4], axis=4)
    x = k.layers.PReLU()(x)

    x = k.layers.UpSampling3D(size=(2, 2, 2))(x)

    x = k.layers.Conv3D(filters=output_size * 3, kernel_size=(3, 3, 3), strides=(1, 1, 1), padding='same')(x)
    x = k.layers.BatchNormalization()(x)
    x = k.layers.PReLU()(x)
    x = k.layers.Dropout(0.5)(x)

    ch, cw, cd = get_crop_shape(x, x_shortcut_3)
    x = k.layers.Cropping3D(cropping=(ch, cw, cd))(x)
    x = k.layers.concatenate([x, x_shortcut_3], axis=4)
    x = k.layers.PReLU()(x)

    x = k.layers.UpSampling3D(size=(2, 2, 2))(x)

    x = k.layers.Conv3D(filters=output_size * 2, kernel_size=(3, 3, 3), strides=(1, 1, 1), padding='same')(x)
    x = k.layers.BatchNormalization()(x)
    x = k.layers.PReLU()(x)
    x = k.layers.Dropout(0.5)(x)

    ch, cw, cd = get_crop_shape(x, x_shortcut_2)
    x = k.layers.Cropping3D(cropping=(ch, cw, cd))(x)
    x = k.layers.concatenate([x, x_shortcut_2], axis=4)
    x = k.layers.PReLU()(x)

    x = k.layers.UpSampling3D(size=(2, 2, 2))(x)

    x = k.layers.Conv3D(filters=output_size, kernel_size=(3, 3, 3), strides=(1, 1, 1), padding='same')(x)
    x = k.layers.BatchNormalization()(x)
    x = k.layers.PReLU()(x)
    x = k.layers.Dropout(0.5)(x)

    ch, cw, cd = get_crop_shape(x, x_shortcut_1)
    x = k.layers.Cropping3D(cropping=(ch, cw, cd))(x)
    x = k.layers.concatenate([x, x_shortcut_1], axis=4)
    x = k.layers.PReLU()(x)

    x = k.layers.Conv3D(filters=output_size, kernel_size=(1, 1, 1), strides=(1, 1, 1), padding='same')(x)
    x = k.layers.BatchNormalization()(x)
    x = k.layers.PReLU()(x)
    x_shortcut = k.layers.Dropout(0.5)(x)

    x = k.layers.Conv3D(filters=output_size, kernel_size=(3, 3, 3), strides=(1, 1, 1), padding='same')(x_shortcut)
    x = k.layers.BatchNormalization()(x)
    x = k.layers.PReLU()(x)
    x = k.layers.Dropout(0.5)(x)

    x = k.layers.Add()([x, x_shortcut])
    x = k.layers.PReLU()(x)

    x = k.layers.AveragePooling3D(pool_size=(2, 2, 2), strides=(2, 2, 2), padding="same")(x)

    x = k.layers.Conv3D(filters=output_size * 2, kernel_size=(1, 1, 1), strides=(1, 1, 1), padding='same')(x)
    x = k.layers.BatchNormalization()(x)
    x = k.layers.PReLU()(x)
    x_shortcut = k.layers.Dropout(0.5)(x)

    x = k.layers.Conv3D(filters=output_size * 2, kernel_size=(3, 3, 3), strides=(1, 1, 1), padding='same')(x_shortcut)
    x = k.layers.BatchNormalization()(x)
    x = k.layers.PReLU()(x)
    x = k.layers.Dropout(0.5)(x)

    x = k.layers.Add()([x, x_shortcut])
    x = k.layers.PReLU()(x)

    x = k.layers.AveragePooling3D(pool_size=(2, 2, 2), strides=(2, 2, 2), padding="same")(x)

    x = k.layers.Conv3D(filters=output_size * 3, kernel_size=(1, 1, 1), strides=(1, 1, 1), padding='same')(x)
    x = k.layers.BatchNormalization()(x)
    x = k.layers.PReLU()(x)
    x_shortcut = k.layers.Dropout(0.5)(x)

    x = k.layers.Conv3D(filters=output_size * 3, kernel_size=(3, 3, 3), strides=(1, 1, 1), padding='same')(x_shortcut)
    x = k.layers.BatchNormalization()(x)
    x = k.layers.PReLU()(x)
    x = k.layers.Dropout(0.5)(x)

    x = k.layers.Add()([x, x_shortcut])
    x = k.layers.PReLU()(x)

    x = k.layers.AveragePooling3D(pool_size=(2, 2, 2), strides=(2, 2, 2), padding="same")(x)

    x = k.layers.Conv3D(filters=output_size * 4, kernel_size=(1, 1, 1), strides=(1, 1, 1), padding='same')(x)
    x = k.layers.BatchNormalization()(x)
    x = k.layers.PReLU()(x)
    x_shortcut = k.layers.Dropout(0.5)(x)

    x = k.layers.Conv3D(filters=output_size * 4, kernel_size=(3, 3, 3), strides=(1, 1, 1), padding='same')(x_shortcut)
    x = k.layers.BatchNormalization()(x)
    x = k.layers.PReLU()(x)
    x = k.layers.Dropout(0.5)(x)

    x = k.layers.Add()([x, x_shortcut])
    x = k.layers.PReLU()(x)

    x = k.layers.AveragePooling3D(pool_size=(2, 2, 2), strides=(2, 2, 2), padding="same")(x)

    x = k.layers.Conv3D(filters=output_size, kernel_size=(1, 1, 1), strides=(1, 1, 1), padding='same')(x)
    x = k.layers.BatchNormalization()(x)
    x = k.layers.PReLU()(x)
    x = k.layers.Dropout(0.5)(x)

    x = deep_fully_connected_rnn(x, 2, True, 0, output_size)

    return x
