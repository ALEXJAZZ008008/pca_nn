import keras as k


def test_activation(x, batch_normalisation_bool, activation, regularisation, dropout):
    if batch_normalisation_bool:
        x = k.layers.BatchNormalization()(x)

    if activation == "lrelu":
        x = k.layers.LeakyReLU(alpha=0.01)(x)
    else:
        if activation == "prelu":
            x = k.layers.PReLU()(x)
        else:
            x = k.layers.Activation(activation)(x)

    if regularisation:
        x = k.layers.AlphaDropout(dropout)(x)

    return x


def test_in(x, filters, initializer, batch_normalisation_bool, activation, feature_downsample_bool, skip_bool,
            dense_bool, regularisation, lone, ltwo, dropout, deep_bool):
    feature_downsample_filters = int(filters / 2)

    x_res_skip_1 = x

    x = k.layers.TimeDistributed(k.layers.Convolution3D(filters=filters,
                                                        kernel_size=(3, 3, 3),
                                                        strides=(1, 1, 1),
                                                        padding="same",
                                                        kernel_initializer=initializer,
                                                        bias_initializer=k.initializers.Constant(0.1),
                                                        name="in"))(x)

    x = test_activation(x, True, activation, False, 0.0)

    if feature_downsample_bool:
        if regularisation:
            x = k.layers.TimeDistributed(k.layers.Convolution3D(filters=feature_downsample_filters,
                                                                kernel_size=(1, 1, 1),
                                                                strides=(1, 1, 1),
                                                                padding="same",
                                                                kernel_initializer=initializer,
                                                                bias_initializer=k.initializers.Constant(0.1),
                                                                kernel_regularizer=k.regularizers.l1_l2(l1=lone,
                                                                                                        l2=ltwo),
                                                                kernel_constraint=k.constraints.UnitNorm(),
                                                                name="in"))(x)

            x = test_activation(x, batch_normalisation_bool, activation, regularisation, dropout)
        else:
            x = k.layers.TimeDistributed(k.layers.Convolution3D(filters=feature_downsample_filters,
                                                                kernel_size=(1, 1, 1),
                                                                strides=(1, 1, 1),
                                                                padding="same",
                                                                kernel_initializer=initializer,
                                                                bias_initializer=k.initializers.Constant(0.1),
                                                                name="in"))(x)

            x = test_activation(x, batch_normalisation_bool, activation, regularisation, dropout)

    if skip_bool:
        if dense_bool:
            x = k.layers.Concatenate(axis=5)([x, x_res_skip_1])

            if not deep_bool:
                if feature_downsample_bool:
                    if regularisation:
                        x = k.layers.TimeDistributed(k.layers.Convolution3D(filters=feature_downsample_filters,
                                                                            kernel_size=(1, 1, 1),
                                                                            strides=(1, 1, 1),
                                                                            padding="same",
                                                                            kernel_initializer=initializer,
                                                                            bias_initializer=k.initializers.Constant(
                                                                                0.1),
                                                                            kernel_regularizer=k.regularizers.l1_l2(
                                                                                l1=lone,
                                                                                l2=ltwo),
                                                                            kernel_constraint=k.constraints.UnitNorm(),
                                                                            name="in"))(x)

                        x = test_activation(x, batch_normalisation_bool, activation, regularisation, dropout)
                    else:
                        x = k.layers.TimeDistributed(k.layers.Convolution3D(filters=feature_downsample_filters,
                                                                            kernel_size=(1, 1, 1),
                                                                            strides=(1, 1, 1),
                                                                            padding="same",
                                                                            kernel_initializer=initializer,
                                                                            bias_initializer=k.initializers.Constant(
                                                                                0.1),
                                                                            name="in"))(x)

                        x = test_activation(x, batch_normalisation_bool, activation, regularisation, dropout)
        else:
            x = k.layers.Add()([x, x_res_skip_1])

    if deep_bool:
        x_res_skip_2 = x

        if regularisation:
            x = k.layers.TimeDistributed(k.layers.Convolution3D(filters=filters,
                                                                kernel_size=(3, 3, 3),
                                                                strides=(1, 1, 1),
                                                                padding="same",
                                                                kernel_initializer=initializer,
                                                                bias_initializer=k.initializers.Constant(0.1),
                                                                kernel_regularizer=k.regularizers.l1_l2(l1=lone,
                                                                                                        l2=ltwo),
                                                                kernel_constraint=k.constraints.UnitNorm(),
                                                                name="in_2"))(x)

            x = test_activation(x, batch_normalisation_bool, activation, regularisation, dropout)
        else:
            x = k.layers.TimeDistributed(k.layers.Convolution3D(filters=filters,
                                                                kernel_size=(3, 3, 3),
                                                                strides=(1, 1, 1),
                                                                padding="same",
                                                                kernel_initializer=initializer,
                                                                bias_initializer=k.initializers.Constant(0.1),
                                                                name="in_2"))(x)

            x = test_activation(x, batch_normalisation_bool, activation, regularisation, dropout)

        if feature_downsample_bool:
            if regularisation:
                x = k.layers.TimeDistributed(k.layers.Convolution3D(filters=feature_downsample_filters,
                                                                    kernel_size=(1, 1, 1),
                                                                    strides=(1, 1, 1),
                                                                    padding="same",
                                                                    kernel_initializer=initializer,
                                                                    bias_initializer=k.initializers.Constant(0.1),
                                                                    kernel_regularizer=k.regularizers.l1_l2(l1=lone,
                                                                                                            l2=ltwo),
                                                                    kernel_constraint=k.constraints.UnitNorm(),
                                                                    name="in"))(x)

                x = test_activation(x, batch_normalisation_bool, activation, regularisation, dropout)
            else:
                x = k.layers.TimeDistributed(k.layers.Convolution3D(filters=feature_downsample_filters,
                                                                    kernel_size=(1, 1, 1),
                                                                    strides=(1, 1, 1),
                                                                    padding="same",
                                                                    kernel_initializer=initializer,
                                                                    bias_initializer=k.initializers.Constant(0.1),
                                                                    name="in"))(x)

                x = test_activation(x, batch_normalisation_bool, activation, regularisation, dropout)

        if skip_bool:
            if dense_bool:
                x = k.layers.Concatenate(axis=5)([x, x_res_skip_1, x_res_skip_2])
            else:
                x = k.layers.Add()([x, x_res_skip_2])

        x_res_skip_3 = x

        if regularisation:
            x = k.layers.TimeDistributed(k.layers.Convolution3D(filters=filters,
                                                                kernel_size=(3, 3, 3),
                                                                strides=(1, 1, 1),
                                                                padding="same",
                                                                kernel_initializer=initializer,
                                                                bias_initializer=k.initializers.Constant(0.1),
                                                                kernel_regularizer=k.regularizers.l1_l2(l1=lone,
                                                                                                        l2=ltwo),
                                                                kernel_constraint=k.constraints.UnitNorm(),
                                                                name="in_3"))(x)

            x = test_activation(x, batch_normalisation_bool, activation, regularisation, dropout)
        else:
            x = k.layers.TimeDistributed(k.layers.Convolution3D(filters=filters,
                                                                kernel_size=(3, 3, 3),
                                                                strides=(1, 1, 1),
                                                                padding="same",
                                                                kernel_initializer=initializer,
                                                                bias_initializer=k.initializers.Constant(0.1),
                                                                name="in_3"))(x)

            x = test_activation(x, batch_normalisation_bool, activation, regularisation, dropout)

        if feature_downsample_bool:
            if regularisation:
                x = k.layers.TimeDistributed(k.layers.Convolution3D(filters=feature_downsample_filters,
                                                                    kernel_size=(1, 1, 1),
                                                                    strides=(1, 1, 1),
                                                                    padding="same",
                                                                    kernel_initializer=initializer,
                                                                    bias_initializer=k.initializers.Constant(0.1),
                                                                    kernel_regularizer=k.regularizers.l1_l2(l1=lone,
                                                                                                            l2=ltwo),
                                                                    kernel_constraint=k.constraints.UnitNorm(),
                                                                    name="in"))(x)

                x = test_activation(x, batch_normalisation_bool, activation, regularisation, dropout)
            else:
                x = k.layers.TimeDistributed(k.layers.Convolution3D(filters=feature_downsample_filters,
                                                                    kernel_size=(1, 1, 1),
                                                                    strides=(1, 1, 1),
                                                                    padding="same",
                                                                    kernel_initializer=initializer,
                                                                    bias_initializer=k.initializers.Constant(0.1),
                                                                    name="in"))(x)

                x = test_activation(x, batch_normalisation_bool, activation, regularisation, dropout)

        if skip_bool:
            if dense_bool:
                x = k.layers.Concatenate(axis=5)([x, x_res_skip_1, x_res_skip_2, x_res_skip_3])
            else:
                x = k.layers.Add()([x, x_res_skip_3])

        x_res_skip_4 = x

        if regularisation:
            x = k.layers.TimeDistributed(k.layers.Convolution3D(filters=filters,
                                                                kernel_size=(3, 3, 3),
                                                                strides=(1, 1, 1),
                                                                padding="same",
                                                                kernel_initializer=initializer,
                                                                bias_initializer=k.initializers.Constant(0.1),
                                                                kernel_regularizer=k.regularizers.l1_l2(l1=lone,
                                                                                                        l2=ltwo),
                                                                kernel_constraint=k.constraints.UnitNorm(),
                                                                name="in_4"))(x)

            x = test_activation(x, batch_normalisation_bool, activation, regularisation, dropout)
        else:
            x = k.layers.TimeDistributed(k.layers.Convolution3D(filters=filters,
                                                                kernel_size=(3, 3, 3),
                                                                strides=(1, 1, 1),
                                                                padding="same",
                                                                kernel_initializer=initializer,
                                                                bias_initializer=k.initializers.Constant(0.1),
                                                                name="in_4"))(x)

            x = test_activation(x, batch_normalisation_bool, activation, regularisation, dropout)

        if feature_downsample_bool:
            if regularisation:
                x = k.layers.TimeDistributed(k.layers.Convolution3D(filters=feature_downsample_filters,
                                                                    kernel_size=(1, 1, 1),
                                                                    strides=(1, 1, 1),
                                                                    padding="same",
                                                                    kernel_initializer=initializer,
                                                                    bias_initializer=k.initializers.Constant(0.1),
                                                                    kernel_regularizer=k.regularizers.l1_l2(l1=lone,
                                                                                                            l2=ltwo),
                                                                    kernel_constraint=k.constraints.UnitNorm(),
                                                                    name="in"))(x)

                x = test_activation(x, batch_normalisation_bool, activation, regularisation, dropout)
            else:
                x = k.layers.TimeDistributed(k.layers.Convolution3D(filters=feature_downsample_filters,
                                                                    kernel_size=(1, 1, 1),
                                                                    strides=(1, 1, 1),
                                                                    padding="same",
                                                                    kernel_initializer=initializer,
                                                                    bias_initializer=k.initializers.Constant(0.1),
                                                                    name="in"))(x)

                x = test_activation(x, batch_normalisation_bool, activation, regularisation, dropout)

        if skip_bool:
            if dense_bool:
                x = k.layers.Concatenate(axis=5)([x, x_res_skip_1, x_res_skip_2, x_res_skip_3, x_res_skip_4])

                if feature_downsample_bool:
                    x = k.layers.TimeDistributed(k.layers.Convolution3D(filters=feature_downsample_filters,
                                                                        kernel_size=(1, 1, 1),
                                                                        strides=(1, 1, 1),
                                                                        padding="same",
                                                                        kernel_initializer=initializer,
                                                                        bias_initializer=k.initializers.Constant(0.1),
                                                                        name="in"))(x)

                    x = test_activation(x, batch_normalisation_bool, activation, regularisation, dropout)
            else:
                x = k.layers.Add()([x, x_res_skip_4])

    return x


def test_module_module_down(x, filters, kernal_size, strides, initializer, batch_normalisation_bool, activation,
                            regularisation, lone, ltwo, dropout, feature_downsample_bool, name, deep_bool, skip_bool,
                            skip_filters_bool):
    feature_downsample_filters = int(filters / 2)

    if regularisation:
        if deep_bool:
            if feature_downsample_bool:
                x_res_skip = k.layers.TimeDistributed(k.layers.Convolution3D(filters=feature_downsample_filters,
                                                                             kernel_size=(
                                                                                 kernal_size, kernal_size, kernal_size),
                                                                             strides=(1, 1, 1),
                                                                             padding="same",
                                                                             kernel_initializer=initializer,
                                                                             bias_initializer=k.initializers.Constant(
                                                                                 0.1),
                                                                             kernel_regularizer=k.regularizers.l1_l2(
                                                                                 l1=lone,
                                                                                 l2=ltwo),
                                                                             kernel_constraint=k.constraints.UnitNorm(),
                                                                             name="down_reg_deep_{0}".format(name)))(x)

                x_res_skip = test_activation(x_res_skip, batch_normalisation_bool, "linear", regularisation, dropout)

                x = k.layers.TimeDistributed(k.layers.Convolution3D(filters=feature_downsample_filters,
                                                                    kernel_size=(kernal_size, kernal_size, kernal_size),
                                                                    strides=(1, 1, 1),
                                                                    padding="same",
                                                                    kernel_initializer=initializer,
                                                                    bias_initializer=k.initializers.Constant(0.1),
                                                                    kernel_regularizer=k.regularizers.l1_l2(l1=lone,
                                                                                                            l2=ltwo),
                                                                    kernel_constraint=k.constraints.UnitNorm(),
                                                                    name="down_reg_deep_{0}".format(name)))(x)

                x = test_activation(x, batch_normalisation_bool, activation, regularisation, dropout)
            else:
                x_res_skip = k.layers.TimeDistributed(k.layers.Convolution3D(filters=filters,
                                                                             kernel_size=(
                                                                                 kernal_size, kernal_size, kernal_size),
                                                                             strides=(1, 1, 1),
                                                                             padding="same",
                                                                             kernel_initializer=initializer,
                                                                             bias_initializer=k.initializers.Constant(
                                                                                 0.1),
                                                                             kernel_regularizer=k.regularizers.l1_l2(
                                                                                 l1=lone,
                                                                                 l2=ltwo),
                                                                             kernel_constraint=k.constraints.UnitNorm(),
                                                                             name="down_reg_deep_{0}".format(name)))(x)

                x_res_skip = test_activation(x_res_skip, batch_normalisation_bool, "linear", regularisation, dropout)

                x = k.layers.TimeDistributed(k.layers.Convolution3D(filters=filters,
                                                                    kernel_size=(kernal_size, kernal_size, kernal_size),
                                                                    strides=(1, 1, 1),
                                                                    padding="same",
                                                                    kernel_initializer=initializer,
                                                                    bias_initializer=k.initializers.Constant(0.1),
                                                                    kernel_regularizer=k.regularizers.l1_l2(l1=lone,
                                                                                                            l2=ltwo),
                                                                    kernel_constraint=k.constraints.UnitNorm(),
                                                                    name="down_reg_deep_{0}".format(name)))(x)

                x = test_activation(x, batch_normalisation_bool, activation, regularisation, dropout)

            if skip_bool and skip_filters_bool:
                x = test_crop(x, x_res_skip)
                x_res_skip = test_crop(x_res_skip, x)
                x = k.layers.Add()([x, x_res_skip])

        x = k.layers.TimeDistributed(k.layers.Convolution3D(filters=filters,
                                                            kernel_size=(kernal_size, kernal_size, kernal_size),
                                                            strides=(strides, strides, strides),
                                                            padding="same",
                                                            kernel_initializer=initializer,
                                                            bias_initializer=k.initializers.Constant(0.1),
                                                            kernel_regularizer=k.regularizers.l1_l2(l1=lone, l2=ltwo),
                                                            kernel_constraint=k.constraints.UnitNorm(),
                                                            name="down_reg_{0}".format(name)))(x)

        x = test_activation(x, batch_normalisation_bool, activation, regularisation, dropout)

        if feature_downsample_bool:
            x = k.layers.TimeDistributed(k.layers.Convolution3D(filters=feature_downsample_filters,
                                                                kernel_size=(1, 1, 1),
                                                                strides=(1, 1, 1),
                                                                padding="same",
                                                                kernel_initializer=initializer,
                                                                bias_initializer=k.initializers.Constant(0.1),
                                                                kernel_regularizer=k.regularizers.l1_l2(l1=lone,
                                                                                                        l2=ltwo),
                                                                kernel_constraint=k.constraints.UnitNorm(),
                                                                name="down_fet_reg_{0}".format(name)))(x)

            x = test_activation(x, batch_normalisation_bool, activation, regularisation, dropout)
    else:
        if deep_bool:
            if feature_downsample_bool:
                x_res_skip = k.layers.TimeDistributed(k.layers.Convolution3D(filters=feature_downsample_filters,
                                                                             kernel_size=(
                                                                                 kernal_size, kernal_size, kernal_size),
                                                                             strides=(1, 1, 1),
                                                                             padding="same",
                                                                             kernel_initializer=initializer,
                                                                             bias_initializer=k.initializers.Constant(
                                                                                 0.1),
                                                                             name="down_reg_deep_{0}".format(name)))(x)

                x_res_skip = test_activation(x_res_skip, batch_normalisation_bool, "linear", regularisation, dropout)

                x = k.layers.TimeDistributed(k.layers.Convolution3D(filters=feature_downsample_filters,
                                                                    kernel_size=(kernal_size, kernal_size, kernal_size),
                                                                    strides=(1, 1, 1),
                                                                    padding="same",
                                                                    kernel_initializer=initializer,
                                                                    bias_initializer=k.initializers.Constant(0.1),
                                                                    name="down_deep_{0}".format(name)))(x)

                x = test_activation(x, batch_normalisation_bool, activation, regularisation, dropout)
            else:
                x_res_skip = k.layers.TimeDistributed(k.layers.Convolution3D(filters=filters,
                                                                             kernel_size=(
                                                                                 kernal_size, kernal_size, kernal_size),
                                                                             strides=(1, 1, 1),
                                                                             padding="same",
                                                                             kernel_initializer=initializer,
                                                                             bias_initializer=k.initializers.Constant(
                                                                                 0.1),
                                                                             name="down_reg_deep_{0}".format(name)))(x)

                x_res_skip = test_activation(x_res_skip, batch_normalisation_bool, "linear", regularisation, dropout)

                x = k.layers.TimeDistributed(k.layers.Convolution3D(filters=filters,
                                                                    kernel_size=(kernal_size, kernal_size, kernal_size),
                                                                    strides=(1, 1, 1),
                                                                    padding="same",
                                                                    kernel_initializer=initializer,
                                                                    bias_initializer=k.initializers.Constant(0.1),
                                                                    name="down_deep_{0}".format(name)))(x)

                x = test_activation(x, batch_normalisation_bool, activation, regularisation, dropout)

            if skip_bool and skip_filters_bool:
                x = test_crop(x, x_res_skip)
                x_res_skip = test_crop(x_res_skip, x)
                x = k.layers.Add()([x, x_res_skip])

        x = k.layers.TimeDistributed(k.layers.Convolution3D(filters=filters,
                                                            kernel_size=(kernal_size, kernal_size, kernal_size),
                                                            strides=(strides, strides, strides),
                                                            padding="same",
                                                            kernel_initializer=initializer,
                                                            bias_initializer=k.initializers.Constant(0.1),
                                                            name="down_{0}".format(name)))(x)

        x = test_activation(x, batch_normalisation_bool, activation, regularisation, dropout)

        if feature_downsample_bool:
            x = k.layers.TimeDistributed(k.layers.Convolution3D(filters=feature_downsample_filters,
                                                                kernel_size=(1, 1, 1),
                                                                strides=(1, 1, 1),
                                                                padding="same",
                                                                kernel_initializer=initializer,
                                                                bias_initializer=k.initializers.Constant(0.1),
                                                                name="down_fet_{0}".format(name)))(x)

            x = test_activation(x, batch_normalisation_bool, activation, regularisation, dropout)

    return x


def test_module_down(x, filters, initializer, batch_normalisation_bool, activation, regularisation, lone, ltwo, dropout,
                     feature_downsample_bool, concat_bool, name, deep_bool, skip_bool, skip_filters_bool):
    if concat_bool:
        concat_filters = int(filters / 4)

        x_1 = test_module_module_down(x, concat_filters, 3, 2, initializer, batch_normalisation_bool, activation,
                                      regularisation, lone, ltwo, dropout, feature_downsample_bool,
                                      "3_{0}".format(name),
                                      deep_bool, skip_bool, skip_filters_bool)

        x_2 = test_module_module_down(x, concat_filters, 5, 2, initializer, batch_normalisation_bool, activation,
                                      regularisation, lone, ltwo, dropout, feature_downsample_bool,
                                      "5_{0}".format(name),
                                      deep_bool, skip_bool, skip_filters_bool)

        x_3 = test_module_module_down(x, concat_filters, 7, 2, initializer, batch_normalisation_bool, activation,
                                      regularisation, lone, ltwo, dropout, feature_downsample_bool,
                                      "7_{0}".format(name),
                                      deep_bool, skip_bool, skip_filters_bool)

        x_4 = test_module_module_down(x, concat_filters, 9, 2, initializer, batch_normalisation_bool, activation,
                                      regularisation, lone, ltwo, dropout, feature_downsample_bool,
                                      "9_{0}".format(name),
                                      deep_bool, skip_bool, skip_filters_bool)

        x = k.layers.Concatenate(axis=5)([x_1, x_2, x_3, x_4])
    else:
        x = test_module_module_down(x, filters, 3, 2, initializer, batch_normalisation_bool, activation,
                                    regularisation, lone, ltwo, dropout, feature_downsample_bool, "3_{0}".format(name),
                                    deep_bool, skip_bool, skip_filters_bool)

    return x


def test_down(x, x_skip, layers, filters, initializer, batch_normalisation_bool, activation,
              regularisation, lone, ltwo, dropout, feature_downsample_bool, concat_bool, skip_bool, ascending_bool,
              deep_bool, skip_filters_bool):
    high_tap = None
    high_tap_skip = None

    mid_tap = None
    mid_tap_skip = None

    for i in range(layers):
        x_skip.append(x)

        x_res_skip = k.layers.TimeDistributed(k.layers.AveragePooling3D(padding="same"))(x)

        ascending_filters = filters

        if ascending_bool:
            ascending_filters = filters * (i + 1)

            feature_downsample_filters = int(ascending_filters / 2)

            if regularisation:
                x_res_skip = k.layers.TimeDistributed(k.layers.Convolution3D(filters=feature_downsample_filters,
                                                                             kernel_size=(1, 1, 1),
                                                                             strides=(1, 1, 1),
                                                                             padding="same",
                                                                             kernel_initializer=initializer,
                                                                             bias_initializer=k.initializers.Constant(
                                                                                 0.1),
                                                                             kernel_regularizer=k.regularizers.l1_l2(
                                                                                 l1=lone, l2=ltwo),
                                                                             kernel_constraint=k.constraints.UnitNorm(),
                                                                             name="ascending_skip_washer_reg"))(
                    x_res_skip)

                x_res_skip = test_activation(x_res_skip, batch_normalisation_bool, "linear", regularisation, dropout)
            else:
                x_res_skip = k.layers.TimeDistributed(k.layers.Convolution3D(filters=feature_downsample_filters,
                                                                             kernel_size=(1, 1, 1),
                                                                             strides=(1, 1, 1),
                                                                             padding="same",
                                                                             kernel_initializer=initializer,
                                                                             bias_initializer=k.initializers.Constant(
                                                                                 0.1),
                                                                             name="ascending_skip_washer"))(x_res_skip)

                x_res_skip = test_activation(x_res_skip, batch_normalisation_bool, "linear", regularisation, dropout)

        x = test_module_down(x, ascending_filters, initializer, batch_normalisation_bool, activation, regularisation,
                             lone, ltwo, dropout, feature_downsample_bool, concat_bool, str(i), deep_bool, skip_bool,
                             skip_filters_bool)
        if ascending_bool:
            if skip_bool and skip_filters_bool:
                x = test_crop(x, x_res_skip)
                x_res_skip = test_crop(x_res_skip, x)
                x = k.layers.Add()([x, x_res_skip])
        else:
            if skip_bool:
                x = test_crop(x, x_res_skip)
                x_res_skip = test_crop(x_res_skip, x)
                x = k.layers.Add()([x, x_res_skip])

        if i == 0:
            high_tap = x
            high_tap_skip = x_skip
        else:
            if i == 2:
                mid_tap = x
                mid_tap_skip = x_skip

    return x, mid_tap, mid_tap_skip, high_tap, high_tap_skip, x_skip


def test_module_out(x):
    x = k.layers.TimeDistributed(k.layers.Flatten())(x)

    return x


def test_in_down_out(x, x_skip, activation, regularisation, filters, initializer, layers,
                     batch_normalisation_bool, lone, ltwo, dropout, feature_downsample_bool, concat_bool, skip_bool,
                     dense_bool, ascending_bool, deep_bool, skip_filters_bool):
    x = test_in(x, filters, initializer, batch_normalisation_bool, activation, feature_downsample_bool, skip_bool,
                dense_bool, regularisation, lone, ltwo, dropout, deep_bool)

    x, mid_tap, mid_tap_skip, high_tap, high_tap_skip, x_skip = test_down(x, x_skip, layers, filters, initializer,
                                                                          batch_normalisation_bool, activation,
                                                                          regularisation, lone, ltwo, dropout,
                                                                          feature_downsample_bool, concat_bool,
                                                                          skip_bool, ascending_bool, deep_bool,
                                                                          skip_filters_bool)

    x = test_module_out(x)

    return x, mid_tap, mid_tap_skip, high_tap, high_tap_skip, x_skip


def test_module_rnn_internal(x, regularisation, units, activation, lone, ltwo, dropout, return_sequences, initializer,
                             recurrent_initializer, unroll, name):
    if regularisation:
        x = k.layers.SimpleRNN(units=units,
                               activation=activation,
                               dropout=dropout,
                               recurrent_dropout=dropout,
                               return_sequences=return_sequences,
                               kernel_initializer=initializer,
                               bias_initializer=k.initializers.Constant(0.1),
                               recurrent_initializer=recurrent_initializer,
                               kernel_regularizer=k.regularizers.l1_l2(l1=lone, l2=ltwo),
                               kernel_constraint=k.constraints.UnitNorm(),
                               unroll=unroll,
                               name="reg_{0}".format(name))(x)
    else:
        x = k.layers.SimpleRNN(units=units,
                               activation=activation,
                               return_sequences=return_sequences,
                               kernel_initializer=initializer,
                               bias_initializer=k.initializers.Constant(0.1),
                               recurrent_initializer=recurrent_initializer,
                               unroll=unroll,
                               name="{0}".format(name))(x)

    return x


def test_module_lstm_internal(x, regularisation, units, activation, return_activation, lone, ltwo, dropout,
                              return_sequences,
                              initializer, recurrent_initializer, unroll, name):
    if regularisation:
        x = k.layers.LSTM(units=units,
                          activation=activation,
                          recurrent_activation=return_activation,
                          dropout=dropout,
                          recurrent_dropout=dropout,
                          return_sequences=return_sequences,
                          kernel_initializer=initializer,
                          bias_initializer=k.initializers.Constant(0.1),
                          recurrent_initializer=recurrent_initializer,
                          kernel_regularizer=k.regularizers.l1_l2(l1=lone, l2=ltwo),
                          kernel_constraint=k.constraints.UnitNorm(),
                          unroll=unroll,
                          name="reg_{0}".format(name))(x)
    else:
        x = k.layers.LSTM(units=units,
                          activation=activation,
                          recurrent_activation=return_activation,
                          return_sequences=return_sequences,
                          kernel_initializer=initializer,
                          bias_initializer=k.initializers.Constant(0.1),
                          recurrent_initializer=recurrent_initializer,
                          unroll=unroll,
                          name="{0}".format(name))(x)

    return x


def test_module_gru_internal(x, regularisation, units, activation, return_activation, lone, ltwo, dropout,
                             return_sequences, initializer, recurrent_initializer, unroll, name):
    if regularisation:
        x = k.layers.GRU(units=units,
                         activation=activation,
                         recurrent_activation=return_activation,
                         dropout=dropout,
                         recurrent_dropout=dropout,
                         return_sequences=return_sequences,
                         kernel_initializer=initializer,
                         bias_initializer=k.initializers.Constant(0.1),
                         recurrent_initializer=recurrent_initializer,
                         kernel_regularizer=k.regularizers.l1_l2(l1=lone, l2=ltwo),
                         kernel_constraint=k.constraints.UnitNorm(),
                         unroll=unroll,
                         name="reg_{0}".format(name))(x)
    else:
        x = k.layers.GRU(units=units,
                         activation=activation,
                         recurrent_activation=return_activation,
                         return_sequences=return_sequences,
                         kernel_initializer=initializer,
                         bias_initializer=k.initializers.Constant(0.1),
                         recurrent_initializer=recurrent_initializer,
                         unroll=unroll,
                         name="{0}".format(name))(x)

    return x


def test_module_rnn(x, units, return_sequences, initializer, recurrent_initializer, unroll, batch_normalisation_bool,
                    activation, regularisation, lone, ltwo, dropout, name):
    if regularisation:
        x = k.layers.SimpleRNN(units=units,
                               return_sequences=return_sequences,
                               kernel_initializer=initializer,
                               bias_initializer=k.initializers.Constant(0.1),
                               recurrent_initializer=recurrent_initializer,
                               kernel_regularizer=k.regularizers.l1_l2(l1=lone, l2=ltwo),
                               kernel_constraint=k.constraints.UnitNorm(),
                               unroll=unroll,
                               name="reg_{0}".format(name))(x)
    else:
        x = k.layers.SimpleRNN(units=units,
                               return_sequences=return_sequences,
                               kernel_initializer=initializer,
                               bias_initializer=k.initializers.Constant(0.1),
                               recurrent_initializer=recurrent_initializer,
                               unroll=unroll,
                               name="{0}".format(name))(x)

    x = test_activation(x, batch_normalisation_bool, activation, regularisation, dropout)

    return x


def test_module_lstm(x, units, return_activation, return_sequences, initializer, recurrent_initializer, unroll,
                     batch_normalisation_bool, activation, regularisation, lone, ltwo, dropout, name):
    if regularisation:
        x = k.layers.LSTM(units=units,
                          recurrent_activation=return_activation,
                          return_sequences=return_sequences,
                          kernel_initializer=initializer,
                          bias_initializer=k.initializers.Constant(0.1),
                          recurrent_initializer=recurrent_initializer,
                          kernel_regularizer=k.regularizers.l1_l2(l1=lone, l2=ltwo),
                          kernel_constraint=k.constraints.UnitNorm(),
                          unroll=unroll,
                          name="reg_{0}".format(name))(x)
    else:
        x = k.layers.LSTM(units=units,
                          recurrent_activation=return_activation,
                          return_sequences=return_sequences,
                          kernel_initializer=initializer,
                          bias_initializer=k.initializers.Constant(0.1),
                          recurrent_initializer=recurrent_initializer,
                          unroll=unroll,
                          name="{0}".format(name))(x)

    x = test_activation(x, batch_normalisation_bool, activation, regularisation, dropout)

    return x


def test_module_gru(x, units, return_activation, return_sequences, initializer, recurrent_initializer, unroll,
                    batch_normalisation_bool, activation, regularisation, lone, ltwo, dropout, name):
    if regularisation:
        x = k.layers.GRU(units=units,
                         recurrent_activation=return_activation,
                         return_sequences=return_sequences,
                         kernel_initializer=initializer,
                         bias_initializer=k.initializers.Constant(0.1),
                         recurrent_initializer=recurrent_initializer,
                         kernel_regularizer=k.regularizers.l1_l2(l1=lone, l2=ltwo),
                         kernel_constraint=k.constraints.UnitNorm(),
                         unroll=unroll,
                         name="reg_{0}".format(name))(x)
    else:
        x = k.layers.GRU(units=units,
                         recurrent_activation=return_activation,
                         return_sequences=return_sequences,
                         kernel_initializer=initializer,
                         bias_initializer=k.initializers.Constant(0.1),
                         recurrent_initializer=recurrent_initializer,
                         unroll=unroll,
                         name="{0}".format(name))(x)

    x = test_activation(x, batch_normalisation_bool, activation, regularisation, dropout)

    return x


def test_module_module_rnn_out(x, rnn_type, internal_bool, units, return_activation, initializer,
                               recurrent_initializer, unroll, batch_normalisation_bool, activation, lone, ltwo, dropout,
                               regularisation, return_sequences, name):
    if internal_bool:
        if rnn_type == "rnn":
            x = test_module_rnn_internal(x, regularisation, units, activation, lone, ltwo, dropout, return_sequences,
                                         recurrent_initializer, initializer, unroll, "rnn_int_{0}".format(name))
        else:
            if rnn_type == "lstm":
                x = test_module_lstm_internal(x, regularisation, units, activation, return_activation, lone, ltwo,
                                              dropout, return_sequences, recurrent_initializer, initializer, unroll,
                                              "lstm_int_{0}".format(name))
            else:
                if rnn_type == "gru":
                    x = test_module_gru_internal(x, regularisation, units, activation, return_activation, lone, ltwo,
                                                 dropout, return_sequences, recurrent_initializer, initializer, unroll,
                                                 "gru_int_{0}".format(name))
    else:
        if rnn_type == "rnn":
            x = test_module_rnn(x, units, return_sequences, recurrent_initializer, initializer, unroll,
                                batch_normalisation_bool, activation, regularisation, lone, ltwo, dropout,
                                "rnn_{0}".format(name))
        else:
            if rnn_type == "lstm":
                x = test_module_lstm(x, units, return_activation, return_sequences, recurrent_initializer, initializer,
                                     unroll, batch_normalisation_bool, activation, regularisation, lone, ltwo, dropout,
                                     "lstm_{0}".format(name))
            else:
                if rnn_type == "gru":
                    x = test_module_gru(x, units, return_activation, return_sequences, recurrent_initializer,
                                        initializer, unroll, batch_normalisation_bool, activation, regularisation, lone,
                                        ltwo, dropout, "gru_{0}".format(name))

    return x


def test_washer(x, filters, initializer, batch_normalisation_bool, activation, regularisation, dropout, name, lone,
                ltwo, skip_bool, skip_filters_bool):
    if regularisation:
        x_res_skip = k.layers.TimeDistributed(k.layers.Convolution3D(filters=filters,
                                                                     kernel_size=(1, 1, 1),
                                                                     strides=(1, 1, 1),
                                                                     padding="same",
                                                                     kernel_initializer=initializer,
                                                                     bias_initializer=k.initializers.Constant(0.1),
                                                                     kernel_regularizer=k.regularizers.l1_l2(l1=lone,
                                                                                                             l2=ltwo),
                                                                     kernel_constraint=k.constraints.UnitNorm(),
                                                                     name="washer_skip_washer_reg"))(x)

        x_res_skip = test_activation(x_res_skip, batch_normalisation_bool, "linear", regularisation, dropout)
    else:
        x_res_skip = k.layers.TimeDistributed(k.layers.Convolution3D(filters=filters,
                                                                     kernel_size=(1, 1, 1),
                                                                     strides=(1, 1, 1),
                                                                     padding="same",
                                                                     kernel_initializer=initializer,
                                                                     bias_initializer=k.initializers.Constant(0.1),
                                                                     name="washer_skip_washer"))(x)

        x_res_skip = test_activation(x_res_skip, batch_normalisation_bool, "linear", regularisation, dropout)

    if regularisation:
        x = k.layers.TimeDistributed(k.layers.Convolution3D(filters=filters,
                                                            kernel_size=(3, 3, 3),
                                                            strides=(1, 1, 1),
                                                            padding="same",
                                                            kernel_initializer=initializer,
                                                            bias_initializer=k.initializers.Constant(0.1),
                                                            kernel_regularizer=k.regularizers.l1_l2(l1=lone, l2=ltwo),
                                                            kernel_constraint=k.constraints.UnitNorm(),
                                                            name="{0}_washer".format(name)))(x)

        x = test_activation(x, batch_normalisation_bool, activation, regularisation, dropout)
    else:
        x = k.layers.TimeDistributed(k.layers.Convolution3D(filters=filters,
                                                            kernel_size=(3, 3, 3),
                                                            strides=(1, 1, 1),
                                                            padding="same",
                                                            kernel_initializer=initializer,
                                                            bias_initializer=k.initializers.Constant(0.1),
                                                            name="{0}_washer".format(name)))(x)

        x = test_activation(x, batch_normalisation_bool, activation, regularisation, dropout)

    if skip_bool and skip_filters_bool:
        x = test_crop(x, x_res_skip)
        x_res_skip = test_crop(x_res_skip, x)
        x = k.layers.Add()([x, x_res_skip])

    return x


def test_module_rnn_out(x, layers, rnn_type, internal_bool, units, return_activation, initializer,
                        recurrent_initializer, unroll, batch_normalisation_bool, activation, lone, ltwo, dropout,
                        regularisation, return_sequences, name, skip_bool, skip_filters_bool):
    x = k.layers.TimeDistributed(k.layers.Flatten())(x)

    for i in range(1, layers):
        x_res_skip = test_module_module_rnn_out(x, rnn_type, internal_bool, units, return_activation, initializer,
                                                recurrent_initializer, unroll, batch_normalisation_bool, "linear", lone,
                                                ltwo,
                                                dropout, regularisation, True, "skip_{0}_{1}".format(name, str(i - 1)))

        x = test_module_module_rnn_out(x, rnn_type, internal_bool, units, return_activation, initializer,
                                       recurrent_initializer, unroll, batch_normalisation_bool, activation, lone, ltwo,
                                       dropout, regularisation, True, "{0}_{1}".format(name, str(i - 1)))

        if skip_bool and skip_filters_bool:
            x = k.layers.Add()([x, x_res_skip])

    x_res_skip = test_module_module_rnn_out(x, rnn_type, internal_bool, units, return_activation, initializer,
                                            recurrent_initializer, unroll, batch_normalisation_bool, "linear", lone,
                                            ltwo, dropout, regularisation, return_sequences,
                                            "skip_{0}_{1}".format(name, str(layers - 1)))

    x = test_module_module_rnn_out(x, rnn_type, internal_bool, units, return_activation, initializer,
                                   recurrent_initializer, unroll, batch_normalisation_bool, activation, lone, ltwo,
                                   dropout, regularisation, return_sequences, "{0}_{1}".format(name, str(layers - 1)))

    if skip_bool and skip_filters_bool:
        x = k.layers.Add()([x, x_res_skip])

    return x


def test_rnn_out(x, layers, rnn_type, internal_bool, units, return_activation, initializer, recurrent_initializer,
                 unroll, batch_normalisation_bool, activation, lone, ltwo, dropout, regularisation, return_sequences,
                 skip_bool, skip_filters_bool):
    x = test_module_rnn_out(x, layers, rnn_type, internal_bool, units, return_activation, recurrent_initializer,
                            initializer, unroll, batch_normalisation_bool, activation, lone, ltwo, dropout,
                            regularisation, return_sequences, "x", skip_bool, skip_filters_bool)

    return x


def test_in_down_rnn_out(x, x_skip, activation, regularisation, lone, ltwo, dropout, filters, output_size, initializer,
                         down_layers, batch_normalisation_bool, out_layers, rnn_type, internal_bool, units,
                         rnn_return_activation, unroll, rnn_batch_normalisation_bool, rnn_initializer,
                         rnn_recurrent_initializer, rnn_activation, rnn_lone, rnn_ltwo, rnn_dropout, rnn_regularisation,
                         rnn_return_sequences, feature_downsample_bool, concat_bool, skip_bool, dense_bool,
                         ascending_bool, deep_bool, skip_filters_bool):
    x = test_in(x, filters, initializer, batch_normalisation_bool, activation, feature_downsample_bool, skip_bool,
                dense_bool, regularisation, lone, ltwo, dropout, deep_bool)

    x, mid_tap, mid_tap_skip, high_tap, high_tap_skip, x_skip = test_down(x, x_skip, down_layers, filters, initializer,
                                                                          batch_normalisation_bool, activation,
                                                                          regularisation, lone, ltwo, dropout,
                                                                          feature_downsample_bool, concat_bool,
                                                                          skip_bool, ascending_bool, deep_bool,
                                                                          skip_filters_bool)

    x = test_washer(x, output_size, initializer, batch_normalisation_bool, activation, regularisation, dropout, "x",
                    lone, ltwo, skip_bool, skip_filters_bool)

    x = test_module_rnn_out(x, out_layers, rnn_type, internal_bool, units, rnn_return_activation, rnn_initializer,
                            rnn_recurrent_initializer, unroll, rnn_batch_normalisation_bool, rnn_activation, rnn_lone,
                            rnn_ltwo, rnn_dropout, rnn_regularisation, rnn_return_sequences, "x", skip_bool,
                            skip_filters_bool)

    return x, mid_tap, mid_tap_skip, high_tap, high_tap_skip, x_skip


def test_module_up(x, regularisation, dropout, filters, initializer, batch_normalisation_bool, activation, name):
    x_shape = x.get_shape()

    x = k.layers.TimeDistributed(k.layers.Deconvolution3D(filters=filters,
                                                          kernel_size=(3, 3, 3),
                                                          strides=(2, 2, 2),
                                                          padding="same",
                                                          kernel_initializer=initializer,
                                                          bias_initializer=k.initializers.Constant(0.1),
                                                          name="up_{0}".format(name)))(x)

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


def test_up(x, x_skip, regularisation, dropout, filters, initializer, batch_normalisation_bool, activation, name,
            skip_bool, lone, ltwo, skip_filters_bool):
    x = test_washer(x, filters, initializer, batch_normalisation_bool, activation, regularisation, dropout, name, lone,
                    ltwo, skip_bool, skip_filters_bool)

    for i in range(len(x_skip)):
        x_res_skip = k.layers.TimeDistributed(k.layers.UpSampling3D())(x)

        x = test_module_up(x, regularisation, dropout, filters, initializer, batch_normalisation_bool, activation,
                           "{0}_{1}".format(name, str(i)))

        x = test_crop(x, x_skip[(len(x_skip) - 1) - i])

        if skip_bool:
            x_res_skip = test_crop(x_res_skip, x)
            x = test_crop(x, x_res_skip)
            x = k.layers.Add()([x, x_res_skip])

    return x


def test_multi_out(x, x_skip, activation, regularisation, lone, ltwo, dropout, filters, initializer, layers,
                   batch_normalisation_bool, up_filters, feature_downsample_bool, concat_bool, skip_bool, dense_bool,
                   ascending_bool, deep_bool, skip_filters_bool):
    x = test_in(x, filters, initializer, batch_normalisation_bool, activation, feature_downsample_bool, skip_bool,
                dense_bool, regularisation, lone, ltwo, dropout, deep_bool)

    x, mid_tap, mid_tap_skip, high_tap, high_tap_skip, x_skip = test_down(x, x_skip, layers, filters, initializer,
                                                                          batch_normalisation_bool, activation,
                                                                          regularisation, lone, ltwo, dropout,
                                                                          feature_downsample_bool, concat_bool,
                                                                          skip_bool, ascending_bool, deep_bool,
                                                                          skip_filters_bool)

    x_2 = test_up(x, x_skip, False, 0.0, up_filters, initializer, batch_normalisation_bool, activation, "x", skip_bool,
                  lone, ltwo, skip_filters_bool)
    x_2_5 = test_up(mid_tap, mid_tap_skip, False, 0.0, up_filters, initializer, batch_normalisation_bool, activation,
                    "tap", skip_bool, lone, ltwo, skip_filters_bool)
    x_2_0 = test_up(high_tap, high_tap_skip, False, 0.0, up_filters, initializer, batch_normalisation_bool, activation,
                    "tap", skip_bool, lone, ltwo, skip_filters_bool)

    x_1 = test_module_out(x)
    x_1_5 = test_module_out(mid_tap)
    x_1_0 = test_module_out(high_tap)

    return x, mid_tap, mid_tap_skip, high_tap, high_tap_skip, x_skip, x_1, x_2, x_1_5, x_2_5, x_1_0, x_2_0


def test_multi_rnn_out(x, x_skip, activation, regularisation, lone, ltwo, dropout, filters, output_size, initializer,
                       down_layers, batch_normalisation_bool, up_filters, out_layers, mid_out_layers, high_out_layers,
                       rnn_type, internal_bool, units, mid_tap_units, high_tap_units, rnn_return_activation,
                       rnn_initializer, rnn_recurrent_initializer, unroll, rnn_batch_normalisation_bool, rnn_activation,
                       rnn_lone, rnn_ltwo, rnn_dropout, rnn_regularisation, rnn_return_sequences,
                       feature_downsample_bool, concat_bool, skip_bool, dense_bool, ascending_bool, deep_bool,
                       skip_filters_bool):
    x = test_in(x, filters, initializer, batch_normalisation_bool, activation, feature_downsample_bool, skip_bool,
                dense_bool, regularisation, lone, ltwo, dropout, deep_bool)

    x, mid_tap, mid_tap_skip, high_tap, high_tap_skip, x_skip = test_down(x, x_skip, down_layers, filters, initializer,
                                                                          batch_normalisation_bool, activation,
                                                                          regularisation, lone, ltwo, dropout,
                                                                          feature_downsample_bool, concat_bool,
                                                                          skip_bool, ascending_bool, deep_bool,
                                                                          skip_filters_bool)

    x_2 = test_up(x, x_skip, False, 0.0, up_filters, initializer, batch_normalisation_bool, activation, "x", skip_bool,
                  lone, ltwo, skip_filters_bool)
    x_2_5 = test_up(mid_tap, mid_tap_skip, False, 0.0, up_filters, initializer, batch_normalisation_bool, activation,
                    "mid_tap", skip_bool, lone, ltwo, skip_filters_bool)
    x_2_0 = test_up(high_tap, high_tap_skip, False, 0.0, up_filters, initializer, batch_normalisation_bool, activation,
                    "high_tap", skip_bool, lone, ltwo, skip_filters_bool)

    x_1 = test_washer(x, output_size, initializer, batch_normalisation_bool, activation, regularisation, dropout, "x",
                      lone, ltwo, skip_bool, skip_filters_bool)
    x_1_5 = test_washer(mid_tap, output_size, initializer, batch_normalisation_bool, activation, regularisation,
                        dropout, "mid_tap", lone, ltwo, skip_bool, skip_filters_bool)
    x_1_0 = test_washer(high_tap, output_size, initializer, batch_normalisation_bool, activation, regularisation,
                        dropout, "high_tap", lone, ltwo, skip_bool, skip_filters_bool)

    x_1 = test_module_rnn_out(x_1, out_layers, rnn_type, internal_bool, units, rnn_return_activation, rnn_initializer,
                              rnn_recurrent_initializer, unroll, rnn_batch_normalisation_bool, rnn_activation, rnn_lone,
                              rnn_ltwo, rnn_dropout, rnn_regularisation, rnn_return_sequences, "x", skip_bool,
                              skip_filters_bool)
    x_1_5 = test_module_rnn_out(x_1_5, mid_out_layers, rnn_type, internal_bool, mid_tap_units, rnn_return_activation,
                                rnn_initializer, rnn_recurrent_initializer, unroll, rnn_batch_normalisation_bool,
                                rnn_activation, rnn_lone, rnn_ltwo, rnn_dropout, rnn_regularisation,
                                rnn_return_sequences, "mid_tap", skip_bool, skip_filters_bool)
    x_1_0 = test_module_rnn_out(x_1_0, high_out_layers, rnn_type, internal_bool, high_tap_units, rnn_return_activation,
                                rnn_initializer, rnn_recurrent_initializer, unroll, rnn_batch_normalisation_bool,
                                rnn_activation, rnn_lone, rnn_ltwo, rnn_dropout, rnn_regularisation,
                                rnn_return_sequences, "high_tap", skip_bool, skip_filters_bool)

    return x, mid_tap, mid_tap_skip, high_tap, high_tap_skip, x_skip, x_1, x_2, x_1_5, x_2_5, x_1_0, x_2_0


def output_module_1(x, internal_bool, rnn_type, washer_units, units, rnn_activation, rnn_initializer,
                    rnn_recurrent_initializer, unroll, rnn_recurrent_activation, initializer, activation,
                    return_sequence, name, regularisation, lone, ltwo):
    x = output_module_1_module(x, internal_bool, rnn_type, washer_units, rnn_activation, rnn_initializer,
                               rnn_recurrent_initializer, unroll, rnn_recurrent_activation, return_sequence, name,
                               regularisation, lone, ltwo)

    if regularisation:
        if return_sequence:
            x = k.layers.TimeDistributed(k.layers.Dense(units=1,
                                                        kernel_initializer=initializer,
                                                        bias_initializer=k.initializers.Constant(0.1),
                                                        kernel_regularizer=k.regularizers.l1_l2(l1=lone, l2=ltwo),
                                                        kernel_constraint=k.constraints.UnitNorm()))(x)

            x = test_activation(x, False, activation, False, 0.0)
        else:
            x = k.layers.Dense(units=units,
                               kernel_initializer=initializer,
                               bias_initializer=k.initializers.Constant(0.1),
                               kernel_regularizer=k.regularizers.l1_l2(l1=lone, l2=ltwo),
                               kernel_constraint=k.constraints.UnitNorm())(x)

            x = test_activation(x, False, activation, False, 0.0)
    else:
        if return_sequence:
            x = k.layers.TimeDistributed(k.layers.Dense(units=1,
                                                        kernel_initializer=initializer,
                                                        bias_initializer=k.initializers.Constant(0.1)))(x)

            x = test_activation(x, False, activation, False, 0.0)
        else:
            x = k.layers.Dense(units=units,
                               kernel_initializer=initializer,
                               bias_initializer=k.initializers.Constant(0.1))(x)

            x = test_activation(x, False, activation, False, 0.0)

    x = k.layers.Reshape(tuple([t for t in x.shape.as_list() if t != 1 and t is not None]), name=name)(x)

    return x


def output_module_1_module(x, internal_bool, rnn_type, units, activation, initializer, recurrent_initializer, unroll,
                           recurrent_activation, return_sequence, name, regularisation, lone, ltwo):
    if regularisation:
        if internal_bool:
            if rnn_type == "rnn":
                x = k.layers.SimpleRNN(units=units,
                                       activation=activation,
                                       return_sequences=return_sequence,
                                       kernel_initializer=initializer,
                                       bias_initializer=k.initializers.Constant(0.1),
                                       recurrent_initializer=recurrent_initializer,
                                       kernel_regularizer=k.regularizers.l1_l2(l1=lone, l2=ltwo),
                                       kernel_constraint=k.constraints.UnitNorm(),
                                       unroll=unroll,
                                       name="{0}_rnn_washer".format(name))(x)
            else:
                if rnn_type == "lstm":
                    x = k.layers.LSTM(units=units,
                                      activation=activation,
                                      recurrent_activation=recurrent_activation,
                                      return_sequences=return_sequence,
                                      kernel_initializer=initializer,
                                      bias_initializer=k.initializers.Constant(0.1),
                                      recurrent_initializer=recurrent_initializer,
                                      kernel_regularizer=k.regularizers.l1_l2(l1=lone, l2=ltwo),
                                      kernel_constraint=k.constraints.UnitNorm(),
                                      unroll=unroll,
                                      name="{0}_rnn_washer".format(name))(x)
                else:
                    if rnn_type == "gru":
                        x = k.layers.GRU(units=units,
                                         activation=activation,
                                         recurrent_activation=recurrent_activation,
                                         return_sequences=return_sequence,
                                         kernel_initializer=initializer,
                                         bias_initializer=k.initializers.Constant(0.1),
                                         recurrent_initializer=recurrent_initializer,
                                         kernel_regularizer=k.regularizers.l1_l2(l1=lone, l2=ltwo),
                                         kernel_constraint=k.constraints.UnitNorm(),
                                         unroll=unroll,
                                         name="{0}_rnn_washer".format(name))(x)
        else:
            if rnn_type == "rnn":
                x = k.layers.SimpleRNN(units=units,
                                       return_sequences=return_sequence,
                                       kernel_initializer=initializer,
                                       bias_initializer=k.initializers.Constant(0.1),
                                       recurrent_initializer=recurrent_initializer,
                                       kernel_regularizer=k.regularizers.l1_l2(l1=lone, l2=ltwo),
                                       kernel_constraint=k.constraints.UnitNorm(),
                                       unroll=unroll,
                                       name="{0}_rnn_washer".format(name))(x)

                x = test_activation(x, False, activation, False, 0.0)
            else:
                if rnn_type == "lstm":
                    x = k.layers.LSTM(units=units,
                                      recurrent_activation=recurrent_activation,
                                      return_sequences=return_sequence,
                                      kernel_initializer=initializer,
                                      bias_initializer=k.initializers.Constant(0.1),
                                      recurrent_initializer=recurrent_initializer,
                                      kernel_regularizer=k.regularizers.l1_l2(l1=lone, l2=ltwo),
                                      kernel_constraint=k.constraints.UnitNorm(),
                                      unroll=unroll,
                                      name="{0}_rnn_washer".format(name))(x)

                    x = test_activation(x, False, activation, False, 0.0)
                else:
                    if rnn_type == "gru":
                        x = k.layers.GRU(units=units,
                                         recurrent_activation=recurrent_activation,
                                         return_sequences=return_sequence,
                                         kernel_initializer=initializer,
                                         bias_initializer=k.initializers.Constant(0.1),
                                         recurrent_initializer=recurrent_initializer,
                                         kernel_regularizer=k.regularizers.l1_l2(l1=lone, l2=ltwo),
                                         kernel_constraint=k.constraints.UnitNorm(),
                                         unroll=unroll,
                                         name="{0}_rnn_washer".format(name))(x)

                        x = test_activation(x, False, activation, False, 0.0)
    else:
        if internal_bool:
            if rnn_type == "rnn":
                x = k.layers.SimpleRNN(units=units,
                                       activation=activation,
                                       return_sequences=return_sequence,
                                       kernel_initializer=initializer,
                                       bias_initializer=k.initializers.Constant(0.1),
                                       recurrent_initializer=recurrent_initializer,
                                       unroll=unroll,
                                       name="{0}_rnn_washer".format(name))(x)
            else:
                if rnn_type == "lstm":
                    x = k.layers.LSTM(units=units,
                                      activation=activation,
                                      recurrent_activation=recurrent_activation,
                                      return_sequences=return_sequence,
                                      kernel_initializer=initializer,
                                      bias_initializer=k.initializers.Constant(0.1),
                                      recurrent_initializer=recurrent_initializer,
                                      unroll=unroll,
                                      name="{0}_rnn_washer".format(name))(x)
                else:
                    if rnn_type == "gru":
                        x = k.layers.GRU(units=units,
                                         activation=activation,
                                         recurrent_activation=recurrent_activation,
                                         return_sequences=return_sequence,
                                         kernel_initializer=initializer,
                                         bias_initializer=k.initializers.Constant(0.1),
                                         recurrent_initializer=recurrent_initializer,
                                         unroll=unroll,
                                         name="{0}_rnn_washer".format(name))(x)
        else:
            if rnn_type == "rnn":
                x = k.layers.SimpleRNN(units=units,
                                       return_sequences=return_sequence,
                                       kernel_initializer=initializer,
                                       bias_initializer=k.initializers.Constant(0.1),
                                       recurrent_initializer=recurrent_initializer,
                                       unroll=unroll,
                                       name="{0}_rnn_washer".format(name))(x)

                x = test_activation(x, False, activation, False, 0.0)
            else:
                if rnn_type == "lstm":
                    x = k.layers.LSTM(units=units,
                                      recurrent_activation=recurrent_activation,
                                      return_sequences=return_sequence,
                                      kernel_initializer=initializer,
                                      bias_initializer=k.initializers.Constant(0.1),
                                      recurrent_initializer=recurrent_initializer,
                                      unroll=unroll,
                                      name="{0}_rnn_washer".format(name))(x)

                    x = test_activation(x, False, activation, False, 0.0)
                else:
                    if rnn_type == "gru":
                        x = k.layers.GRU(units=units,
                                         recurrent_activation=recurrent_activation,
                                         return_sequences=return_sequence,
                                         kernel_initializer=initializer,
                                         bias_initializer=k.initializers.Constant(0.1),
                                         recurrent_initializer=recurrent_initializer,
                                         unroll=unroll,
                                         name="{0}_rnn_washer".format(name))(x)

                        x = test_activation(x, False, activation, False, 0.0)

    return x


def output_module_2(x, initializer, activation, name):
    x = k.layers.TimeDistributed(k.layers.Convolution3D(filters=1,
                                                        kernel_size=(3, 3, 3),
                                                        strides=(1, 1, 1),
                                                        padding='same',
                                                        kernel_initializer=initializer,
                                                        bias_initializer=k.initializers.Constant(0.1)))(x)

    x = test_activation(x, False, activation, False, 0.0)

    x = k.layers.Reshape(tuple([t for t in x.shape.as_list() if t is not None]), name=name)(x)

    return x
