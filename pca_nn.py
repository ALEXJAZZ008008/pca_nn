from __future__ import division, print_function
import os
import time
import math
from random import seed
from random import randint
import numpy as np
import scipy.io
import scipy.stats
from sklearn.preprocessing import RobustScaler
import keras as k

import test_2


def get_x_stochastic(input_path, start_position, data_window_size, window_size, data_size, window_stride_size,
                     cut_list):
    out_of_bounds_bool = False

    if start_position + data_window_size + window_size >= data_size:
        start_position = data_size - data_window_size

        out_of_bounds_bool = True

    print("Getting x")

    x = []

    sinos_array = np.load(input_path)

    data_load_out_of_bounds_bool = False

    stochastic_i_list = []

    seed(int(time.time()))

    for i in range(start_position, start_position + data_window_size, window_stride_size):
        x_window = []

        stochastic_i = data_size
        stochastic_i_bool = True

        while stochastic_i_bool:
            stochastic_i = randint(0, (data_size - window_size) - 1)

            stochastic_i_bool = False

            for j in range(len(cut_list)):
                if stochastic_i >= cut_list[j] - window_size and stochastic_i <= cut_list[j]:
                    stochastic_i_bool = True

                    break

        stochastic_i_list.append(stochastic_i)

        for j in range(window_size):
            if out_of_bounds_bool:
                if stochastic_i + j >= data_size:
                    data_load_out_of_bounds_bool = True

                    break

            x_window.append(sinos_array[stochastic_i + j].T)

        if data_load_out_of_bounds_bool:
            break

        x.append(np.asfarray(x_window))

    print("Got x")

    return np.nan_to_num(np.expand_dims(np.asfarray(x), axis=5)).astype(
        np.float32), start_position, out_of_bounds_bool, stochastic_i_list


def get_y_stochastic(input_path, window_size, data_size, out_of_bounds_bool, stochastic_i_list):
    print("Get y")

    y = []

    test_array = np.load(input_path)

    data_load_out_of_bounds_bool = False

    for i in range(len(stochastic_i_list)):
        y_window = []

        for j in range(window_size):
            if out_of_bounds_bool:
                if stochastic_i_list[i] + j >= data_size:
                    data_load_out_of_bounds_bool = True

                    break

            y_window.append(test_array[stochastic_i_list[i] + j])

        if data_load_out_of_bounds_bool:
            break

        y.append(np.squeeze(np.asfarray(y_window)))

    print("Got y")

    return np.nan_to_num(np.asfarray(y)).astype(np.float32)


def get_x(input_path, start_position, data_window_size, window_size, data_size, window_stride_size):
    out_of_bounds_bool = False

    if start_position + data_window_size + window_size >= data_size:
        start_position = data_size - data_window_size

        out_of_bounds_bool = True

    print("Getting x")

    x = []

    sinos_array = np.load(input_path)

    data_load_out_of_bounds_bool = False

    for i in range(start_position, start_position + data_window_size, window_stride_size):
        x_window = []

        for j in range(window_size):
            if out_of_bounds_bool:
                if i + j >= data_size:
                    data_load_out_of_bounds_bool = True

                    break

            x_window.append(sinos_array[i + j].T)

        if data_load_out_of_bounds_bool:
            break

        x.append(np.asfarray(x_window))

    print("Got x")

    return np.nan_to_num(np.expand_dims(np.asfarray(x), axis=5)).astype(np.float32), start_position, out_of_bounds_bool


def get_y(input_path, start_position, data_window_size, window_size, data_size, window_stride_size, out_of_bounds_bool):
    print("Get y")

    y = []

    test_array = np.load(input_path)

    data_load_out_of_bounds_bool = False

    for i in range(start_position, start_position + data_window_size, window_stride_size):
        y_window = []

        for j in range(window_size):
            if out_of_bounds_bool:
                if i + j >= data_size:
                    data_load_out_of_bounds_bool = True

                    break

            y_window.append(test_array[i + j])

        if data_load_out_of_bounds_bool:
            break

        y.append(np.squeeze(np.asfarray(y_window)))

    print("Got y")

    return np.nan_to_num(np.asfarray(y)).astype(np.float32)


def fit_model(input_model,
              save_bool,
              load_bool,
              plot_bool,
              apply_bool,
              x_input_path,
              y_input_path,
              path,
              start_position,
              data_window_size,
              window_size,
              data_size,
              window_stride_size,
              output_path,
              tof_bool,
              stochastic_bool,
              passthrough_bool,
              epochs,
              output_all_bool,
              number_of_bins,
              cut_list,
              mid_tap_bool,
              high_tap_bool):
    print("Get training data")

    if stochastic_bool:
        x_train, start_position, out_of_bounds_bool, stochastic_i_list = get_x_stochastic(x_input_path,
                                                                                          start_position,
                                                                                          data_window_size,
                                                                                          window_size,
                                                                                          data_size,
                                                                                          window_stride_size,
                                                                                          cut_list)
        y_train = get_y_stochastic(y_input_path, window_size, data_size, out_of_bounds_bool, stochastic_i_list)
    else:
        x_train, start_position, out_of_bounds_bool = get_x(x_input_path,
                                                            start_position,
                                                            data_window_size,
                                                            window_size,
                                                            data_size,
                                                            window_stride_size)
        y_train = get_y(y_input_path,
                        start_position,
                        data_window_size,
                        window_size,
                        data_size,
                        window_stride_size,
                        out_of_bounds_bool)

    if input_model is None:
        print("No input model")

        if load_bool:
            print("Load model from file")

            model = k.models.load_model(output_path + "/model.h5")
        else:
            print("Generate new model")

            output_size = y_train.shape[1:][0]

            input_x = k.layers.Input(x_train.shape[1:])

            x = input_x
            x_skip = []

            regularisation = True
            rnn_units = 40
            rnn_mid_tap_units = 40
            rnn_high_tap_units = 40
            batch_normalisation_bool = True
            rnn_lone = 0.0001
            rnn_ltwo = 0.0001

            if regularisation:
                x, mid_tap, mid_tap_skip, high_tap, high_tap_skip, x_skip, x_1, x_2, x_1_5, x_2_5, x_1_0, x_2_0 = test_2.test_multi_rnn_out(
                    x, x_skip, "selu", regularisation, 0.0001, 0.0001, 0.0, 8, "lecun_normal", 7, True, 4, 1, 1, 1, "lstm",
                    True, rnn_units * 2, rnn_mid_tap_units * 2, rnn_high_tap_units * 2, "sigmoid", "glorot_normal",
                    "glorot_uniform", False, batch_normalisation_bool, "tanh", rnn_lone, rnn_ltwo, 0.5, regularisation,
                    True, True, False, False, False, False, False, False)
            else:
                x, mid_tap, mid_tap_skip, high_tap, high_tap_skip, x_skip, x_1, x_2, x_1_5, x_2_5, x_1_0, x_2_0 = test_2.test_multi_rnn_out(
                    x, x_skip, "selu", regularisation, 0.0, 0.0, 0.0, 8, "lecun_normal", 7, True, 4, 1, 1, 1, "lstm",
                    True, rnn_units, rnn_mid_tap_units, rnn_high_tap_units, "sigmoid", "glorot_normal",
                    "glorot_uniform", False, batch_normalisation_bool, "tanh", rnn_lone, rnn_ltwo, 0.0, regularisation,
                    True, True, False, False, False, False, False, False)

            x_1 = test_2.output_module_1(x_1, True, "lstm", rnn_units, output_size, "tanh", "glorot_normal",
                                         "glorot_uniform", False, "sigmoid", "glorot_normal", "linear", True,
                                         "output_1", regularisation, rnn_lone, rnn_ltwo, batch_normalisation_bool)
            x_2 = test_2.output_module_2(x_2, "glorot_normal", "linear", "output_2")

            x_1_5 = test_2.output_module_1(x_1_5, True, "lstm", rnn_mid_tap_units, output_size, "tanh", "glorot_normal",
                                           "glorot_uniform", False, "sigmoid", "glorot_normal", "linear", True,
                                           "output_3", regularisation, rnn_lone, rnn_ltwo, batch_normalisation_bool)
            x_2_5 = test_2.output_module_2(x_2_5, "glorot_normal", "linear", "output_4")

            x_1_0 = test_2.output_module_1(x_1_0, True, "lstm", rnn_high_tap_units, output_size, "tanh",
                                           "glorot_normal", "glorot_uniform", False, "sigmoid", "glorot_normal",
                                           "linear", False, "output_5", regularisation, rnn_lone, rnn_ltwo,
                                           batch_normalisation_bool)
            x_2_0 = test_2.output_module_2(x_2_0, "glorot_normal", "linear", "output_6")

            #if mid_tap_bool:
            #    if high_tap_bool:
            #        model = k.Model(inputs=input_x, outputs=[x_1, x_2, x_1_5, x_2_5, x_1_0, x_2_0])
            #    else:
            #        model = k.Model(inputs=input_x, outputs=[x_1, x_2, x_1_5, x_2_5])
            #else:
            #    model = k.Model(inputs=input_x, outputs=[x_1, x_2])

            model = k.Model(inputs=input_x, outputs=x_1)

            lr = 0.01

            #if mid_tap_bool:
            #    if high_tap_bool:
            #        model.compile(
            #            optimizer=k.optimizers.SGD(learning_rate=lr, momentum=0.99, nesterov=True, clipnorm=1.0),
            #            loss=k.losses.mean_squared_error, loss_weights=[1.0, 0.1, 1.0, 0.1, 1.0, 0.1])
            #    else:
            #        model.compile(
            #            optimizer=k.optimizers.SGD(learning_rate=lr, momentum=0.99, nesterov=True, clipnorm=1.0),
            #            loss=k.losses.mean_squared_error, loss_weights=[1.0, 0.1, 1.0, 0.1])
            #else:
            #    model.compile(optimizer=k.optimizers.SGD(learning_rate=lr, momentum=0.99, nesterov=True, clipnorm=1.0),
            #                  loss=k.losses.mean_squared_error, loss_weights=[1.0, 0.1])

            model.compile(optimizer=k.optimizers.SGD(learning_rate=lr, momentum=0.99, nesterov=True, clipnorm=1.0), loss=k.losses.mean_squared_error)

            with open(output_path + "/lr", "w") as file:
                file.write(str(lr))

            batch_size = 1

            with open(output_path + "/batch_size", "w") as file:
                file.write(str(batch_size))
    else:
        print("Using input model")

        model = input_model

    model.summary()

    if plot_bool:
        k.utils.plot_model(model, output_path + "model.png")

    if not passthrough_bool:
        print("Fitting model")

        with open(output_path + "/lr", "r") as file:
            lr = float(file.read())

        k.backend.set_value(model.optimizer.lr, lr)

        print("lr: " + str(k.backend.get_value(model.optimizer.lr)))

        with open(output_path + "/batch_size", "r") as file:
            batch_size = int(file.read())

        reduce_lr = k.callbacks.ReduceLROnPlateau(monitor="loss", factor=0.9, patience=epochs - 1, verbose=1,
                                                  cooldown=1)
        tensorboard_callback = k.callbacks.TensorBoard(log_dir=output_path + "/log")

        #if mid_tap_bool:
        #    if high_tap_bool:
        #        model.fit(x_train, {"output_1": y_train, "output_2": x_train, "output_3": y_train, "output_4": x_train,
        #                            "output_5": y_train, "output_6": x_train}, batch_size=batch_size, epochs=epochs,
        #                  callbacks=[reduce_lr], verbose=1)
        #    else:
        #        model.fit(x_train, {"output_1": y_train, "output_2": x_train, "output_3": y_train, "output_4": x_train},
        #                  batch_size=batch_size, epochs=epochs, callbacks=[reduce_lr], verbose=1)
        #else:
        #    model.fit(x_train, {"output_1": y_train, "output_2": x_train}, batch_size=batch_size, epochs=epochs,
        #              callbacks=[reduce_lr], verbose=1)

        model.fit(x_train, y_train, batch_size=batch_size, epochs=epochs, verbose=1)

        output_lr = float(k.backend.get_value(model.optimizer.lr))

        if not math.isclose(output_lr, lr, rel_tol=1e-3):
            with open(output_path + "/lr", "w") as file:
                file.write(str(output_lr))

            max_batch_size = int(data_window_size / 10)

            if batch_size <= max_batch_size:
                batch_size = batch_size + 1

            if batch_size > max_batch_size:
                batch_size = max_batch_size

            with open(output_path + "/batch_size", "w") as file:
                file.write(str(int(batch_size)))

        print("lr: " + str(k.backend.get_value(model.optimizer.lr)))

    if save_bool:
        print("Saving model")

        model.save(output_path + "/model.h5")

    if not passthrough_bool:
        if apply_bool:
            test_model(model,
                       x_input_path,
                       y_input_path,
                       path,
                       start_position,
                       data_window_size,
                       window_size,
                       data_size,
                       window_stride_size,
                       output_path,
                       output_path,
                       output_all_bool,
                       number_of_bins)

    return model, start_position, out_of_bounds_bool


def evaluate_model(input_model,
                   x_input_path,
                   y_input_path,
                   start_position,
                   data_window_size,
                   window_size,
                   data_size,
                   window_stride_size,
                   model_input_path,
                   cut_list,
                   mid_tap_bool,
                   high_tap_bool):
    print("Get test data")

    x_test, start_position, out_of_bounds_bool, stochastic_i_list = get_x_stochastic(x_input_path,
                                                                                     start_position,
                                                                                     data_window_size,
                                                                                     window_size,
                                                                                     data_size,
                                                                                     window_stride_size,
                                                                                     cut_list)
    y_test = get_y_stochastic(y_input_path, window_size, data_size, out_of_bounds_bool, stochastic_i_list)

    if input_model is None:
        print("No input model")
        print("Load model from file")

        model = k.models.load_model(model_input_path + "/model.h5")
    else:
        model = input_model

    print("Applying model")

    if mid_tap_bool:
        if high_tap_bool:
            output = model.evaluate(x_test,
                                    {"output_1": y_test, "output_2": x_test, "output_3": y_test, "output_4": x_test,
                                     "output_5": y_test, "output_6": x_test}, batch_size=1, verbose=1)
        else:
            output = model.evaluate(x_test,
                                    {"output_1": y_test, "output_2": x_test, "output_3": y_test, "output_4": x_test},
                                    batch_size=1, verbose=1)
    else:
        output = model.evaluate(x_test, {"output_1": y_test, "output_2": x_test}, batch_size=1, verbose=1)

    #if mid_tap_bool:
    #    if high_tap_bool:
    #        output_string = 'Test loss: {0}, Test loss output_1: {1}, Test loss output_2: {2}, Test loss output_3: {3}, Test loss output_4: {4}, Test loss output_5: {3}, Test loss output_6: {4}'.format(
    #            output[0], output[1], output[2], output[3], output[4], output[5], output[6])
    #    else:
    #        output_string = 'Test loss: {0}, Test loss output_1: {1}, Test loss output_2: {2}, Test loss output_3: {3}, Test loss output_4: {4}'.format(
    #            output[0], output[1], output[2], output[3], output[4])
    #else:
    #    output_string = 'Test loss: {0}, Test loss output_1: {1}, Test loss output_2: {2}'.format(output[0], output[1],
    #                                                                                              output[2])

    output_string = 'Test loss: {0}'.format(output)

    print(output_string)

    return model, output_string


def write_to_file(file, data):
    for i in range(len(data)):
        output_string = ""

        for j in range(len(data[i])):
            output_string = output_string + str(data[i][j]) + ","

        output_string = output_string[:-1] + "\n"

        file.write(output_string)


def test_model(input_model,
               x_input_path,
               y_input_path,
               path,
               start_position,
               data_window_size,
               window_size,
               data_size,
               window_stride_size,
               model_input_path,
               output_path,
               output_all_bool,
               number_of_bins):
    print("Get test data")

    x_test, start_position, out_of_bounds_bool = get_x(x_input_path,
                                                       start_position,
                                                       data_window_size,
                                                       window_size,
                                                       data_size,
                                                       window_stride_size)
    y_test = get_y(y_input_path,
                   start_position,
                   data_window_size,
                   window_size,
                   data_size,
                   window_stride_size,
                   out_of_bounds_bool)

    if input_model is None:
        print("No input model")
        print("Load model from file")

        model = k.models.load_model(model_input_path + "/model.h5")
    else:
        model = input_model

    print("Applying model")

    output = model.predict(x_test, batch_size=1, verbose=1)

    if len(output) > 1:
        if len(output) > 2:
            if output_all_bool:
                for i in range(len(output[1])):
                    downsample_histogram_equalisation_and_standardise_input_data(output[1], number_of_bins,
                                                                                 output_path + "/test_estimated_input_" + str(
                                                                                     1) + "_" + str(
                                                                                     path) + "_" + str(
                                                                                     start_position) + "_" + str(i))

                for i in range(len(output[3])):
                    downsample_histogram_equalisation_and_standardise_input_data(output[1], number_of_bins,
                                                                                 output_path + "/test_estimated_input_" + str(
                                                                                     1) + "_" + str(
                                                                                     path) + "_" + str(
                                                                                     start_position) + "_" + str(i))

            output = [output[0], output[2]]

        else:
            if output_all_bool:
                for i in range(len(output[1])):
                    downsample_histogram_equalisation_and_standardise_input_data(output[1], number_of_bins,
                                                                                 output_path + "/test_estimated_input_" + str(
                                                                                     1) + "_" + str(
                                                                                     path) + "_" + str(
                                                                                     start_position) + "_" + str(i))

            output = [output[0]]

    for i in range(len(output)):
        current_output = output[i]

        with open(output_path + "/test_estimated_signal_" + str(i) + ".csv", "w") as file:
            write_to_file(file, current_output)

        difference_matrix = current_output - y_test
        difference_vector = np.abs(difference_matrix.flatten())

        print("Output " + str(i) + " max difference: " + str(difference_vector.max()))
        print("Output " + str(i) + " mean difference: " + str(difference_vector.mean()))

        boolean_difference = []

        for j in range(len(current_output)):
            for l in range(len(current_output[j])):
                if abs(current_output[j][l] - y_test[j][l]) < (np.max(y_test) - np.min(y_test)) / 10.0:
                    boolean_difference.append(np.array(0))
                else:
                    boolean_difference.append(np.array(1))

        absolute_difference = sum(boolean_difference)

        print("Output " + str(i) + " absolute boolean difference: " + str(absolute_difference) + "/" + str(
            len(y_test) * window_size))
        print("Output " + str(i) + " relative boolean difference: " + str(
            (((float(absolute_difference) / float(window_size)) / float(len(y_test))) * float(100))) + "%")

        with open(output_path + "/difference_" + str(i) + ".csv", "w") as file:
            write_to_file(file, difference_matrix)

    return output, start_position, out_of_bounds_bool


def histogram_equalisation(data_array, number_of_bins):
    hist, bins = np.histogram(data_array.flatten(), number_of_bins, density=True)

    cdf = hist.cumsum()
    cdf = 255 * cdf / cdf[-1]  # normalize

    output = np.interp(data_array.flatten(), bins[:-1], cdf).reshape(data_array.shape)

    return output


def downsample_histogram_equalisation_and_standardise_input_data(input_data, number_of_bins, output_path):
    data_array = histogram_equalisation(input_data, number_of_bins)

    data_array = scipy.stats.zscore(data_array)

    np.save(output_path, data_array.astype(np.float32))


def downsample_histogram_equalisation_and_standardise(input_path, tof_bool, number_of_bins, output_path):
    for i in range(len(input_path)):
        data = scipy.io.loadmat(input_path[i])

        data_array = data.get(list(data.keys())[3])

        if not tof_bool:
            data_array = np.mean(data_array, 3)

        data_array = data_array.T

        data_array = histogram_equalisation(data_array, number_of_bins)

        data_array_shape = data_array.shape

        data_array = np.reshape(data_array.flatten(), [-1, 1])

        transformer = RobustScaler().fit(data_array)
        data_array = transformer.transform(data_array)

        data_array = np.reshape(data_array, data_array_shape)

        np.save(output_path[i], data_array.astype(np.float32))


def downsample_histogram_equalisation_and_zscore(input_path, tof_bool, number_of_bins, output_path):
    for i in range(len(input_path)):
        data = scipy.io.loadmat(input_path[i])

        data_array = data.get(list(data.keys())[3])

        if not tof_bool:
            data_array = np.mean(data_array, 3)

        data_array = data_array.T

        data_array = histogram_equalisation(data_array, number_of_bins)

        data_array = scipy.stats.zscore(data_array)

        np.save(output_path[i], data_array.astype(np.float32))


def downsample_and_standardise(input_path, tof_bool, output_path):
    for i in range(len(input_path)):
        data = scipy.io.loadmat(input_path[i])

        data_array = data.get(list(data.keys())[3])

        if not tof_bool:
            data_array = np.nanmean(data_array, 3)

        data_array = data_array.T

        data_array_shape = data_array.shape

        np.reshape(data_array.flatten(), [-1, 1])

        transformer = RobustScaler().fit(data_array)
        data_array = transformer.transform(data_array)

        data_array = np.reshape(data_array, data_array_shape)

        np.save(output_path[i], data_array.astype(np.float32))


def downsample_and_zscore(input_path, tof_bool, output_path):
    for i in range(len(input_path)):
        data = scipy.io.loadmat(input_path[i])

        data_array = data.get(list(data.keys())[3])

        if not tof_bool:
            data_array = np.nanmean(data_array, 3)

        data_array = data_array.T

        data_array = scipy.stats.zscore(data_array)

        np.save(output_path[i], data_array.astype(np.float32))


# https://stackoverflow.com/questions/36000843/scale-numpy-array-to-certain-range
def rescale_linear(array, new_min, new_max):
    """Rescale an arrary linearly."""
    minimum, maximum = np.min(array), np.max(array)
    m = (new_max - new_min) / (maximum - minimum)
    b = new_min - m * minimum
    return m * array + b


def standardise(input_path, output_path):
    for i in range(len(input_path)):
        data = scipy.io.loadmat(input_path[i])
        data_array = data.get(list(data.keys())[3])

        data_array_shape = data_array.shape

        np.reshape(data_array.flatten(), [-1, 1])

        transformer = RobustScaler().fit(data_array)
        data_array = transformer.transform(data_array)

        data_array = np.reshape(data_array, data_array_shape)

        np.save(output_path[i], data_array.astype(np.float32))


def zscore(input_path, output_path):
    for i in range(len(input_path)):
        data = scipy.io.loadmat(input_path[i])
        data_array = data.get(list(data.keys())[3])

        data_array = scipy.stats.zscore(data_array)

        np.save(output_path[i], data_array.astype(np.float32))


def concat_array_list(data):
    data_list = []
    cut_list = []

    for i in range(len(data)):
        temp_list = []

        new_data = np.load(data[i])

        for j in range(len(data_list)):
            temp_list.append(data_list[j])

        cut_list.append(len(data_list))

        for j in range(len(new_data)):
            temp_list.append(new_data[j])

        data_list = temp_list

    cut_list.pop(0)

    return np.asfarray(data_list), cut_list


def concat_one_input(x, y):
    x_output = "x_one_input.npy"
    y_output = "y_one_input.npy"

    x_data, cut_list = concat_array_list(x)
    y_data, cut_list_y = concat_array_list(y)

    np.save(x_output, x_data.astype(np.float32))
    np.save(y_output, y_data.astype(np.float32))

    return [x_output], [y_output], cut_list


def main(fit_model_bool, while_bool, load_bool):
    save_bool = True
    plot_bool = True
    apply_bool = False
    passthrough_bool = False
    single_input_bool = False
    concat_one_input_bool = True
    reload_data = False
    tof_bool = False
    stochastic_bool = True
    flip_bool = False

    output_all_bool = False
    number_of_bins = 1000000

    cv_bool = True
    cv_pos = 0

    output_to_output = 0

    output_path = "./results/"
    output_prefix = ".csv"
    cv_tracker_path = None

    if not os.path.exists(output_path):
        os.makedirs(output_path)

    if single_input_bool:
        x_path_orig_list = ["./normalised_sinos_1.mat"]
        y_path_orig_list = ["./output_signal_1.mat"]

        x_path_list = ["normalised_sinos_preprocessed_1.npy"]
        y_path_list = ["output_signal_preprocessed_1.npy"]

        output_file_name = ["estimated_signal_1"]
    else:
        x_path_orig_list = ["./normalised_sinos_1.mat",
                            "./normalised_sinos_3.mat",
                            "./normalised_sinos_5.mat",
                            "./normalised_sinos_6.mat",
                            "./normalised_sinos_7.mat",
                            "./normalised_sinos_8.mat",
                            "./normalised_sinos_9.mat",
                            "./normalised_sinos_15.mat",
                            "./normalised_sinos_19.mat",
                            "./normalised_sinos_20.mat"]
        y_path_orig_list = ["./output_signal_1.mat",
                            "./output_signal_3.mat",
                            "./output_signal_5.mat",
                            "./output_signal_6.mat",
                            "./output_signal_7.mat",
                            "./output_signal_8.mat",
                            "./output_signal_9.mat",
                            "./output_signal_15.mat",
                            "./output_signal_19.mat",
                            "./output_signal_20.mat"]

        x_path_list = ["normalised_sinos_preprocessed_1.npy",
                       "normalised_sinos_preprocessed_3.npy",
                       "normalised_sinos_preprocessed_5.npy",
                       "normalised_sinos_preprocessed_6.npy",
                       "normalised_sinos_preprocessed_7.npy",
                       "normalised_sinos_preprocessed_8.npy",
                       "normalised_sinos_preprocessed_9.npy",
                       "normalised_sinos_preprocessed_15.npy",
                       "normalised_sinos_preprocessed_19.npy",
                       "normalised_sinos_preprocessed_20.npy"]
        y_path_list = ["output_signal_preprocessed_1.npy",
                       "output_signal_preprocessed_3.npy",
                       "output_signal_preprocessed_5.npy",
                       "output_signal_preprocessed_6.npy",
                       "output_signal_preprocessed_7.npy",
                       "output_signal_preprocessed_8.npy",
                       "output_signal_preprocessed_9.npy",
                       "output_signal_preprocessed_15.npy",
                       "output_signal_preprocessed_19.npy",
                       "output_signal_preprocessed_20.npy"]

        output_file_name = ["estimated_signal_1",
                            "estimated_signal_3",
                            "estimated_signal_5",
                            "estimated_signal_6",
                            "estimated_signal_7",
                            "estimated_signal_8",
                            "estimated_signal_9",
                            "estimated_signal_15",
                            "estimated_signal_19",
                            "estimated_signal_20"]

    test_x_path_orig_list = x_path_orig_list
    test_y_path_orig_list = y_path_orig_list

    test_x_path_list = x_path_list
    test_y_path_list = y_path_list

    test_output_file_name = output_file_name

    if reload_data:
        print("Getting data")

        downsample_histogram_equalisation_and_standardise(x_path_orig_list, tof_bool, number_of_bins, x_path_list)
        standardise(y_path_orig_list, y_path_list)

        print("Got data")

    if cv_bool:
        output_path = "./cv_{0}_results/".format(str(cv_pos))

        if not os.path.exists(output_path):
            os.makedirs(output_path)

        cv_tracker_path = "{0}/cv_tracker".format(output_path)

        if not os.path.isfile(cv_tracker_path):
            os.mknod(cv_tracker_path)

        temp_x_path_orig_list = x_path_orig_list
        temp_y_path_orig_list = y_path_orig_list

        temp_x_path_list = x_path_list
        temp_y_path_list = y_path_list

        temp_output_file_name = output_file_name

        x_path_orig_list = []
        y_path_orig_list = []

        x_path_list = []
        y_path_list = []

        output_file_name = []

        test_x_path_orig_list = []
        test_y_path_orig_list = []

        test_x_path_list = []
        test_y_path_list = []

        test_output_file_name = []

        for i in range(len(temp_x_path_orig_list)):
            if i == cv_pos:
                test_x_path_orig_list.append(temp_x_path_orig_list[i])
                test_y_path_orig_list.append(temp_y_path_orig_list[i])

                test_x_path_list.append(temp_x_path_list[i])
                test_y_path_list.append(temp_y_path_list[i])

                test_output_file_name.append(temp_output_file_name[i])
            else:
                x_path_orig_list.append(temp_x_path_orig_list[i])
                y_path_orig_list.append(temp_y_path_orig_list[i])

                x_path_list.append(temp_x_path_list[i])
                y_path_list.append(temp_y_path_list[i])

                output_file_name.append(temp_output_file_name[i])

        if len(x_path_orig_list) < 1:
            x_path_orig_list = test_x_path_orig_list
            y_path_orig_list = test_y_path_orig_list

            x_path_list = test_x_path_list
            y_path_list = test_y_path_list

            output_file_name = test_output_file_name

        if len(test_x_path_orig_list) < 1:
            test_x_path_orig_list = x_path_orig_list
            test_path_orig_list = y_path_orig_list

            test_x_path_list = x_path_list
            test_y_path_list = y_path_list

            test_output_file_name = output_file_name

    window_size = 40
    window_stride_size = math.floor(window_size / 2.0)

    if fit_model_bool:
        window_stride_size = 1
        epochs = 1

        cut_list = []

        mid_tap_bool = False
        high_tap_bool = False

        if concat_one_input_bool:
            x_path_list, y_path_list, cut_list = concat_one_input(x_path_list, y_path_list)

        if load_bool:
            with open(output_path + "/path_start_point", "r") as file:
                path_start_point = int(file.read())

            with open(output_path + "/data_start_point", "r") as file:
                data_start_point = int(file.read())

            with open(output_path + "/cv_index", "r") as file:
                cv_index = int(file.read())
        else:
            path_start_point = 0

            with open(output_path + "/path_start_point", "w") as file:
                file.write(str(path_start_point))

            data_start_point = 0

            with open(output_path + "/data_start_point", "w") as file:
                file.write(str(data_start_point))

            cv_index = 0

            with open(output_path + "/cv_index", "w") as file:
                file.write(str(cv_index))

        path_length = len(x_path_list)

        while_model = None

        while True:
            for i in range(path_start_point, path_length, 1):
                with open(output_path + "/path_start_point", "w") as file:
                    file.write(str(i))

                print("Path: " + str(i) + "/" + str(path_length))

                data_size = np.load(y_path_list[i]).shape[0]

                test_data_size = None

                if cv_bool:
                    test_data_size = np.load(test_y_path_list[i]).shape[0]

                if tof_bool:
                    ideal_data_window_size = window_size
                else:
                    ideal_data_window_size = (data_size / len(x_path_orig_list)) / 10

                if ideal_data_window_size >= data_size:
                    ideal_data_window_size = data_size - 1

                data_window_size = 0

                while data_window_size < ideal_data_window_size:
                    data_window_size = data_window_size + window_size

                if data_window_size >= data_size:
                    data_window_size = data_size - 1

                data_window_size = int(data_window_size)

                data_window_stride_size = data_window_size

                for j in range(data_start_point, data_size, data_window_stride_size):

                    print("Data: " + str(j) + "/" + str(data_size))

                    if not concat_one_input_bool:
                        cut_list = [data_size]

                    out_of_bounds_bool = False

                    while_model, start_position, out_of_bounds_bool = fit_model(while_model,
                                                                                save_bool,
                                                                                load_bool,
                                                                                plot_bool,
                                                                                apply_bool,
                                                                                x_path_list[i],
                                                                                y_path_list[i],
                                                                                i,
                                                                                j,
                                                                                data_window_size,
                                                                                window_size,
                                                                                data_size,
                                                                                window_stride_size,
                                                                                output_path,
                                                                                tof_bool,
                                                                                stochastic_bool,
                                                                                passthrough_bool,
                                                                                epochs,
                                                                                output_all_bool,
                                                                                number_of_bins,
                                                                                cut_list,
                                                                                mid_tap_bool,
                                                                                high_tap_bool)

                    if cv_bool:
                        cv_index = cv_index + 1

                        while_model, output = evaluate_model(while_model, test_x_path_list[0], test_y_path_list[0], 0,
                                                             data_window_size, window_size, test_data_size,
                                                             window_stride_size, output_path, [test_data_size],
                                                             mid_tap_bool, high_tap_bool)

                        with open(cv_tracker_path, 'a') as file:
                            file.write("{0}\n".format(str(output)))

                        while_model.save(output_path + "/cv_model_{0}.h5".format(str(cv_index)))

                        with open(output_path + "/cv_index", "w") as file:
                            file.write(str(cv_index))

                    with open(output_path + "/data_start_point", "w") as file:
                        file.write(str(j))

                    if out_of_bounds_bool:
                        break

                data_start_point = 0

                with open(output_path + "/data_start_point", "w") as file:
                    file.write(str(data_start_point))

            path_start_point = 0

            with open(output_path + "/path_start_point", "w") as file:
                file.write(str(path_start_point))

            print("Done")

            if not while_bool:
                break
    else:
        print("Test model")
        print("Load model from file")

        model = k.models.load_model(output_path + "/model.h5")

        path_length = len(x_path_list)

        for i in range(path_length):
            print("Path: " + str(i) + "/" + str(path_length))

            data_array = np.load(y_path_list[i])
            data_size = data_array.shape[0]

            if tof_bool:
                ideal_data_window_size = window_size
            else:
                ideal_data_window_size = (data_size / len(x_path_orig_list)) / 10

            if ideal_data_window_size >= data_size:
                ideal_data_window_size = data_size - 1

            data_window_size = 0

            while data_window_size < ideal_data_window_size:
                data_window_size = data_window_size + window_size

            if data_window_size >= data_size:
                data_window_size = data_size - 1

            data_window_size = int(data_window_size)

            data_window_stride_size = data_window_size

            output_list = []

            for j in range(0, data_size, data_window_stride_size):
                print("Data: " + str(j) + "/" + str(data_size))

                current_output, start_position, out_of_bounds_bool = test_model(model,
                                                                                test_x_path_list[i],
                                                                                test_y_path_list[i],
                                                                                i,
                                                                                j,
                                                                                data_window_size,
                                                                                window_size,
                                                                                data_size,
                                                                                window_stride_size,
                                                                                output_path,
                                                                                output_path,
                                                                                output_all_bool,
                                                                                number_of_bins)

                current_output = current_output[output_to_output]

                for l in range(len(current_output)):
                    if output_all_bool:
                        with open(output_path + output_file_name[i] + "_part_" + str(i) + "_" + str(j) + "_" + str(
                                l) + output_prefix, "w") as file:
                            write_to_file(file, current_output[l].reshape(current_output[l].size, 1))

                    current_output[l] = scipy.stats.zscore(current_output[l])

                current_output_list = current_output.tolist()

                for l in range(len(current_output_list)):
                    current_output_list[l] = current_output_list[l]

                    for p in range(l * window_stride_size):
                        current_output_list[l].insert(0, np.nan)

                    for p in range(start_position):
                        current_output_list[l].insert(0, np.nan)

                    for p in range(len(current_output_list[l]), data_size):
                        current_output_list[l].append(np.nan)

                    output_list.append(current_output_list[l])

                if out_of_bounds_bool:
                    break

            output_array = np.asfarray(output_list)

            if flip_bool:
                flipped = True

                while flipped:
                    flipped = False

                    for j in range(2, len(output_array)):
                        cc = np.ma.corrcoef(np.ma.masked_invalid(output_array[i - 1]),
                                            np.ma.masked_invalid(output_array[i]))

                        if cc.data[0][1] < 0 or cc.data[1][0] < 0:
                            flipped = True

                            output_array[i] = output_array[i] * -1

            output = np.nanmean(np.asfarray(output_array), axis=0)
            output = scipy.stats.zscore(output)

            with open(output_path + output_file_name[i] + output_prefix, "w") as file:
                write_to_file(file, output.reshape(output.size, 1))

        print("Done")


if __name__ == "__main__":
    main(True, True, False)
