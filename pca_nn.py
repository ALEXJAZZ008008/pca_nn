# Copyright University College London 2019, 2020
# Author: Alexander Whitehead, Institute of Nuclear Medicine, UCL
# For internal research only.


from __future__ import division, print_function
import os
import time
import math
from random import seed
from random import randint
import numpy as np
import matplotlib

matplotlib.use("TKAgg")

import matplotlib.pyplot as plt
import scipy.io
import scipy.stats
from sklearn.preprocessing import RobustScaler
from sklearn.preprocessing import PowerTransformer
import tensorflow as tf

gpus = tf.config.experimental.list_physical_devices('GPU')

if gpus:
    try:
        tf.config.experimental.set_virtual_device_configuration(gpus[0], [
            tf.config.experimental.VirtualDeviceConfiguration(memory_limit=6144)])
    except RuntimeError as e:
        print(e)

import keras as k

import pca_nn_paths
import optimise
import test_3


def get_x_stochastic(input_path, start_position, data_window_size, window_size, data_size, window_stride_size,
                     cut_list, noise_factor):
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

    x_array = np.nan_to_num(np.expand_dims(np.asfarray(x), axis=5))
    x_array_noisy = (x_array + (noise_factor * np.random.normal(loc=0.0, scale=1.0, size=x_array.shape))) / (
            1.0 + noise_factor)

    print("Got x")

    return x_array, x_array_noisy, start_position, out_of_bounds_bool, stochastic_i_list


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

    return np.nan_to_num(np.asfarray(y))


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

    return np.nan_to_num(np.expand_dims(np.asfarray(x), axis=5)), start_position, out_of_bounds_bool


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

    return np.nan_to_num(np.asfarray(y))


def generate_model(y_train, x_train_noisy, window_size, tof_bool):
    print("Generate new model")

    input_size = int(math.ceil(np.prod(x_train_noisy.shape[1:]) / window_size))
    output_size = y_train.shape[1:][0]
    output_shape = x_train_noisy.shape[2:]

    input_x = k.layers.Input(x_train_noisy.shape[1:])

    x = input_x

    # network settings
    network_to_data_normalisation_multiplier = 1.0

    # dense

    stride_multiplier = 4
    stride = math.pow(2, stride_multiplier)
    stride = int(math.floor(stride))
    layers = 1
    layers_layers = 1

    # cnn

    base_units_multiplier = 5
    base_units = math.pow(2, base_units_multiplier)
    base_units = int(math.floor(base_units * network_to_data_normalisation_multiplier))

    cnn_bool = True
    cnn_start_units = base_units
    cnn_layers = 3
    cnn_increase_layer_density_bool = False
    cnn_layer_layers = 1
    cnn_pool_bool = True
    cnn_max_pool_bool = True
    cnn_deconvolution_bool = False

    rnn_multiplier = 2
    rnn_multiplier = rnn_multiplier * network_to_data_normalisation_multiplier
    rnn_units = int(math.floor(window_size * rnn_multiplier))
    rnn_layers = 1

    regularisation_epsilon = 0.0
    lone = regularisation_epsilon
    ltwo = regularisation_epsilon
    dropout = 0.0
    auto_encoder_weight = 1.0

    # end network settings

    optimised_rnn_units = optimise.optimise_value(int(rnn_units), float(dropout))
    print("Output: {0}".format(optimised_rnn_units))

    if cnn_bool:
        if tof_bool:
            base_units_multiplier = 1
            base_units = math.pow(2, base_units_multiplier)
            base_units = int(math.floor(base_units * network_to_data_normalisation_multiplier))

            cnn_start_units = base_units
            cnn_layers = 1
            cnn_increase_layer_density_bool = True
            cnn_layer_layers = 0

            rnn_multiplier = 1
            rnn_multiplier = rnn_multiplier * network_to_data_normalisation_multiplier
            rnn_units = int(math.floor(window_size * rnn_multiplier))
            rnn_layers = 1

            regularisation_epsilon = 0.0
            lone = regularisation_epsilon
            ltwo = regularisation_epsilon
            dropout = 0.0
            auto_encoder_weight = 0.0

            x_1, x_2 = test_3.tof_crnn_dynamic_signal_extractor(x, cnn_start_units, cnn_layers,
                                                                cnn_increase_layer_density_bool, cnn_layer_layers, lone,
                                                                ltwo, rnn_layers, rnn_units, dropout, output_size)
        else:
            x_1, x_2 = test_3.crnn_dynamic_signal_extractor(x, cnn_start_units, cnn_layers,
                                                            cnn_increase_layer_density_bool, cnn_layer_layers, lone,
                                                            ltwo, cnn_pool_bool, cnn_max_pool_bool,
                                                            cnn_deconvolution_bool, rnn_layers, optimised_rnn_units,
                                                            dropout, output_size)
    else:
        x_1, x_2 = test_3.dense_dynamic_signal_extractor(x, stride, layers, layers_layers, input_size, lone, ltwo,
                                                         rnn_layers, rnn_units, dropout, output_size, output_shape)

    model = k.Model(inputs=input_x, outputs=[x_1, x_2])

    model.compile(optimizer=k.optimizers.Nadam(clipnorm=1.0),
                  loss={"output_1": k.losses.mean_squared_error, "output_2": k.losses.mean_absolute_error},
                  loss_weights=[1.0 - auto_encoder_weight, auto_encoder_weight])

    return model


def fit_model(input_model,
              save_bool,
              load_bool,
              load_full_model_bool,
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
              stochastic_bool,
              noisy_bool,
              tof_bool,
              passthrough_bool,
              epochs,
              output_all_bool,
              plot_output_bool,
              cut_list,
              noise_factor):
    print("Get training data")

    x_train_noisy = None

    if stochastic_bool:
        x_train, x_train_noisy, start_position, out_of_bounds_bool, stochastic_i_list = get_x_stochastic(x_input_path,
                                                                                                         start_position,
                                                                                                         data_window_size,
                                                                                                         window_size,
                                                                                                         data_size,
                                                                                                         window_stride_size,
                                                                                                         cut_list,
                                                                                                         noise_factor)
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

    if noisy_bool:
        x_train_noisy = None

    if x_train_noisy is None:
        x_train_noisy = x_train

    if input_model is None:
        print("No input model")

        if load_bool:
            if load_full_model_bool:
                print("Load model from file")

                model = k.models.load_model(output_path + "/model")
            else:
                model = generate_model(y_train, x_train_noisy, window_size, tof_bool)

                model.load_weights(output_path + "/model_weights")
        else:
            model = generate_model(y_train, x_train_noisy, window_size, tof_bool)

        model.summary()

        if plot_bool:
            k.utils.plot_model(model, output_path + "model.png")
    else:
        print("Using input model")

        model = input_model

    if not passthrough_bool:
        print("Fitting model")

        model.fit(x_train_noisy, {"output_1": y_train, "output_2": x_train}, epochs=epochs, batch_size=1, shuffle=True,
                  verbose=1)

    if save_bool:
        print("Saving model")

        model.save(output_path + "/model")
        model.save_weights(output_path + "/model_weights")

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
                       plot_output_bool)

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
                   cut_list):
    print("Get test data")

    x_test, x_test_noisy, start_position, out_of_bounds_bool, stochastic_i_list = get_x_stochastic(x_input_path,
                                                                                                   start_position,
                                                                                                   data_window_size,
                                                                                                   window_size,
                                                                                                   data_size,
                                                                                                   window_stride_size,
                                                                                                   cut_list,
                                                                                                   noise_factor=0.0)
    y_test = get_y_stochastic(y_input_path, window_size, data_size, out_of_bounds_bool, stochastic_i_list)

    if input_model is None:
        print("No input model")
        print("Load model from file")

        model = k.models.load_model(model_input_path + "/model")
    else:
        model = input_model

    print("Applying model")

    output = model.evaluate(x_test, {"output_1": y_test, "output_2": x_test}, batch_size=1, verbose=1)
    output_string = "Test loss: {0}, Test loss output_1: {1}, Test loss output_2: {2}".format(output[0], output[1],
                                                                                              output[2])

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
               plot_output_bool):
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

        model = k.models.load_model(model_input_path + "/model")
    else:
        model = input_model

    print("Applying model")

    output = model.predict(x_test, batch_size=1, verbose=1)

    output_volume_array = None

    if len(output) > 1:
        if output_all_bool:
            for i in range(len(output[1])):
                output_volume_array = normalise_and_save(output[1],
                                                         output_path + "/test_estimated_input_" + str(1) + "_" + str(
                                                             path) + "_" + str(start_position) + "_" + str(i))

        output = [output[0]]

    if output_all_bool and plot_output_bool:
        plt.imshow(np.squeeze(output_volume_array[0, 0, :, 0, :, 0]))
        plt.show()

        plt.imshow(np.squeeze(x_test[0, 0, :, 0, :, 0]))
        plt.show()

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

    if plot_output_bool:
        plt.plot(output[0][0, :])
        plt.show()

        plt.plot(y_test[0][0, :])
        plt.show()

    return output, start_position, out_of_bounds_bool


def normalise_and_save(input_data, output_path):
    data_array = rescale_linear(input_data, 0.0, 1.0)

    np.save(output_path, data_array)

    return data_array


def downsample_and_power_transform_normalise(input_path, tof_bool, new_min, new_max, output_path):
    for i in range(len(input_path)):
        try:
            data = scipy.io.loadmat(input_path[i])
            data_array = data.get(list(data.keys())[3])
        except:
            data_array = np.load(input_path[i])

        if not tof_bool:
            data_array = np.nanmean(data_array, 3)

        data_array = data_array.T
        data_array_shape = data_array.shape

        data_array = np.reshape(data_array.flatten(), [-1, 1])

        pt = PowerTransformer()
        data_array = pt.fit_transform(data_array)

        data_array = np.reshape(data_array, data_array_shape)

        data_array = rescale_linear(data_array, new_min, new_max)

        np.save(output_path[i], data_array)

    return output_path


def power_transform(input_path, output_path):
    for i in range(len(input_path)):
        try:
            data = scipy.io.loadmat(input_path[i])
            data_array = data.get(list(data.keys())[3])
        except:
            data_array = np.load(input_path[i])

        data_array_shape = data_array.shape

        data_array = np.reshape(data_array.flatten(), [-1, 1])

        pt = PowerTransformer()
        data_array = pt.fit_transform(data_array)

        data_array = np.reshape(data_array, data_array_shape)

        np.save(output_path[i], data_array)

    return output_path


def histogram_equalisation(data_array, number_of_bins):
    hist, bins = np.histogram(data_array.flatten(), number_of_bins, density=True)

    cdf = hist.cumsum()
    cdf = 255 * cdf / cdf[-1]  # normalize

    output = np.interp(data_array.flatten(), bins[:-1], cdf).reshape(data_array.shape)

    return output


def contrast_equalisation(input_path, number_of_bins, output_path):
    for i in range(len(input_path)):
        try:
            data = scipy.io.loadmat(input_path[i])
            data_array = data.get(list(data.keys())[3])
        except:
            data_array = np.load(input_path[i])

        data_array = data_array.T

        data_array = histogram_equalisation(data_array, number_of_bins)

        np.save(output_path[i], data_array)

    return output_path


def standardise(input_path, output_path):
    for i in range(len(input_path)):
        try:
            data = scipy.io.loadmat(input_path[i])
            data_array = data.get(list(data.keys())[3])
        except:
            data_array = np.load(input_path[i])

        data_array_shape = data_array.shape

        data_array = np.reshape(data_array.flatten(), [-1, 1])

        transformer = RobustScaler().fit(data_array)
        data_array = transformer.transform(data_array)

        data_array = np.reshape(data_array, data_array_shape)

        np.save(output_path[i], data_array)

    return output_path


def zscore_normalise(input_path, new_min, new_max, output_path):
    for i in range(len(input_path)):
        try:
            data = scipy.io.loadmat(input_path[i])
            data_array = data.get(list(data.keys())[3])
        except:
            data_array = np.load(input_path[i])

        data_array = scipy.stats.zscore(data_array)
        data_array = rescale_linear(data_array, new_min, new_max)

        np.save(output_path[i], data_array)

    return output_path


def zscore(input_path, output_path):
    for i in range(len(input_path)):
        try:
            data = scipy.io.loadmat(input_path[i])
            data_array = data.get(list(data.keys())[3])
        except:
            data_array = np.load(input_path[i])

        data_array = scipy.stats.zscore(data_array)

        np.save(output_path[i], data_array)

    return output_path


# https://stackoverflow.com/questions/36000843/scale-numpy-array-to-certain-range
def rescale_linear(array, new_min, new_max):
    """Rescale an arrary linearly."""
    minimum, maximum = np.min(array), np.max(array)
    m = (new_max - new_min) / (maximum - minimum)
    b = new_min - m * minimum
    return m * array + b


def normalise(input_path, output_path, new_min, new_max):
    for i in range(len(input_path)):
        try:
            data = scipy.io.loadmat(input_path[i])
            data_array = data.get(list(data.keys())[3])
        except:
            data_array = np.load(input_path[i])

        data_array = rescale_linear(data_array, new_min, new_max)

        np.save(output_path[i], data_array)

    return output_path


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


def concat_one_input(x, y, x_output, y_output, cut_list_output):
    x_data, cut_list = concat_array_list(x)
    np.save(x_output, x_data)

    y_data, cut_list_y = concat_array_list(y)
    np.save(y_output, y_data)

    np.save(cut_list_output, np.asarray(cut_list).astype(np.int))

    return [x_output], [y_output], cut_list


def split_one_input(x, y, x_output, y_output, test_x_output, test_y_output, cut_list_output, split, window_size):
    x_data = np.load(x[0])
    y_data = np.load(y[0])

    test_data_index = []
    cut_list = []

    data_size = len(y_data)
    target_data_size = data_size * split

    current_data_size = len(test_data_index)

    seed(int(time.time()))

    while current_data_size < target_data_size:
        stochastic_i = randint(0, (data_size - window_size) - 1)
        stochastic_i_range = range(stochastic_i, stochastic_i + window_size)

        found = False

        for i in range(len(stochastic_i_range)):
            for j in range(len(test_data_index)):
                if stochastic_i_range[i] == test_data_index[j]:
                    found = True

                    break

            if found:
                break

        if found:
            continue

        for i in range(len(stochastic_i_range)):
            test_data_index.append(stochastic_i_range[i])

        current_data_size = len(test_data_index)

    test_x_data = x_data.take(test_data_index, axis=0)
    test_y_data = y_data.take(test_data_index, axis=0)

    data_index = [x for x in range(data_size) if x not in test_data_index]

    x_data = x_data.take(data_index, axis=0)
    y_data = y_data.take(data_index, axis=0)

    np.save(test_x_output, test_x_data)
    np.save(test_y_output, test_y_data)
    np.save(x_output, x_data)
    np.save(y_output, y_data)

    partial_data_list = [data_index[0]]
    previous_value = data_index[0]

    for i in range(1, len(data_index)):
        partial_data_list.append(data_index[i])

        if previous_value - data_index[i] > 1:
            cut_list.append(len(partial_data_list))

    np.save(cut_list_output, np.asarray(cut_list).astype(np.int))

    return [x_output], [y_output], [test_x_output], [test_y_output], cut_list


def main(fit_model_bool, while_bool, load_bool):
    save_bool = True
    load_full_model_bool = True
    plot_bool = True
    plot_output_bool = False
    apply_bool = False
    passthrough_bool = False
    single_input_bool = True
    autoencoder_input_bool = False
    dynamic_bool = True
    static_bool = True
    concat_one_input_bool = True
    reload_data = not load_bool
    force_no_reload_bool = True
    tof_bool = False
    stochastic_bool = True
    noisy_bool = False
    flip_bool = True

    output_all_bool = True
    number_of_bins = 10000000
    noise_factor = 0.0

    cv_bool = False
    cv_simple_bool = False
    cv_simple_test_bool = False
    cv_pos = 0

    output_to_output = 0

    output_path = "./results/"
    output_prefix = ".csv"
    cv_tracker_path = None

    cut_list = []

    if not os.path.exists(output_path):
        os.makedirs(output_path)

    x_path_orig_list, x_path_list, test_x_path_list, y_path_orig_list, test_y_path_list, y_path_list, output_file_name = pca_nn_paths.get_paths(
        single_input_bool, static_bool, autoencoder_input_bool, dynamic_bool)

    if reload_data and not force_no_reload_bool:
        print("Getting data")

        x_path_list = downsample_and_power_transform_normalise(x_path_orig_list, tof_bool, -1.0, 1.0, x_path_list)
        y_path_list = zscore_normalise(y_path_orig_list, -1.0, 1.0, y_path_list)

        print("Got data")

    window_size = 40
    window_stride_size = math.floor(window_size / 2.0)

    if cv_bool:
        output_path = "./cv_{0}_results/".format(str(cv_pos))

        if not os.path.exists(output_path):
            os.makedirs(output_path)

        cv_tracker_path = "{0}/cv_tracker".format(output_path)

        if not os.path.isfile(cv_tracker_path):
            os.mknod(cv_tracker_path)

        if cv_simple_bool:
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
                x_path_list = test_x_path_list
                y_path_list = test_y_path_list

                output_file_name = test_output_file_name

            if len(test_x_path_orig_list) < 1:
                test_x_path_list = x_path_list
                test_y_path_list = y_path_list
        else:
            temp_test_x_path_list = test_x_path_list
            temp_test_y_path_list = test_y_path_list

            x_output = "x_one_input.npy"
            y_output = "y_one_input.npy"
            cut_list_output = "cut_list_input.npy"

            if reload_data and not force_no_reload_bool:
                concat_one_input(x_path_list, y_path_list, x_output, y_output, cut_list_output)

            x_path_list = [x_output]
            y_path_list = [y_output]

            x_output = "split_x_one_input.npy"
            y_output = "split_y_one_input.npy"
            test_x_output = "split_test_x_one_input.npy"
            test_y_output = "split_test_y_one_input.npy"
            cut_list_output = "split_cut_list_input.npy"

            if reload_data and not force_no_reload_bool:
                x_path_list, y_path_list, test_x_path_list, test_y_path_list, cut_list = split_one_input(x_path_list,
                                                                                                         y_path_list,
                                                                                                         x_output,
                                                                                                         y_output,
                                                                                                         test_x_output,
                                                                                                         test_y_output,
                                                                                                         cut_list_output,
                                                                                                         0.5,
                                                                                                         window_size)
            else:
                cut_list = np.load(cut_list_output)

            x_path_list = [x_output]
            y_path_list = [y_output]
            test_x_path_list = [test_x_output]
            test_y_path_list = [test_y_output]

            if cv_simple_test_bool:
                test_x_path_list = temp_test_x_path_list
                test_y_path_list = temp_test_y_path_list

    if fit_model_bool:
        window_stride_size = 1
        epochs = 1

        if cv_simple_bool:
            if concat_one_input_bool:
                x_output = "x_one_input.npy"
                y_output = "y_one_input.npy"
                cut_list_output = "cut_list_input.npy"

                if reload_data and not force_no_reload_bool:
                    x_path_list, y_path_list, cut_list = concat_one_input(x_path_list, y_path_list, x_output, y_output,
                                                                          cut_list_output)
                else:
                    cut_list = np.load(cut_list_output)

                x_path_list = [x_output]
                y_path_list = [y_output]

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
                    ideal_data_window_size_multiplier = 6
                    ideal_data_window_size = window_size * ideal_data_window_size_multiplier

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

                    if not cv_bool and not concat_one_input_bool:
                        cut_list = [data_size]

                    while_model, start_position, out_of_bounds_bool = fit_model(while_model,
                                                                                save_bool,
                                                                                load_bool,
                                                                                load_full_model_bool,
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
                                                                                stochastic_bool,
                                                                                noisy_bool,
                                                                                tof_bool,
                                                                                passthrough_bool,
                                                                                epochs,
                                                                                output_all_bool,
                                                                                plot_output_bool,
                                                                                cut_list,
                                                                                noise_factor)

                    if cv_bool:
                        cv_index = cv_index + 1

                        while_model, output = evaluate_model(while_model, test_x_path_list[0], test_y_path_list[0], 0,
                                                             data_window_size, window_size, test_data_size,
                                                             window_stride_size, output_path, [test_data_size])

                        with open(cv_tracker_path, 'a') as file:
                            file.write("{0}\n".format(str(output)))

                        while_model.save(output_path + "/cv_model_{0}".format(str(cv_index)))

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

        model = k.models.load_model(output_path + "/model")

        path_length = len(x_path_list)

        for i in range(path_length):
            print("Path: " + str(i) + "/" + str(path_length))

            data_array = np.load(test_y_path_list[i])
            data_size = data_array.shape[0]

            if tof_bool:
                ideal_data_window_size = window_size
            else:
                ideal_data_window_size_multiplier = 6
                ideal_data_window_size = window_size * ideal_data_window_size_multiplier

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
                                                                                plot_output_bool)

                current_output = current_output[output_to_output]

                for l in range(len(current_output)):
                    if output_all_bool:
                        with open(output_path + output_file_name[i] + "_part_" + str(i) + "_" + str(j) + "_" + str(
                                l) + output_prefix, "w") as file:
                            write_to_file(file, current_output[l].reshape(current_output[l].size, 1))

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

            with open(output_path + output_file_name[i] + output_prefix, "w") as file:
                write_to_file(file, output.reshape(output.size, 1))

        print("Done")


if __name__ == "__main__":
    main(True, True, False)
