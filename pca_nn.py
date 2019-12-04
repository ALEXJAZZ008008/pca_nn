from __future__ import division, print_function
from random import seed
from random import randint
import time
import keras as k
import numpy as np
import scipy.io
import scipy.stats

import network
import test
import test_2


def get_x_stochastic(input_path, start_position, data_window_size, window_size, data_size, window_stride_size):
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

        stochastic_i = randint(0, (data_size - window_size) - 1)

        print("stochastic_i: " + str(stochastic_i))

        stochastic_i_list.append(stochastic_i)

        for j in range(window_size):
            if out_of_bounds_bool:
                if stochastic_i + j >= data_size:
                    data_load_out_of_bounds_bool = True

                    break

            x_window.append(sinos_array[stochastic_i + j])

        if data_load_out_of_bounds_bool:
            break

        x.append(np.asfarray(x_window).T)

    print("Got x")

    return np.nan_to_num(np.asfarray(x)).astype(np.float32), start_position, out_of_bounds_bool, stochastic_i_list


def get_y_stochastic(input_path, window_size, data_size, out_of_bounds_bool, stochastic_i_list):
    print("Get y")

    y = []

    test_array = np.load(input_path)

    data_load_out_of_bounds_bool = False

    for i in range(len(stochastic_i_list)):
        y_window = []

        print("stochastic_i: " + str(stochastic_i_list[i]))

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

            x_window.append(sinos_array[i + j])

        if data_load_out_of_bounds_bool:
            break

        x.append(np.asfarray(x_window).T)

    print("Got x")

    return np.nan_to_num(np.asfarray(x)).astype(np.float32), start_position, out_of_bounds_bool


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
              start_position,
              data_window_size,
              window_size,
              data_size,
              window_stride_size,
              output_path,
              tof_bool,
              stochastic_bool,
              passthrough_bool,
              epochs):
    print("Get training data")

    if stochastic_bool:
        x_train, start_position, out_of_bounds_bool, stochastic_i_list = get_x_stochastic(x_input_path,
                                                                                          start_position,
                                                                                          data_window_size,
                                                                                          window_size,
                                                                                          data_size,
                                                                                          window_stride_size)
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

            # x = test.test_in_down_out(input_x, 40, "he_uniform", True, "prelu")
            # x = test.test_rnn_out(input_x, tof_bool, 1, "lstm", 40, "hard_sigmoid", "glorot_uniform", False)
            # x = test.test_in_rnn_down_rnn_out(input_x, 40, "he_uniform", tof_bool, 2, "lstm", 40, "hard_sigmoid", "glorot_uniform", False, True, "prelu", 7, True)

            # x = test_2.test_in_down_out(input_x, "relu", 40, "he_uniform", 5, True)
            x = test_2.test_in_down_rnn_out(input_x, "relu", 40, "he_uniform", 5, True, tof_bool, 0, "lstm", 40, "hard_sigmoid", "glorot_uniform", True)

            x = network.output_module(x, output_size, "tanh", "lecun_normal")

            model = k.Model(inputs=input_x, outputs=x)

            lr = 0.01

            model.compile(optimizer=k.optimizers.SGD(learning_rate=lr, momentum=0.99, nesterov=True, clipnorm=1.0,
                                                     clipvalue=0.5),
                          loss=k.losses.mean_squared_error)

            with open(output_path + "/lr", "w") as file:
                file.write(str(lr))
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

        batch_size = 1

        reduce_lr = k.callbacks.ReduceLROnPlateau(monitor="loss", factor=0.99, patience=1, verbose=1, cooldown=1)

        model.fit(x_train, y_train, batch_size=batch_size, epochs=epochs, callbacks=[reduce_lr], verbose=1)
        
        lr = k.backend.get_value(model.optimizer.lr)

        with open(output_path + "/lr", "w") as file:
            file.write(str(lr))

        print("lr: " + str(k.backend.get_value(model.optimizer.lr)))

    if save_bool:
        print("Saving model")

        model.save(output_path + "/model.h5")

    if not passthrough_bool:
        if apply_bool:
            test_model(model,
                       x_input_path,
                       y_input_path,
                       start_position,
                       data_window_size,
                       window_size,
                       data_size,
                       window_stride_size,
                       output_path,
                       output_path)

    return model, start_position, out_of_bounds_bool


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
               start_position,
               data_window_size,
               window_size,
               data_size,
               window_stride_size,
               model_input_path,
               output_path):
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

    output = model.predict(x_test)

    with open(output_path + "/test_estimated_signal.csv", "w") as file:
        write_to_file(file, output)

    difference_matrix = output - y_test
    difference_vector = np.abs(difference_matrix.flatten())

    print("Max difference: " + str(difference_vector.max()))
    print("Mean difference: " + str(difference_vector.mean()))

    boolean_difference = []

    for i in range(len(output)):
        for j in range(len(output[i])):
            if abs(output[i][j] - y_test[i][j]) < 2.0 / 50.0:
                boolean_difference.append(np.array(0))
            else:
                boolean_difference.append(np.array(1))

    absolute_difference = sum(boolean_difference)

    print("Absolute boolean difference: " + str(absolute_difference) + "/" + str(len(y_test) * window_size))
    print("Relative boolean difference: " + str(((float(absolute_difference) / float(window_size)) / float(len(y_test)))
                                                * float(100)) + "%")

    with open(output_path + "/difference.csv", "w") as file:
        write_to_file(file, difference_matrix)

    return output, start_position, out_of_bounds_bool


def histogram_equalisation(data_array, number_of_bins):
    hist, bins = np.histogram(data_array.flatten(), number_of_bins, density=True)

    cdf = hist.cumsum()
    cdf = 255 * cdf / cdf[-1]  # normalize

    output = np.interp(data_array.flatten(), bins[:-1], cdf).reshape(data_array.shape)

    return output


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


def downsample_and_zscore(input_path, tof_bool, output_path):
    for i in range(len(input_path)):
        data = scipy.io.loadmat(input_path[i])

        data_array = data.get(list(data.keys())[3])

        if not tof_bool:
            data_array = np.mean(data_array, 3)

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


def rescale(input_path, output_path):
    data_array = []

    for i in range(len(input_path)):
        data = scipy.io.loadmat(input_path[i])
        data_array.append(data.get(list(data.keys())[3]))

    data_array = rescale_linear(np.asfarray(data_array), 0.0, 1.0)

    for i in range(len(data_array)):
        np.save(output_path[i], data_array[i])


def main(fit_model_bool, while_bool, load_bool):
    passthrough_bool = False
    single_input_bool = True
    tof_bool = False
    stochastic_bool = True

    number_of_bins = 1000000

    output_path = "./results/"

    if single_input_bool:
        x_path_orig_list = ["./normalised_sinos_1.mat"]
        y_path_orig_list = ["./output_signal_1.mat"]

        x_path_list = ["normalised_sinos_downsample_and_rescale_1.npy"]
        y_path_list = ["output_signal_rescaled_1.npy"]

        output_file_name = ["estimated_signal_1.csv"]
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

        x_path_list = ["normalised_sinos_downsample_and_rescale_1.npy",
                       "normalised_sinos_downsample_and_rescale_3.npy",
                       "normalised_sinos_downsample_and_rescale_5.npy",
                       "normalised_sinos_downsample_and_rescale_6.npy",
                       "normalised_sinos_downsample_and_rescale_7.npy",
                       "normalised_sinos_downsample_and_rescale_8.npy",
                       "normalised_sinos_downsample_and_rescale_9.npy",
                       "normalised_sinos_downsample_and_rescale_15.npy",
                       "normalised_sinos_downsample_and_rescale_19.npy",
                       "normalised_sinos_downsample_and_rescale_20.npy"]
        y_path_list = ["output_signal_rescaled_1.npy",
                       "output_signal_rescaled_3.npy",
                       "output_signal_rescaled_5.npy",
                       "output_signal_rescaled_6.npy",
                       "output_signal_rescaled_7.npy",
                       "output_signal_rescaled_8.npy",
                       "output_signal_rescaled_9.npy",
                       "output_signal_rescaled_15.npy",
                       "output_signal_rescaled_19.npy",
                       "output_signal_rescaled_20.npy"]

        output_file_name = ["estimated_signal_1.csv",
                            "estimated_signal_3.csv",
                            "estimated_signal_5.csv",
                            "estimated_signal_6.csv",
                            "estimated_signal_7.csv",
                            "estimated_signal_8.csv",
                            "estimated_signal_9.csv",
                            "estimated_signal_15.csv",
                            "estimated_signal_19.csv",
                            "estimated_signal_20.csv"]

    print("Getting data")

    downsample_histogram_equalisation_and_zscore(x_path_orig_list, tof_bool, number_of_bins, x_path_list)
    rescale(y_path_orig_list, y_path_list)

    print("Got data")

    window_size = 40
    window_stride_size = window_size

    if tof_bool:
        data_window_size = window_size
    else:
        data_window_size = window_size * 10

    data_window_stride_size = data_window_size

    if fit_model_bool:
        window_stride_size = 1
        epochs = 2

        if load_bool:
            with open(output_path + "/path_start_point", "r") as file:
                path_start_point = int(file.read())

            with open(output_path + "/data_start_point", "r") as file:
                data_start_point = int(file.read())
        else:
            path_start_point = 0

            with open(output_path + "/path_start_point", "w") as file:
                file.write(str(path_start_point))

            data_start_point = 0

            with open(output_path + "/data_start_point", "w") as file:
                file.write(str(data_start_point))

        path_length = len(x_path_list)

        while_model = None

        while True:
            for i in range(path_start_point, path_length, 1):

                with open(output_path + "/path_start_point", "w") as file:
                    file.write(str(i))

                print("Path: " + str(i) + "/" + str(path_length))

                data_array = np.load(y_path_list[i])
                data_size = data_array.shape[0]

                if data_window_size >= data_size:
                    data_window_size = data_size - 1

                for j in range(data_start_point, data_size, data_window_stride_size):

                    print("Data: " + str(j) + "/" + str(data_size))

                    out_of_bounds_bool = True

                    # try:
                    while_model, start_position, out_of_bounds_bool = fit_model(while_model,
                                                                                True,
                                                                                load_bool,
                                                                                True,
                                                                                True,
                                                                                x_path_list[i],
                                                                                y_path_list[i],
                                                                                j,
                                                                                data_window_size,
                                                                                window_size,
                                                                                data_size,
                                                                                window_stride_size,
                                                                                output_path,
                                                                                tof_bool,
                                                                                stochastic_bool,
                                                                                passthrough_bool,
                                                                                epochs)
                    # except:
                    # print("Error fitting model: continuing")

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
        flip_bool = False

        print("Test model")
        print("Load model from file")

        model = k.models.load_model(output_path + "/model.h5")

        path_length = len(x_path_list)

        for i in range(path_length):
            print("Path: " + str(i) + "/" + str(path_length))

            data_array = np.load(y_path_list[i])
            data_size = data_array.shape[0]

            if data_window_size >= data_size:
                data_window_size = data_size - 1

            output_list = []

            for j in range(0, data_size, data_window_stride_size):
                print("Data: " + str(j) + "/" + str(data_size))

                current_output, start_position, out_of_bounds_bool = test_model(model,
                                                                                x_path_list[i],
                                                                                y_path_list[i],
                                                                                j,
                                                                                data_window_size,
                                                                                window_size,
                                                                                data_size,
                                                                                window_stride_size,
                                                                                output_path,
                                                                                output_path)

                current_output_list = current_output.tolist()

                for l in range(len(current_output_list)):
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

            output = scipy.stats.zscore(np.nanmean(np.asfarray(output_array), axis=0))

            with open("./results/" + output_file_name[i], "w") as file:
                write_to_file(file, output.reshape(output.size, 1))

        print("Done")


if __name__ == "__main__":
    main(True, True, False)
