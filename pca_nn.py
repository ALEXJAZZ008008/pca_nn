from __future__ import division, print_function
import math
import keras as k
import numpy as np
import scipy.io
from scipy import stats

import network


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
              epochs):
    print("Get training data")

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

            x = network.test_rnn_down_rnn_out(input_x)

            x = network.output_module(x, output_size, "tanh")

            model = k.Model(inputs=input_x, outputs=x)

            model.compile(optimizer=k.optimizers.Adadelta(),
                          loss=k.losses.mean_squared_error,
                          metrics=["mean_absolute_error"])
    else:
        print("Using input model")

        model = input_model

    model.summary()

    if plot_bool:
        k.utils.plot_model(model, output_path + "model.png")

    print("Fitting model")

    y_train_len = len(y_train)
    batch_size = int(y_train_len / 10)

    if batch_size <= 0:
        batch_size = 1

    model.fit(x_train, y_train, batch_size=batch_size, epochs=epochs, verbose=1)

    loss = model.evaluate(x_train, y_train, batch_size=batch_size, verbose=1)

    print("Metrics: ", model.metrics_names)
    print("Train loss, acc:", loss)

    if save_bool:
        print("Saving model")

        model.save(output_path + "/model.h5")

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

    with open(output_path + "/output_signal.csv", "w") as file:
        write_to_file(file, output)

    difference_matrix = output - y_test
    difference_vector = np.abs(difference_matrix.flatten())
    
    print("Max difference: " + str(difference_vector.max()))
    print("Mean difference: " + str(difference_vector.mean()))

    boolean_difference = []

    for i in range(len(output)):
        for j in range(len(output[i])):
            if output[i][j] - y_test[i][j] < 2.0 / 100.0:
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


def standardise(input_path, output_path):
    for i in range(len(input_path)):
        data = scipy.io.loadmat(input_path[i])
        data_array = np.mean(data.get(list(data.keys())[3]), 3).T

        data_array = (data_array - data_array.mean()) / data_array.std()

        np.save(output_path[i], data_array)


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

    data_array = rescale_linear(np.asfarray(data_array), -1.0, 1.0)

    for i in range(len(data_array)):
        np.save(output_path[i], data_array[i])


def main(fit_model_bool, while_bool, load_bool):
    x_path_orig_list = ["./sinos_0.mat"]
    y_path_orig_list = ["./output_signal_0.mat"]
    x_path_list = ["sinos_standardised_0.npy"]
    y_path_list = ["output_signal_rescaled_0.npy"]

    standardise(x_path_orig_list, x_path_list)
    rescale(y_path_orig_list, y_path_list)

    window_size = 40
    data_window_size = window_size * 4
    data_window_stride_size = data_window_size - window_size

    if fit_model_bool:
        window_stride_size = 1

        while_model = None

        epoch_size = 10

        while True:
            path_length = len(x_path_list)

            for i in range(path_length):
                print("Path: " + str(i) + "/" + str(path_length))

                data_array = np.load(y_path_list[i])
                data_size = data_array.shape[0]

                if data_window_size >= data_size:
                    data_window_size = data_size - 1

                for j in range(0, data_size, data_window_stride_size):
                    print("Data: " + str(j) + "/" + str(data_size))

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
                                                                                "./results/",
                                                                                epoch_size)

                    if out_of_bounds_bool:
                        break

            print("Done")

            if not while_bool:
                break
    else:
        window_stride_size = math.floor(window_size / 2)

        print("Test model")
        print("Load model from file")

        model = k.models.load_model("./results/" + "/model.h5")
        
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
                                                                                "./results/",
                                                                                "./results/")

                for l in range(len(current_output)):
                    current_output[l] = stats.zscore(current_output[l])

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

            flipped = True

            while flipped:
                flipped = False

                for j in range(2, len(output_array)):
                    cc = np.ma.corrcoef(np.ma.masked_invalid(output_array[i - 1]), np.ma.masked_invalid(output_array[i]))

                    if cc.data[0][1] < 0 or cc.data[1][0] < 0:
                        flipped = True

                        output_array[i] = output_array[i] * -1

            output = np.nanmean(np.asfarray(output_array), axis=0)
            output = stats.zscore(output)

            with open("./results/" + "/signal_" + str(i) + ".csv", "w") as file:
                write_to_file(file, output.reshape(output.size, 1))

        print("Done")


if __name__ == "__main__":
    main(True, False, False)
