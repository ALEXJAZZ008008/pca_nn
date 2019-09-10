from __future__ import division, print_function
import tensorflow as tf
from tensorflow import keras as k
import numpy as np
import scipy.io
from scipy import stats

import network


# https://stackoverflow.com/questions/36000843/scale-numpy-array-to-certain-range
def rescale_linear(array, new_min, new_max):
    """Rescale an arrary linearly."""
    minimum, maximum = np.min(array), np.max(array)
    m = (new_max - new_min) / (maximum - minimum)
    b = new_min - m * minimum
    return m * array + b


def get_x(input_path, start_position, data_window_size, window_size, window_stride_size):
    print("Getting x")

    x = []

    sinos = scipy.io.loadmat(input_path)
    sinos_array = np.mean(sinos.get(list(sinos.keys())[3]), 3).T

    for i in range(start_position, start_position + data_window_size, window_stride_size):
        x_window = []

        for j in range(window_size):
            x_window.append(sinos_array[i + j])

        x.append(np.asarray(x_window).T)

    print("Got x")

    return np.nan_to_num(np.asarray(x))


def get_y(input_path, start_position, data_window_size, window_size, window_stride_size):
    print("Get y")

    y = []

    test = scipy.io.loadmat(input_path)
    test_array = rescale_linear(test.get(list(test.keys())[3]), -1.0, 1.0)

    for i in range(start_position, start_position + data_window_size, window_stride_size):
        y_window = []

        for j in range(window_size):
            y_window.append(test_array[i + j])

        y.append(np.squeeze(np.asarray(y_window)))

    print("Got y")

    return np.nan_to_num(np.asarray(y))


# https://stackoverflow.com/questions/43855162/rmse-rmsle-loss-function-in-keras
def root_mean_squared_error(y_true, y_pred):
    return k.backend.sqrt(k.backend.mean(k.backend.square(y_pred - y_true)))


# https://stackoverflow.com/questions/46619869/how-to-specify-the-correlation-coefficient-as-the-loss-function-in-keras
def correlation_coefficient_loss(y_true, y_pred):
    x = y_true
    y = y_pred
    mx = k.backend.mean(x)
    my = k.backend.mean(y)
    xm, ym = x-mx, y-my
    r_num = k.backend.sum(tf.multiply(xm, ym))
    r_den = k.backend.sqrt(tf.multiply(k.backend.sum(k.backend.square(xm)), k.backend.sum(k.backend.square(ym))))
    r = r_num / r_den

    r = k.backend.maximum(k.backend.minimum(r, 1.0), -1.0)
    return 1 - k.backend.square(r)


# https://stackoverflow.com/questions/51625357/i-want-to-use-tensorflows-metricpearson-correlation-in-keras
def tf_pearson(y_true, y_pred):
    return tf.contrib.metrics.streaming_pearson_correlation(y_pred, y_true)[1]


def fit_model(input_model,
              save_bool,
              load_bool,
              apply_bool,
              x_input_path,
              y_input_path,
              start_position,
              data_window_size,
              window_size,
              output_path,
              epochs):
    print("Get training data")

    x_train = get_x(x_input_path, start_position, data_window_size, window_size, 1)
    y_train = get_y(y_input_path, start_position, data_window_size, window_size, 1)

    if input_model is None:
        print("No input model")

        if load_bool:
            print("Load model from file")

            model = k.models.load_model(output_path + "/model.h5")
        else:
            print("Generate new model")

            output_size = y_train.shape[1:][0]

            input_x = k.layers.Input(x_train.shape[1:])

            x = network.resnet_rnn(input_x, output_size)

            x = network.output_module(x, output_size)

            model = k.Model(inputs=input_x, outputs=x)

            model.compile(optimizer=k.optimizers.Adam(), loss=k.losses.mean_squared_error, metrics=["accuracy"])
    else:
        print("Using input model")

        model = input_model

    model.summary()
    k.utils.plot_model(model, output_path + "model.png")

    print("Fitting model")

    k.backend.get_session().run(tf.local_variables_initializer())
    model.fit(x_train, y_train, epochs=epochs, verbose=1)

    loss = model.evaluate(x_train, y_train, verbose=0)
    print("Train loss:", loss)

    print("Saving model")

    if save_bool:
        model.save(output_path + "/model.h5")

    if apply_bool:
        test_model(model,
                   x_input_path,
                   y_input_path,
                   start_position,
                   data_window_size,
                   window_size,
                   output_path,
                   output_path)

    return model


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
               model_input_path,
               output_path):
    print("Get test data")

    x_test = get_x(x_input_path, start_position, data_window_size, window_size, window_size)
    y_test = get_y(y_input_path, start_position, data_window_size, window_size, window_size)

    if input_model is None:
        print("No input model")
        print("Load model from file")

        model = k.models.load_model(model_input_path + "/model.h5",
                                    custom_objects={'root_mean_squared_error': root_mean_squared_error})
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
            if output[i][j] - y_test[i][j] < 0.001:
                boolean_difference.append(np.array(0))
            else:
                boolean_difference.append(np.array(1))

    absolute_difference = sum(boolean_difference)

    print("Absolute boolean difference: " + str(absolute_difference) + "/" + str(len(y_test) * 20))
    print("Relative boolean difference: " + str(((absolute_difference / 20.0) / len(y_test)) * 100) + "%")

    with open(output_path + "/difference.csv", "w") as file:
        write_to_file(file, difference_matrix)

    return output


def main(fit_model_bool, while_bool, load_bool):
    y_path = "./output_signal.mat"

    data = scipy.io.loadmat(y_path)
    data_array = data.get(list(data.keys())[3])

    data_size = data_array.shape[0]
    window_size = 20
    epoch_size = 100

    if epoch_size >= data_size:
        epoch_size = data_size
        data_window_size = epoch_size - 1
        data_stride_size = 1
    else:
        data_window_size = epoch_size + window_size
        data_stride_size = data_window_size

    if fit_model_bool:
        while_model = None

        while True:
            print("Fit model")

            for i in range(0, data_size, 1):
                while_model = fit_model(while_model,
                                        True,
                                        load_bool,
                                        True,
                                        "./sinos.mat",
                                        y_path,
                                        i,
                                        data_window_size,
                                        window_size,
                                        "./results/",
                                        epoch_size)

            print("Done")

            if not while_bool:
                break
    else:
        print("Test model")

        output = np.empty((0, 20))

        for i in range(0, data_size, data_stride_size):
            output = np.concatenate((output, stats.zscore(test_model(None,
                                                                     "./sinos.mat",
                                                                     y_path,
                                                                     i,
                                                                     data_window_size,
                                                                     window_size,
                                                                     "./results/",
                                                                     "./results/"))))

        output = stats.zscore(output.flatten())

        with open("./results/" + "/signal.csv", "w") as file:
            write_to_file(file, output.reshape(output.size, 1))

        print("Done")


if __name__ == "__main__":
    main(True, True, True)
