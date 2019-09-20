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


def get_x_mat(input_path, start_position, data_window_size, window_size, data_size, window_stride_size):
    print("Getting x")

    x = []

    sinos = scipy.io.loadmat(input_path)
    sinos_array = np.mean(sinos.get(list(sinos.keys())[3]), 3).T

    limit = start_position + data_window_size

    if limit > data_size:
        limit = (data_size - window_size) + 1

    for i in range(start_position, limit, window_stride_size):
        x_window = []

        for j in range(window_size):
            x_window.append(sinos_array[i + j])

        x.append(np.asfarray(x_window).T)

    print("Got x")

    return np.nan_to_num(np.asfarray(x, dtype="float16"))


def get_y_mat(input_path, start_position, data_window_size, window_size, data_size, window_stride_size):
    print("Get y")

    y = []

    test = scipy.io.loadmat(input_path)
    test_array = rescale_linear(test.get(list(test.keys())[3]), -1.0, 1.0)

    limit = start_position + data_window_size

    if limit > data_size:
        limit = (data_size - window_size) + 1

    for i in range(start_position, limit, window_stride_size):
        y_window = []

        for j in range(window_size):
            y_window.append(test_array[i + j])

        y.append(np.squeeze(np.asfarray(y_window)))

    print("Got y")

    return np.nan_to_num(np.asfarray(y, dtype="float16"))


def get_x(input_path, start_position, data_window_size, window_size, data_size, window_stride_size):
    print("Getting x")

    x = []

    sinos_array = np.load(input_path)

    limit = start_position + data_window_size

    if limit > data_size:
        limit = (data_size - window_size) + 1

    for i in range(start_position, limit, window_stride_size):
        x_window = []

        for j in range(window_size):
            x_window.append(sinos_array[i + j])

        x.append(np.asfarray(x_window).T)

    print("Got x")

    return np.nan_to_num(np.asfarray(x, dtype="float16"))


def get_y(input_path, start_position, data_window_size, window_size, data_size, window_stride_size):
    print("Get y")

    y = []

    test_array = np.load(input_path)

    limit = start_position + data_window_size

    if limit > data_size:
        limit = (data_size - window_size) + 1

    for i in range(start_position, limit, window_stride_size):
        y_window = []

        for j in range(window_size):
            y_window.append(test_array[i + j])

        y.append(np.squeeze(np.asfarray(y_window)))

    print("Got y")

    return np.nan_to_num(np.asfarray(y, dtype="float16"))


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
              epochs,
              lr,
              lr_factor):
    print("Get training data")

    x_train = get_x(x_input_path, start_position, data_window_size, window_size, data_size, window_stride_size)
    y_train = get_y(y_input_path, start_position, data_window_size, window_size, data_size, window_stride_size)

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

            model.compile(optimizer=k.optimizers.SGD(lr=lr),
                          loss=k.losses.mean_squared_error,
                          metrics=["mean_absolute_error"])
    else:
        print("Using input model")

        model = input_model

    tf.compat.v1.keras.backend.set_value(model.optimizer.lr, lr)

    print("lr: " + str(k.backend.eval(model.optimizer.lr)))

    model.summary()

    if plot_bool:
        k.utils.plot_model(model, output_path + "model.png")

    print("Fitting model")

    tf.compat.v1.keras.backend.get_session().run(tf.compat.v1.local_variables_initializer())

    y_train_len = len(y_train)
    batch_size = int(y_train_len / 4)

    if batch_size <= 0:
        batch_size = 1

    patience = int(epochs / 11)

    if patience <= 0:
        patience = 1

    reduce_lr = k.callbacks.ReduceLROnPlateau(monitor="mean_absolute_error",
                                              factor=lr_factor,
                                              patience=patience,
                                              cooldown=1,
                                              verbose=0)

    loss = model.evaluate(x_train, y_train, batch_size=batch_size, verbose=0)

    print("Metrics: ", model.metrics_names)
    print("Train loss, acc:", loss)

    model.fit(x_train, y_train, batch_size=batch_size, epochs=epochs, callbacks=[reduce_lr], verbose=1)

    loss = model.evaluate(x_train, y_train, batch_size=batch_size, verbose=0)

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

    return model, k.backend.eval(model.optimizer.lr)


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

    x_test = get_x(x_input_path, start_position, data_window_size, window_size, data_size, window_stride_size)
    y_test = get_y(y_input_path, start_position, data_window_size, window_size, data_size, window_stride_size)

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

    return output


def standardise(input_path, output_path):
    for i in range(len(input_path)):
        data = scipy.io.loadmat(input_path[i])
        data_array = np.mean(data.get(list(data.keys())[3]), 3).T

        data_array = (data_array - data_array.mean()) / data_array.std()

        np.save(output_path[i], data_array)


def rescale(input_path, output_path):
    data_array = []

    for i in range(len(input_path)):
        data = scipy.io.loadmat(input_path[i])
        data_array.append(data.get(list(data.keys())[3]))

    data_array = rescale_linear(np.asfarray(data_array), -1.0, 1.0)

    for i in range(len(data_array)):
        np.save(output_path[i], data_array[i])


def main(fit_model_bool, while_bool, load_bool):
    config = tf.compat.v1.ConfigProto()
    config.gpu_options.allow_growth = True
    sess = tf.compat.v1.Session(config=config)

    tf.compat.v1.keras.backend.set_session(sess)

    tf.compat.v1.keras.backend.set_floatx("float16")

    x_path_orig_list = ["./sinos.mat"]
    y_path_orig_list = ["./output_signal.mat"]
    x_path_list = ["sinos_standardised.npy"]
    y_path_list = ["output_signal_rescaled.npy"]

    standardise(x_path_orig_list, x_path_list)
    rescale(y_path_orig_list, y_path_list)

    window_size = 40
    window_stride_size = 1
    data_window_size = 100
    data_window_stride_size = 1

    if fit_model_bool:
        while_model = None

        epoch_size = 1000

        lr = 1.0
        lr_factor = 0.9

        while True:
            path_length = range(len(x_path_list))

            for i in path_length:
                print("Path: " + str(i) + "/" + str(path_length))

                data_array = np.load(y_path_list[i])
                data_size = data_array.shape[0]

                if data_window_size >= data_size:
                    data_window_size = data_size - 1

                for j in range(0, data_size, data_window_stride_size):
                    print("Path: " + str(j) + "/" + str(data_size))

                    while_model, lr = fit_model(while_model,
                                                True,
                                                load_bool,
                                                False,
                                                True,
                                                x_path_list[i],
                                                y_path_list[i],
                                                j,
                                                data_window_size,
                                                window_size,
                                                data_size,
                                                window_stride_size,
                                                "./results/",
                                                epoch_size,
                                                lr,
                                                lr_factor)

            print("Done")

            if not while_bool:
                break
    else:
        print("Test model")
        print("Load model from file")

        model = k.models.load_model("./results/" + "/model.h5")
        
        path_length = range(len(x_path_list))
        
        for i in path_length:
            print("Path: " + str(i) + "/" + str(path_length))

            data_array = np.load(y_path_list[i])
            data_size = data_array.shape[0]

            if data_window_size >= data_size:
                data_window_size = data_size - 1

            output_list = []

            for j in range(0, data_size, data_window_stride_size):
                print("Path: " + str(j) + "/" + str(data_size))

                current_output = test_model(model,
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

                    for p in range(j):
                        current_output_list[l].insert(0, np.nan)

                    for p in range(len(current_output_list[l]), data_size):
                        current_output_list[l].append(np.nan)

                    output_list.append(current_output_list[l])

            output = np.nanmean(np.asfarray(output_list), axis=0)
            output = stats.zscore(output)

            with open("./results/" + "/signal_" + str(i) + ".csv", "w") as file:
                write_to_file(file, output.reshape(output.size, 1))

        print("Done")


if __name__ == "__main__":
    main(False, False, True)
