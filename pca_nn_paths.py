# Copyright University College London 2020
# Author: Alexander Whitehead, Institute of Nuclear Medicine, UCL
# For internal research only.


def get_paths(single_input_bool, static_bool, autoencoder_input_bool, dynamic_bool):
    if single_input_bool:
        if static_bool:
            x_path_orig_list = [
                "/home/alex/Documents/jrmomo/moving_window_pet_pca/output/Scan2/COAG001/Baseline/sinos_1.mat"]
            y_path_orig_list = [
                "/home/alex/Documents/jrmomo/moving_window_pet_pca/output/Scan2/COAG001/Baseline/output_signal_1.mat"]

            x_path_list = ["normalised_sinos_preprocessed_static_1.npy"]
            y_path_list = ["output_signal_preprocessed_static_1.npy"]

            output_file_name = ["estimated_signal_static_1"]
        else:
            x_path_orig_list = [
                "/home/alex/Documents/jrmomo/moving_window_pet_pca/output/Scan1/COAG001/Baseline/sinos_1.mat"]
            y_path_orig_list = [
                "/home/alex/Documents/jrmomo/moving_window_pet_pca/output/Scan1/COAG001/Baseline/output_signal_1.mat"]

            x_path_list = ["normalised_sinos_preprocessed_dynamic_1.npy"]
            y_path_list = ["output_signal_preprocessed_dynamic_1.npy"]

            output_file_name = ["estimated_signal_dynamic_1"]
    else:
        x_path_orig_list = []
        y_path_orig_list = []

        x_path_list = []
        y_path_list = []

        output_file_name = []

        if autoencoder_input_bool:
            if dynamic_bool:
                x_path_orig_list.extend(
                    ["/home/alex/Documents/jrmomo/moving_window_pet_pca/output/Scan1/COAG001/Baseline/sinos_1.mat",
                     "/home/alex/Documents/jrmomo/moving_window_pet_pca/output/Scan1/COAG002/Baseline/sinos_2.mat",
                     "/home/alex/Documents/jrmomo/moving_window_pet_pca/output/Scan1/COAG003/Baseline/sinos_3.mat",
                     "/home/alex/Documents/jrmomo/moving_window_pet_pca/output/Scan1/COAG004/Baseline/sinos_4.mat",
                     "/home/alex/Documents/jrmomo/moving_window_pet_pca/output/Scan1/COAG005/Baseline/sinos_5.mat",
                     "/home/alex/Documents/jrmomo/moving_window_pet_pca/output/Scan1/COAG006/Baseline/sinos_6.mat",
                     "/home/alex/Documents/jrmomo/moving_window_pet_pca/output/Scan1/COAG007/Baseline/sinos_7.mat",
                     "/home/alex/Documents/jrmomo/moving_window_pet_pca/output/Scan1/COAG008/Baseline/sinos_8.mat",
                     "/home/alex/Documents/jrmomo/moving_window_pet_pca/output/Scan1/COAG009/Baseline/sinos_9.mat",
                     "/home/alex/Documents/jrmomo/moving_window_pet_pca/output/Scan1/COAG001/PostTreatment/sinos_12.mat",
                     "/home/alex/Documents/jrmomo/moving_window_pet_pca/output/Scan1/COAG002/PostTreatment/sinos_13.mat",
                     "/home/alex/Documents/jrmomo/moving_window_pet_pca/output/Scan1/COAG003/PostTreatment/sinos_14.mat",
                     "/home/alex/Documents/jrmomo/moving_window_pet_pca/output/Scan1/COAG004/PostTreatment/sinos_15.mat",
                     "/home/alex/Documents/jrmomo/moving_window_pet_pca/output/Scan1/COAG005/PostTreatment/sinos_16.mat",
                     "/home/alex/Documents/jrmomo/moving_window_pet_pca/output/Scan1/COAG006/PostTreatment/sinos_17.mat",
                     "/home/alex/Documents/jrmomo/moving_window_pet_pca/output/Scan1/COAG007/PostTreatment/sinos_18.mat",
                     "/home/alex/Documents/jrmomo/moving_window_pet_pca/output/Scan1/COAG008/PostTreatment/sinos_19.mat"])
                y_path_orig_list.extend(["/home/alex/Documents/jrmomo/moving_window_pet_pca/output/Scan1/COAG001/Baseline/output_signal_1.mat",
                                         "/home/alex/Documents/jrmomo/moving_window_pet_pca/output/Scan1/COAG001/Baseline/output_signal_1.mat",
                                         "/home/alex/Documents/jrmomo/moving_window_pet_pca/output/Scan1/COAG001/Baseline/output_signal_1.mat",
                                         "/home/alex/Documents/jrmomo/moving_window_pet_pca/output/Scan1/COAG001/Baseline/output_signal_1.mat",
                                         "/home/alex/Documents/jrmomo/moving_window_pet_pca/output/Scan1/COAG001/Baseline/output_signal_1.mat",
                                         "/home/alex/Documents/jrmomo/moving_window_pet_pca/output/Scan1/COAG001/Baseline/output_signal_1.mat",
                                         "/home/alex/Documents/jrmomo/moving_window_pet_pca/output/Scan1/COAG001/Baseline/output_signal_1.mat",
                                         "/home/alex/Documents/jrmomo/moving_window_pet_pca/output/Scan1/COAG001/Baseline/output_signal_1.mat",
                                         "/home/alex/Documents/jrmomo/moving_window_pet_pca/output/Scan1/COAG001/Baseline/output_signal_1.mat",
                                         "/home/alex/Documents/jrmomo/moving_window_pet_pca/output/Scan1/COAG001/Baseline/output_signal_1.mat",
                                         "/home/alex/Documents/jrmomo/moving_window_pet_pca/output/Scan1/COAG001/Baseline/output_signal_1.mat",
                                         "/home/alex/Documents/jrmomo/moving_window_pet_pca/output/Scan1/COAG001/Baseline/output_signal_1.mat",
                                         "/home/alex/Documents/jrmomo/moving_window_pet_pca/output/Scan1/COAG001/Baseline/output_signal_1.mat",
                                         "/home/alex/Documents/jrmomo/moving_window_pet_pca/output/Scan1/COAG001/Baseline/output_signal_1.mat",
                                         "/home/alex/Documents/jrmomo/moving_window_pet_pca/output/Scan1/COAG001/Baseline/output_signal_1.mat",
                                         "/home/alex/Documents/jrmomo/moving_window_pet_pca/output/Scan1/COAG001/Baseline/output_signal_1.mat",
                                         "/home/alex/Documents/jrmomo/moving_window_pet_pca/output/Scan1/COAG001/Baseline/output_signal_1.mat"])

                x_path_list.extend(["normalised_sinos_preprocessed_dynamic_1.npy",
                                    "normalised_sinos_preprocessed_dynamic_2.npy",
                                    "normalised_sinos_preprocessed_dynamic_3.npy",
                                    "normalised_sinos_preprocessed_dynamic_4.npy",
                                    "normalised_sinos_preprocessed_dynamic_5.npy",
                                    "normalised_sinos_preprocessed_dynamic_6.npy",
                                    "normalised_sinos_preprocessed_dynamic_7.npy",
                                    "normalised_sinos_preprocessed_dynamic_8.npy",
                                    "normalised_sinos_preprocessed_dynamic_9.npy",
                                    "normalised_sinos_preprocessed_dynamic_12.npy",
                                    "normalised_sinos_preprocessed_dynamic_13.npy",
                                    "normalised_sinos_preprocessed_dynamic_14.npy",
                                    "normalised_sinos_preprocessed_dynamic_15.npy",
                                    "normalised_sinos_preprocessed_dynamic_16.npy",
                                    "normalised_sinos_preprocessed_dynamic_17.npy",
                                    "normalised_sinos_preprocessed_dynamic_18.npy",
                                    "normalised_sinos_preprocessed_dynamic_19.npy"])
                y_path_list.extend(["output_signal_preprocessed_dynamic_1.npy",
                                    "output_signal_preprocessed_dynamic_1.npy",
                                    "output_signal_preprocessed_dynamic_1.npy",
                                    "output_signal_preprocessed_dynamic_1.npy",
                                    "output_signal_preprocessed_dynamic_1.npy",
                                    "output_signal_preprocessed_dynamic_1.npy",
                                    "output_signal_preprocessed_dynamic_1.npy",
                                    "output_signal_preprocessed_dynamic_1.npy",
                                    "output_signal_preprocessed_dynamic_1.npy",
                                    "output_signal_preprocessed_dynamic_1.npy",
                                    "output_signal_preprocessed_dynamic_1.npy",
                                    "output_signal_preprocessed_dynamic_1.npy",
                                    "output_signal_preprocessed_dynamic_1.npy",
                                    "output_signal_preprocessed_dynamic_1.npy",
                                    "output_signal_preprocessed_dynamic_1.npy",
                                    "output_signal_preprocessed_dynamic_1.npy",
                                    "output_signal_preprocessed_dynamic_1.npy"])

                output_file_name.extend(["estimated_signal_dynamic_1",
                                         "estimated_signal_dynamic_1",
                                         "estimated_signal_dynamic_1",
                                         "estimated_signal_dynamic_1",
                                         "estimated_signal_dynamic_1",
                                         "estimated_signal_dynamic_1",
                                         "estimated_signal_dynamic_1",
                                         "estimated_signal_dynamic_1",
                                         "estimated_signal_dynamic_1",
                                         "estimated_signal_dynamic_1",
                                         "estimated_signal_dynamic_1",
                                         "estimated_signal_dynamic_1",
                                         "estimated_signal_dynamic_1",
                                         "estimated_signal_dynamic_1",
                                         "estimated_signal_dynamic_1",
                                         "estimated_signal_dynamic_1",
                                         "estimated_signal_dynamic_1"])

            if static_bool:
                x_path_orig_list.extend(
                    ["/home/alex/Documents/jrmomo/moving_window_pet_pca/output/Scan2/COAG001/Baseline/sinos_1.mat",
                     "/home/alex/Documents/jrmomo/moving_window_pet_pca/output/Scan2/COAG002/Baseline/sinos_2.mat",
                     "/home/alex/Documents/jrmomo/moving_window_pet_pca/output/Scan2/COAG003/Baseline/sinos_3.mat",
                     "/home/alex/Documents/jrmomo/moving_window_pet_pca/output/Scan2/COAG004/Baseline/sinos_4.mat",
                     "/home/alex/Documents/jrmomo/moving_window_pet_pca/output/Scan2/COAG006/Baseline/sinos_6.mat",
                     "/home/alex/Documents/jrmomo/moving_window_pet_pca/output/Scan2/COAG007/Baseline/sinos_7.mat",
                     "/home/alex/Documents/jrmomo/moving_window_pet_pca/output/Scan2/COAG008/Baseline/sinos_8.mat",
                     "/home/alex/Documents/jrmomo/moving_window_pet_pca/output/Scan2/COAG001/PostTreatment/sinos_12.mat",
                     "/home/alex/Documents/jrmomo/moving_window_pet_pca/output/Scan2/COAG003/PostTreatment/sinos_14.mat",
                     "/home/alex/Documents/jrmomo/moving_window_pet_pca/output/Scan2/COAG004/PostTreatment/sinos_15.mat",
                     "/home/alex/Documents/jrmomo/moving_window_pet_pca/output/Scan2/COAG006/PostTreatment/sinos_17.mat",
                     "/home/alex/Documents/jrmomo/moving_window_pet_pca/output/Scan2/COAG007/PostTreatment/sinos_18.mat",
                     "/home/alex/Documents/jrmomo/moving_window_pet_pca/output/Scan2/COAG008/PostTreatment/sinos_19.mat"])
                y_path_orig_list.extend([
                                            "/home/alex/Documents/jrmomo/moving_window_pet_pca/output/Scan2/COAG001/Baseline/output_signal_1.mat",
                                            "/home/alex/Documents/jrmomo/moving_window_pet_pca/output/Scan2/COAG001/Baseline/output_signal_1.mat",
                                            "/home/alex/Documents/jrmomo/moving_window_pet_pca/output/Scan2/COAG001/Baseline/output_signal_1.mat",
                                            "/home/alex/Documents/jrmomo/moving_window_pet_pca/output/Scan2/COAG001/Baseline/output_signal_1.mat",
                                            "/home/alex/Documents/jrmomo/moving_window_pet_pca/output/Scan2/COAG001/Baseline/output_signal_1.mat",
                                            "/home/alex/Documents/jrmomo/moving_window_pet_pca/output/Scan2/COAG001/Baseline/output_signal_1.mat",
                                            "/home/alex/Documents/jrmomo/moving_window_pet_pca/output/Scan2/COAG001/Baseline/output_signal_1.mat",
                                            "/home/alex/Documents/jrmomo/moving_window_pet_pca/output/Scan2/COAG001/Baseline/output_signal_1.mat",
                                            "/home/alex/Documents/jrmomo/moving_window_pet_pca/output/Scan2/COAG001/Baseline/output_signal_1.mat",
                                            "/home/alex/Documents/jrmomo/moving_window_pet_pca/output/Scan2/COAG001/Baseline/output_signal_1.mat",
                                            "/home/alex/Documents/jrmomo/moving_window_pet_pca/output/Scan2/COAG001/Baseline/output_signal_1.mat",
                                            "/home/alex/Documents/jrmomo/moving_window_pet_pca/output/Scan2/COAG001/Baseline/output_signal_1.mat",
                                            "/home/alex/Documents/jrmomo/moving_window_pet_pca/output/Scan2/COAG001/Baseline/output_signal_1.mat"])

                x_path_list.extend(["normalised_sinos_preprocessed_static_1.npy",
                                    "normalised_sinos_preprocessed_static_2.npy",
                                    "normalised_sinos_preprocessed_static_3.npy",
                                    "normalised_sinos_preprocessed_static_4.npy",
                                    "normalised_sinos_preprocessed_static_6.npy",
                                    "normalised_sinos_preprocessed_static_7.npy",
                                    "normalised_sinos_preprocessed_static_8.npy",
                                    "normalised_sinos_preprocessed_static_12.npy",
                                    "normalised_sinos_preprocessed_static_14.npy",
                                    "normalised_sinos_preprocessed_static_15.npy",
                                    "normalised_sinos_preprocessed_static_17.npy",
                                    "normalised_sinos_preprocessed_static_18.npy",
                                    "normalised_sinos_preprocessed_static_19.npy"])
                y_path_list.extend(["output_signal_preprocessed_static_1.npy",
                                    "output_signal_preprocessed_static_1.npy",
                                    "output_signal_preprocessed_static_1.npy",
                                    "output_signal_preprocessed_static_1.npy",
                                    "output_signal_preprocessed_static_1.npy",
                                    "output_signal_preprocessed_static_1.npy",
                                    "output_signal_preprocessed_static_1.npy",
                                    "output_signal_preprocessed_static_1.npy",
                                    "output_signal_preprocessed_static_1.npy",
                                    "output_signal_preprocessed_static_1.npy",
                                    "output_signal_preprocessed_static_1.npy",
                                    "output_signal_preprocessed_static_1.npy",
                                    "output_signal_preprocessed_static_1.npy"])

                output_file_name.extend(["estimated_signal_static_1",
                                         "estimated_signal_static_1",
                                         "estimated_signal_static_1",
                                         "estimated_signal_static_1",
                                         "estimated_signal_static_1",
                                         "estimated_signal_static_1",
                                         "estimated_signal_static_1",
                                         "estimated_signal_static_1",
                                         "estimated_signal_static_1",
                                         "estimated_signal_static_1",
                                         "estimated_signal_static_1",
                                         "estimated_signal_static_1",
                                         "estimated_signal_static_1"])
        else:
            if dynamic_bool:
                x_path_orig_list.extend(
                    ["/home/alex/Documents/jrmomo/moving_window_pet_pca/output/Scan1/COAG001/Baseline/sinos_1.mat",
                     "/home/alex/Documents/jrmomo/moving_window_pet_pca/output/Scan1/COAG003/Baseline/sinos_3.mat",
                     "/home/alex/Documents/jrmomo/moving_window_pet_pca/output/Scan1/COAG005/Baseline/sinos_5.mat",
                     "/home/alex/Documents/jrmomo/moving_window_pet_pca/output/Scan1/COAG006/Baseline/sinos_6.mat",
                     "/home/alex/Documents/jrmomo/moving_window_pet_pca/output/Scan1/COAG007/Baseline/sinos_7.mat",
                     "/home/alex/Documents/jrmomo/moving_window_pet_pca/output/Scan1/COAG008/Baseline/sinos_8.mat",
                     "/home/alex/Documents/jrmomo/moving_window_pet_pca/output/Scan1/COAG009/Baseline/sinos_9.mat",
                     "/home/alex/Documents/jrmomo/moving_window_pet_pca/output/Scan1/COAG003/PostTreatment/sinos_14.mat",
                     "/home/alex/Documents/jrmomo/moving_window_pet_pca/output/Scan1/COAG007/PostTreatment/sinos_18.mat",
                     "/home/alex/Documents/jrmomo/moving_window_pet_pca/output/Scan1/COAG008/PostTreatment/sinos_19.mat"])
                y_path_orig_list.extend(
                    [
                        "/home/alex/Documents/jrmomo/moving_window_pet_pca/output/Scan1/COAG001/Baseline/output_signal_1.mat",
                        "/home/alex/Documents/jrmomo/moving_window_pet_pca/output/Scan1/COAG003/Baseline/output_signal_3.mat",
                        "/home/alex/Documents/jrmomo/moving_window_pet_pca/output/Scan1/COAG005/Baseline/output_signal_5.mat",
                        "/home/alex/Documents/jrmomo/moving_window_pet_pca/output/Scan1/COAG006/Baseline/output_signal_6.mat",
                        "/home/alex/Documents/jrmomo/moving_window_pet_pca/output/Scan1/COAG007/Baseline/output_signal_7.mat",
                        "/home/alex/Documents/jrmomo/moving_window_pet_pca/output/Scan1/COAG008/Baseline/output_signal_8.mat",
                        "/home/alex/Documents/jrmomo/moving_window_pet_pca/output/Scan1/COAG009/Baseline/output_signal_9.mat",
                        "/home/alex/Documents/jrmomo/moving_window_pet_pca/output/Scan1/COAG003/PostTreatment/output_signal_14.mat",
                        "/home/alex/Documents/jrmomo/moving_window_pet_pca/output/Scan1/COAG007/PostTreatment/output_signal_18.mat",
                        "/home/alex/Documents/jrmomo/moving_window_pet_pca/output/Scan1/COAG008/PostTreatment/output_signal_19.mat"])

                x_path_list.extend(["normalised_sinos_preprocessed_dynamic_1.npy",
                                    "normalised_sinos_preprocessed_dynamic_3.npy",
                                    "normalised_sinos_preprocessed_dynamic_5.npy",
                                    "normalised_sinos_preprocessed_dynamic_6.npy",
                                    "normalised_sinos_preprocessed_dynamic_7.npy",
                                    "normalised_sinos_preprocessed_dynamic_8.npy",
                                    "normalised_sinos_preprocessed_dynamic_9.npy",
                                    "normalised_sinos_preprocessed_dynamic_14.npy",
                                    "normalised_sinos_preprocessed_dynamic_18.npy",
                                    "normalised_sinos_preprocessed_dynamic_19.npy"])
                y_path_list.extend(["output_signal_preprocessed_dynamic_1.npy",
                                    "output_signal_preprocessed_dynamic_3.npy",
                                    "output_signal_preprocessed_dynamic_5.npy",
                                    "output_signal_preprocessed_dynamic_6.npy",
                                    "output_signal_preprocessed_dynamic_7.npy",
                                    "output_signal_preprocessed_dynamic_8.npy",
                                    "output_signal_preprocessed_dynamic_9.npy",
                                    "output_signal_preprocessed_dynamic_14.npy",
                                    "output_signal_preprocessed_dynamic_18.npy",
                                    "output_signal_preprocessed_dynamic_19.npy"])

                output_file_name.extend(["estimated_signal_dynamic_1",
                                         "estimated_signal_dynamic_3",
                                         "estimated_signal_dynamic_5",
                                         "estimated_signal_dynamic_6",
                                         "estimated_signal_dynamic_7",
                                         "estimated_signal_dynamic_8",
                                         "estimated_signal_dynamic_9",
                                         "estimated_signal_dynamic_15",
                                         "estimated_signal_dynamic_19",
                                         "estimated_signal_dynamic_19"])

            if static_bool:
                x_path_orig_list.extend(
                    ["/home/alex/Documents/jrmomo/moving_window_pet_pca/output/Scan2/COAG001/Baseline/sinos_1.mat",
                     "/home/alex/Documents/jrmomo/moving_window_pet_pca/output/Scan2/COAG006/Baseline/sinos_6.mat",
                     "/home/alex/Documents/jrmomo/moving_window_pet_pca/output/Scan2/COAG007/Baseline/sinos_7.mat",
                     "/home/alex/Documents/jrmomo/moving_window_pet_pca/output/Scan2/COAG008/Baseline/sinos_8.mat",
                     "/home/alex/Documents/jrmomo/moving_window_pet_pca/output/Scan2/COAG001/PostTreatment/sinos_12.mat",
                     "/home/alex/Documents/jrmomo/moving_window_pet_pca/output/Scan2/COAG004/PostTreatment/sinos_15.mat",
                     "/home/alex/Documents/jrmomo/moving_window_pet_pca/output/Scan2/COAG007/PostTreatment/sinos_18.mat",
                     "/home/alex/Documents/jrmomo/moving_window_pet_pca/output/Scan2/COAG008/PostTreatment/sinos_19.mat"])
                y_path_orig_list.extend(
                    [
                        "/home/alex/Documents/jrmomo/moving_window_pet_pca/output/Scan2/COAG001/Baseline/output_signal_1.mat",
                        "/home/alex/Documents/jrmomo/moving_window_pet_pca/output/Scan2/COAG006/Baseline/output_signal_6.mat",
                        "/home/alex/Documents/jrmomo/moving_window_pet_pca/output/Scan2/COAG007/Baseline/output_signal_7.mat",
                        "/home/alex/Documents/jrmomo/moving_window_pet_pca/output/Scan2/COAG008/Baseline/output_signal_8.mat",
                        "/home/alex/Documents/jrmomo/moving_window_pet_pca/output/Scan2/COAG001/PostTreatment/output_signal_12.mat",
                        "/home/alex/Documents/jrmomo/moving_window_pet_pca/output/Scan2/COAG004/PostTreatment/output_signal_15.mat",
                        "/home/alex/Documents/jrmomo/moving_window_pet_pca/output/Scan2/COAG007/PostTreatment/output_signal_18.mat",
                        "/home/alex/Documents/jrmomo/moving_window_pet_pca/output/Scan2/COAG008/PostTreatment/output_signal_19.mat"])

                x_path_list.extend(["normalised_sinos_preprocessed_static_1.npy",
                                    "normalised_sinos_preprocessed_static_6.npy",
                                    "normalised_sinos_preprocessed_static_7.npy",
                                    "normalised_sinos_preprocessed_static_8.npy",
                                    "normalised_sinos_preprocessed_static_12.npy",
                                    "normalised_sinos_preprocessed_static_15.npy",
                                    "normalised_sinos_preprocessed_static_18.npy",
                                    "normalised_sinos_preprocessed_static_19.npy"])
                y_path_list.extend(["output_signal_preprocessed_static_1.npy",
                                    "output_signal_preprocessed_static_6.npy",
                                    "output_signal_preprocessed_static_7.npy",
                                    "output_signal_preprocessed_static_8.npy",
                                    "output_signal_preprocessed_static_12.npy",
                                    "output_signal_preprocessed_static_15.npy",
                                    "output_signal_preprocessed_static_18.npy",
                                    "output_signal_preprocessed_static_19.npy"])

                output_file_name.extend(["estimated_signal_static_1",
                                         "estimated_signal_static_6",
                                         "estimated_signal_static_7",
                                         "estimated_signal_static_8",
                                         "estimated_signal_static_12",
                                         "estimated_signal_static_15",
                                         "estimated_signal_static_18",
                                         "estimated_signal_static_19"])

    test_x_path_list = x_path_list
    test_y_path_list = y_path_list

    return x_path_orig_list, x_path_list, test_x_path_list, y_path_orig_list, test_y_path_list, y_path_list, output_file_name
