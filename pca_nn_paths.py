# Copyright University College London 2020
# Author: Alexander Whitehead, Institute of Nuclear Medicine, UCL
# For internal research only.


def get_paths(single_input_bool, static_bool, autoencoder_input_bool, dynamic_bool, tested_input_bool, original_bool,
              pca_bool, conservative_bool):
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
                y_path_orig_list.extend([
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
            if tested_input_bool:
                if dynamic_bool:
                    x_path_orig_list.extend(
                        ["/home/alex/Documents/jrmomo/moving_window_pet_pca/output/Scan1/COAG001/Baseline/sinos_1.mat",
                         "/home/alex/Documents/jrmomo/moving_window_pet_pca/output/Scan1/COAG007/Baseline/sinos_7.mat",
                         "/home/alex/Documents/jrmomo/moving_window_pet_pca/output/Scan1/COAG008/Baseline/sinos_8.mat",
                         "/home/alex/Documents/jrmomo/moving_window_pet_pca/output/Scan1/COAG007/PostTreatment/sinos_18.mat",
                         "/home/alex/Documents/jrmomo/moving_window_pet_pca/output/Scan1/COAG008/PostTreatment/sinos_19.mat"])
                    y_path_orig_list.extend(
                        [
                            "/home/alex/Documents/jrmomo/moving_window_pet_pca/output/Scan1/COAG001/Baseline/output_signal_1.mat",
                            "/home/alex/Documents/jrmomo/moving_window_pet_pca/output/Scan1/COAG007/Baseline/output_signal_7.mat",
                            "/home/alex/Documents/jrmomo/moving_window_pet_pca/output/Scan1/COAG008/Baseline/output_signal_8.mat",
                            "/home/alex/Documents/jrmomo/moving_window_pet_pca/output/Scan1/COAG007/PostTreatment/output_signal_18.mat",
                            "/home/alex/Documents/jrmomo/moving_window_pet_pca/output/Scan1/COAG008/PostTreatment/output_signal_19.mat"])

                    x_path_list.extend(["normalised_sinos_preprocessed_dynamic_1.npy",
                                        "normalised_sinos_preprocessed_dynamic_7.npy",
                                        "normalised_sinos_preprocessed_dynamic_8.npy",
                                        "normalised_sinos_preprocessed_dynamic_18.npy",
                                        "normalised_sinos_preprocessed_dynamic_19.npy"])
                    y_path_list.extend(["output_signal_preprocessed_dynamic_1.npy",
                                        "output_signal_preprocessed_dynamic_7.npy",
                                        "output_signal_preprocessed_dynamic_8.npy",
                                        "output_signal_preprocessed_dynamic_18.npy",
                                        "output_signal_preprocessed_dynamic_19.npy"])

                    output_file_name.extend(["estimated_signal_dynamic_1",
                                             "estimated_signal_dynamic_7",
                                             "estimated_signal_dynamic_8",
                                             "estimated_signal_dynamic_18",
                                             "estimated_signal_dynamic_19"])

                    if original_bool:
                        x_path_orig_list.extend(
                            [
                                "/home/alex/Documents/jrmomo/moving_window_pet_pca/output/Scan1/COAG001/Baseline/sinos_1.mat",
                                "/home/alex/Documents/jrmomo/moving_window_pet_pca/output/Scan1/COAG007/Baseline/sinos_7.mat",
                                "/home/alex/Documents/jrmomo/moving_window_pet_pca/output/Scan1/COAG008/Baseline/sinos_8.mat",
                                "/home/alex/Documents/jrmomo/moving_window_pet_pca/output/Scan1/COAG007/PostTreatment/sinos_18.mat",
                                "/home/alex/Documents/jrmomo/moving_window_pet_pca/output/Scan1/COAG008/PostTreatment/sinos_19.mat"])
                        y_path_orig_list.extend(
                            [
                                "/home/alex/Documents/jrmomo/moving_window_pet_pca/output/Scan1/COAG001/Baseline/output_signal_original_1.mat",
                                "/home/alex/Documents/jrmomo/moving_window_pet_pca/output/Scan1/COAG007/Baseline/output_signal_original_7.mat",
                                "/home/alex/Documents/jrmomo/moving_window_pet_pca/output/Scan1/COAG008/Baseline/output_signal_original_8.mat",
                                "/home/alex/Documents/jrmomo/moving_window_pet_pca/output/Scan1/COAG007/PostTreatment/output_signal_original_18.mat",
                                "/home/alex/Documents/jrmomo/moving_window_pet_pca/output/Scan1/COAG008/PostTreatment/output_signal_original_19.mat"])

                        x_path_list.extend(["normalised_sinos_preprocessed_dynamic_1.npy",
                                            "normalised_sinos_preprocessed_dynamic_7.npy",
                                            "normalised_sinos_preprocessed_dynamic_8.npy",
                                            "normalised_sinos_preprocessed_dynamic_18.npy",
                                            "normalised_sinos_preprocessed_dynamic_19.npy"])
                        y_path_list.extend(["output_signal_original_preprocessed_dynamic_1.npy",
                                            "output_signal_original_preprocessed_dynamic_7.npy",
                                            "output_signal_original_preprocessed_dynamic_8.npy",
                                            "output_signal_original_preprocessed_dynamic_18.npy",
                                            "output_signal_original_preprocessed_dynamic_19.npy"])

                        output_file_name.extend(["estimated_signal_dynamic_1",
                                                 "estimated_signal_dynamic_7",
                                                 "estimated_signal_dynamic_8",
                                                 "estimated_signal_dynamic_18",
                                                 "estimated_signal_dynamic_19"])

                    if pca_bool:
                        x_path_orig_list.extend(
                            [
                                "/home/alex/Documents/jrmomo/moving_window_pet_pca/output/Scan1/COAG001/Baseline/sinos_1.mat",
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
                        y_path_orig_list.extend(
                            [
                                "/home/alex/Documents/jrmomo/moving_window_pet_pca/output/Scan1/COAG001/Baseline/mean_output_one_pc_w_1.mat",
                                "/home/alex/Documents/jrmomo/moving_window_pet_pca/output/Scan1/COAG002/Baseline/mean_output_one_pc_w_2.mat",
                                "/home/alex/Documents/jrmomo/moving_window_pet_pca/output/Scan1/COAG003/Baseline/mean_output_one_pc_w_3.mat",
                                "/home/alex/Documents/jrmomo/moving_window_pet_pca/output/Scan1/COAG004/Baseline/mean_output_one_pc_w_4.mat",
                                "/home/alex/Documents/jrmomo/moving_window_pet_pca/output/Scan1/COAG005/Baseline/mean_output_one_pc_w_5.mat",
                                "/home/alex/Documents/jrmomo/moving_window_pet_pca/output/Scan1/COAG006/Baseline/mean_output_one_pc_w_6.mat",
                                "/home/alex/Documents/jrmomo/moving_window_pet_pca/output/Scan1/COAG007/Baseline/mean_output_one_pc_w_7.mat",
                                "/home/alex/Documents/jrmomo/moving_window_pet_pca/output/Scan1/COAG008/Baseline/mean_output_one_pc_w_8.mat",
                                "/home/alex/Documents/jrmomo/moving_window_pet_pca/output/Scan1/COAG009/Baseline/mean_output_one_pc_w_9.mat",
                                "/home/alex/Documents/jrmomo/moving_window_pet_pca/output/Scan1/COAG001/PostTreatment/mean_output_one_pc_w_12.mat",
                                "/home/alex/Documents/jrmomo/moving_window_pet_pca/output/Scan1/COAG002/PostTreatment/mean_output_one_pc_w_13.mat",
                                "/home/alex/Documents/jrmomo/moving_window_pet_pca/output/Scan1/COAG003/PostTreatment/mean_output_one_pc_w_14.mat",
                                "/home/alex/Documents/jrmomo/moving_window_pet_pca/output/Scan1/COAG004/PostTreatment/mean_output_one_pc_w_15.mat",
                                "/home/alex/Documents/jrmomo/moving_window_pet_pca/output/Scan1/COAG005/PostTreatment/mean_output_one_pc_w_16.mat",
                                "/home/alex/Documents/jrmomo/moving_window_pet_pca/output/Scan1/COAG006/PostTreatment/mean_output_one_pc_w_17.mat",
                                "/home/alex/Documents/jrmomo/moving_window_pet_pca/output/Scan1/COAG007/PostTreatment/mean_output_one_pc_w_18.mat",
                                "/home/alex/Documents/jrmomo/moving_window_pet_pca/output/Scan1/COAG008/PostTreatment/mean_output_one_pc_w_19.mat"])

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
                        y_path_list.extend(["mean_output_one_pc_w_preprocessed_dynamic_1.npy",
                                            "mean_output_one_pc_w_preprocessed_dynamic_2.npy",
                                            "mean_output_one_pc_w_preprocessed_dynamic_3.npy",
                                            "mean_output_one_pc_w_preprocessed_dynamic_4.npy",
                                            "mean_output_one_pc_w_preprocessed_dynamic_5.npy",
                                            "mean_output_one_pc_w_preprocessed_dynamic_6.npy",
                                            "mean_output_one_pc_w_preprocessed_dynamic_7.npy",
                                            "mean_output_one_pc_w_preprocessed_dynamic_8.npy",
                                            "mean_output_one_pc_w_preprocessed_dynamic_9.npy",
                                            "mean_output_one_pc_w_preprocessed_dynamic_12.npy",
                                            "mean_output_one_pc_w_preprocessed_dynamic_13.npy",
                                            "mean_output_one_pc_w_preprocessed_dynamic_14.npy",
                                            "mean_output_one_pc_w_preprocessed_dynamic_15.npy",
                                            "mean_output_one_pc_w_preprocessed_dynamic_16.npy",
                                            "mean_output_one_pc_w_preprocessed_dynamic_17.npy",
                                            "mean_output_one_pc_w_preprocessed_dynamic_18.npy",
                                            "mean_output_one_pc_w_preprocessed_dynamic_19.npy"])

                        output_file_name.extend(["estimated_signal_dynamic_1",
                                                 "estimated_signal_dynamic_2",
                                                 "estimated_signal_dynamic_3",
                                                 "estimated_signal_dynamic_4",
                                                 "estimated_signal_dynamic_5",
                                                 "estimated_signal_dynamic_6",
                                                 "estimated_signal_dynamic_7",
                                                 "estimated_signal_dynamic_8",
                                                 "estimated_signal_dynamic_9",
                                                 "estimated_signal_dynamic_12",
                                                 "estimated_signal_dynamic_13",
                                                 "estimated_signal_dynamic_14",
                                                 "estimated_signal_dynamic_15",
                                                 "estimated_signal_dynamic_16",
                                                 "estimated_signal_dynamic_17",
                                                 "estimated_signal_dynamic_18",
                                                 "estimated_signal_dynamic_19"])

                    if not conservative_bool:
                        x_path_orig_list.extend(
                            [
                                "/home/alex/Documents/jrmomo/moving_window_pet_pca/output/Scan1/COAG006/Baseline/sinos_6.mat"])
                        y_path_orig_list.extend(
                            [
                                "/home/alex/Documents/jrmomo/moving_window_pet_pca/output/Scan1/COAG006/Baseline/output_signal_6.mat"])

                        x_path_list.extend(["normalised_sinos_preprocessed_dynamic_6.npy"])
                        y_path_list.extend(["output_signal_preprocessed_dynamic_6.npy"])

                        output_file_name.extend(["estimated_signal_dynamic_6"])

                        if original_bool:
                            x_path_orig_list.extend(
                                [
                                    "/home/alex/Documents/jrmomo/moving_window_pet_pca/output/Scan1/COAG006/Baseline/sinos_6.mat"])
                            y_path_orig_list.extend(
                                [
                                    "/home/alex/Documents/jrmomo/moving_window_pet_pca/output/Scan1/COAG006/Baseline/output_signal_original_6.mat"])

                            x_path_list.extend(["normalised_sinos_preprocessed_dynamic_6.npy"])
                            y_path_list.extend(["output_signal_original_preprocessed_dynamic_6.npy"])

                            output_file_name.extend(["estimated_signal_dynamic_6"])

                        if pca_bool:
                            x_path_orig_list.extend(
                                [
                                    "/home/alex/Documents/jrmomo/moving_window_pet_pca/output/Scan1/COAG001/Baseline/sinos_1.mat",
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
                            y_path_orig_list.extend(
                                [
                                    "/home/alex/Documents/jrmomo/moving_window_pet_pca/output/Scan1/COAG001/Baseline/mean_output_1.mat",
                                    "/home/alex/Documents/jrmomo/moving_window_pet_pca/output/Scan1/COAG002/Baseline/mean_output_2.mat",
                                    "/home/alex/Documents/jrmomo/moving_window_pet_pca/output/Scan1/COAG003/Baseline/mean_output_3.mat",
                                    "/home/alex/Documents/jrmomo/moving_window_pet_pca/output/Scan1/COAG004/Baseline/mean_output_4.mat",
                                    "/home/alex/Documents/jrmomo/moving_window_pet_pca/output/Scan1/COAG005/Baseline/mean_output_5.mat",
                                    "/home/alex/Documents/jrmomo/moving_window_pet_pca/output/Scan1/COAG006/Baseline/mean_output_6.mat",
                                    "/home/alex/Documents/jrmomo/moving_window_pet_pca/output/Scan1/COAG007/Baseline/mean_output_7.mat",
                                    "/home/alex/Documents/jrmomo/moving_window_pet_pca/output/Scan1/COAG008/Baseline/mean_output_8.mat",
                                    "/home/alex/Documents/jrmomo/moving_window_pet_pca/output/Scan1/COAG009/Baseline/mean_output_9.mat",
                                    "/home/alex/Documents/jrmomo/moving_window_pet_pca/output/Scan1/COAG001/PostTreatment/mean_output_12.mat",
                                    "/home/alex/Documents/jrmomo/moving_window_pet_pca/output/Scan1/COAG002/PostTreatment/mean_output_13.mat",
                                    "/home/alex/Documents/jrmomo/moving_window_pet_pca/output/Scan1/COAG003/PostTreatment/mean_output_14.mat",
                                    "/home/alex/Documents/jrmomo/moving_window_pet_pca/output/Scan1/COAG004/PostTreatment/mean_output_15.mat",
                                    "/home/alex/Documents/jrmomo/moving_window_pet_pca/output/Scan1/COAG005/PostTreatment/mean_output_16.mat",
                                    "/home/alex/Documents/jrmomo/moving_window_pet_pca/output/Scan1/COAG006/PostTreatment/mean_output_17.mat",
                                    "/home/alex/Documents/jrmomo/moving_window_pet_pca/output/Scan1/COAG007/PostTreatment/mean_output_18.mat",
                                    "/home/alex/Documents/jrmomo/moving_window_pet_pca/output/Scan1/COAG008/PostTreatment/mean_output_19.mat"])

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
                            y_path_list.extend(["mean_output_preprocessed_dynamic_1.npy",
                                                "mean_output_preprocessed_dynamic_2.npy",
                                                "mean_output_preprocessed_dynamic_3.npy",
                                                "mean_output_preprocessed_dynamic_4.npy",
                                                "mean_output_preprocessed_dynamic_5.npy",
                                                "mean_output_preprocessed_dynamic_6.npy",
                                                "mean_output_preprocessed_dynamic_7.npy",
                                                "mean_output_preprocessed_dynamic_8.npy",
                                                "mean_output_preprocessed_dynamic_9.npy",
                                                "mean_output_preprocessed_dynamic_12.npy",
                                                "mean_output_preprocessed_dynamic_13.npy",
                                                "mean_output_preprocessed_dynamic_14.npy",
                                                "mean_output_preprocessed_dynamic_15.npy",
                                                "mean_output_preprocessed_dynamic_16.npy",
                                                "mean_output_preprocessed_dynamic_17.npy",
                                                "mean_output_preprocessed_dynamic_18.npy",
                                                "mean_output_preprocessed_dynamic_19.npy"])

                            output_file_name.extend(["estimated_signal_dynamic_1",
                                                     "estimated_signal_dynamic_2",
                                                     "estimated_signal_dynamic_3",
                                                     "estimated_signal_dynamic_4",
                                                     "estimated_signal_dynamic_5",
                                                     "estimated_signal_dynamic_6",
                                                     "estimated_signal_dynamic_7",
                                                     "estimated_signal_dynamic_8",
                                                     "estimated_signal_dynamic_9",
                                                     "estimated_signal_dynamic_12",
                                                     "estimated_signal_dynamic_13",
                                                     "estimated_signal_dynamic_14",
                                                     "estimated_signal_dynamic_15",
                                                     "estimated_signal_dynamic_16",
                                                     "estimated_signal_dynamic_17",
                                                     "estimated_signal_dynamic_18",
                                                     "estimated_signal_dynamic_19"])

                if static_bool:
                    x_path_orig_list.extend(
                        ["/home/alex/Documents/jrmomo/moving_window_pet_pca/output/Scan2/COAG007/Baseline/sinos_7.mat",
                         "/home/alex/Documents/jrmomo/moving_window_pet_pca/output/Scan2/COAG004/PostTreatment/sinos_15.mat",
                         "/home/alex/Documents/jrmomo/moving_window_pet_pca/output/Scan2/COAG007/PostTreatment/sinos_18.mat",
                         "/home/alex/Documents/jrmomo/moving_window_pet_pca/output/Scan2/COAG008/PostTreatment/sinos_19.mat"])
                    y_path_orig_list.extend(
                        [
                            "/home/alex/Documents/jrmomo/moving_window_pet_pca/output/Scan2/COAG007/Baseline/output_signal_7.mat",
                            "/home/alex/Documents/jrmomo/moving_window_pet_pca/output/Scan2/COAG004/PostTreatment/output_signal_15.mat",
                            "/home/alex/Documents/jrmomo/moving_window_pet_pca/output/Scan2/COAG007/PostTreatment/output_signal_18.mat",
                            "/home/alex/Documents/jrmomo/moving_window_pet_pca/output/Scan2/COAG008/PostTreatment/output_signal_19.mat"])

                    x_path_list.extend(["normalised_sinos_preprocessed_static_7.npy",
                                        "normalised_sinos_preprocessed_static_15.npy",
                                        "normalised_sinos_preprocessed_static_18.npy",
                                        "normalised_sinos_preprocessed_static_19.npy"])
                    y_path_list.extend(["output_signal_preprocessed_static_7.npy",
                                        "output_signal_preprocessed_static_15.npy",
                                        "output_signal_preprocessed_static_18.npy",
                                        "output_signal_preprocessed_static_19.npy"])

                    output_file_name.extend(["estimated_signal_static_7",
                                             "estimated_signal_static_15",
                                             "estimated_signal_static_18",
                                             "estimated_signal_static_19"])

                    if original_bool:
                        x_path_orig_list.extend(
                            [
                                "/home/alex/Documents/jrmomo/moving_window_pet_pca/output/Scan2/COAG007/Baseline/sinos_7.mat",
                                "/home/alex/Documents/jrmomo/moving_window_pet_pca/output/Scan2/COAG004/PostTreatment/sinos_15.mat",
                                "/home/alex/Documents/jrmomo/moving_window_pet_pca/output/Scan2/COAG007/PostTreatment/sinos_18.mat",
                                "/home/alex/Documents/jrmomo/moving_window_pet_pca/output/Scan2/COAG008/PostTreatment/sinos_19.mat"])
                        y_path_orig_list.extend(
                            [
                                "/home/alex/Documents/jrmomo/moving_window_pet_pca/output/Scan2/COAG007/Baseline/output_signal_original_7.mat",
                                "/home/alex/Documents/jrmomo/moving_window_pet_pca/output/Scan2/COAG004/PostTreatment/output_signal_original_15.mat",
                                "/home/alex/Documents/jrmomo/moving_window_pet_pca/output/Scan2/COAG007/PostTreatment/output_signal_original_18.mat",
                                "/home/alex/Documents/jrmomo/moving_window_pet_pca/output/Scan2/COAG008/PostTreatment/output_signal_original_19.mat"])

                        x_path_list.extend(["normalised_sinos_preprocessed_static_7.npy",
                                            "normalised_sinos_preprocessed_static_15.npy",
                                            "normalised_sinos_preprocessed_static_18.npy",
                                            "normalised_sinos_preprocessed_static_19.npy"])
                        y_path_list.extend(["output_signal_original_preprocessed_static_7.npy",
                                            "output_signal_original_preprocessed_static_15.npy",
                                            "output_signal_original_preprocessed_static_18.npy",
                                            "output_signal_original_preprocessed_static_19.npy"])

                        output_file_name.extend(["estimated_signal_static_7",
                                                 "estimated_signal_static_15",
                                                 "estimated_signal_static_18",
                                                 "estimated_signal_static_19"])

                    if pca_bool:
                        x_path_orig_list.extend(
                            [
                                "/home/alex/Documents/jrmomo/moving_window_pet_pca/output/Scan2/COAG001/Baseline/sinos_1.mat",
                                "/home/alex/Documents/jrmomo/moving_window_pet_pca/output/Scan2/COAG002/Baseline/sinos_2.mat",
                                "/home/alex/Documents/jrmomo/moving_window_pet_pca/output/Scan2/COAG003/Baseline/sinos_3.mat",
                                "/home/alex/Documents/jrmomo/moving_window_pet_pca/output/Scan2/COAG004/Baseline/sinos_4.mat",
                                "/home/alex/Documents/jrmomo/moving_window_pet_pca/output/Scan2/COAG006/Baseline/sinos_6.mat",
                                "/home/alex/Documents/jrmomo/moving_window_pet_pca/output/Scan2/COAG007/Baseline/sinos_7.mat",
                                "/home/alex/Documents/jrmomo/moving_window_pet_pca/output/Scan2/COAG008/Baseline/sinos_8.mat",
                                "/home/alex/Documents/jrmomo/moving_window_pet_pca/output/Scan2/COAG009/Baseline/sinos_9.mat",
                                "/home/alex/Documents/jrmomo/moving_window_pet_pca/output/Scan2/COAG001/PostTreatment/sinos_12.mat",
                                "/home/alex/Documents/jrmomo/moving_window_pet_pca/output/Scan2/COAG002/PostTreatment/sinos_13.mat",
                                "/home/alex/Documents/jrmomo/moving_window_pet_pca/output/Scan2/COAG003/PostTreatment/sinos_14.mat",
                                "/home/alex/Documents/jrmomo/moving_window_pet_pca/output/Scan2/COAG004/PostTreatment/sinos_15.mat",
                                "/home/alex/Documents/jrmomo/moving_window_pet_pca/output/Scan2/COAG005/PostTreatment/sinos_16.mat",
                                "/home/alex/Documents/jrmomo/moving_window_pet_pca/output/Scan2/COAG006/PostTreatment/sinos_17.mat",
                                "/home/alex/Documents/jrmomo/moving_window_pet_pca/output/Scan2/COAG007/PostTreatment/sinos_18.mat",
                                "/home/alex/Documents/jrmomo/moving_window_pet_pca/output/Scan2/COAG008/PostTreatment/sinos_19.mat"])
                        y_path_orig_list.extend(
                            [
                                "/home/alex/Documents/jrmomo/moving_window_pet_pca/output/Scan2/COAG001/Baseline/static_zscore_output_1.mat",
                                "/home/alex/Documents/jrmomo/moving_window_pet_pca/output/Scan2/COAG002/Baseline/static_zscore_output_2.mat",
                                "/home/alex/Documents/jrmomo/moving_window_pet_pca/output/Scan2/COAG003/Baseline/static_zscore_output_3.mat",
                                "/home/alex/Documents/jrmomo/moving_window_pet_pca/output/Scan2/COAG004/Baseline/static_zscore_output_4.mat",
                                "/home/alex/Documents/jrmomo/moving_window_pet_pca/output/Scan2/COAG006/Baseline/static_zscore_output_6.mat",
                                "/home/alex/Documents/jrmomo/moving_window_pet_pca/output/Scan2/COAG007/Baseline/static_zscore_output_7.mat",
                                "/home/alex/Documents/jrmomo/moving_window_pet_pca/output/Scan2/COAG008/Baseline/static_zscore_output_8.mat",
                                "/home/alex/Documents/jrmomo/moving_window_pet_pca/output/Scan2/COAG009/Baseline/static_zscore_output_9.mat",
                                "/home/alex/Documents/jrmomo/moving_window_pet_pca/output/Scan2/COAG001/PostTreatment/static_zscore_output_12.mat",
                                "/home/alex/Documents/jrmomo/moving_window_pet_pca/output/Scan2/COAG002/PostTreatment/static_zscore_output_13.mat",
                                "/home/alex/Documents/jrmomo/moving_window_pet_pca/output/Scan2/COAG003/PostTreatment/static_zscore_output_14.mat",
                                "/home/alex/Documents/jrmomo/moving_window_pet_pca/output/Scan2/COAG004/PostTreatment/static_zscore_output_15.mat",
                                "/home/alex/Documents/jrmomo/moving_window_pet_pca/output/Scan2/COAG005/PostTreatment/static_zscore_output_16.mat",
                                "/home/alex/Documents/jrmomo/moving_window_pet_pca/output/Scan2/COAG006/PostTreatment/static_zscore_output_17.mat",
                                "/home/alex/Documents/jrmomo/moving_window_pet_pca/output/Scan2/COAG007/PostTreatment/static_zscore_output_18.mat",
                                "/home/alex/Documents/jrmomo/moving_window_pet_pca/output/Scan2/COAG008/PostTreatment/static_zscore_output_19.mat"])

                        x_path_list.extend(["normalised_sinos_preprocessed_static_1.npy",
                                            "normalised_sinos_preprocessed_static_2.npy",
                                            "normalised_sinos_preprocessed_static_3.npy",
                                            "normalised_sinos_preprocessed_static_4.npy",
                                            "normalised_sinos_preprocessed_static_6.npy",
                                            "normalised_sinos_preprocessed_static_7.npy",
                                            "normalised_sinos_preprocessed_static_8.npy",
                                            "normalised_sinos_preprocessed_static_9.npy",
                                            "normalised_sinos_preprocessed_static_12.npy",
                                            "normalised_sinos_preprocessed_static_13.npy",
                                            "normalised_sinos_preprocessed_static_14.npy",
                                            "normalised_sinos_preprocessed_static_15.npy",
                                            "normalised_sinos_preprocessed_static_16.npy",
                                            "normalised_sinos_preprocessed_static_17.npy",
                                            "normalised_sinos_preprocessed_static_18.npy",
                                            "normalised_sinos_preprocessed_static_19.npy"])
                        y_path_list.extend(["static_zscore_output_preprocessed_static_1.npy",
                                            "static_zscore_output_preprocessed_static_2.npy",
                                            "static_zscore_output_preprocessed_static_3.npy",
                                            "static_zscore_output_preprocessed_static_4.npy",
                                            "static_zscore_output_preprocessed_static_6.npy",
                                            "static_zscore_output_preprocessed_static_7.npy",
                                            "static_zscore_output_preprocessed_static_8.npy",
                                            "static_zscore_output_preprocessed_static_9.npy",
                                            "static_zscore_output_preprocessed_static_12.npy",
                                            "static_zscore_output_preprocessed_static_13.npy",
                                            "static_zscore_output_preprocessed_static_14.npy",
                                            "static_zscore_output_preprocessed_static_15.npy",
                                            "static_zscore_output_preprocessed_static_16.npy",
                                            "static_zscore_output_preprocessed_static_17.npy",
                                            "static_zscore_output_preprocessed_static_18.npy",
                                            "static_zscore_output_preprocessed_static_19.npy"])

                        output_file_name.extend(["estimated_signal_static_1",
                                                 "estimated_signal_static_2",
                                                 "estimated_signal_static_3",
                                                 "estimated_signal_static_4",
                                                 "estimated_signal_static_6",
                                                 "estimated_signal_static_7",
                                                 "estimated_signal_static_8",
                                                 "estimated_signal_static_9",
                                                 "estimated_signal_static_12",
                                                 "estimated_signal_static_13",
                                                 "estimated_signal_static_14",
                                                 "estimated_signal_static_15",
                                                 "estimated_signal_static_16",
                                                 "estimated_signal_static_17",
                                                 "estimated_signal_static_18",
                                                 "estimated_signal_static_19"])

                        if original_bool:
                            x_path_orig_list.extend(
                                [
                                    "/home/alex/Documents/jrmomo/moving_window_pet_pca/output/Scan2/COAG001/Baseline/sinos_1.mat",
                                    "/home/alex/Documents/jrmomo/moving_window_pet_pca/output/Scan2/COAG002/Baseline/sinos_2.mat",
                                    "/home/alex/Documents/jrmomo/moving_window_pet_pca/output/Scan2/COAG003/Baseline/sinos_3.mat",
                                    "/home/alex/Documents/jrmomo/moving_window_pet_pca/output/Scan2/COAG004/Baseline/sinos_4.mat",
                                    "/home/alex/Documents/jrmomo/moving_window_pet_pca/output/Scan2/COAG006/Baseline/sinos_6.mat",
                                    "/home/alex/Documents/jrmomo/moving_window_pet_pca/output/Scan2/COAG007/Baseline/sinos_7.mat",
                                    "/home/alex/Documents/jrmomo/moving_window_pet_pca/output/Scan2/COAG008/Baseline/sinos_8.mat",
                                    "/home/alex/Documents/jrmomo/moving_window_pet_pca/output/Scan2/COAG009/Baseline/sinos_9.mat",
                                    "/home/alex/Documents/jrmomo/moving_window_pet_pca/output/Scan2/COAG001/PostTreatment/sinos_12.mat",
                                    "/home/alex/Documents/jrmomo/moving_window_pet_pca/output/Scan2/COAG002/PostTreatment/sinos_13.mat",
                                    "/home/alex/Documents/jrmomo/moving_window_pet_pca/output/Scan2/COAG003/PostTreatment/sinos_14.mat",
                                    "/home/alex/Documents/jrmomo/moving_window_pet_pca/output/Scan2/COAG004/PostTreatment/sinos_15.mat",
                                    "/home/alex/Documents/jrmomo/moving_window_pet_pca/output/Scan2/COAG005/PostTreatment/sinos_16.mat",
                                    "/home/alex/Documents/jrmomo/moving_window_pet_pca/output/Scan2/COAG006/PostTreatment/sinos_17.mat",
                                    "/home/alex/Documents/jrmomo/moving_window_pet_pca/output/Scan2/COAG007/PostTreatment/sinos_18.mat",
                                    "/home/alex/Documents/jrmomo/moving_window_pet_pca/output/Scan2/COAG008/PostTreatment/sinos_19.mat"])
                            y_path_orig_list.extend(
                                [
                                    "/home/alex/Documents/jrmomo/moving_window_pet_pca/output/Scan2/COAG001/Baseline/static_output_1.mat",
                                    "/home/alex/Documents/jrmomo/moving_window_pet_pca/output/Scan2/COAG002/Baseline/static_output_2.mat",
                                    "/home/alex/Documents/jrmomo/moving_window_pet_pca/output/Scan2/COAG003/Baseline/static_output_3.mat",
                                    "/home/alex/Documents/jrmomo/moving_window_pet_pca/output/Scan2/COAG004/Baseline/static_output_4.mat",
                                    "/home/alex/Documents/jrmomo/moving_window_pet_pca/output/Scan2/COAG006/Baseline/static_output_6.mat",
                                    "/home/alex/Documents/jrmomo/moving_window_pet_pca/output/Scan2/COAG007/Baseline/static_output_7.mat",
                                    "/home/alex/Documents/jrmomo/moving_window_pet_pca/output/Scan2/COAG008/Baseline/static_output_8.mat",
                                    "/home/alex/Documents/jrmomo/moving_window_pet_pca/output/Scan2/COAG009/Baseline/static_output_9.mat",
                                    "/home/alex/Documents/jrmomo/moving_window_pet_pca/output/Scan2/COAG001/PostTreatment/static_output_12.mat",
                                    "/home/alex/Documents/jrmomo/moving_window_pet_pca/output/Scan2/COAG002/PostTreatment/static_output_13.mat",
                                    "/home/alex/Documents/jrmomo/moving_window_pet_pca/output/Scan2/COAG003/PostTreatment/static_output_14.mat",
                                    "/home/alex/Documents/jrmomo/moving_window_pet_pca/output/Scan2/COAG004/PostTreatment/static_output_15.mat",
                                    "/home/alex/Documents/jrmomo/moving_window_pet_pca/output/Scan2/COAG005/PostTreatment/static_output_16.mat",
                                    "/home/alex/Documents/jrmomo/moving_window_pet_pca/output/Scan2/COAG006/PostTreatment/static_output_17.mat",
                                    "/home/alex/Documents/jrmomo/moving_window_pet_pca/output/Scan2/COAG007/PostTreatment/static_output_18.mat",
                                    "/home/alex/Documents/jrmomo/moving_window_pet_pca/output/Scan2/COAG008/PostTreatment/static_output_19.mat"])

                            x_path_list.extend(["normalised_sinos_preprocessed_static_1.npy",
                                                "normalised_sinos_preprocessed_static_2.npy",
                                                "normalised_sinos_preprocessed_static_3.npy",
                                                "normalised_sinos_preprocessed_static_4.npy",
                                                "normalised_sinos_preprocessed_static_6.npy",
                                                "normalised_sinos_preprocessed_static_7.npy",
                                                "normalised_sinos_preprocessed_static_8.npy",
                                                "normalised_sinos_preprocessed_static_9.npy",
                                                "normalised_sinos_preprocessed_static_12.npy",
                                                "normalised_sinos_preprocessed_static_13.npy",
                                                "normalised_sinos_preprocessed_static_14.npy",
                                                "normalised_sinos_preprocessed_static_15.npy",
                                                "normalised_sinos_preprocessed_static_16.npy",
                                                "normalised_sinos_preprocessed_static_17.npy",
                                                "normalised_sinos_preprocessed_static_18.npy",
                                                "normalised_sinos_preprocessed_static_19.npy"])
                            y_path_list.extend(["static_output_preprocessed_static_1.npy",
                                                "static_output_preprocessed_static_2.npy",
                                                "static_output_preprocessed_static_3.npy",
                                                "static_output_preprocessed_static_4.npy",
                                                "static_output_preprocessed_static_6.npy",
                                                "static_output_preprocessed_static_7.npy",
                                                "static_output_preprocessed_static_8.npy",
                                                "static_output_preprocessed_static_9.npy",
                                                "static_output_preprocessed_static_12.npy",
                                                "static_output_preprocessed_static_13.npy",
                                                "static_output_preprocessed_static_14.npy",
                                                "static_output_preprocessed_static_15.npy",
                                                "static_output_preprocessed_static_16.npy",
                                                "static_output_preprocessed_static_17.npy",
                                                "static_output_preprocessed_static_18.npy",
                                                "static_output_preprocessed_static_19.npy"])

                            output_file_name.extend(["estimated_signal_static_1",
                                                     "estimated_signal_static_2",
                                                     "estimated_signal_static_3",
                                                     "estimated_signal_static_4",
                                                     "estimated_signal_static_6",
                                                     "estimated_signal_static_7",
                                                     "estimated_signal_static_8",
                                                     "estimated_signal_static_9",
                                                     "estimated_signal_static_12",
                                                     "estimated_signal_static_13",
                                                     "estimated_signal_static_14",
                                                     "estimated_signal_static_15",
                                                     "estimated_signal_static_16",
                                                     "estimated_signal_static_17",
                                                     "estimated_signal_static_18",
                                                     "estimated_signal_static_19"])

                    if not conservative_bool:
                        x_path_orig_list.extend(
                            [
                                "/home/alex/Documents/jrmomo/moving_window_pet_pca/output/Scan2/COAG001/Baseline/sinos_1.mat",
                                "/home/alex/Documents/jrmomo/moving_window_pet_pca/output/Scan2/COAG008/Baseline/sinos_8.mat",
                                "/home/alex/Documents/jrmomo/moving_window_pet_pca/output/Scan2/COAG001/PostTreatment/sinos_12.mat"])
                        y_path_orig_list.extend(
                            [
                                "/home/alex/Documents/jrmomo/moving_window_pet_pca/output/Scan2/COAG001/Baseline/output_signal_1.mat",
                                "/home/alex/Documents/jrmomo/moving_window_pet_pca/output/Scan2/COAG008/Baseline/output_signal_8.mat",
                                "/home/alex/Documents/jrmomo/moving_window_pet_pca/output/Scan2/COAG001/PostTreatment/output_signal_12.mat"])

                        x_path_list.extend(["normalised_sinos_preprocessed_static_1.npy",
                                            "normalised_sinos_preprocessed_static_8.npy",
                                            "normalised_sinos_preprocessed_static_12.npy"])
                        y_path_list.extend(["output_signal_preprocessed_static_1.npy",
                                            "output_signal_preprocessed_static_8.npy",
                                            "output_signal_preprocessed_static_12.npy"])

                        output_file_name.extend(["estimated_signal_static_1",
                                                 "estimated_signal_static_8",
                                                 "estimated_signal_static_12"])

                        if pca_bool:
                            x_path_orig_list.extend(
                                [
                                    "/home/alex/Documents/jrmomo/moving_window_pet_pca/output/Scan2/COAG001/Baseline/sinos_1.mat",
                                    "/home/alex/Documents/jrmomo/moving_window_pet_pca/output/Scan2/COAG002/Baseline/sinos_2.mat",
                                    "/home/alex/Documents/jrmomo/moving_window_pet_pca/output/Scan2/COAG003/Baseline/sinos_3.mat",
                                    "/home/alex/Documents/jrmomo/moving_window_pet_pca/output/Scan2/COAG004/Baseline/sinos_4.mat",
                                    "/home/alex/Documents/jrmomo/moving_window_pet_pca/output/Scan2/COAG006/Baseline/sinos_6.mat",
                                    "/home/alex/Documents/jrmomo/moving_window_pet_pca/output/Scan2/COAG007/Baseline/sinos_7.mat",
                                    "/home/alex/Documents/jrmomo/moving_window_pet_pca/output/Scan2/COAG008/Baseline/sinos_8.mat",
                                    "/home/alex/Documents/jrmomo/moving_window_pet_pca/output/Scan2/COAG009/Baseline/sinos_9.mat",
                                    "/home/alex/Documents/jrmomo/moving_window_pet_pca/output/Scan2/COAG001/PostTreatment/sinos_12.mat",
                                    "/home/alex/Documents/jrmomo/moving_window_pet_pca/output/Scan2/COAG002/PostTreatment/sinos_13.mat",
                                    "/home/alex/Documents/jrmomo/moving_window_pet_pca/output/Scan2/COAG003/PostTreatment/sinos_14.mat",
                                    "/home/alex/Documents/jrmomo/moving_window_pet_pca/output/Scan2/COAG004/PostTreatment/sinos_15.mat",
                                    "/home/alex/Documents/jrmomo/moving_window_pet_pca/output/Scan2/COAG005/PostTreatment/sinos_16.mat",
                                    "/home/alex/Documents/jrmomo/moving_window_pet_pca/output/Scan2/COAG006/PostTreatment/sinos_17.mat",
                                    "/home/alex/Documents/jrmomo/moving_window_pet_pca/output/Scan2/COAG007/PostTreatment/sinos_18.mat",
                                    "/home/alex/Documents/jrmomo/moving_window_pet_pca/output/Scan2/COAG008/PostTreatment/sinos_19.mat"])
                            y_path_orig_list.extend(
                                [
                                    "/home/alex/Documents/jrmomo/moving_window_pet_pca/output/Scan2/COAG001/Baseline/mean_output_one_pc_w_1.mat",
                                    "/home/alex/Documents/jrmomo/moving_window_pet_pca/output/Scan2/COAG002/Baseline/mean_output_one_pc_w_2.mat",
                                    "/home/alex/Documents/jrmomo/moving_window_pet_pca/output/Scan2/COAG003/Baseline/mean_output_one_pc_w_3.mat",
                                    "/home/alex/Documents/jrmomo/moving_window_pet_pca/output/Scan2/COAG004/Baseline/mean_output_one_pc_w_4.mat",
                                    "/home/alex/Documents/jrmomo/moving_window_pet_pca/output/Scan2/COAG006/Baseline/mean_output_one_pc_w_6.mat",
                                    "/home/alex/Documents/jrmomo/moving_window_pet_pca/output/Scan2/COAG007/Baseline/mean_output_one_pc_w_7.mat",
                                    "/home/alex/Documents/jrmomo/moving_window_pet_pca/output/Scan2/COAG008/Baseline/mean_output_one_pc_w_8.mat",
                                    "/home/alex/Documents/jrmomo/moving_window_pet_pca/output/Scan2/COAG009/Baseline/mean_output_one_pc_w_9.mat",
                                    "/home/alex/Documents/jrmomo/moving_window_pet_pca/output/Scan2/COAG001/PostTreatment/mean_output_one_pc_w_12.mat",
                                    "/home/alex/Documents/jrmomo/moving_window_pet_pca/output/Scan2/COAG002/PostTreatment/mean_output_one_pc_w_13.mat",
                                    "/home/alex/Documents/jrmomo/moving_window_pet_pca/output/Scan2/COAG003/PostTreatment/mean_output_one_pc_w_14.mat",
                                    "/home/alex/Documents/jrmomo/moving_window_pet_pca/output/Scan2/COAG004/PostTreatment/mean_output_one_pc_w_15.mat",
                                    "/home/alex/Documents/jrmomo/moving_window_pet_pca/output/Scan2/COAG005/PostTreatment/mean_output_one_pc_w_16.mat",
                                    "/home/alex/Documents/jrmomo/moving_window_pet_pca/output/Scan2/COAG006/PostTreatment/mean_output_one_pc_w_17.mat",
                                    "/home/alex/Documents/jrmomo/moving_window_pet_pca/output/Scan2/COAG007/PostTreatment/mean_output_one_pc_w_18.mat",
                                    "/home/alex/Documents/jrmomo/moving_window_pet_pca/output/Scan2/COAG008/PostTreatment/mean_output_one_pc_w_19.mat"])

                            x_path_list.extend(["normalised_sinos_preprocessed_static_1.npy",
                                                "normalised_sinos_preprocessed_static_2.npy",
                                                "normalised_sinos_preprocessed_static_3.npy",
                                                "normalised_sinos_preprocessed_static_4.npy",
                                                "normalised_sinos_preprocessed_static_6.npy",
                                                "normalised_sinos_preprocessed_static_7.npy",
                                                "normalised_sinos_preprocessed_static_8.npy",
                                                "normalised_sinos_preprocessed_static_9.npy",
                                                "normalised_sinos_preprocessed_static_12.npy",
                                                "normalised_sinos_preprocessed_static_13.npy",
                                                "normalised_sinos_preprocessed_static_14.npy",
                                                "normalised_sinos_preprocessed_static_15.npy",
                                                "normalised_sinos_preprocessed_static_16.npy",
                                                "normalised_sinos_preprocessed_static_17.npy",
                                                "normalised_sinos_preprocessed_static_18.npy",
                                                "normalised_sinos_preprocessed_static_19.npy"])
                            y_path_list.extend(["mean_output_one_pc_w_preprocessed_static_1.npy",
                                                "mean_output_one_pc_w_preprocessed_static_2.npy",
                                                "mean_output_one_pc_w_preprocessed_static_3.npy",
                                                "mean_output_one_pc_w_preprocessed_static_4.npy",
                                                "mean_output_one_pc_w_preprocessed_static_6.npy",
                                                "mean_output_one_pc_w_preprocessed_static_7.npy",
                                                "mean_output_one_pc_w_preprocessed_static_8.npy",
                                                "mean_output_one_pc_w_preprocessed_static_9.npy",
                                                "mean_output_one_pc_w_preprocessed_static_12.npy",
                                                "mean_output_one_pc_w_preprocessed_static_13.npy",
                                                "mean_output_one_pc_w_preprocessed_static_14.npy",
                                                "mean_output_one_pc_w_preprocessed_static_15.npy",
                                                "mean_output_one_pc_w_preprocessed_static_16.npy",
                                                "mean_output_one_pc_w_preprocessed_static_17.npy",
                                                "mean_output_one_pc_w_preprocessed_static_18.npy",
                                                "mean_output_one_pc_w_preprocessed_static_19.npy"])

                            output_file_name.extend(["estimated_signal_static_1",
                                                     "estimated_signal_static_2",
                                                     "estimated_signal_static_3",
                                                     "estimated_signal_static_4",
                                                     "estimated_signal_static_6",
                                                     "estimated_signal_static_7",
                                                     "estimated_signal_static_8",
                                                     "estimated_signal_static_9",
                                                     "estimated_signal_static_12",
                                                     "estimated_signal_static_13",
                                                     "estimated_signal_static_14",
                                                     "estimated_signal_static_15",
                                                     "estimated_signal_static_16",
                                                     "estimated_signal_static_17",
                                                     "estimated_signal_static_18",
                                                     "estimated_signal_static_19"])

                        if original_bool:
                            x_path_orig_list.extend(
                                [
                                    "/home/alex/Documents/jrmomo/moving_window_pet_pca/output/Scan2/COAG001/Baseline/sinos_1.mat",
                                    "/home/alex/Documents/jrmomo/moving_window_pet_pca/output/Scan2/COAG008/Baseline/sinos_8.mat",
                                    "/home/alex/Documents/jrmomo/moving_window_pet_pca/output/Scan2/COAG001/PostTreatment/sinos_12.mat"])
                            y_path_orig_list.extend(
                                [
                                    "/home/alex/Documents/jrmomo/moving_window_pet_pca/output/Scan2/COAG001/Baseline/output_signal_original_1.mat",
                                    "/home/alex/Documents/jrmomo/moving_window_pet_pca/output/Scan2/COAG008/Baseline/output_signal_original_8.mat",
                                    "/home/alex/Documents/jrmomo/moving_window_pet_pca/output/Scan2/COAG001/PostTreatment/output_signal_original_12.mat"])

                            x_path_list.extend(["normalised_sinos_preprocessed_static_1.npy",
                                                "normalised_sinos_preprocessed_static_8.npy",
                                                "normalised_sinos_preprocessed_static_12.npy"])
                            y_path_list.extend(["output_signal_original_preprocessed_static_1.npy",
                                                "output_signal_original_preprocessed_static_8.npy",
                                                "output_signal_original_preprocessed_static_12.npy"])

                            output_file_name.extend(["estimated_signal_static_1",
                                                     "estimated_signal_static_8",
                                                     "estimated_signal_static_12"])

                            if pca_bool:
                                x_path_orig_list.extend(
                                    [
                                        "/home/alex/Documents/jrmomo/moving_window_pet_pca/output/Scan2/COAG001/Baseline/sinos_1.mat",
                                        "/home/alex/Documents/jrmomo/moving_window_pet_pca/output/Scan2/COAG002/Baseline/sinos_2.mat",
                                        "/home/alex/Documents/jrmomo/moving_window_pet_pca/output/Scan2/COAG003/Baseline/sinos_3.mat",
                                        "/home/alex/Documents/jrmomo/moving_window_pet_pca/output/Scan2/COAG004/Baseline/sinos_4.mat",
                                        "/home/alex/Documents/jrmomo/moving_window_pet_pca/output/Scan2/COAG006/Baseline/sinos_6.mat",
                                        "/home/alex/Documents/jrmomo/moving_window_pet_pca/output/Scan2/COAG007/Baseline/sinos_7.mat",
                                        "/home/alex/Documents/jrmomo/moving_window_pet_pca/output/Scan2/COAG008/Baseline/sinos_8.mat",
                                        "/home/alex/Documents/jrmomo/moving_window_pet_pca/output/Scan2/COAG009/Baseline/sinos_9.mat",
                                        "/home/alex/Documents/jrmomo/moving_window_pet_pca/output/Scan2/COAG001/PostTreatment/sinos_12.mat",
                                        "/home/alex/Documents/jrmomo/moving_window_pet_pca/output/Scan2/COAG002/PostTreatment/sinos_13.mat",
                                        "/home/alex/Documents/jrmomo/moving_window_pet_pca/output/Scan2/COAG003/PostTreatment/sinos_14.mat",
                                        "/home/alex/Documents/jrmomo/moving_window_pet_pca/output/Scan2/COAG004/PostTreatment/sinos_15.mat",
                                        "/home/alex/Documents/jrmomo/moving_window_pet_pca/output/Scan2/COAG005/PostTreatment/sinos_16.mat",
                                        "/home/alex/Documents/jrmomo/moving_window_pet_pca/output/Scan2/COAG006/PostTreatment/sinos_17.mat",
                                        "/home/alex/Documents/jrmomo/moving_window_pet_pca/output/Scan2/COAG007/PostTreatment/sinos_18.mat",
                                        "/home/alex/Documents/jrmomo/moving_window_pet_pca/output/Scan2/COAG008/PostTreatment/sinos_19.mat"])
                                y_path_orig_list.extend(
                                    [
                                        "/home/alex/Documents/jrmomo/moving_window_pet_pca/output/Scan2/COAG001/Baseline/mean_output_one_pc_w_1.mat",
                                        "/home/alex/Documents/jrmomo/moving_window_pet_pca/output/Scan2/COAG002/Baseline/mean_output_one_pc_w_2.mat",
                                        "/home/alex/Documents/jrmomo/moving_window_pet_pca/output/Scan2/COAG003/Baseline/mean_output_one_pc_w_3.mat",
                                        "/home/alex/Documents/jrmomo/moving_window_pet_pca/output/Scan2/COAG004/Baseline/mean_output_one_pc_w_4.mat",
                                        "/home/alex/Documents/jrmomo/moving_window_pet_pca/output/Scan2/COAG006/Baseline/mean_output_one_pc_w_6.mat",
                                        "/home/alex/Documents/jrmomo/moving_window_pet_pca/output/Scan2/COAG007/Baseline/mean_output_one_pc_w_7.mat",
                                        "/home/alex/Documents/jrmomo/moving_window_pet_pca/output/Scan2/COAG008/Baseline/mean_output_one_pc_w_8.mat",
                                        "/home/alex/Documents/jrmomo/moving_window_pet_pca/output/Scan2/COAG009/Baseline/mean_output_one_pc_w_9.mat",
                                        "/home/alex/Documents/jrmomo/moving_window_pet_pca/output/Scan2/COAG001/PostTreatment/mean_output_one_pc_w_12.mat",
                                        "/home/alex/Documents/jrmomo/moving_window_pet_pca/output/Scan2/COAG002/PostTreatment/mean_output_one_pc_w_13.mat",
                                        "/home/alex/Documents/jrmomo/moving_window_pet_pca/output/Scan2/COAG003/PostTreatment/mean_output_one_pc_w_14.mat",
                                        "/home/alex/Documents/jrmomo/moving_window_pet_pca/output/Scan2/COAG004/PostTreatment/mean_output_one_pc_w_15.mat",
                                        "/home/alex/Documents/jrmomo/moving_window_pet_pca/output/Scan2/COAG005/PostTreatment/mean_output_one_pc_w_16.mat",
                                        "/home/alex/Documents/jrmomo/moving_window_pet_pca/output/Scan2/COAG006/PostTreatment/mean_output_one_pc_w_17.mat",
                                        "/home/alex/Documents/jrmomo/moving_window_pet_pca/output/Scan2/COAG007/PostTreatment/mean_output_one_pc_w_18.mat",
                                        "/home/alex/Documents/jrmomo/moving_window_pet_pca/output/Scan2/COAG008/PostTreatment/mean_output_one_pc_w_19.mat"])

                                x_path_list.extend(["normalised_sinos_preprocessed_static_1.npy",
                                                    "normalised_sinos_preprocessed_static_2.npy",
                                                    "normalised_sinos_preprocessed_static_3.npy",
                                                    "normalised_sinos_preprocessed_static_4.npy",
                                                    "normalised_sinos_preprocessed_static_6.npy",
                                                    "normalised_sinos_preprocessed_static_7.npy",
                                                    "normalised_sinos_preprocessed_static_8.npy",
                                                    "normalised_sinos_preprocessed_static_9.npy",
                                                    "normalised_sinos_preprocessed_static_12.npy",
                                                    "normalised_sinos_preprocessed_static_13.npy",
                                                    "normalised_sinos_preprocessed_static_14.npy",
                                                    "normalised_sinos_preprocessed_static_15.npy",
                                                    "normalised_sinos_preprocessed_static_16.npy",
                                                    "normalised_sinos_preprocessed_static_17.npy",
                                                    "normalised_sinos_preprocessed_static_18.npy",
                                                    "normalised_sinos_preprocessed_static_19.npy"])
                                y_path_list.extend(["mean_output_one_pc_w_preprocessed_static_1.npy",
                                                    "mean_output_one_pc_w_preprocessed_static_2.npy",
                                                    "mean_output_one_pc_w_preprocessed_static_3.npy",
                                                    "mean_output_one_pc_w_preprocessed_static_4.npy",
                                                    "mean_output_one_pc_w_preprocessed_static_6.npy",
                                                    "mean_output_one_pc_w_preprocessed_static_7.npy",
                                                    "mean_output_one_pc_w_preprocessed_static_8.npy",
                                                    "mean_output_one_pc_w_preprocessed_static_9.npy",
                                                    "mean_output_one_pc_w_preprocessed_static_12.npy",
                                                    "mean_output_one_pc_w_preprocessed_static_13.npy",
                                                    "mean_output_one_pc_w_preprocessed_static_14.npy",
                                                    "mean_output_one_pc_w_preprocessed_static_15.npy",
                                                    "mean_output_one_pc_w_preprocessed_static_16.npy",
                                                    "mean_output_one_pc_w_preprocessed_static_17.npy",
                                                    "mean_output_one_pc_w_preprocessed_static_18.npy",
                                                    "mean_output_one_pc_w_preprocessed_static_19.npy"])

                                output_file_name.extend(["estimated_signal_static_1",
                                                         "estimated_signal_static_2",
                                                         "estimated_signal_static_3",
                                                         "estimated_signal_static_4",
                                                         "estimated_signal_static_6",
                                                         "estimated_signal_static_7",
                                                         "estimated_signal_static_8",
                                                         "estimated_signal_static_9",
                                                         "estimated_signal_static_12",
                                                         "estimated_signal_static_13",
                                                         "estimated_signal_static_14",
                                                         "estimated_signal_static_15",
                                                         "estimated_signal_static_16",
                                                         "estimated_signal_static_17",
                                                         "estimated_signal_static_18",
                                                         "estimated_signal_static_19"])
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
                                             "estimated_signal_dynamic_18",
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
