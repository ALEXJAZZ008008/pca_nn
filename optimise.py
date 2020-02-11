# Copyright University College London 2020
# Author: Alexander Whitehead, Institute of Nuclear Medicine, UCL
# For internal research only.


import sys
import math
import scipy.optimize


def objective_function(new_input_value, input_value, scale_factor):
    objective_value = (input_value - (new_input_value - (new_input_value * scale_factor))) ** 2.0

    print("Objective value: {0}".format(objective_value[0]))

    return objective_value


def optimise_value(input_value, scale_fator):
    new_input_factor = input_value

    new_input_factor = scipy.optimize.minimize(objective_function, new_input_factor, args=(input_value, scale_fator)).x

    return math.ceil(new_input_factor)


if __name__ == "__main__":
    print("Output: {0}".format(optimise_value(int(sys.argv[1]), float(sys.argv[2]))))
