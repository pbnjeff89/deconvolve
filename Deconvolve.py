"""
Deconvolve takes convolved data for tunneling measurements
and returns back the best fit for the nonequilibrium distribution
function.

Usage:

In the terminal, input

python Deconvolve.py [path_to_data] [column_of_voltage_bias] ...
    ... [column_of_differential_conductance]

Output:
    deconvolved_[origin_data_file_name].txt

        This contains the best fit of the nonequilibrium distribution
        function.
"""

import time
import numpy as np

def deconvolve(file_path):

    start_time = time.time()

    data = []

    with open(file_path) as f:

        header = f.readline()

        for line in f:

            data.append(line)

        print(header)
        print(data)

    print('Total time: ' +  str(time.time() - start_time)
            + ' seconds.')

    return None