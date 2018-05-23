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
import os

import numpy as np
from calculations import *

def deconvolve(file_path, superconducting_gap,
                temperature, transport_type,
                nonequilibrium_voltage, probe_location,
                learning_rate = 0.1):
    """
    deconvolve finds the best possible nonequilibrium
    distribution function by least squares.

    :param filepath: This is a file of two columns containing the bias
    voltage and their associated differential conductances. The unit
    of bias voltage is in Volts, and the unit of differential
    conductance is a multiple of e^2/h. The first column must be
    the voltage bias and the second column is the differential
    conductance.

    :param superconducting_gap: This is the energy gap in units of
    electron volts (eV). Used in calculating the differential
    conductance.

    :param temperature: The temperature of the experiment, in units of
    Kelvin (K). Used in calculating the Fermi-Dirac distribution
    function of the superconductor and calculating the temperature
    of electrons in the electron-electron scattering case for the
    initial non-equilibrium distribution

    :param transport_type: A string that should be in this list:
        'b'  (ballistic)
        'ph' (phonon scattering)
        'ee' (electron-electron scattering)
    This directly determines the initial non-equilibrium distribution
    function. The meanings of these particular terms is outside the
    scope of the code and should be looked for in an textbook about
    condensed matter systems.
    """
    start_time = time.time()

    (directory, file_name) = os.path.split(file_path)
    (file_base_name, file_extension) = (os.path.splitext(file_name[0]),
                                        os.path.splitext(file_name[1]))

    data = np.loadtxt(file_path,delimiter='\t',skiprows=2)

    # Take out later...temporarily added in for test.txt
    data[:,0] = data[:,0] / 1000.0

    target_differential_conductance = data[:,1]

    biases = data[:,0]
    energies = _get_energies(data[:,0])

    model_density_of_states = np.ones((energies.shape[0],)) * 1e-3
    model_nonequilibrium_distribution = get_energy_distribution(
                                            energies,
                                            temperature=temperature,
                                            transport_type=transport_type,
                                            nonequilibrium_voltage = nonequilibrium_voltage,
                                            probe_location = probe_location)

    error_still_decreasing = True
    previous_error = 10000.0

    while error_still_decreasing:

        (model_differential_conductance, square_errors) = calculate_square_errors(
                                            biases, energies,
                                            model_density_of_states,
                                            model_nonequilibrium_distribution,
                                            temperature,
                                            superconducting_gap,
                                            target_differential_conductance)
        
        error_sum = np.sum(square_errors)

        if (error_sum - previous_error_sum) < tolerance:
            error_still_decreasing = False
        else:
            previous_error_sum = error_sum

            dos_error_derivatives = calculate_error_derivatives(
                                perturbed_function_name='dos',
                                model_density_of_states,
                                model_nonequilibrium_distribution,
                                energies, square_errors,
                                biases, target_differential_conductance,
                                temperature, superconducting_gap):
            dist_error_derivatives = calculate_error_derivatives(
                                perturbed_function_name='f',
                                model_nonequilibrium_distribution,
                                model_density_of_states,
                                energies, square_errors,
                                biases, target_differential_conductance,
                                temperature, superconducting_gap):

            # N.b. that the probability distribution must only be
            # monotonically decreasing
            model_density_of_states = update(model_density_of_states,
                                            learning_rate,
                                            dos_error_derivatives)
            model_nonequilibrium_distribution = update(model_nonequilibrium_distribution,
                                                    learning_rate,
                                                    dist_error_derivatives)
            model_nonequilibrium_distribution = np.minimum.accumulate(
                                                        model_nonequilibrium_distribution)

    energies = energies.reshape(energies.shape[0],1)
    model_density_of_states = model_density_of_states.reshape(
                                model_density_of_states.shape[0],1)
    model_nonequilibrium_distribution = model_nonequilibrium_distribution.reshape(
                                model_nonequilibrium_distribution.shape[0],1)

    output_density_of_states = np.concatenate(
        (energies,model_density_of_states), axis=1)
    output_nonequilibrium_distribution = np.concatenate(
        (energies,model_nonequilibrium_distribution), axis=1)

    save_directory = directory + '/' + file_base_name

    if not os.path.exists(save_directory):
        os.mkdir(save_directory)

    np.savetxt(save_directory + '/dos.txt',
                output_density_of_states,
                delimiter = ',')

    np.savetxt(save_directory + '/nonequilibrium_f.txt',
                output_nonequilibrium_distribution,
                delimiter = ',')

    print('Total time: ' +  str(time.time() - start_time)
            + ' seconds.')

    return None

def _get_energies(voltages):
    """
    Given the raw data, establish the energies used in the models.

    Input:

    voltages: This should be the numpy array of the voltages
    that come from the raw data.

    Returns:

    A numpy array of the energies
    """
    voltage_step = abs(voltages[1] - voltages[0])
    energy_max = abs(np.amax(voltages))

    # This was in Nick's code; it forces misalignment with the
    # voltages. Not entirely obvious (yet) why it's here.
    energy_step = (2 * energy_max
                    /round((2 * energy_max / voltage_step) - 1))

    energies = np.arange(-energy_max, energy_max, energy_step)

    return energies