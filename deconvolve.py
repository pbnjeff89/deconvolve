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
from calculations import *

def deconvolve(file_path, superconducting_gap,
                temperature, transport_type,
                nonequilibrium_voltage, probe_location):
    """
    deconvolve finds the best possible nonequilibrium
    distribution function given the differential conductance
    data using a MSE metric.

    :param filepath: This is a file of two columns containing the bias
    voltage and their associated differential conductances. The unit
    of bias voltage is in Volts, and the unit of differential
    conductance is a multiple of e^2/h. The first column must be
    the voltage bias and the second column is the differential
    conductance.

    :param superconducting_gap: This is the energy gap in units of
    electron volts (eV). Used in calculating the differential
    conductance.

    temperature: The temperature of the experiment, in units of
    Kelvin (K). Used in calculating the Fermi-Dirac distribution
    function of the superconductor and calculating the temperature
    of electrons in the electron-electron scattering case for the
    initial non-equilibrium distribution

    transport_type: A string that should be in this list:
        'b'  (ballistic)
        'ph' (phonon scattering)
        'ee' (electron-electron scattering)
    This directly determines the initial non-equilibrium distribution
    function. The meanings of these particular terms is outside the
    scope of the code and should be looked for in an textbook about
    condensed matter systems.
    """
    start_time = time.time()

    data = np.loadtxt(file_path,delimiter='\t',skiprows=2)

    # Take out later...temporarily added in for test.txt
    data[:,0] = data[:,0] / 1000.0

    target_differential_conductance = data[:,1]

    biases = data[:,0]
    energies = _get_energies(data[:,0])

    model_density_of_states = np.ones((energies.shape[0],1))
    model_nonequilibrium_distribution = get_energy_distribution(
                                            energies,
                                            temperature,
                                            transport_type=transport_type,
                                            nonequilibrium_voltage = nonequilibrium_voltage,
                                            probe_location = probe_location)

    model_differential_conductance = (
        calculate_differential_conductance(biases, energies, 
        model_density_of_states, model_nonequilibrium_distribution, 
        temperature, superconducting_gap)
        )

    #print(model_density_of_states)
    #print(model_differential_conductance)
    #print(model_nonequilibrium_distribution)

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

    energies = []
    energy = -energy_max

    while energy <= energy_max:
        energies.append(energy)
        energy += energy_step

    return np.asarray(energies)