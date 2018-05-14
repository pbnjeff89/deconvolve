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

from initialize import (get_initial_energy_distribution,
                        get_initial_density_of_states)

def deconvolve(file_path, superconducting_gap, temperature, transport_type):
    """
    deconvolve finds the best possible nonequilibrium
    distribution function given the differential conductance
    data using a MSE metric.

    Input:

    filepath: This is a file of two columns containing the bias
    voltage and their associated differential conductances. The unit
    of bias voltage is in Volts, and the unit of differential
    conductance is a multiple of e^2/h.

    superconducting_gap: This is the energy gap in units of
    electron volts (eV). Used in calculating the differential
    conductance.

    temperature: The temperature of the experiment, in units of
    Kelvin (K). Used in calculating the Fermi-Dirac distribution
    function of the superconductor and calculating the temperature
    of electrons in the electron-electron scattering case for the
    initial non-equilibrium distribution

    transport_type: A string that should be in this list:
        'ballistic'
        'ph scattering' (phonon scattering)
        'ee scattering' (electron-electron scattering)
    This directly determines the initial non-equilibrium distribution
    function.
    """
    start_time = time.time()

    data = np.loadtxt(file_path,delimiter='\t',skiprows=2)

    # Take out later...temporarily added in for test.txt
    data[:,0] = data[:,0] / 1000.0

    energies = _get_energies(data[:,0])

    (model_density_of_states, model_nonequilibrium_distribution,
        model_differential_conductance) = _initialize_models(data[:,0],
                                                energies)

    print(model_density_of_states)
    print(model_differential_conductance)
    print(model_nonequilibrium_distribution)

    print('Total time: ' +  str(time.time() - start_time)
            + ' seconds.')

    return None

def _get_energies(voltages):
    """
    Given the raw data, establish the energies used in the models.

    Input:

    voltages: This should be the numpy array of the voltages
    that come from the raw data.
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

    return energies

def _initialize_models(voltages, energies):
    """
    Creates numpy arrays for the initial models of the differential
    conductance, the density of states, and the non-equilibrium
    distribution functions.
    """
    num_voltages = voltages.shape[0]
    num_energies = len(energies)

    model_differential_conductance = np.zeros((num_voltages,2))
    model_density_of_states = np.zeros((num_energies,2))
    model_nonequilibrium_distribution = np.zeros((num_energies,2))

    model_differential_conductance[:,0] = voltages
    model_density_of_states[:,0] = energies
    model_nonequilibrium_distribution[:,0] = energies

    return (model_density_of_states,
            model_nonequilibrium_distribution,
            model_differential_conductance)