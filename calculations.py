import numpy as np
from distributions import *

from matplotlib import pyplot as plt

def calculate_differential_conductance(bias, energies, 
        density_of_states, energy_distribution, temperature,
        superconducting_gap):
    
    # Boltzmann constant in eV/K
    kb = 8.6173303e-5

    # + 0j is necessary so that numpy knows it is processing
    # a complex number
    superconductor_dos = np.real(np.absolute(energies) / 
                            np.sqrt(energies ** 2 - superconducting_gap ** 2 + 0j))

    superconductor_dos_derivative = np.gradient(superconductor_dos)
    superconductor_distribution = get_energy_distribution(energies)

    return differential_conductance

def shift_density_of_states(energies, dos, voltage_shift):
    shifted_dos = shift(energies, dos, voltage_shift, 0.0)
    return shifted_dos

def shift_distribution(energies, distribution, voltage_shift):
    shifted_distribution = shift(energies, distribution, voltage_shift, 1.0)
    return shifted_distribution

def shift(energies, function, voltage_shift, tail):
    """
    Returns a shifted function, useful for shifting the density of
    states and energy distribution. Each function defaults to zeros
    for each energy, and the tail value assists in taking care
    of the function values in the far negative regions.

    Input:

    energies: A numpy array of the energies.

    function: A function of energy, like the density of states or
    the energy distribution of electrons

    voltage_shift: The voltage shift that the function needs to shift
    by. (N.b. the electron charge is set to 1.)

    tail: The value of the function that should be set.

    Returns:

    shifted: A numpy array of the shifted function.
    """
    shifted = np.zeros(function.shape[0])

    shift_energy = (voltage_shift
                        / np.absolute(energies[1] - energies[0]))
    # indices must take an int
    shift_index = int(np.floor(voltage_shift
                        / np.absolute(energies[1] - energies[0])))
    weight = shift_energy - shift_index

    # TODO:
    # Write this in a more Pythonic way
    # This seems to suggest
    num_energies = int(function.shape[0])

    for i in range(0, num_energies):
        if ( (i + shift_index < num_energies)
                and (i + shift_index + 1 < num_energies)
                and (i + shift_index >= 0) ):
            shifted[i] = ( (1 - weight) * function[i + shift_index]
                            + weight * function[i + shift_index + 1] )
        elif (i + shift_index) < 0:
            shifted[i] = tail

    return shifted