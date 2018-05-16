import numpy as np
from .distributions import *

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


def calculate_mse(data, model):
    pass

def shift_density_of_states(energies, dos, shift):
    shifted_dos = np.zeros(dos.shape[0])
    return shifted_dos

def shift_distribution(energies, function, shift):
    shifted_distribution = np.zeros(dos.shape[0])
    return shifted_distribution