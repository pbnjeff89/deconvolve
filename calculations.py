import numpy as np
import warnings
from matplotlib import pyplot as plt

def calculate_differential_conductance(bias, energies, 
        sample_density_of_states, nonequilibrium_energy_distribution,
        temperature, superconducting_gap):
    
    # Boltzmann constant in eV/K
    kb = 8.6173303e-5

    # + 0j is necessary so that numpy knows it is processing
    # a complex number
    superconductor_dos = np.real(np.absolute(energies) / 
                            np.sqrt(energies ** 2 - superconducting_gap ** 2 + 0j))

    superconductor_dos_derivative = np.gradient(superconductor_dos)
    superconductor_distribution = get_energy_distribution(energies, temperature)
    
    shifted_sample_dos = shift_density_of_states(energies,
                                                sample_density_of_states,
                                                bias)
    shifted_sample_distribution = shift_distribution(energies,
                                                    nonequilibrium_energy_distribution,
                                                    bias)

    differential_conductance = (
        np.trapz(superconductor_dos_derivative * shifted_sample_dos
            * (shifted_sample_distribution - superconductor_distribution))
        )

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

def get_energy_distribution(energies, temperature,
                            transport_type='fd',
                            nonequilibrium_voltage = 0,
                            probe_location = 0):
    """
    Returns the specified energy distribution.

    :param energies: Numpy array of energies

    :param temperature: Lattice temperature, which should be close to the
    temperature measured at the sample level.

    :param transport_type: If input is user-given, then it should be in:
        'b'  (ballistic)
        'ph' (phonon scattering)
        'ee' (electron-electron scattering)
    Otherwise, it defaults to 'fd' (Fermi-Dirac).

    :param non_equilibrium_voltage: Nonequilibrium voltage across the normal leads of the
    system. (N.b. not the same as the voltage across the tunnel
    probes.)

    :param probe_location: This should be a value between 0 and 1. This describes
    where along the 1D system the probe is, as a multiple of the length.
    (e.g. probe_location = 0.5 says that the probe is halfway between the
    ends of the system.)

    :returns distribution:
    """
    # There is an overflow when calculating exponentials, but it
    # seems to be quite harmless as the numerics that this attempts
    # to calculate are simply the zeroes
    warnings.simplefilter(action='ignore', category='RuntimeWarning')

    # Book-keeping for making the variable names shorter
    kb = 8.6173303e-5
    T = temperature
    x = probe_location
    U = nonequilibrium_voltage
    fermi_dirac_distribution = 1 / (1 + np.exp(energies / (kb * T)))

    if transport_type is 'fd': 
        warnings.resetwarnings()
        return fermi_dirac_distribution
    elif transport_type is 'b':
        low_distribution = fermi_dirac_distribution
        high_distribution = shift(energies, fermi_dirac_distribution,
                                    U, 1.0)
        distribution =  ( (1 - x) * low_distribution
                            + x * high_distribution )
        warnings.resetwarnings()
        return distribution
    elif transport_type is 'ph':
        distribution = shift(energies, fermi_dirac_distribution,
                                -U * x, 1.0)
        warnings.resetwarnings()
        return distribution
    else:
        # implicitly, this ought to be the case 'ee'

        # Lorentz number
        # N.b. electron charge is set to 1
        L = np.pi ** 2 / 3 * kb ** 2

        # temperature in the hot-electron regime
        T = np.sqrt(T^2 + x * (1-x) * U ** 2 / L)
        distribution = 1 / (1 + np.exp(energies / (kb * T)))
        distribution = shift(energies, distribution,
                                -U * x, 1.0)
        warnings.resetwarnings()
        return distribution