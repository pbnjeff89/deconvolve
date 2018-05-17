import warnings
from calculations import shift

def get_energy_distribution(energies, temperature,
                            transport_type='fd',
                            nonequilibrium_voltage = 0,
                            probe_location = 0):
    """
    Returns the energy distribution as specified.

    Input:

    energies: Numpy array of energies

    temperature: Lattice temperature, which should be close to the
    temperature measured at the sample level.

    transport_type: If input is user-given, then it should be in:
        'b'  (ballistic)
        'ph' (phonon scattering)
        'ee' (electron-electron scattering)
    Otherwise, it defaults to 'fd' (Fermi-Dirac).

    non_equilibrium_voltage: Nonequilibrium voltage across the normal leads of the
    system. (N.b. not the same as the voltage across the tunnel
    probes.)

    probe_location: This should be a value between 0 and 1. This describes
    where along the 1D system the probe is, as a multiple of the length.
    (e.g. probe_location = 0.5 says that the probe is halfway between the
    ends of the system.)
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
        distribution = shift(energies, fermi_dirac_distribution
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