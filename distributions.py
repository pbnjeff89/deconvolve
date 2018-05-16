import warnings

def get_energy_distribution(energies, temperature, transport_type='fd'):
    """
    Returns the energy distribution as specified.

    Input:

    energies: Numpy array of energies

    transport_type: If input is user-given, then it should be in:
        'b'  (ballistic)
        'ph' (phonon scattering)
        'ee' (electron-electron scattering)
    Otherwise, it defaults to 'fd' (Fermi-Dirac).
    """
    # There is an overflow when calculating exponentials, but it
    # seems to be quite harmless as the numerics that this attempts
    # to calculate are simply the zeroes
    warnings.simplefilter(action='ignore', category='RuntimeWarning')

    kb = 8.6173303e-5

    if transport_type is 'fd':
        distribution = 1 / (1 + np.exp(energies / (kb * T)))
        warnings.resetwarnings()
        return distribution
    elif transport_type is 'b':
        warnings.resetwarnings()
        return 1
    elif transport_type is 'ph':
        warnings.resetwarnings()
        return 1
    else:
        # implicitly, this ought to be the case 'ee'
        warnings.resetwarnings()
        return 1