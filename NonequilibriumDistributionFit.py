import numpy

def find_best_fit(initial_distribution, measured_distribution,
                    learning_rate = 0.2, error_tolerance = 0.1):
    chi_squared = 10.0
    calculated_distribution = initial_distribution

    while chi_squared > error_tolerance:
        calculated_conductance = calculate_conductance(calculated_distribution)
        chi_squared = calculate_chi_squared(calculated_conductance, measured_distribution)
        if chi_squared > error_tolerance:
            calculated_distribution = calculate_distribution(calculated_distribution,
                                                                learning_rate,
                                                                chi_squared)

    return calculated_distribution

def calculate_conductance(calculated_distribution):
    pass

def calculate_chi_squared(calculated_conductance, measured_conductance):
    pass

def calculate_distribution(distribution, learning_rate, chi_squared)
    pass