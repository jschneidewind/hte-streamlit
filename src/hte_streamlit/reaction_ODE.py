import numpy as np
from scipy.integrate import odeint, solve_ivp
import matplotlib.pyplot as plt
import re
from collections import defaultdict
import pprint as pp

def parse_reactions(reactions):
    '''
    Parse a list of chemical reaction strings into structured dictionaries.
    
    This function processes reaction strings with the format:
    "[A] + 2 [B] > [C] + [D], k1 ; hv1, sigma1"
    
    Where:
    - Chemical species are enclosed in square brackets
    - Stoichiometric coefficients can be integers or decimals (e.g., 0.5 [A])
    - Reactants and products are separated by ">"
    - The rate constant identifier follows after a comma
    - Optional additional parameters follow a semicolon and are comma-separated
    
    Parameters
    ----------
    reactions : list of str
        List of reaction strings to parse.
        
    Returns
    -------
    tuple
        A tuple containing:
        - parsed_reactions: list of dict
            Each dictionary contains:
            * 'reactants': dict mapping species to stoichiometric coefficients
            * 'products': dict mapping species to stoichiometric coefficients
            * 'rate_constant': str, identifier of the rate constant
            * 'other_multipliers': list of str, optional parameters for the reaction
        - sorted_species: list
            Alphabetically sorted list of all unique chemical species in the reaction network
            
    Examples
    --------
    >>> reactions = ['[A] + 2 [B] > [C], k1', '[C] > [A] + [B], k2 ; hv1, sigma1']
    >>> parsed, species = parse_reactions(reactions)
    >>> parsed
    [{'reactants': {'[A]': 1.0, '[B]': 2.0}, 'products': {'[C]': 1.0}, 
      'rate_constant': 'k1', 'other_multipliers': []},
     {'reactants': {'[C]': 1.0}, 'products': {'[A]': 1.0, '[B]': 1.0}, 
      'rate_constant': 'k2', 'other_multipliers': ['hv1', 'sigma1']}]
    >>> species
    ['[A]', '[B]', '[C]']'''

    parsed_reactions = []
    species_set = set()
    
    for reaction in reactions:
        reaction_dict = {'reactants': {}, 'products': {}, 'rate_constant': '', 'other_multipliers': []}
        
        # Split the reaction into main components
        reaction_part, rate_part = reaction.split(',', 1)
        rate_details = [x.strip() for x in rate_part.split(';')]

        reaction_dict['rate_constant'] = rate_details[0]  # First element is the rate constant
       
        if len(rate_details) > 1:
            reaction_dict['other_multipliers'] = [item.strip() for item in rate_details[1].split(',')]
        
        # Split reactants and products
        reactants_str, products_str = reaction_part.split('>')
        
        def parse_species(side):
            species_count = defaultdict(float)
            species_matches = re.findall(r'(?:([\d\.]+)\s*)?(\[[^\]]+\])', side)

            for count, species in species_matches:
                count = float(count) if count else 1.0
                species_count[species] += count
                species_set.add(species)
            
            return dict(species_count)
        
        reaction_dict['reactants'] = parse_species(reactants_str)
        reaction_dict['products'] = parse_species(products_str)
        
        parsed_reactions.append(reaction_dict)
    
    return parsed_reactions, sorted(species_set)


def build_ode_system(parsed_reactions, species, rate_constants, other_multipliers = {}):
    """
    Build the system of ordinary differential equations.
    
    Parameters:
    -----------
    parsed_reactions : list
        List of dictionaries with keys 'reactants', 'products', 'rate_constant', and optional 'photon_flux', 'sigma'
    species : list
        Sorted list of all unique chemical species
    rate_constants : dict
        Dictionary mapping rate constant identifiers to values
    other_multipliers : dict, optional
        Dictionary mapping other multipliers to values
        
    Returns:
    --------
    function
        A function that computes the derivatives for each species.
    """
    
    def ode_system(y, t):
        """
        Compute derivatives for each species.
        
        Parameters:
        -----------
        y : array-like
            Current concentrations of each species.
        t : float
            Current time (not used explicitly in autonomous systems).
            
        Returns:
        --------
        array-like
            Derivatives for each species.
        """
        dydt = np.zeros(len(species))
        
        # Create a dictionary mapping species to their current concentrations
        conc = {spec: y[i] for i, spec in enumerate(species)}
        
        # Compute contribution from each reaction
        for reaction in parsed_reactions:

            # Start with base rate constant
            rate = rate_constants[reaction['rate_constant']]

            # Apply other multipliers if present
            for multiplier in reaction['other_multipliers']:
                mult = other_multipliers[multiplier]

                # If the multiplier is a function, resolve its arguments, call it, and multiply the rate
                if isinstance(mult, dict) and 'function' in mult:
                    arguments = mult['arguments']  
                    # resolve sources into keyword args
                    kwargs = {}
                    for parameter, source in arguments.items():
                        if source in conc:
                            kwargs[parameter] = conc[source]
                        elif source in other_multipliers and not isinstance(other_multipliers[source], dict):
                            kwargs[parameter] = other_multipliers[source]
                        elif source in rate_constants:
                            kwargs[parameter] = rate_constants[source]
                        else:
                            raise KeyError(f"Cannot resolve argument source '{source}' for multiplier '{multiplier}'")
                    rate *= mult['function'](**kwargs)

                # If the multiplier is a number, multiply directly
                else:
                    rate *= mult
            
            # Calculate concentration-dependent rate
            for reactant, stoich in reaction['reactants'].items():
                rate *= conc[reactant] ** stoich
            
            # Update derivatives for reactants (consumption)
            for reactant, stoich in reaction['reactants'].items():
                idx = species.index(reactant)
                dydt[idx] -= stoich * rate
            
            # Update derivatives for products (production)
            for product, stoich in reaction['products'].items():
                idx = species.index(product)
                dydt[idx] += stoich * rate
        
        return dydt
    
    return ode_system


def solve_ode_system(parsed_reactions, 
                     species, 
                     rate_constants, 
                     initial_conditions, 
                     times, 
                     other_multipliers = {}):
    """
    Solve the system of ODEs.
    
    Parameters:
    -----------
    parsed_reactions : list
        List of dictionaries with keys 'reactants', 'products', 'rate_constant', and optional 'photon_flux', 'sigma'
    species : list
        Sorted list of all unique chemical species
    rate_constants : dict
        Dictionary mapping rate constant identifiers to values
    initial_conditions : dict
        Dictionary mapping species to their initial concentrations
    times : array-like
        Time points at which to solve the ODEs
    other_multipliers : dict, optional
        Dictionary mapping other multipliers to values
        
    Returns:
    --------
    array-like
        Solution array with shape (len(times), len(species)).
    """
    # Convert initial conditions to array
    y0 = np.zeros(len(species))  # Initial concentrations default to zero
    for spec, conc in initial_conditions.items():
        if spec in species:
            idx = species.index(spec)
            y0[idx] = conc
        else:
            print(f'Warning: {spec} not in species list')
    
    # Build ODE system
    ode_system = build_ode_system(parsed_reactions, species, rate_constants, other_multipliers)
    
    # Solve ODEs
        
    solution = odeint(ode_system, y0, times, 
                        rtol=1e-8, atol=1e-10,  # Tighter tolerances
                        mxstep=5000)            # More steps allowed
    
    # except:
    #     # Fallback to a more robust solver
    #     print('Falling back')
    #     sol_obj = solve_ivp(lambda t, y: ode_system(y, t), 
    #                        [times[0], times[-1]], y0, 
    #                        t_eval=times, 
    #                        method='LSODA',          # Good for stiff systems
    #                        rtol=1e-6, atol=1e-9)
    #     solution = sol_obj.y.T

    # alternative:

    # solution = solve_ivp(
    #                 lambda t, y: ode_system(y, t),
    #                 [times[0], times[-1]], y0,
    #                 t_eval=times,
    #                 method='Radau',    # Implicit method for stiff systems
    #                 rtol=1e-6, atol=1e-9)

    return solution


def plot_solution(species, times, solution, exclude_species = []):
    """
    Plot the solution.
    
    Parameters:
    -----------
    species : list
        Sorted list of all unique chemical species
    times : array-like
        Time points at which the ODEs were solved
    solution : array-like
        Solution array with shape (len(times), len(species))
    exclude_species : list, optional
        List of species names to exclude from plotting. Default is None.
    """
    plt.figure(figsize=(10, 6))
    
    for i, spec in enumerate(species):
        if spec not in exclude_species:
            plt.plot(times, solution[:, i], label=spec)
    
    plt.xlabel('Time')
    plt.ylabel('Concentration')
    plt.title('Chemical Reaction Network Dynamics')
    plt.legend()
    plt.grid(True)
    #plt.show()


def example_function(A = None, 
                     B = None, 
                     sigma = None):

    return A * B * sigma


def calculate_excitations_per_second(photon_flux = None, 
                                    concentration = None, 
                                    extinction_coefficient = None,
                                    pathlength = None):
    """
    Calculate the number of excitations per Ru per second based on photon flux and concentration.

    Parameters
    ----------
    photon_flux : float
        Photon flux in photons cm^-2 s^-1.
    concentration : float
        Concentration of the species in micromolar (uM).
    extinction_coefficient : float
        Extinction coefficient of species in M^-1 cm^-1.
    pathlength : float
        Path length of the sample in cm (e.g., the distance light travels through the sample).  

    Returns
    -------
    float
        Number of excitations per Ru per second.
    """

    AVOGADRO_NUMBER = 6.022e23  # Avogadro's number in mol^-1

    concentration_M = concentration * 1e-6  # Convert concentration from uM to M
    volume_L = (pathlength * 1) / 1000 # Assuming a unit area (1 cm2) for simplicity, converting from cm3 to L
    photon_flux_mol = photon_flux / AVOGADRO_NUMBER  # Convert photon flux to mol/s
    
    absorbance = concentration_M * extinction_coefficient * pathlength  # Calculation of absorbance using Beer-Lambert law
    absorbed_fraction = 1 - 10**(-absorbance)  # Fraction of photons absorbed

    excitations_per_Ru = (photon_flux_mol * absorbed_fraction) / (volume_L * concentration_M)

    return excitations_per_Ru

def calculate_excitations_per_second_competing(photon_flux,
                                               concentration_A,
                                               concentration_B,
                                               extinction_coefficient_A,
                                               extinction_coefficient_B,
                                               pathlength):
    '''
    Calculate the number of excitations per A per second for two competing species A and B.
    
    Parameters
    ----------
    photon_flux : float
        Photon flux in photons cm^-2 s^-1.
    concentration_A : float
        Concentration of species A in micromolar (uM).
    concentration_B : float
        Concentration of species B in micromolar (uM).
    extinction_coefficient_A : float
        Extinction coefficient of species A in M^-1 cm^-1.
    extinction_coefficient_B : float
        Extinction coefficient of species B in M^-1 cm^-1.
    pathlength : float
        Path length of the sample in cm (e.g., the distance light travels through the sample.

    Returns
    -------
    float
        Number of excitations per A per second.
        '''
    
    AVOGADRO_NUMBER = 6.022e23  # Avogadro's number in mol^-1

    concentration_A_M = concentration_A * 1e-6  # Convert from uM to M
    concentration_B_M = concentration_B * 1e-6  # Convert from uM to M
    volume_L = (pathlength * 1) / 1000 # Assuming a unit area (1 cm2) for simplicity, converting from cm3 to L
    photon_flux_mol = photon_flux / AVOGADRO_NUMBER  # Convert photon flux to mol/s

    mu_A = concentration_A_M * extinction_coefficient_A
    mu_B = concentration_B_M * extinction_coefficient_B

    if mu_A + mu_B > 0:
        fractional_absorbance_A = mu_A / (mu_A + mu_B)  # Fraction of total absorbance due to A
    else:
        fractional_absorbance_A = 0

    absorbance_total = (mu_A + mu_B) * pathlength  # Total absorbance
    absorbed_fraction = 1 - 10**(-absorbance_total)  # Fraction of photons absorbed

    if concentration_A_M > 0:
        excitations_per_A = (photon_flux_mol * absorbed_fraction * fractional_absorbance_A) / (volume_L * concentration_A_M)
    else:
        excitations_per_A = 0

    return excitations_per_A










def main():
        # Define reactions with photochemical parameters
    # reactions = [
    #     'B > A, k1',
    #     'A > B, k2, hv1, sigma1',
    #     'B + B + B > C, k3',
    #     'C > D, k4, hv1, sigma1'
    # ]

    # reactions = ['[A] + 0.1 [A] + [B] > [C], k1 ; hv1, sigma1',
    #              '[C] > [D] + [Ru1] + [Ru1], k2 ; other1, test3',
    #              '[Ru1] > [Ru2], k3;test1']

    reactions = ['[A] + [Cat] > [B] + [Cat], k1 ; function_test',
                 '2 [Cat] > [Cat-Dimer], k2']

    # Set rate constants
    # rate_constants = {
    #     'k1': 0.5,
    #     'k2': 0.5
    # }


    # Set rate constants
    rate_constants = {
        'k1': 1,
        'k2': 0.0,
        'k3': 0.01,
        'k4': 0.02
    }
    
    other_multipliers = {
        'hv1': 2.3e17,
        'sigma1': 0.5e-18,
        'extinction_coefficient': 8500, 
        'pathlength': 2.25,
        'other1': 0.5,
        'test3': 0.2,
        'test1': 0.1,
        'function_test': {
            'function': calculate_excitations_per_second,
            'arguments': {
                'photon_flux': 'hv1',
                'concentration': '[A]',
                'extinction_coefficient': 'extinction_coefficient',
                'pathlength': 'pathlength'
            }
        }
    }


    
    # # Set initial conditions
    # initial_conditions = {
    #     '[A]': 0.5,
    #     '[B]': 1.0
    # }

        # Set initial conditions
    initial_conditions = {
        '[A]': 100,
        '[B]': 0,
        '[Cat]': 1
    }
    
    
    # Define time points
    times = np.linspace(0, 2, 1000)

    parsed_reactions, species = parse_reactions(reactions)



    solution = solve_ode_system(parsed_reactions, species, rate_constants, 
                                initial_conditions, times, other_multipliers)

    # Plot results
    plot_solution(species, times, solution)


    
    # Print final concentrations
    # print("Final concentrations:")
    # for i, spec in enumerate(species):
    #     print(f"{spec}: {solution[-1, i]:.6f}")

    # conc_A = 1000
    # ex_A = 8700
    # conc_B = 10
    # ex_B = 4000
    # pathlength = 1.0  # cm
    # photon_flux = 2.3e17  # photons cm^-2 s^-1

    # print('A', calculate_excitations_per_second_competing(photon_flux,
    #                                                        conc_A,
    #                                                        conc_B,
    #                                                        ex_A,
    #                                                        ex_B,
    #                                                        pathlength))
    # print('B', calculate_excitations_per_second_competing(photon_flux,
    #                                                           conc_B,
    #                                                           conc_A,
    #                                                           ex_B,
    #                                                           ex_A,
    #                                                           pathlength))




    plt.show()

def catalyst_dimerization():

    # reactions = ['[RuII] + [S2O8] > [RuIII] + [SO4], k1',
    #              '[RuIII] + [RuII] > [Ru-Dimer], k2',
    #              '[Ru-Dimer] + [RuIII] > [O2] + [Ru-Dimer] + [RuII], k3',
    #              '[Ru-Dimer] + [RuIII] > [Ru-Trimer], k4',
    #              '[RuIII] > [RuII], k5']
                    
    # reactions = ['[RuII] + [S2O8] > [RuIII] + [SO4], k1',
    #              '[RuIII] > [H2O2] + [RuII], k2',
    #              '2 [RuIII] > [Ru-Dimer], k3',
    #              '2 [RuIII] + [Ru-Dimer] > 2 [Ru-Dimer], k4',
    #              '[H2O2] > [O2], k5',
    #              '[RuIII] > [Inactive], k6',
    #              '[RuIII] + [SO4] > [RuII] + [S2O8], k7'
    #             ]
    
    # reactions = ['[RuII] + [S2O8] > [RuIII] + [SO4], k1 ; hv1',
    #              '[RuIII] > [H2O2] + [RuII], k2 ; hv2',
    #              '2 [RuIII] > [Ru-Dimer], k3',
    #              '2 [RuIII] + [Ru-Dimer] > 2 [Ru-Dimer], k4',
    #              '[H2O2] > [O2], k5',
    #              '[RuIII] > [Inactive], k6']
    
    # reactions = ['[RuII] > [RuII-ex], k1 ; hv1',
    #              '[RuII-ex] + [S2O8] > [RuIII] + [SO4], k7',
    #              '[RuIII] > [H2O2] + [RuII], k2 ; hv2',
    #              '2 [RuIII] > [Ru-Dimer], k3',
    #              '2 [RuIII] + [Ru-Dimer] > 2 [Ru-Dimer], k4',
    #              '[H2O2] > [O2], k5',
    #              '[RuIII] > [Inactive], k6']
    
    reactions = ['[RuII] > [RuII-ex], k1 ; hv_functionA',
                 '[RuII-ex] > [RuII], k8',
                 '[RuII-ex] + [S2O8] > [RuIII] + [SO4], k7',
                 '[RuIII] > [H2O2] + [RuII], k2 ; hv_function_B',
                 '2 [RuIII] > [Ru-Dimer], k3',
                 '2 [RuIII] + [Ru-Dimer] > 2 [Ru-Dimer], k4',
                 '[H2O2] > [O2], k5',
                 '[RuIII] > [Inactive], k6']
    
    # rate_constants = {
    #     'k1': 1.0E-4,
    #     'k2': 3.0E-2,
    #     'k3': 5.0E-6,
    #     'k4': 1.0E-3,
    #     'k5': 1.0E-1,
    #     'k6': 1.0E-2,
    #     'k7': 1.0E-3
    # }
    
    # rate_constants = {
    #     'k1': 3.479e-01,
    #     'k2': 3.100e+00,
    #     'k3': 3.943e-02,
    #     'k4': 1.068e-02,
    #     'k5': 1.135e-02,
    #     'k6': 2.910e-02,
    #     'k7': 0.000e-01
    # }

    # rate_constants = {
    #     'k1': 9.644e-01,
    #     'k2': 9.983e-01,
    #     'k3': 6.406e-03,
    #     'k4': 2.428e-03,
    #     'k5': 2.293e-02,
    #     'k6': 4.493e-03,
    #     'k7': 9.092e+01,
    #     'k8': 1/650e-9
    # }

    # rate_constants = {
    #     'k1': 5.647e-01,
    #     'k2': 9.960e-01,
    #     'k3': 7.019e-03,
    #     'k4': 2.492e-03,
    #     'k5': 2.453e-02,
    #     'k6': 3.662e-03,
    #     'k7': 8.276e+00,
    #     'k8': 1/650e-9
    # }

    rate_constants = {
        'k1': 9.995e-01,
        'k2': 9.886e-01,
        'k3': 7.407e-03,
        'k4': 3.437e-03,
        'k5': 2.739e-02,
        'k6': 4.762e-03,
        'k7': 5.918e+01,
        'k8': 1/650e-9
    }



    initial_conditions = {
        '[RuII]': 10,
        '[S2O8]': 6000
    }

    times = np.linspace(0, 350, 10000)

    # other_multipliers = {
    #     'hv1': 7.6,
    #     'hv2': 0.47
    # }

    other_multipliers = {
        'pathlength': 2.25,
        'photon_flux': 2.3e17,
        'Ru_II_extinction_coefficient': 8500,
        'Ru_III_extinction_coefficient': 540,
        'hv_functionA': {
            'function': calculate_excitations_per_second_competing,
            'arguments': {
                'photon_flux': 'photon_flux',
                'concentration_A': '[RuII]',
                'concentration_B': '[RuIII]',
                'extinction_coefficient_A': 'Ru_II_extinction_coefficient',
                'extinction_coefficient_B': 'Ru_III_extinction_coefficient',
                'pathlength': 'pathlength'
            }
        },
        'hv_function_B': {
            'function': calculate_excitations_per_second_competing,
            'arguments': {
                'photon_flux': 'photon_flux',
                'concentration_A': '[RuIII]',
                'concentration_B': '[RuII]',
                'extinction_coefficient_A': 'Ru_III_extinction_coefficient',
                'extinction_coefficient_B': 'Ru_II_extinction_coefficient',
                'pathlength': 'pathlength'
            }
        }
    }

    parsed_reactions, species = parse_reactions(reactions)


    solution = solve_ode_system(parsed_reactions, species, rate_constants, 
                                initial_conditions, times, other_multipliers) 

    # Plot results
    # plot_solution(species, times, solution, exclude_species = ['[S2O8]', '[SO4]'])

    #ox_concentrations = np.linspace(0, 10000, 50)
    #ru_concentrations = np.linspace(0, 100, 50)
    photon_flux = np.linspace(1e17, 1e18, 30)

    max_rates = []

    #for ox_conc in ox_concentrations:
    for flux in photon_flux:
    #for ru_conc in ru_concentrations:


        #initial_conditions['[S2O8]'] = ox_conc
        #initial_conditions['[RuII]'] = ru_conc
        other_multipliers['photon_flux'] = flux

        solution = solve_ode_system(parsed_reactions, species, rate_constants, 
                                    initial_conditions, times, other_multipliers) 
        
        o2 = solution[:, species.index('[O2]')]
        rate = np.diff(o2) / np.diff(times)
        max_rate = np.amax(rate)

        max_rates.append(max_rate)

    #plt.plot(ox_concentrations, max_rates, 'o-')
    plt.plot(photon_flux, max_rates, 'o-')
    #plt.plot(ru_concentrations, max_rates, 'o-')


    plt.show()
    

        
        




def two_photon_water_splitting():

    reactions = ['2 [A-Mono_S0] > 2 [A_S0], k1',
                 '[A_S0] > 2 [A-Mono_S0], k2',
                 '[A_S0] > [A_Sn], k3 ; hv1, sigma1',
                 '[A_Sn] > [A_S0], k4',
                 '[A_Sn] > [A_S0], k19',
                 '[A_Sn] > [B_T0], k5',
                 '[B_T0] > [A_S0], k6',
                 '[B_T0] > [B_T2], k7 ; hv2, sigma2',
                 '[B_T2] > [C_T0], k8',
                 '[C_T0] > [D_T0] + [F_S0], k9',
                 '[D_T0] + [F_S0] > [C_T0], k10',
                 '[D_T0] > [E_S0] + [O2], k11',
                 '[E_S0] > [F_S0], k12',
                 '[F_S0] > [E_S0], k13',
                 '[O2] + [A_Sn] > [X], k14',
                 '[O2] + [B_T0] > [X], k15',
                 '[A_Sn] > [A-Trans_S0], k16',
                 '[A-Trans_S0] > [A_S0], k17',
                 '[F_S0] > [F-Trans_S0], k18'
                 ]

    rate_constants = {
        'k1': 6e+12,
        'k2': 36e+12,
        'k3': 1,
        'k4': 1.43e+11,
        'k5': 6.21e+12,
        'k6': 1.54e+07,
        'k7': 1,
        'k8': 6.21e+12,
        'k9': 1.42e+03,
        'k10': 1.57e-02,
        'k11': 1.47e+04,
        'k12': 6.21e+12,
        'k13': 2.66e+10,
        'k14': 1.0e+15,
        'k15': 1.0e+15,
        'k16': 1.0e+10,
        'k17': 7.26e-07,
        'k18': 7.26e-07,
        'k19': 2.5e+8
    }

    other_multipliers = {
        'hv1': 1e17,
        'hv2': 1e17,
        'sigma1': 1e-17,
        'sigma2': 1e-17,
    }

    # in mol/L
    initial_conditions = {
        '[A-Mono_S0]': 0.004,
        '[A_S0]': 0.001
    }

    # in seconds
    times = np.linspace(0, 100, 1000)

    parsed_reactions, species = parse_reactions(reactions)

    solution = solve_ode_system(parsed_reactions, species, rate_constants, 
                                initial_conditions, times, other_multipliers)

    # Plot results
    plot_solution(species, times, solution)


# Example usage
if __name__ == "__main__":
    #main()
    catalyst_dimerization()
    #two_photon_water_splitting()