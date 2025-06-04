import numpy as np
from scipy.integrate import odeint
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
                rate *= other_multipliers[multiplier]
            
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
    solution = odeint(ode_system, y0, times)

    return solution


def plot_solution(species, times, solution):
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
    """
    plt.figure(figsize=(10, 6))
    
    for i, spec in enumerate(species):
        plt.plot(times, solution[:, i], label=spec)
    
    plt.xlabel('Time')
    plt.ylabel('Concentration')
    plt.title('Chemical Reaction Network Dynamics')
    plt.legend()
    plt.grid(True)
    plt.show()


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

    reactions = ['[A] + [Cat] > [B] + [Cat], k1',
                 '2 [Cat] > [Cat-Dimer], k2']

    # Set rate constants
    # rate_constants = {
    #     'k1': 0.5,
    #     'k2': 0.5
    # }


    # Set rate constants
    rate_constants = {
        'k1': 10,
        'k2': 0.05,
        'k3': 0.01,
        'k4': 0.02
    }
    
    other_multipliers = {
        'hv1': 10.0e18,
        'sigma1': 0.5e-18,
        'other1': 0.5,
        'test3': 0.2,
        'test1': 100
    }


    
    # # Set initial conditions
    # initial_conditions = {
    #     '[A]': 0.5,
    #     '[B]': 1.0
    # }

        # Set initial conditions
    initial_conditions = {
        '[A]': 1,
        '[B]': 0,
        '[Cat]': 0.1
    }
    
    
    # Define time points
    times = np.linspace(0, 100, 1000)

    parsed_reactions, species = parse_reactions(reactions)


    solution = solve_ode_system(parsed_reactions, species, rate_constants, 
                                initial_conditions, times, other_multipliers)

    # Plot results
    plot_solution(species, times, solution)
    
    # Print final concentrations
    # print("Final concentrations:")
    # for i, spec in enumerate(species):
    #     print(f"{spec}: {solution[-1, i]:.6f}")

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
    main()
    #two_photon_water_splitting()