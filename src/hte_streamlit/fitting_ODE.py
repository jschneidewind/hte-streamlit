import numpy as np
from scipy.optimize import differential_evolution, dual_annealing, minimize
from dataclasses import dataclass
from types import SimpleNamespace

from hte_streamlit.reaction_ODE import solve_ode_system, parse_reactions, calculate_excitations_per_second_competing
from hte_streamlit.experiments_database import ExperimentalDataset

import pprint as pp

import matplotlib.pyplot as plt
from matplotlib.figure import Figure
from typing import Dict, Optional, List, Union



def resolve_experiment_attributes(template_dict, experiment):
    """
    Resolve attribute paths in a dictionary to actual values from an experiment.
    
    Parameters
    ----------
    template_dict : dict
        Dictionary with string attribute paths as values or nested dictionaries
        containing attribute paths.
    experiment : Experiment
        Experiment instance containing the attributes to look up.
        
    Returns
    -------
    dict
        New dictionary with the same keys but resolved attribute values.
    """
    result_dict = {}
    
    for key, value in template_dict.items():
        # Check if the value is a nested dictionary
        if isinstance(value, dict) and 'function' not in value:
            # Recursively resolve the nested dictionary
            result_dict[key] = resolve_experiment_attributes(value, experiment)
        elif isinstance(value, str):
            # Handle string attribute paths
            path_components = value.split('.')
            
            # Start from the experiment object
            current_obj = experiment
            
            # Navigate through the attribute path
            for component in path_components:
                if hasattr(current_obj, component):
                    current_obj = getattr(current_obj, component)
                else:
                    raise AttributeError(f"'{type(current_obj).__name__}' object has no attribute '{component}'")
            
            # Store the final resolved value
            result_dict[key] = current_obj
        else:
            # For other types (e.g., numbers, lists), just copy the value
            result_dict[key] = value
    
    return result_dict


def square_loss_time_series(model_data, experimental_data, **kwargs):
    '''
    Placeholder'''

    return np.sum((np.array(model_data) - np.array(experimental_data['y'])) ** 2), model_data

def square_loss_max_rate_ydiff(model_data, experimental_data, times, **kwargs):
    '''
    Placeholder'''

    model_data_ydiff = np.diff(model_data) / np.diff(times)

    model_data_max_rate_ydiff = np.amax(model_data_ydiff) 

    return (model_data_max_rate_ydiff - experimental_data['y']) ** 2, model_data_max_rate_ydiff

def square_loss_ydiff(model_data, experimental_data, times, **kwargs):
    '''
    Placeholder'''

    model_data_ydiff = np.diff(model_data) / np.diff(times)

    return np.sum((model_data_ydiff - experimental_data['y']) ** 2), model_data_ydiff

def robust_loss_ydiff(model_data, experimental_data, times, **kwargs):
    '''
    Robust loss function using Huber loss to handle outliers
    '''
    model_data_ydiff = np.diff(model_data) / np.diff(times)
    
    # Check for invalid values
    if np.any(np.isnan(model_data_ydiff)) or np.any(np.isinf(model_data_ydiff)):
        return 1e6, model_data_ydiff
    
    residuals = model_data_ydiff - experimental_data['y']
    
    # Huber loss parameter
    delta = np.std(experimental_data['y'])
    
    # Apply Huber loss
    huber_loss = np.where(np.abs(residuals) <= delta,
                         0.5 * residuals**2,
                         delta * (np.abs(residuals) - 0.5 * delta))
    
    return np.sum(huber_loss), model_data_ydiff

class Fitting_Model:
    '''Placeholder
    Doctring
    '''

    def __init__(self, reaction_network: list):
        self.reaction_network = reaction_network
        self.fixed_rate_constants: dict = {}
        self.rate_constants_to_optimize: dict = {}
        self.data_to_be_fitted: dict = {}
        self.initial_conditions: dict = {}
        self.other_multipliers: dict = {}
        self.times: dict = {}
        self.experiments: list = []
        self.loss_function = None
        self.x0 = None

        self.parsed_reactions, self.species = parse_reactions(self.reaction_network)

    def optimize(self, 
                 workers = -1, disp = True):

        bounds = list(self.rate_constants_to_optimize.values())

        self.result = differential_evolution(
            objective_function,
            bounds = bounds,
            args = (model,),
            workers = workers,
            disp = disp,
            updating = 'deferred',
            x0 = self.x0)

        print(self.result)

    def optimize_dual_annealing(self):
        
        bounds = list(self.rate_constants_to_optimize.values())

        self.result = dual_annealing(
            objective_function,
            bounds= bounds,
            args = (model,)
        )

        print(self.result)

    def minimize(self, x0, method = 'L-BFGS-B'):

        self.result = minimize(
            objective_function,
            method = method,
            x0 = x0,
            args = (model,)
            )

        print(self.result)




def objective_function(rate_constants_to_optimize, model: Fitting_Model, return_full = False):
    '''
    Placeholder
    '''

    rate_constants = dict(zip(model.rate_constants_to_optimize.keys(), rate_constants_to_optimize))
    rate_constants |= model.fixed_rate_constants

    total_error = 0.0 
    full_output = {}
    time_series = {}

    for experiment_entry in model.experiments:

        if isinstance(experiment_entry, tuple):
            experiment, weight = experiment_entry
        else:
            experiment, weight = experiment_entry, 1.0 # Default weight of 1.0 if not specified

        experiment_name = experiment.experiment_metadata.experiment_name
        full_output[experiment_name] = {}

        # Resolve the rate constants, initial conditions, other multipliers, and times and data to be fitted
        initial_conditions = resolve_experiment_attributes(model.initial_conditions, experiment)
        other_multipliers = resolve_experiment_attributes(model.other_multipliers, experiment)
        times = resolve_experiment_attributes(model.times, experiment)
        data_to_be_fitted = resolve_experiment_attributes(model.data_to_be_fitted, experiment)

        # Solve the ODE system
        model_result = solve_ode_system(model.parsed_reactions,
                                        model.species,
                                        rate_constants,
                                        initial_conditions,
                                        times['times'],
                                        other_multipliers)
        time_series[experiment_name] = model_result
               
        # Calculate the error between the model and the data for each species to be fitted
        experiment_error = 0.0

        for species, data in data_to_be_fitted.items():
            idx = model.species.index(species)
            model_data = model_result[:,idx]

            error, model_data_transformed = model.loss_function(model_data, data, times = times['times'])
            
            full_output[experiment_name][species] = model_data_transformed

            experiment_error += error
        
        total_error += experiment_error * weight

    if return_full is True:
        return total_error, full_output
    elif return_full == 'All':
        return total_error, full_output, time_series
    else:
        return total_error

def visualize_optimization_results(model: Fitting_Model) -> Figure:
    """
    Visualize the optimization results by plotting model predictions against experimental data.
    
    Parameters
    ----------
    model : Fitting_Model
        The fitted model with optimization results.
 
    Returns
    -------
    matplotlib.figure.Figure
        The generated figure object.
    """
    if not hasattr(model, 'result') or model.result is None:
        raise ValueError("Model has not been optimized yet. Call model.optimize() first.")
    

    error, model_results = objective_function(model.result.x, model, return_full = True)

    fig, ax = plt.subplots(2, 2, figsize=(12, 8))
    fig.tight_layout(pad=3.0)

    colors = plt.rcParams['axes.prop_cycle'].by_key()['color']

    for i, experiment_entry in enumerate(model.experiments):
        if isinstance(experiment_entry, tuple):
            experiment, weight = experiment_entry
        else:
            experiment= experiment_entry

        color = colors[i % len(colors)] 
        experimental_data = resolve_experiment_attributes(model.data_to_be_fitted, experiment)
        
        #for species, data in model.data_to_be_fitted.items():
        for species, data in experimental_data.items():

            ax[0][0].scatter(data['x'], data['y'], color = color)

            model_data = model_results[experiment.experiment_metadata.experiment_name][species]

            ax[0][0].plot(data['x'], model_data, color = color)
            # Then add a single legend entry for both
            ax[0][0].plot([], [], color=color, marker='o', linestyle='-', 
                    label=f'{species} - {experiment.experiment_metadata.experiment_name}')
            
            OX_conc = experiment.experiment_metadata.oxidant_concentration_uM
            Ru_conc = experiment.experiment_metadata.ru_concentration_uM
            power = experiment.experiment_metadata.power_output
            max_rate_ydiff = experiment.analysis_metadata.max_rate_ydiff
            
            ax[0][1].plot(power, max_rate_ydiff, 'o-', color=color)
            ax[0][1].plot(power, np.amax(model_data), 'x', color=color)

            ax[1][0].plot(OX_conc, max_rate_ydiff, 'o-', color=color)
            ax[1][0].plot(OX_conc, np.amax(model_data), 'x', color=color)

            ax[1][1].plot(Ru_conc, max_rate_ydiff, 'o-', color=color)
            ax[1][1].plot(Ru_conc, np.amax(model_data), 'x', color=color)

    ax[0][0].legend()

    return fig


def visualize_optimization_results(model: Fitting_Model, show_groups=None) -> Figure:
    """
    Parameters
    ----------
    show_groups : list, optional
        List of groups to show. Options: ['power', 'oxidant', 'ruthenium']
        Default: ['power', 'oxidant', 'ruthenium'] (show all)
    """

    if not hasattr(model, 'result') or model.result is None:
        raise ValueError("Model has not been optimized yet. Call model.optimize() first.")
    
    error, model_results = objective_function(model.result.x, model, return_full = True)
    print('Fun:', error)

    fig, ax = plt.subplots(2, 2, figsize=(12, 8))
    fig.tight_layout(pad=3.0)
    colors = plt.rcParams['axes.prop_cycle'].by_key()['color']

    if show_groups is None:
        show_groups = ['power', 'oxidant', 'ruthenium']
    
    # Define experiment groups based on your experimental design
    experiment_groups = {
        'power': ['MRG-059-ZO-3-1', 'MRG-059-ZN-14-1', 'MRG-059-ZN-13-1', 'MRG-059-ZN-11-1', 'MRG-059-ZO-1-1'],
        'oxidant': ['MRG-059-ZN-10-1', 'MRG-059-ZN-9-1', 'MRG-059-ZN-8-1', 'MRG-059-ZO-1-1', 'MRG-059-ZO-9-1'],
        'ruthenium': ['MRG-059-ZO-2-1', 'MRG-059-ZN-7-1', 'MRG-059-ZN-4-1', 'MRG-059-ZN-3-1', 'MRG-059-ZN-2-1', 'MRG-059-ZN-1-1', 'MRG-059-ZO-1-1', 'MRG-059-ZO-8-1']
    }
    
    for i, experiment_entry in enumerate(model.experiments):
        if isinstance(experiment_entry, tuple):
            experiment, weight = experiment_entry
        else:
            experiment = experiment_entry

        exp_name = experiment.experiment_metadata.experiment_name
        color = colors[i % len(colors)]
        experimental_data = resolve_experiment_attributes(model.data_to_be_fitted, experiment)
        
        for species, data in experimental_data.items():
            model_data = model_results[exp_name][species]
            
            # Time series plot (always show)
            ax[0][0].scatter(data['x'], data['y'], color=color)
            ax[0][0].plot(data['x'], model_data, color=color)
            ax[0][0].plot([], [], color=color, marker='o', linestyle='-', 
                         label=f'{species} - {exp_name}')
            
            # Get experiment parameters
            OX_conc = experiment.experiment_metadata.oxidant_concentration_uM
            Ru_conc = experiment.experiment_metadata.ru_concentration_uM
            power = experiment.experiment_metadata.power_output
            max_rate_ydiff = experiment.analysis_metadata.max_rate_ydiff
            
            # Conditional plotting based on groups
            if 'power' in show_groups and exp_name in experiment_groups['power']:
                ax[0][1].plot(power, max_rate_ydiff, 'o-', color=color)
                ax[0][1].plot(power, np.amax(model_data), 'x', color=color)

            if 'oxidant' in show_groups and exp_name in experiment_groups['oxidant']:
                ax[1][0].plot(OX_conc, max_rate_ydiff, 'o-', color=color)
                ax[1][0].plot(OX_conc, np.amax(model_data), 'x', color=color)

            if 'ruthenium' in show_groups and exp_name in experiment_groups['ruthenium']:
                ax[1][1].plot(Ru_conc, max_rate_ydiff, 'o-', color=color)
                ax[1][1].plot(Ru_conc, np.amax(model_data), 'x', color=color)

    ax[0][0].legend()
    return fig

if __name__ == '__main__':

    dataset = ExperimentalDataset.load_from_hdf5('data/250608_HTE.h5')
    dataset.update_reaction_data()

    #print(dataset.experiments['MRG-059-ZN-10-1'].analysis_metadata.max_rate_ydiff)

    # model = Fitting_Model(['[S2O8] + [RuII] > [RuIII] + [SO4], k1 ; irradiation_power',
    #                        '[S2O8] + [RuIII] > [RuIV] + [SO4], k2 ; irradiation_power',
    #                        '2 [RuIV] > [Ru-Dimer], k3',
    #                        '3 [RuIV] > [Ru-Trimer], k4',
    #                        '[RuIII] > [RuII] + [O2], k5',
    #                        '[S2O8] > [SO4], k6'])


    #### Substrate model that works
    # model = Fitting_Model(['[S2O8] + [RuII] > [RuIII] + [SO4], k1 ; irradiation_power',
    #                        '[RuIII] + [RuII] > [Ru-Dimer], k2',
    #                        '[RuIII] > [RuII] + [O2], k3',
    #                        '[S2O8] > [SO4], k4',
    #                        '2 [RuII] > [Ru-Dimer], k5'])
    
    # model = Fitting_Model(['[RuII] + [S2O8] > [O2] + [RuII], k1',
    #                        '2 [RuII] > [Ru-Dimer], k2',
    #                        '[Ru-Dimer] > 2 [RuII], k3',
    #                        '[RuII] + [Ru-Dimer] > [Ru-Trimer], k4',
    #                        '[Ru-Trimer] > [RuII] + [Ru-Dimer], k5'
                           
    #                         ])
                           

    ### Irreversible first reaction
    # model = Fitting_Model(['[RuII] + [S2O8] > [RuIII] + [SO4], k1 ; hv, epsilon',
    #                        '[RuIII] > [H2O2] + [RuII], k2; hv, epsilon',
    #                        '2 [RuIII] > [Ru-Dimer], k3',
    #                        '2 [RuIII] + [Ru-Dimer] > 2 [Ru-Dimer], k4',
    #                        '[H2O2] > [O2], k5',
    #                        '[RuIII] > [Inactive], k6'
    #              ])
    
    # model = Fitting_Model(['[RuII] + [S2O8] > [RuIII] + [SO4], k1 ; hv, sigma_RuII',
    #                        '[RuIII] > [H2O2] + [RuII], k2; hv, sigma_RuIII',
    #                        '2 [RuIII] > [Ru-Dimer], k3',
    #                        '2 [RuIII] + [Ru-Dimer] > 2 [Ru-Dimer], k4',
    #                        '[H2O2] > [O2], k5',
    #                        '[RuIII] > [Inactive], k6'
    #              ])
    
    ### Current king
    # model = Fitting_Model(['[RuII] > [RuII-ex], k1 ; hv, sigma_RuII',
    #                        '[RuII-ex] + [S2O8] > [RuIII] + [SO4], k7',
    #                        '[RuIII] > [H2O2] + [RuII], k2 ; hv, sigma_RuIII',
    #                        '2 [RuIII] > [Ru-Dimer], k3',
    #                        '2 [RuIII] + [Ru-Dimer] > 2 [Ru-Dimer], k4',
    #                        '[H2O2] > [O2], k5',
    #                        '[RuIII] > [Inactive], k6'])
    
    ### Accurate light absorption model
    # model = Fitting_Model(['[RuII] > [RuII-ex], k1 ; hv_functionA',
    #                         '[RuII-ex] + [S2O8] > [RuIII] + [SO4], k7',
    #                         '[RuIII] > [H2O2] + [RuII], k2 ; hv_function_B',
    #                         '2 [RuIII] > [Ru-Dimer], k3',
    #                         '2 [RuIII] + [Ru-Dimer] > 2 [Ru-Dimer], k4',
    #                         '[H2O2] > [O2], k5',
    #                         '[RuIII] > [Inactive], k6'])
    
    ### inlcuding decay of [RuII-ex] back to ground state
    model = Fitting_Model(['[RuII] > [RuII-ex], k1 ; hv_functionA',
                            '[RuII-ex] > [RuII], k8',
                            '[RuII-ex] + [S2O8] > [RuIII] + [SO4], k7',
                            '[RuIII] > [H2O2] + [RuII], k2 ; hv_function_B',
                            '2 [RuIII] > [Ru-Dimer], k3',
                            '2 [RuIII] + [Ru-Dimer] > 2 [Ru-Dimer], k4',
                            '[H2O2] > [O2], k5',
                            '[RuIII] > [Inactive], k6'])
                           
    ### Reversible first reaction
    # model = Fitting_Model(['[RuII] + [S2O8] > [RuIII] + [SO4], k1',
    #                        '[RuIII] > [H2O2] + [RuII], k2',
    #                        '2 [RuIII] > [Ru-Dimer], k3',
    #                        '2 [RuIII] + [Ru-Dimer] > 2 [Ru-Dimer], k4',
    #                        '[H2O2] > [O2], k5',
    #                        '[RuIII] > [Inactive], k6',
    #                        '[RuIII] + [SO4] > [RuII] + [S2O8], k7'
    #                        ])




    # model = Fitting_Model(['[Oxidant] + [Cat] > [O2] + [Cat], k1 ; irradiation_power',
    #                        '[Oxidant] > [Waste], k3',
    #                        '2 [Cat] > [Cat-Dimer], k2'])
    
    ### Ru experiments
    # model.experiments = [dataset.experiments['MRG-059-ZO-2-1'],
    #                      dataset.experiments['MRG-059-ZN-7-1'],
    #                      dataset.experiments['MRG-059-ZN-4-1'],
    #                      dataset.experiments['MRG-059-ZN-3-1'],
    #                      dataset.experiments['MRG-059-ZN-2-1'],
    #                      dataset.experiments['MRG-059-ZN-1-1'],
    #                      dataset.experiments['MRG-059-ZO-1-1'],
    #                      dataset.experiments['MRG-059-ZO-8-1']]
    
    ### Substrate experiments
    # model.experiments = [(dataset.experiments['MRG-059-ZN-10-1'], 10),
    #                      (dataset.experiments['MRG-059-ZN-9-1'], 10),
    #                      (dataset.experiments['MRG-059-ZN-8-1'], 10),
    #                      (dataset.experiments['MRG-059-ZO-1-1'], 10),
    #                      (dataset.experiments['MRG-059-ZO-9-1'], 10)]
    
    ## Irradiation power experiments
    # model.experiments = [dataset.experiments['MRG-059-ZO-3-1'],
    #                      dataset.experiments['MRG-059-ZN-14-1'],
    #                      dataset.experiments['MRG-059-ZN-13-1'],
    #                      dataset.experiments['MRG-059-ZN-11-1'],
    #                      dataset.experiments['MRG-059-ZO-1-1']]
    


    ## Ru + substrate experiments
    # model.experiments = [dataset.experiments['MRG-059-ZO-2-1'],
    #                      dataset.experiments['MRG-059-ZN-7-1'],
    #                      dataset.experiments['MRG-059-ZN-4-1'],
    #                      dataset.experiments['MRG-059-ZN-3-1'],
    #                      dataset.experiments['MRG-059-ZN-2-1'],
    #                      dataset.experiments['MRG-059-ZN-1-1'],
    #                      dataset.experiments['MRG-059-ZO-1-1'],
    #                      dataset.experiments['MRG-059-ZO-8-1'],
    #                      (dataset.experiments['MRG-059-ZN-10-1'], 1),
    #                      (dataset.experiments['MRG-059-ZN-9-1'], 1),
    #                      (dataset.experiments['MRG-059-ZN-8-1'], 1),
    #                      (dataset.experiments['MRG-059-ZO-9-1'], 1)]

    ### Ru + substrate + irradiation power experiments
    model.experiments = [dataset.experiments['MRG-059-ZO-2-1'],
                         dataset.experiments['MRG-059-ZN-7-1'],
                         dataset.experiments['MRG-059-ZN-4-1'],
                         dataset.experiments['MRG-059-ZN-3-1'],
                         (dataset.experiments['MRG-059-ZN-2-1'], 1),
                         dataset.experiments['MRG-059-ZN-1-1'],
                         dataset.experiments['MRG-059-ZO-1-1'],
                         dataset.experiments['MRG-059-ZO-8-1'],
                         (dataset.experiments['MRG-059-ZN-10-1'], 1),
                         (dataset.experiments['MRG-059-ZN-9-1'], 1),
                         (dataset.experiments['MRG-059-ZN-8-1'], 1),
                         (dataset.experiments['MRG-059-ZO-9-1'], 1),
                         dataset.experiments['MRG-059-ZO-3-1'],
                         (dataset.experiments['MRG-059-ZN-14-1'], 0.1),
                         dataset.experiments['MRG-059-ZN-13-1'],
                         dataset.experiments['MRG-059-ZN-11-1']]
    

    # model.experiments = [dataset.experiments['MRG-059-ZN-10-1']]

    ### Fixed based on substrate model solution
    # model.fixed_rate_constants = {
    #     'k1': 1.752e-04,
    #     'k7': 7.568e-03
    # }

    ### Intermediate case, ratio as in substrate model
    # model.fixed_rate_constants = {
    #     'k1': 1.00e-03,
    #     'k7': 4.32e-02
    # }

    model.fixed_rate_constants = {
        'k8': 1/650e-9
    }

    # model.rate_constants_to_optimize = {'k1': (0.0, 10.0),
    #                                     'k2': (0.0, 10.0),
    #                                     'k3': (0.0, 10.0),
    #                                     'k4': (0.0, 10.0),
    #                                     'k5': (0.0, 10.0),
    #                                     'k6': (0.0, 10.0)} 
    
    # model.rate_constants_to_optimize = {'k1': (1E-4, 1E-2),
    #                                     'k2': (1E-1, 1E+1),
    #                                     'k3': (1E-4, 1E-2),
    #                                     'k4': (1E-4, 1E-2),
    #                                     'k5': (1E-3, 1E-1),
    #                                     'k6': (1E-4, 1E-2),
    #                                     'k7': (1E-4, 1E-2)} 

    # Bounds for model with irradiation power and toy extinction coefficient
    # model.rate_constants_to_optimize = {'k1': (5E-2, 5E-1),
    #                                     'k2': (1E-1, 1E+1),
    #                                     'k3': (1E-3, 1E-1),
    #                                     'k4': (1E-3, 1E-1),
    #                                     'k5': (1E-3, 5E-1),
    #                                     'k6': (1E-3, 5E-1)} 
    
    # Bounds for model with photon flux and absorption cross sections
    # model.rate_constants_to_optimize = {'k1': (1E-1, 1E-0),
    #                                     'k2': (1E-1, 1E-0),
    #                                     'k3': (1E-3, 1E-1),
    #                                     'k4': (1E-3, 1E-1),
    #                                     'k5': (1E-3, 5E-1),
    #                                     'k6': (1E-3, 5E-1)} 
    
    # Bounds for model with separate Ru(II) excitation 
    # model.rate_constants_to_optimize = {'k1': (1E-1, 1E-0),
    #                                     'k2': (1E-1, 1E-0),
    #                                     'k3': (1E-3, 1E-1),
    #                                     'k4': (1E-3, 1E-1),
    #                                     'k5': (1E-3, 5E-1),
    #                                     'k6': (1E-3, 5E-1),
    #                                     'k7': (1E+1, 1E+4)}
    
    ### Bounds for king model with [RuII-ex] decay
    model.rate_constants_to_optimize = {'k1': (1E-1, 1E-0),
                                        'k2': (1E-1, 1E-0),
                                        'k3': (1E-3, 1E-1),
                                        'k4': (1E-3, 1E-1),
                                        'k5': (1E-3, 5E-1),
                                        'k6': (1E-3, 5E-1),
                                        'k7': (1E+0, 6E+1)}  
    
    # model.rate_constants_to_optimize = {'k1': (1E-1, 1E-0),
    #                                     'k2': (1E-1, 1E-0),
    #                                     'k3': (1E-3, 1E-1),
    #                                     'k4': (1E-3, 1E-1),
    #                                     'k5': (1E-3, 5E-1),
    #                                     'k6': (1E-3, 5E-1),
    #                                     'k7': (1E+0, 8E+0)}  
        

        # model.result.x = np.array([9.734e-02,
        #                        1.659e+00,
        #                        9.588e-03,
        #                        8.055e-03,
        #                        1.314e-02,
        #                        3.367e-02])
    


    # model.rate_constants_to_optimize = {
    #                                     'k2': (1E-1, 1E+1),
    #                                     'k3': (1E-4, 1E-2),
    #                                     'k4': (1E-4, 1E-2),
    #                                     'k5': (1E-3, 1E-1),
    #                                     'k6': (1E-4, 1E-2)
    #                                     } 




    # model.data_to_be_fitted = {
    #     '[B]': 'time_series_data.data_reaction',
    #     '[Cat-Dimer]': 'time_series_data.y_fit'
    # }
    
    # model.data_to_be_fitted = {
    #     '[O2]': {'x': 'time_series_data.time_reaction',
    #              'y': 'time_series_data.data_reaction_molar'}
    # }


    # model.data_to_be_fitted = {
    #     '[O2]': {'x': 'time_series_data.time_reaction',
    #              'y': 'time_series_data.data_reaction'}
    # }

    model.data_to_be_fitted = {
        '[O2]': {'x': 'time_series_data.x_diff',
                 'y': 'time_series_data.y_diff'}
        }
    

    model.initial_conditions = {
            '[S2O8]': 'experiment_metadata.oxidant_concentration_uM',
            '[RuII]': 'experiment_metadata.ru_concentration_uM'
        }
    
    # model.other_multipliers = {
    #         'hv': 'experiment_metadata.power_output',
    #         'epsilon': 'experiment_metadata.extinction_coefficient'
    #     }
    
    # model.other_multipliers = {
    #         'hv': 'experiment_metadata.photon_flux',
    #         'sigma_RuII': 'experiment_metadata.sigma_RuII',
    #         'sigma_RuIII': 'experiment_metadata.sigma_RuIII' 
    #     }
    
    model.other_multipliers = {
        'pathlength': 2.25,
        'photon_flux': 'experiment_metadata.photon_flux',
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

    model.times = {
            'times': 'time_series_data.time_reaction'
        }

    model.loss_function = square_loss_ydiff
    #model.loss_function = robust_loss_ydiff
    #model.loss_function = square_loss_time_series

    # model.x0 = np.array([9.734e-02,
    #                      1.659e+00,
    #                      9.588e-03,
    #                      8.055e-03,
    #                      1.314e-02,
    #                      3.367e-02])

    #model.optimize()
    #model.optimize_dual_annealing()
    # model.minimize(x0 = np.array([9.734e-02,
    #                                 1.659e+00,
    #                                 9.588e-03,
    #                                 8.055e-03,
    #                                 1.314e-02,
    #                                 3.367e-02,
    #                                 1.000E+1]))

    model.result = SimpleNamespace()

    # # Ru conc model - king model
    # Fun: 21.69570383156908 (for 0.1 weights on extreme experiments)
    # model.result.x = np.array([9.734e-02,
    #                            1.659e+00,
    #                            9.588e-03,
    #                            8.055e-03,
    #                            1.314e-02,
    #                            3.367e-02])


    ### Ru conc model - new king?
    # Fun: 4.020374330260097 (on Ru experiments)
    # Fun: 19.58325743988357 (for 0.1 weights on extreme experiments)
    # model.result.x = np.array([3.479e-01,
    #                            3.100e+00,
    #                            3.943e-02,
    #                            1.068e-02,
    #                            1.135e-02,
    #                            2.910e-02])

    ### Ru conc model - with photon flux + extinction coefficients
    # Fun = 20.55440694082596
    # model.result.x = np.array([5.801e-01,
    #                            1.000e+00,
    #                            4.001e-03,
    #                            2.191e-03,
    #                            2.025e-02,
    #                            7.603e-03])

    ### Ru conc model - with separate Ru(II) excitation

    # model.result.x = np.array([9.721e-01,
    #                            9.990e-01,
    #                            3.840e-03,
    #                            2.528e-03,
    #                            2.025e-02,
    #                            8.189e-03,
    #                            1.524e+02])
                       




    # # Substrate model
    # model.result.x = np.array([2.013e-04,
    #                            1.006e+00,
    #                            8.262e-03,
    #                            3.431e-04,
    #                            2.905e-02,
    #                            9.982e-02,
    #                            9.971e-02])

    # [ 1.752e-04  1.809e-01  1.615e-03  1.000e-04  6.755e-02  1.000e-02  7.568e-03]
    # [ 1.752e-04  1.809e-01  1.615e-03  1.000e-04  6.755e-02  1.000e-02  7.568e-03]

    # Ru and substrate model
    # model.result.x = np.array([9.928e-03,
    #                            8.371e-01,
    #                            9.842e-03,
    #                            3.439e-03,
    #                            1.657e-02,
    #                            9.827e-03,
    #                            2.611e-04])

    ### Ru and substrate model, fixed k1 and k7 based on substrate model
    # model.result.x = np.array([4.101e-01,
    #                            3.434e-03,
    #                            4.294e-03,
    #                            2.650e-02,
    #                            9.754e-03])

    ### Ru and substrate model, fixed k1 and k7 based on ratio from substrate model
    # model.result.x = np.array([4.731e-01,
    #                            4.573e-03,
    #                            3.369e-03,
    #                            2.345e-02,
    #                            8.552e-03])


    ### Ru + substrate + Irradiation power model

    # Fun = 24.967884116691593 for 0.1 weights on extreme experiments
    # model.result.x = np.array([7.613e-02,
    #                            1.383e+00,
    #                            8.109e-03,
    #                            1.448e-02,
    #                            1.408e-02,
    #                            4.243e-02])

    ### bounds pushed for Ru + substrate  + Irradiation power model
    # [ 9.108e-02  1.003e+00  1.000e-02  1.000e-02
    #                     1.730e-02  1.338e-02]

    # [ 3.058e-01  1.200e+00  2.337e-02  1.030e-02
    #                     1.963e-02  5.113e-03]

    # Fun = 18.276 for 0.1 weights on extreme experiments
    # model.result.x = np.array([2.454e-01,
    #                            1.726e+00,
    #                            3.057e-02,
    #                            6.410e-03,
    #                            1.475e-02,
    #                            1.345e-02])

    # Fun = 19.281776452283587 for 0.1 weight on extreme experiments
    # model.result.x = np.array([8.088e-01,
    #                            9.978e-01,
    #                            6.566e-03,
    #                            2.103e-03,
    #                            2.278e-02,
    #                            4.029e-03])

    # Fun = 19.375732732820012 - new king model
    # model.result.x = np.array([9.644e-01,
    #                             9.983e-01,
    #                             6.406e-03,
    #                             2.428e-03,
    #                             2.293e-02,
    #                             4.493e-03,
    #                             9.092e+01])
                        
    ### Model with accurate absorption modelling, 0.1 weights on extreme experiments, king model as of 20.06.2025
            # fun: 19.297290352982493
    # model.result.x = np.array([8.037e-01,
    #                            9.894e-01,
    #                            6.578e-03,
    #                            2.223e-03,
    #                            2.350e-02,
    #                            5.004e-03,
    #                            8.685e+02])
          

    ### Model with accurate absorption modelling, 1 weights on extreme experiments
        #    fun: 39.728328723659075
        #            x: [ 1.236e-01  1.000e+00  8.621e-03  6.974e-03
        #                 3.075e-02  1.000e-03  7.220e+02]      


    ### model with decay of [RuII-ex] back to ground state, 0.1 weights on extreme experiments
    # fun: 19.312020374196262
    # [5.647e-01  9.960e-01  7.019e-03  2.492e-03
    #                     2.453e-02  3.662e-03  8.276e+03]


    ### model with deacy of [RuII-ex] back to ground state, 0.1 weights, k7 bound to 6e+1
    # fun: 20.499670751421295    
    # model.result.x = np.array([9.568e-01,
    #                            9.638e-01,
    #                            8.715e-03,
    #                            2.769e-03,
    #                            2.937e-02,
    #                            3.670e-03,
    #                            5.994e+01])

    ### model with deacy of [RuII-ex] back to ground state, 0.1 weight only on irradiance extreme, k7 bound to 6e+1, king model as of 24.06.2025
    # fun:  21.146351667114644
    model.result.x = np.array([9.995e-01,
                        9.886e-01,
                        7.407e-03,
                        3.437e-03,
                        2.739e-02,
                        4.762e-03,
                        5.918e+01])

    ### model with model with deacy of [RuII-ex] back to ground state, 0.1 weight only on irradiance extreme, k7 bound to 8e+0
    #  fun: 24.927777855276123
    # model.result.x = np.array([9.865e-01,
    #                            9.947e-01,
    #                            8.748e-03,  
    #                            9.838e-03,
    #                            5.942e-02, 
    #                            4.959e-03,  
    #                            7.792e+00])






    # model.result = dataclass()
    # model.result.x = np.array([9.993e+00, 5.243e-06, 3.213e+00, 7.569e+00, 1.059e-01, 1.473e-02])
    # model.result.x = np.array([9.999e+00, 9.547e+00, 8.560e-02, 1.864e-02, 4.518e+00, 2.681e+00])    


    fig = visualize_optimization_results(model)


    plt.show()


