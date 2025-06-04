import numpy as np
from scipy.optimize import differential_evolution
from dataclasses import dataclass

from hte_streamlit.reaction_ODE import solve_ode_system, parse_reactions
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
        if isinstance(value, dict):
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


def square_loss_time_series(model_data, experimental_data):
    '''
    Placeholder'''

    return np.sum((np.array(model_data) - np.array(experimental_data['y'])) ** 2), model_data

def square_loss_max_rate_ydiff(model_data, experimental_data):
    '''
    Placeholder'''

    model_data_max_rate_ydiff = np.amax(np.diff(model_data))

    return (model_data_max_rate_ydiff - experimental_data['y']) ** 2, model_data_max_rate_ydiff

def square_loss_ydiff(model_data, experimental_data):
    '''
    Placeholder'''

    model_data_ydiff = np.diff(model_data)

    return np.sum((model_data_ydiff - experimental_data['y']) ** 2), model_data_ydiff

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
            updating = 'deferred')

        print(self.result)

def objective_function(rate_constants_to_optimize, model: Fitting_Model, return_full = False):
    '''
    Placeholder
    '''

    rate_constants = dict(zip(model.rate_constants_to_optimize.keys(), rate_constants_to_optimize))
    rate_constants |= model.fixed_rate_constants

    total_error = 0.0 
    full_output = {}

    for experiment in model.experiments:

        full_output[experiment.experiment_metadata.experiment_name] = {}

        # Resolve the rate constants, initial conditions, other multipliers, and times
        initial_conditions = resolve_experiment_attributes(model.initial_conditions, experiment)
        other_multipliers = resolve_experiment_attributes(model.other_multipliers, experiment)
        times = resolve_experiment_attributes(model.times, experiment)

        # Resolve the data to be fitted
        data_to_be_fitted = resolve_experiment_attributes(model.data_to_be_fitted, experiment)

        # Solve the ODE system
        model_result = solve_ode_system(model.parsed_reactions,
                                        model.species,
                                        rate_constants,
                                        initial_conditions,
                                        times['times'],
                                        other_multipliers)
        
        # Calculate the error between the model and the data for each species to be fitted
        for species, data in data_to_be_fitted.items():
            idx = model.species.index(species)
            model_data = model_result[:,idx]

            error, model_data_transformed = model.loss_function(model_data, data)
            
            full_output[experiment.experiment_metadata.experiment_name][species] = model_data_transformed

            total_error += error


    if return_full:
        return total_error, full_output
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

    fig, ax = plt.subplots()

    colors = plt.rcParams['axes.prop_cycle'].by_key()['color']

    for i, experiment in enumerate(model.experiments):
        color = colors[i % len(colors)] 
        experimental_data = resolve_experiment_attributes(model.data_to_be_fitted, experiment)
        
        #for species, data in model.data_to_be_fitted.items():
        for species, data in experimental_data.items():

            ax.scatter(data['x'], data['y'], color = color)

            
            ax.plot(data['x'], model_results[experiment.experiment_metadata.experiment_name][species], color = color)
            # Then add a single legend entry for both
            ax.plot([], [], color=color, marker='o', linestyle='-', 
                    label=f'{species} - {experiment.experiment_metadata.experiment_name}')
    
    ax.legend()

    return fig



if __name__ == '__main__':

    dataset = ExperimentalDataset.load_from_hdf5('data/HTE-overview_250204.h5')
    dataset.update_reaction_data()

    # model = Fitting_Model(['[S2O8] + [RuII] > [RuIII] + [SO4], k1 ; irradiation_power',
    #                        '[S2O8] + [RuIII] > [RuIV] + [SO4], k2 ; irradiation_power',
    #                        '2 [RuIV] > [Ru-Dimer], k3',
    #                        '3 [RuIV] > [Ru-Trimer], k4',
    #                        '[RuIII] > [RuII] + [O2], k5',
    #                        '[S2O8] > [SO4], k6'])

    model = Fitting_Model(['[S2O8] + [RuII] > [RuIII] + [SO4], k1 ; irradiation_power',
                           '2 [RuIII] > [Ru-Dimer], k2',
                           '[RuIII] > [RuII] + [O2], k3',
                           '[S2O8] > [SO4], k4'])

    # model = Fitting_Model(['[Oxidant] + [Cat] > [O2] + [Cat], k1 ; irradiation_power',
    #                        '[Oxidant] > [Waste], k3',
    #                        '2 [Cat] > [Cat-Dimer], k2'])
    
    # model.experiments = [dataset.experiments['MRG-059-ZO-2-1'],
    #                      dataset.experiments['MRG-059-ZN-7-1'],
    #                      dataset.experiments['MRG-059-ZN-4-1'],
    #                      dataset.experiments['MRG-059-ZN-3-1'],
    #                      dataset.experiments['MRG-059-ZN-2-1'],
    #                      dataset.experiments['MRG-059-ZN-1-1'],
    #                      dataset.experiments['MRG-059-ZO-1-1'],
    #                      dataset.experiments['MRG-059-ZO-8-1']]
    
    model.experiments = [dataset.experiments['MRG-059-ZN-10-1'],
                         dataset.experiments['MRG-059-ZN-9-1'],
                         dataset.experiments['MRG-059-ZN-8-1'],
                         dataset.experiments['MRG-059-ZO-1-1']]

    # model.experiments = [dataset.experiments['MRG-059-ZN-10-1']]

    # model.fixed_rate_constants = {
    #     'k1': 0.1
    # }

    model.rate_constants_to_optimize = {'k1': (0.0, 10.0),
                                        'k2': (0.0, 10.0),
                                        'k3': (0.0, 10.0),
                                        'k4': (0.0, 10.0),
                                        'k5': (0.0, 10.0),
                                        'k6': (0.0, 10.0)} 

    # model.data_to_be_fitted = {
    #     '[B]': 'time_series_data.data_reaction',
    #     '[Cat-Dimer]': 'time_series_data.y_fit'
    # }
    
    # model.data_to_be_fitted = {
    #     '[O2]': {'x': 'time_series_data.time_reaction',
    #              'y': 'time_series_data.data_reaction_molar'}
    # }


    model.data_to_be_fitted = {
        '[O2]': {'x': 'time_series_data.x_diff',
                 'y': 'time_series_data.y_diff_molar'}
        }
    

    model.initial_conditions = {
            '[S2O8]': 'experiment_metadata.oxidant_concentration',
            '[RuII]': 'experiment_metadata.ru_concentration'
        }
    
    model.other_multipliers = {
            'irradiation_power': 'experiment_metadata.power_output'
        }

    model.times = {
            'times': 'time_series_data.time_reaction'
        }

    model.loss_function = square_loss_ydiff
    #model.loss_function = square_loss_time_series


    model.optimize()



    # model.result = dataclass()
    # model.result.x = np.array([9.993e+00, 5.243e-06, 3.213e+00, 7.569e+00, 1.059e-01, 1.473e-02])
    # model.result.x = np.array([9.999e+00, 9.547e+00, 8.560e-02, 1.864e-02, 4.518e+00, 2.681e+00])    


    fig = visualize_optimization_results(model)


    plt.show()




