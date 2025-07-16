import pandas as pd
import numpy as np
import copy
from types import SimpleNamespace
import matplotlib.image as mpimg
from matplotlib.offsetbox import OffsetImage, AnnotationBbox
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
from matplotlib.figure import Figure
from matplotlib import rcParams
rcParams['font.family'] = 'sans-serif'
rcParams['font.sans-serif'] = ['Arial']
rcParams['font.size'] = 12
rcParams['mathtext.fontset'] = 'custom'
rcParams['mathtext.rm'] = 'Arial'
rcParams['mathtext.it'] = 'Arial:italic'
rcParams['mathtext.bf'] = 'Arial:bold'

import pprint as pp

from hte_streamlit.experiments_database import ExperimentalDataset
from hte_streamlit.reaction_ODE import calculate_excitations_per_second_competing
from hte_streamlit.fitting_ODE import Fitting_Model, objective_function, resolve_experiment_attributes, square_loss_ydiff

def get_attribute_range(experiments, attribute_path):
    """
    Get the maximum and minimum values of an attribute across a set of experiments.
    
    Parameters
    ----------
    experiments : list
        List of experiment objects or tuples (experiment, weight)
    attribute_path : str
        Dot-separated attribute path (e.g., 'experiment_metadata.power_output')
        
    Returns
    -------
    tuple
        (min_value, max_value) of the attribute across all experiments
        
    Raises
    ------
    ValueError
        If no experiments provided or attribute not found in any experiment
    AttributeError
        If the attribute path is invalid for any experiment
    """
    if not experiments:
        raise ValueError("No experiments provided")
    
    values = []
    
    for experiment_entry in experiments:
        # Handle both plain experiments and (experiment, weight) tuples
        if isinstance(experiment_entry, tuple):
            experiment, weight = experiment_entry
        else:
            experiment = experiment_entry
        
        # Create a template dict to resolve the attribute
        template_dict = {'value': attribute_path}
        
        try:
            resolved_dict = resolve_experiment_attributes(template_dict, experiment)
            value = resolved_dict['value']
            values.append(value)
        except AttributeError as e:
            exp_name = getattr(experiment, 'experiment_metadata', {})
            exp_name = getattr(exp_name, 'experiment_name', 'Unknown')
            raise AttributeError(f"Could not resolve '{attribute_path}' for experiment {exp_name}: {e}")
    
    if not values:
        raise ValueError(f"No valid values found for attribute '{attribute_path}'")
    
    return min(values), max(values)

def average_reproductions(df_cleaned, 
                          rate_column = 'max rate ydiff'):

    df_cleaned = df_cleaned.dropna(subset=[rate_column])

    grouped_df = df_cleaned.groupby('group').agg(
        mean_rate=(rate_column, 'mean'),
        min_rate=(rate_column, 'min'),
        max_rate=(rate_column, 'max')
    ).reset_index()

    base_experiment_df = df_cleaned[df_cleaned['Experiment'].str.endswith('-1')].merge(grouped_df, on='group')

    base_experiment_df['Experiment'] = base_experiment_df['group']
    base_experiment_df = base_experiment_df.drop(columns = ['group', 'rate', 'annotations'])

    return base_experiment_df

def filter_data(df, 
                experiment_list, 
                prefix="MRG-059-"):
    """
    Complete experiment names with prefix and return a DataFrame with only those experiments.
    
    Parameters
    ----------
    df : pandas.DataFrame
        DataFrame containing all experiments
    experiment_list : list
        List of experiment short names (e.g., ['ZN-10', 'ZN-9'])
    prefix : str, default="MRG-059-"
        Prefix to add to each experiment code
        
    Returns
    -------
    pandas.DataFrame
        Filtered DataFrame containing only the specified experiments
    """

    full_experiment_names = [f"{prefix}{code}" for code in experiment_list]
    mask = df['Experiment'].isin(full_experiment_names)
    filtered_df = df[mask].copy()
    
    return filtered_df

def create_experiment_list(dataset, short_codes, prefix='MRG-059-', ending='-1'):
    """
    Create a list of experiment objects from short codes.
    
    Parameters
    ----------
    dataset : ExperimentalDataset
        The dataset containing experiments
    short_codes : list
        List of short experiment codes (e.g., ['ZO-3', 'ZN-14'])
    prefix : str
        Prefix to add to each code
    ending : str
        Ending to add to each code
        
    Returns
    -------
    list
        List of experiment objects
    """
    full_names = [f"{prefix}{code}{ending}" for code in short_codes]
    return [dataset.experiments[name] for name in full_names]

def plot_max_rate_data(df_averaged,
              experiment_list, 
              column_name, 
              results = None,
              legend = True, 
              ax = None, 
              fig = None, 
              axis_label = None, 
              color = 'black'):
    '''
    Plot maximum rate data with error bars for specified experiments.

    Parameters
    ----------
    df_averaged : pandas.DataFrame
        DataFrame containing averaged experimental data
    experiment_list : list
        List of experiment short names (e.g., ['ZN-10', 'ZN-9'])
    column_name : str
        Column name to use for x-axis values (e.g., 'c([Ru(bpy(3]Cl2) [M]')
    legend : bool, default True 
        Whether to display the legend
    ax : matplotlib.axes.Axes, optional
        Axes to plot on; if None, a new figure and axes will be created
    fig : matplotlib.figure.Figure, optional
        Figure object; if None, a new figure will be created
    axis_label : str, optional
        Label for the x-axis; if None, column_name will be used
    color : str, default 'black'
        Color for the plot markers and error bars

    Returns
    -------
    fig : matplotlib.figure.Figure
        The figure object containing the plot
    ax : matplotlib.axes.Axes
        The axes object containing the plot
    '''

    filtered_df= filter_data(df_averaged, experiment_list)

    data_label_added = False

    for index, row in filtered_df.iterrows():
        x = row[column_name]
        y = row['mean_rate']
        yerr = [[row['mean_rate'] - row['min_rate']], [row['max_rate'] - row['mean_rate']]]

        label = 'Data' if not data_label_added else None
  
        ax.errorbar(x, y, yerr = yerr, fmt='o', ecolor=color, 
                 capsize=3, capthick=1, markersize=5,label = label, color = color) 
        
        data_label_added = True

    if results is not None:
        ax.plot(results[:,0], results[:, 1], color = color, linewidth = 2, label = 'Kinetic model')
     
    ax.set_xlabel(axis_label, color = color)
    ax.set_ylabel(r'Max. rate / $\mathrm{\mu M(O_2) \, s^{-1}}$')

    if legend:
        ax.legend()
        #ax.legend(title='Base Experiment', bbox_to_anchor=(1.02, 1.01), loc='upper left')

    return fig, ax


def plot_model_fit(model: Fitting_Model,
                   model_results: dict,
                   attribute_of_interest: str,
                   axis_label,
                   ax = None, 
                   fig: Figure = None,
                   colormap = 'viridis',
                   y_axis_label = r'Rate / $\mathrm{\mu M(O_2) \, s^{-1}}$',
                   unit = None) -> Figure:
    """
    Plot model fit with colors based on column_of_interest values
    
    Parameters
    ----------
    model : Fitting_Model
        The fitted model
    attribute_of_interest : str
        Attribute path to use for color mapping and sorting
    axis_label : str
        Label for x-axis
    ax : matplotlib.axes.Axes, optional
        Axes to plot on
    fig : matplotlib.figure.Figure, optional
        Figure object
    colormap : str, default 'viridis'
        Colormap to use for coloring experiments
    y_axis_label : str
        Label for y-axis
    unit : str, optional
        Unit string for legend labels
    """

    # Get attribute values for all experiments and sort them
    experiment_values = []
    for experiment in model.experiments:        
        experimental_value = resolve_experiment_attributes({'Value': attribute_of_interest}, experiment)['Value']
        experiment_values.append((experimental_value, experiment))
    
    # Sort experiments by their attribute values
    experiment_values.sort(key=lambda x: x[0])
    
    # Create evenly spaced color positions
    num_experiments = len(experiment_values)
    if num_experiments == 1:
        color_positions = [0.5]  # Single experiment gets middle color
    else:
        color_positions = [i / (num_experiments - 1) for i in range(num_experiments)]
    
    # Create colormap
    cmap = plt.get_cmap(colormap)

    # Plot experiments in sorted order with evenly spaced colors
    for idx, ((experimental_value, experiment), color_pos) in enumerate(zip(experiment_values, color_positions)):

        experimental_data = resolve_experiment_attributes(model.data_to_be_fitted, experiment)
        exp_name = experiment.experiment_metadata.experiment_name

        # Use evenly spaced color position instead of normalized value
        color = cmap(color_pos)

        for species, data in experimental_data.items():
            model_data = model_results[exp_name][species]
            
            ax.scatter(data['x'], data['y'], color=color, s = 10)
            ax.plot(data['x'], model_data, color=color, linewidth=2)
            ax.plot([], [], color=color, marker='o', linestyle='-', 
                   label=f'{experimental_value:.0f} {unit}' if unit else f'{experimental_value:.0f}')

    ax.set_title(axis_label)        
    ax.set_xlabel('Time / s')
    ax.set_ylabel(y_axis_label)
    ax.legend()

    return fig

class KineticPlotter:
    # Class-level attributes for shared data
    _shared_df = None
    _shared_model = None
    _shared_dataset = None
    _shared_reference_experiment = None
    
    @classmethod
    def set_shared_data(cls, 
                        df_averaged=None, 
                        kinetic_model: Fitting_Model = None,
                        dataset: ExperimentalDataset = None,
                        reference_experiment=None):
        """Set shared data for all KineticPlotter instances"""
        if df_averaged is not None:
            cls._shared_df = df_averaged
        if kinetic_model is not None:
            cls._shared_model = kinetic_model
        if dataset is not None:
            cls._shared_dataset = dataset
        if reference_experiment is not None:
            cls._shared_reference_experiment = reference_experiment

    def __init__(self, 
                 experiment_list, 
                 column_of_interest, 
                 attribute_of_interest,
                 axis_label = None, 
                 color = 'black',
                 df_averaged = None,
                 kinetic_model: Fitting_Model = None,
                 dataset: ExperimentalDataset = None,
                 unit = None,
                 datapoints = 30,
                 reference_experiment = None):
        
        # Use instance-specific data if provided, otherwise use shared data
        self.df_averaged = df_averaged or self._shared_df
        self.dataset = dataset or self._shared_dataset
        self.reference_experiment = reference_experiment or self._shared_reference_experiment
        
        if self.df_averaged is None or self.dataset is None:
            raise ValueError("Either provide data directly or set shared data with set_shared_data()")
        
        self.experiment_list = experiment_list
        self.column_of_interest = column_of_interest
        self.attribute_of_interest = attribute_of_interest
        self.axis_label = axis_label or column_of_interest
        self.color = color
        self.unit = unit

        # Create independent copy of kinetic model
        base_model = kinetic_model or self._shared_model
        if base_model is not None:
            self.kinetic_model = copy.deepcopy(base_model)
            self.kinetic_model.experiments = create_experiment_list(
                self.dataset, self.experiment_list)
        else:
            self.kinetic_model = None

        self.temp_model = copy.deepcopy(self.kinetic_model)
        self.temp_model.experiments = self.generate_parameter_sweep_experiments(datapoints)

        error, self.experiment_model_results, self.time_series = objective_function(
            self.kinetic_model.result.x, self.kinetic_model, return_full='All')

        error, self.sweep_model_results, self.time_series_sweep = objective_function(
            self.temp_model.result.x, self.temp_model, return_full='All')
        
        self.experiment_results = self.get_sweep_results(self.kinetic_model,
                                                        self.experiment_model_results)
        
        self.sweep_results = self.get_sweep_results(self.temp_model, 
                                                    self.sweep_model_results)   
    
    def generate_parameter_sweep_experiments(self, 
                                             datapoints = 20,
                                             power_space = True):
        """
        Generate synthetic experiments by varying the attribute of interest.
        
        Parameters
        ----------
        datapoints : int, default 20
            Number of synthetic experiments to generate
        """
        
        min_value, max_value = get_attribute_range(self.kinetic_model.experiments, 
                                    self.attribute_of_interest)
        
        parameter_values = np.linspace(min_value, max_value, datapoints)

        power = 4 # Adjust this: higher values = more concentration at start
        t = np.linspace(0, 1, datapoints)
        quadratic_space = min_value + (max_value - min_value) * (t ** power)

        if power_space is True:
            parameter_values = quadratic_space

        synthetic_experiments = []
        
        for i, param_value in enumerate(parameter_values):

            synthetic_experiment = copy.deepcopy(self.reference_experiment)
            self._set_nested_attribute(synthetic_experiment, self.attribute_of_interest, param_value)
            
            # Update experiment name to make it unique
            original_name = synthetic_experiment.experiment_metadata.experiment_name
            synthetic_experiment.experiment_metadata.experiment_name = f"{original_name}_sweep_{i:03d}"
            
            synthetic_experiments.append(synthetic_experiment)
        
        # Store the synthetic experiments
        self.synthetic_experiments = synthetic_experiments
        self.sweep_parameter_values = parameter_values


        return synthetic_experiments
        
    def _set_nested_attribute(self, obj, attr_path, value):
        """
        Set a nested attribute using dot notation.
        
        Parameters
        ----------
        obj : object
            The object to modify
        attr_path : str
            Dot-separated attribute path (e.g., 'experiment_metadata.ru_concentration_uM')
        value : Any
            The value to set
        """
        attributes = attr_path.split('.')
        current_obj = obj
        
        # Navigate to the parent of the final attribute
        for attr in attributes[:-1]:
            current_obj = getattr(current_obj, attr)
        
        # Set the final attribute
        setattr(current_obj, attributes[-1], value)

    def get_sweep_results(self, model, results_dict):

        results = []

        for experiment in model.experiments:

            experiment_results = []

            exp_name = experiment.experiment_metadata.experiment_name
            parameter_value = resolve_experiment_attributes(
                {'Value': self.attribute_of_interest}, 
                experiment)['Value']
            
            experiment_results.append(parameter_value)

            experimental_data = resolve_experiment_attributes(model.data_to_be_fitted, experiment)
            
            for species, data in experimental_data.items():
                model_data = results_dict[exp_name][species]
                experiment_results.append(np.amax(model_data))

            results.append(experiment_results)

        results.sort(key=lambda x: x[0])
        
        return np.asarray(results)
                                                        
    def plot_max_rates(self, 
                       ax = None, 
                       fig = None, 
                       sweep_results = True):

        if ax is None:
            fig, ax = plt.subplots()

        if sweep_results is True:
            results = self.sweep_results
        elif sweep_results is False:
            results = self.experiment_results
        else:
            results = None

        plot_max_rate_data(self.df_averaged,
                            self.experiment_list,
                            self.column_of_interest, 
                            results = results,
                            legend = True, 
                            ax = ax, 
                            fig = fig, 
                            axis_label = self.axis_label, 
                            color = self.color)
        
    def plot_kinetic_model_fit(self, ax = None, fig = None):

        if ax is None:
            fig, ax = plt.subplots()
        
        plot_model_fit(self.kinetic_model, 
                       self.experiment_model_results,
                       self.attribute_of_interest,
                       self.axis_label,
                       ax = ax,
                       fig = fig,
                       unit = self.unit)
        
    def plot_time_series(self, experiment,
                         exclude_species = [],
                         ylim = None,
                         ax = None, fig = None,
                         show_experimental_data = True,
                         plot_order = []):

        if ax is None:
            fig, ax = plt.subplots()

        solution = self.time_series[experiment.experiment_metadata.experiment_name]
        times = resolve_experiment_attributes(self.kinetic_model.times, experiment)['times']
        species = self.kinetic_model.species

        colors = plt.cm.Paired(range(len(species)))
        color_map = dict(zip(species, colors))

        if show_experimental_data:
            ax.plot(times, 
                    experiment.time_series_data.data_reaction, 
                    'o', alpha = 0.3, label = '[O2] Experimental data',
                    color = 'grey')
            

        # Plot species in desired order
        for spec in plot_order:
            if spec in species and spec not in exclude_species:
                i = species.index(spec)
                ax.plot(times, solution[:, i], label=spec, 
                        color = color_map[spec])

        # Plot any remaining species not in the defined order
        for i, spec in enumerate(species):
            if spec not in exclude_species and spec not in plot_order:
                ax.plot(times, solution[:, i], label=spec, 
                        color = color_map[spec])
        

        # for i, spec in enumerate(species):
        #     if spec not in exclude_species:
        #         ax.plot(times, solution[:, i], label=spec, 
        #                 color = color_map[spec])



        ax.set_xlabel('Time / s')
        ax.set_ylabel(r'Concentration /  $\mathrm{\mu M}$')

        if ylim is not None:
            ax.set_ylim(ylim)

        ax.legend()


def plot_max_rates_parameters(Ru_model, 
                              OX_model,
                              Irradiation_model,
                              pH_model):
    
    fig, ax = plt.subplots(nrows = 2, ncols = 3, figsize=(11, 5))
    fig.subplots_adjust(left = 0.08, right = 0.8, top = 0.93, bottom = 0.12,
                        wspace = 0.4, hspace = 0.4)
    
    Ru_model.plot_max_rates(ax = ax[0,0], fig = fig)
    OX_model.plot_max_rates(ax = ax[0,1], fig = fig)
    Irradiation_model.plot_max_rates(ax = ax[1,0], fig = fig, sweep_results = False)
    pH_model.plot_max_rates(ax = ax[1,1], fig = fig, sweep_results = None)


    #bbox_props = dict(boxstyle="square,pad=0.1", facecolor='white', edgecolor='black', linewidth=1)

    ax[0, 0].text(-0.35, 1.04, 'A', transform=ax[0, 0].transAxes, fontsize=22, fontweight='bold')
    ax[0, 1].text(-0.35, 1.04, 'B', transform=ax[0, 1].transAxes, fontsize=22, fontweight='bold')
    ax[1, 0].text(-0.35, 1.04, 'C', transform=ax[1, 0].transAxes, fontsize=22, fontweight='bold')
    ax[1, 1].text(-0.35, 1.04, 'D', transform=ax[1, 1].transAxes, fontsize=22, fontweight='bold')
    ax[0, 2].text(-0.35, 1.04, 'E', transform=ax[0, 2].transAxes, fontsize=22, fontweight='bold')      

    img = mpimg.imread('/Users/jacob/Documents/Water_Splitting/Projects/HTE_Photocatalysis/Ru_Reaction_Scheme.png')
    imagebox = OffsetImage(img, zoom=0.145) 
    ab = AnnotationBbox(imagebox, (0.85, -0.2), frameon=False, 
                        xycoords=ax[0,2].transAxes)
    ax[0,2].add_artist(ab)
    
    ax[0,2].axis('off')
    ax[1,2].axis('off')

    return fig

def plot_kinetic_fit_parameters(Ru_model,
                                OX_model,
                                Irradiation_model):
    
    fig = plt.figure(figsize=(11, 8))
    
    ax0 = plt.subplot2grid((2, 4), (0, 0), colspan=2)  # Top left, spans 2 columns
    ax1 = plt.subplot2grid((2, 4), (0, 2), colspan=2)  # Top right, spans 2 columns
    ax2 = plt.subplot2grid((2, 4), (1, 1), colspan=2)  # Bottom center, spans 2 columns (starts at column 1)
    
    fig.subplots_adjust(left=0.08, right=0.98, top=0.95, bottom=0.07,
                        wspace=0.6, hspace=0.3)

    Ru_model.plot_kinetic_model_fit(ax=ax0, fig=fig)
    OX_model.plot_kinetic_model_fit(ax=ax1, fig=fig)
    Irradiation_model.plot_kinetic_model_fit(ax=ax2, fig=fig)

    ax0.text(-0.15, 1.0, 'A', transform=ax0.transAxes, fontsize=22, fontweight='bold')
    ax1.text(-0.15, 1.0, 'B', transform=ax1.transAxes, fontsize=22, fontweight='bold')
    ax2.text(-0.15, 1.0, 'C', transform=ax2.transAxes, fontsize=22, fontweight='bold')

    return fig

def plot_time_series(Ru_model, reference_experiment):

    fig, ax = plt.subplots(1, 2, figsize=(10, 6))
    fig.tight_layout()
    fig.subplots_adjust(left = 0.08, right = 0.98, top = 0.9, bottom = 0.15,
                        wspace= 0.3)

    Ru_model.plot_time_series(reference_experiment,
                              exclude_species= ['[S2O8]', '[SO4]'], 
                              fig = fig, ax = ax[0],
                              plot_order = ['[O2]', '[H2O2]', '[Ru-Dimer]', '[Inactive]', '[RuII]', '[RuIII]'],)
    
    Ru_model.plot_time_series(reference_experiment,
                                exclude_species= ['[S2O8]', '[SO4]', '[O2]', '[H2O2]'], 
                                fig = fig, ax = ax[1], show_experimental_data = False,
                                plot_order = ['[RuII]', '[RuIII]', '[Ru-Dimer]', '[Inactive]'])
    
    ax[0].text(-0.15, 1.0, 'A', transform=ax[0].transAxes, fontsize=22, fontweight='bold')
    ax[1].text(-0.15, 1.0, 'B', transform=ax[1].transAxes, fontsize=22, fontweight='bold')

    return fig


    
if __name__ == '__main__':

    dataset = ExperimentalDataset.load_from_hdf5('data/250608_HTE.h5')
    dataset.update_reaction_data()
    df = dataset.overview_df

    df['c([Ru(bpy(3]Cl2) [M]'] *= 1e6 # convert to uM
    df['c(Na2S2O8) [M]'] *= 1e6 # convert uM
    df_averaged = average_reproductions(df)

    reference_experiment = dataset.experiments['MRG-059-ZO-1-1']

    OX_experiments = ['ZN-10', 'ZN-9', 'ZN-8', 'ZO-1', 'ZO-9']
    Irradiation_experiments = ['ZO-1', 'ZO-3', 'ZN-14', 'ZN-13', 'ZN-11']
    Ru_experiments = ['ZO-1', 'ZO-8', 'ZO-2', 'ZN-7', 'ZN-4', 'ZN-3', 'ZN-2', 'ZN-1']
    pH_experiments = ['ZO-1', 'ZO-7', 'ZO-6', 'ZO-5', 'ZO-4']

    viridis = plt.cm.viridis
    colors = [viridis(i) for i in np.linspace(0, 0.85, 4)]

    # model = Fitting_Model(['[RuII] > [RuII-ex], k1 ; hv, sigma_RuII',
    #                         '[RuII-ex] + [S2O8] > [RuIII] + [SO4], k7',
    #                         '[RuIII] > [H2O2] + [RuII], k2 ; hv, sigma_RuIII',
    #                         '2 [RuIII] > [Ru-Dimer], k3',
    #                         '2 [RuIII] + [Ru-Dimer] > 2 [Ru-Dimer], k4',
    #                         '[H2O2] > [O2], k5',
    #                         '[RuIII] > [Inactive], k6'])
    
    ### Correct absorption
    # model = Fitting_Model(['[RuII] > [RuII-ex], k1 ; hv_functionA',
    #                         '[RuII-ex] + [S2O8] > [RuIII] + [SO4], k7',
    #                         '[RuIII] > [H2O2] + [RuII], k2 ; hv_function_B',
    #                         '2 [RuIII] > [Ru-Dimer], k3',
    #                         '2 [RuIII] + [Ru-Dimer] > 2 [Ru-Dimer], k4',
    #                         '[H2O2] > [O2], k5',
    #                         '[RuIII] > [Inactive], k6'])

    model = Fitting_Model(['[RuII] > [RuII-ex], k1 ; hv_functionA',
                            '[RuII-ex] > [RuII], k8',
                            '[RuII-ex] + [S2O8] > [RuIII] + [SO4], k7',
                            '[RuIII] > [H2O2] + [RuII], k2 ; hv_function_B',
                            '2 [RuIII] > [Ru-Dimer], k3',
                            '2 [RuIII] + [Ru-Dimer] > 2 [Ru-Dimer], k4',
                            '[H2O2] > [O2], k5',
                            '[RuIII] > [Inactive], k6'])
    
    model.rate_constants_to_optimize = {'k1': (1E-1, 1E-0),
                                        'k2': (1E-1, 1E-0),
                                        'k3': (1E-3, 1E-1),
                                        'k4': (1E-3, 1E-1),
                                        'k5': (1E-3, 5E-1),
                                        'k6': (1E-3, 5E-1),
                                        'k7': (1E-0, 1E+3)} 
    
    model.fixed_rate_constants = {
        'k8': 1/650e-9
    }

    model.data_to_be_fitted = {
        '[O2]': {'x': 'time_series_data.x_diff',
                 'y': 'time_series_data.y_diff'}
        }

    model.initial_conditions = {
            '[S2O8]': 'experiment_metadata.oxidant_concentration_uM',
            '[RuII]': 'experiment_metadata.ru_concentration_uM'
        }

    model.other_multipliers = {
            'hv': 'experiment_metadata.photon_flux',
            'sigma_RuII': 'experiment_metadata.sigma_RuII',
            'sigma_RuIII': 'experiment_metadata.sigma_RuIII' 
        }
    
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

    model.result = SimpleNamespace()
    # model.result.x = np.array([9.644e-01,
    #                             9.983e-01,
    #                             6.406e-03,
    #                             2.428e-03,
    #                             2.293e-02,
    #                             4.493e-03,
    #                             9.092e+01])
    ### Correct absorption
    # model.result.x = np.array([8.037e-01,
    #                            9.894e-01,
    #                            6.578e-03,
    #                            2.223e-03,
    #                            2.350e-02,
    #                            5.004e-03,
    #                            8.685e+02])
    ### Relaxation of [RuII-ex]
    # model.result.x = np.array([9.568e-01,
    #                            9.638e-01,
    #                            8.715e-03,
    #                            2.769e-03,
    #                            2.937e-02,
    #                            3.670e-03,
    #                            5.994e+01])

    ### Relaxation of [RuII-ex], weight 1 on extreme [Ru] conc
    model.result.x = np.array([9.995e-01,
                        9.886e-01,
                        7.407e-03,
                        3.437e-03,
                        2.739e-02,
                        4.762e-03,
                        5.918e+01])




    KineticPlotter.set_shared_data(
        df_averaged = df_averaged,
        kinetic_model = model,
        dataset = dataset,
        reference_experiment = reference_experiment
    )

    Ru_model = KineticPlotter(Ru_experiments, 
                            'c([Ru(bpy(3]Cl2) [M]', 
                            'experiment_metadata.ru_concentration_uM',
                            axis_label = r'$\mathrm{[Ru(bpy)_3]Cl_2}$ / $\mathrm{\mu M}$', 
                            color = colors[0],
                            unit = r'$\mathrm{\mu M}$')
                            
    OX_model = KineticPlotter(OX_experiments, 
                            'c(Na2S2O8) [M]', 
                            'experiment_metadata.oxidant_concentration_uM',
                            axis_label = r'$\mathrm{Na_2S_2O_8}$ / $\mathrm{\mu M}$',
                            color = colors[1],
                            unit = r'$\mathrm{\mu M}$')
                            
    Irradiation_model = KineticPlotter(Irradiation_experiments, 
                                       'Power output [W/m^2]', 
                                       'experiment_metadata.power_output',
                                        axis_label = r'Irradiance / W$\mathrm{m^{-2}}$', 
                                        color = colors[2],
                                        unit = r'W$\mathrm{m^{-2}}$')
                                       
    pH_model = KineticPlotter(pH_experiments, 
                              'pH [-]', 
                              'experiment_metadata.pH',
                                axis_label = 'pH', 
                                color = colors[3])
                             


    #fig_rate = plot_max_rates_parameters(Ru_model, OX_model, Irradiation_model, pH_model)
    #fig_rate.savefig('Figures/kinetic_max_rates.pdf', dpi = 500)


    fig_fit = plot_kinetic_fit_parameters(Ru_model, OX_model, Irradiation_model)
    fig_fit.savefig('Figures/kinetic_fit.png', dpi = 500)

    #fig_time = plot_time_series(Ru_model, reference_experiment)
    #fig_time.savefig('Figures/kinetic_time_series.pdf', dpi = 500)

            
    plt.show()
    







