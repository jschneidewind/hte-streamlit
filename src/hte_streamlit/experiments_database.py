import pandas as pd
import numpy as np
from pathlib import Path
import h5py
from dataclasses import dataclass, asdict, field
from typing import List, Dict
import colorsys

def import_data(file_name):
    df = pd.read_csv(file_name, low_memory = False)
    return df

def import_overview_excel(file_name):
    df = pd.read_excel(file_name, sheet_name='Tabelle1')
    return df

def get_experimental_metadata(experiment_name, metadata_df):

    row_values = metadata_df.loc[metadata_df['Experiment'] == experiment_name]

    if row_values.empty:
        raise ValueError(f"No experiment found in overview_df with name '{experiment_name}'")

    experimental_metadata = {'experiment_name': experiment_name,
                             'power_output': row_values['Power output [W/m^2]'].values[0],
                             'ru_concentration': row_values['c([Ru(bpy(3]Cl2) [M]'].values[0],
                             'oxidant_concentration': row_values['c(Na2S2O8) [M]'].values[0],
                             'buffer_concentration': row_values['c(buffer) [M]'].values[0],
                             'pH': row_values['pH [-]'].values[0],
                             'buffer_used': row_values['buffer used'].values[0],
                             'annotations': row_values['annotations'].values[0]                       
    }

    return experimental_metadata

@dataclass
class ExperimentMetadata:
    """Store experimental conditions and metadata"""
    experiment_name: str
    power_output: float
    ru_concentration: float
    oxidant_concentration: float
    buffer_concentration: float
    pH: float
    buffer_used: int 
    annotations: str = ""
    color: str = "#cc0a7c"

@dataclass
class AnalysisMetadata:
    """Store analysis metadata"""
    p: np.ndarray
    max_rate: float
    max_rate_ydiff: float
    initial_state: np.ndarray
    matrix: str
    rate_constant: float
    rxn_start: int
    rxn_end: int
    residual: np.ndarray
    idx_for_fitting: int
    
@dataclass
class DataSets:
    """Store datasets"""    
    data_corrected: np.ndarray

@dataclass
class TimeSeriesData:
    """Store time series data"""   
    time_reaction: np.ndarray
    data_reaction: np.ndarray
    y_fit: np.ndarray
    baseline_y: np.ndarray
    lbc_fit_y: np.ndarray
    full_x_values: np.ndarray
    full_y_corrected: np.ndarray 
    x_diff: np.ndarray
    y_diff: np.ndarray
    y_diff_smoothed: np.ndarray
    y_diff_fit: np.ndarray
    time_full: np.ndarray
    data_full: np.ndarray
    
@dataclass
class ExperimentalData:
    """Container for individual experiment's data"""
    time_series_data: TimeSeriesData
    experiment_metadata: ExperimentMetadata
    analysis_metadata: AnalysisMetadata
    datasets: DataSets

@dataclass
class ExperimentalDataset:
    """Container for all experimental data types and metadata for multiple experiments"""
    experiments: Dict[str, 'ExperimentalData'] = field(default_factory=dict)
    overview_df: pd.DataFrame = field(default_factory=lambda: pd.DataFrame())

    def add_experiment(self, name: str, experimental_data: 'ExperimentalData'):
        """Add an experiment to the dataset"""
        self.experiments[name] = experimental_data
        self.insert_experiment_results_in_df(experimental_data)

    def insert_experiment_results_in_df(self, experimental_data):

        name = experimental_data.experiment_metadata.experiment_name
        analysis = experimental_data.analysis_metadata
        
        self.overview_df.loc[self.overview_df['Experiment'] == name, 'max rate'] = analysis.max_rate
        self.overview_df.loc[self.overview_df['Experiment'] == name, 'max rate ydiff'] = analysis.max_rate_ydiff
        self.overview_df.loc[self.overview_df['Experiment'] == name, 'rate constant'] = analysis.rate_constant

    def update_metadata(self, file_name):

        new_overview_df = import_overview_excel(file_name)
        self.overview_df = new_overview_df

        for experiment_name, experiment_data in self.experiments.items():
            new_experimental_metadata = get_experimental_metadata(experiment_name, self.overview_df)

            experiment_data.experiment_metadata = ExperimentMetadata(**new_experimental_metadata, 
                                            color = get_experiment_color(new_experimental_metadata['experiment_name']))
            
            self.insert_experiment_results_in_df(experiment_data)

    def save_to_hdf5(self, filename: str):
        """Save all experiments to a single HDF5 file"""

        if not self.overview_df.empty:
            self.overview_df.to_hdf(filename, key='overview_df', mode='w', format='table')
    
        with h5py.File(filename, 'a') as f:
            for exp_name, experimental_data in self.experiments.items():
                if exp_name in f:
                    print(f"Experiment {exp_name} already exists. Overwriting...")
                    del f[exp_name]

                # Create a group for each experiment
                exp_grp = f.create_group(exp_name)
                
                time_series_grp = exp_grp.create_group('time_series_data')

                for key, value in asdict(experimental_data.time_series_data).items():
                    time_series_grp.attrs[key] = value
                
                exp_meta_grp = exp_grp.create_group('experiment_metadata')

                for key, value in asdict(experimental_data.experiment_metadata).items():
                    exp_meta_grp.attrs[key] = value

                analysis_meta_grp = exp_grp.create_group('analysis_metadata')

                for key, value in asdict(experimental_data.analysis_metadata).items():
                    analysis_meta_grp.attrs[key] = value

                datasets_grp = exp_grp.create_group('datasets')

                for key, value in asdict(experimental_data.datasets).items():
                    datasets_grp.attrs[key] = value
                
                print(f"Experiment {exp_name} added successfully.")
    
    def print_experiments(self):
        """
        Print all experiments in the dataset in a formatted way
        """
        if not self.experiments:
            print("No experiments in dataset")
            return
            
        print(f"Dataset contains {len(self.experiments)} experiments:")
        for i, name in enumerate(self.list_experiments(), 1):
            print(f"{i}. {name}")

    @classmethod
    def load_from_hdf5(cls, filename: str):
        """Load experiments from HDF5 file"""
        dataset = cls()

        try:
            dataset.overview_df = pd.read_hdf(filename, key='overview_df')
        except (KeyError, ValueError):
            print("No overview DataFrame found in file")
            dataset.overview_df = pd.DataFrame()

        with h5py.File(filename, 'r') as f:
            for exp_name in f.keys():
                if exp_name == 'overview_df':  # Skip the overview_df group
                    continue
                # Load experimental data

                time_series_dict = dict(f[f'{exp_name}/time_series_data'].attrs)
                time_series_data = TimeSeriesData(**time_series_dict)
               
                # Load metadata
                exp_metadata_dict = dict(f[f'{exp_name}/experiment_metadata'].attrs)
                experiment_metadata = ExperimentMetadata(**exp_metadata_dict)

                analysis_metadata_dict = dict(f[f'{exp_name}/analysis_metadata'].attrs)
                analysis_metadata = AnalysisMetadata(**analysis_metadata_dict)

                datasets_dict = dict(f[f'{exp_name}/datasets'].attrs)
                datasets = DataSets(**datasets_dict)

                # Create ExperimentalData and add to dataset
                experimental_data = ExperimentalData(
                    time_series_data, experiment_metadata, analysis_metadata, datasets
                )
                dataset.add_experiment(exp_name, experimental_data)
        
        return dataset

def get_experiment_color(experiment_name):
    """
    Generate a color for an experiment based on its group and subgroup with enhanced distinction.
    
    Args:
        experiment_name (str): Experiment name in format 'MRG-059-XX-N-M'
        
    Returns:
        str: Hex color code
    """
    # Split the experiment name to extract group and numbers
    parts = experiment_name.split('-')
    if len(parts) < 4:
        raise ValueError("Invalid experiment name format")
    
    # Extract group identifier (e.g., 'V', 'ZA', 'ZB')
    group = parts[2]
    
    # Extract subgroup number (first number after letters)
    subgroup_num = int(parts[3].split('-')[0])
    
    # Extract last number
    last_num = int(parts[-1])
    
    # Generate base hue for the group using prime numbers for better distribution
    group_value = 0
    for i, char in enumerate(reversed(group)):
        group_value += (ord(char) - ord('A') + 26 if char >= 'A' else ord(char) - ord('A')) * (31 ** i)
    
    # Use golden ratio conjugate for more distinct group color separation
    golden_conjugate = 0.618033988749895
    base_hue = (group_value * golden_conjugate) % 1.0
    
    # Create more distinct subgroup colors by shifting hue significantly
    # but staying within the same color family
    subgroup_hue = (base_hue + (subgroup_num - 1) * 0.1) % 1.0
    
    # High saturation for vivid colors
    # Vary saturation more dramatically between subgroups
    saturation = min(1.0, 0.8 + (subgroup_num * 0.15) % 0.2)
    
    # Brightness variations for the last number
    # Using cosine function for smooth but distinct transitions
    #value = 0.8 + 0.2 * np.cos(last_num * np.pi / 2)
    value = 1 - (last_num / 10)
    
    # Convert HSV to RGB
    rgb = colorsys.hsv_to_rgb(subgroup_hue, saturation, value)
    
    # Convert RGB to hex
    hex_color = "#{:02x}{:02x}{:02x}".format(
        int(rgb[0] * 255),
        int(rgb[1] * 255),
        int(rgb[2] * 255)
    )
    
    return hex_color