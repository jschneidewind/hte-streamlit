import pandas as pd
import numpy as np
from pathlib import Path
import h5py
from dataclasses import dataclass, asdict, field
from typing import List, Dict
import colorsys
import re
import hashlib

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
    color: str = "#ce1480"

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

    def update_reaction_data(self):

        ENERGY_PER_PHOTON_J = 4.22648E-19 # J
        M2_TO_CM2 = 10000

        ### Making this a permanent change to the dataset

        for experiment_name, experiment_data in self.experiments.items():
            experiment_data.time_series_data.data_reaction_molar = experiment_data.time_series_data.data_reaction * 1e-6
            experiment_data.time_series_data.y_diff_molar = np.diff(experiment_data.time_series_data.data_reaction_molar) / np.diff(experiment_data.time_series_data.time_reaction)

            experiment_data.experiment_metadata.ru_concentration_uM = experiment_data.experiment_metadata.ru_concentration * 1e6
            experiment_data.experiment_metadata.oxidant_concentration_uM = experiment_data.experiment_metadata.oxidant_concentration * 1e6 
            experiment_data.experiment_metadata.extinction_coefficient = 1./991 # toy extinction coefficient so that extinction coefficient * power_output = 1

            experiment_data.experiment_metadata.photon_flux = (experiment_data.experiment_metadata.power_output / M2_TO_CM2) / ENERGY_PER_PHOTON_J

            experiment_data.experiment_metadata.sigma_RuII = 3.25066E-17 # 8500 M^-1 cm^-1 * 3.82431E-21
            experiment_data.experiment_metadata.sigma_RuIII = 2.06513E-18 # 540 M^-1 cm^-1 * 3.82431E-21

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
                
    def list_experiments(self) -> List[str]:
        """
        List all experiments in the dataset
        
        Returns:
            List[str]: A sorted list of experiment names
        """
        return sorted(self.experiments.keys())

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
    Generates a consistent color for an experiment based on its name using HSV color space.
    The function parses experiment names in the format 'MRG-X-GROUP-SUBGROUP-EXPNUM'
    and generates a color by:
    1. Creating a hash from the group-subgroup combination for the hue
    2. Using fixed high saturation
    3. Varying the value based on experiment number
    Parameters:
        experiment_name (str): Name of the experiment in format 'MRG-X-GROUP-SUBGROUP-EXPNUM'
    Returns:
        str: Hex color code (e.g., '#FF0000' for red)
            Returns '#808080' (gray) if the experiment name doesn't match the expected pattern
    Example:
        >>> get_experiment_color('MRG-1-ABC-01-03')
        '#7b2e9b'  # Returns a consistent color for this experiment
    """
    # Parse experiment name
    pattern = r'MRG-\d+-([A-Z]+)-(\d+)-(\d+)'
    match = re.match(pattern, experiment_name)
    
    if not match:
        return "#808080"  # Default gray
        
    group, subgroup_num, exp_num = match.groups()
    
    # Create hash from group-subgroup combination
    hash_input = f"{group}-{subgroup_num}"
    hash_value = int(hashlib.md5(hash_input.encode()).hexdigest(), 16)

    # Map hash to hue (0-1)
    hue = (hash_value % 1000) / 1000.0
    
    # Fixed high saturation
    saturation = 0.9
    
    # Map experiment number to cyclical value (0.5-1.0)
    value = 0.5 + 0.5 * ((int(exp_num) % 10) / 10.0)
    
    # Convert HSV to RGB
    rgb = colorsys.hsv_to_rgb(hue, saturation, value)
    
    # Convert RGB to hex
    hex_color = "#{:02x}{:02x}{:02x}".format(
        int(rgb[0] * 255),
        int(rgb[1] * 255),
        int(rgb[2] * 255)
    )
    
    return hex_color