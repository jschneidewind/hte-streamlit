import os
import pandas as pd
import matplotlib.pyplot as plt
from concurrent.futures import ProcessPoolExecutor
import multiprocessing
from functools import partial
import traceback

from hte_streamlit.fit import fit_data
from hte_streamlit.experiments_database import TimeSeriesData, ExperimentMetadata, AnalysisMetadata, DataSets, ExperimentalData, ExperimentalDataset, get_experiment_color, import_data, import_overview_excel, get_experimental_metadata

def generate_list_of_files(keywords, directory):
    files = [file for file in os.listdir(f'{directory}/csv') if any(keyword in file for keyword in keywords)]
    return files

def analyze_single_file(file, directory, dataset, plotting = True, plot_baseline = True,
                        ax = None, fig = None):
    
    experiment_name = file.split('_')[1]
    experiment_name = experiment_name.split('.')[0]
    
    if plotting is True:
        filename = f'{directory}/png/{file}_fit.png'
    else:
        filename = None

    df = import_data(f'{directory}/csv/{file}')

    try:
        analysis_metadata, dataframes, time_series_data = fit_data(df, 
                        filename = filename, 
                        plotting = plotting, plot_baseline = plot_baseline, ax = ax, fig = fig, 
                        return_full = True)
        
        max_rate_print_out = analysis_metadata['max_rate']
        print(f'{file} analyzed, rate: {max_rate_print_out}')

        experimental_metadata = get_experimental_metadata(experiment_name, dataset.overview_df)

        time_series_data_class = TimeSeriesData(**time_series_data)
        experiment_metadata_class = ExperimentMetadata(**experimental_metadata, 
                                color = get_experiment_color(experimental_metadata['experiment_name']))
        analysis_metadata_class = AnalysisMetadata(**analysis_metadata)
        datasets_class = DataSets(**dataframes)

        experimental_data_class = ExperimentalData(time_series_data = time_series_data_class,
                                        experiment_metadata = experiment_metadata_class,
                                        analysis_metadata = analysis_metadata_class,
                                        datasets = datasets_class)
        
        return {
            'success': True,
            'experiment_name': experimental_metadata['experiment_name'],
            'data': experimental_data_class
        }

    except Exception as e:
        tb = traceback.format_exc()
        print(f'{file} analysis failed, not added to dataset, error: {str(e)}')
        print("Full traceback:")
        print(tb)
        return {
            'success': False,
            'file': file,
            'error': f"{str(e)}\n\nFull traceback:\n{tb}"
        }

def analyze_files_multiprocessing(keywords, directory, dataset, plotting = True, plot_baseline = True,
                                  ax = None, fig = None): 

    files = generate_list_of_files(keywords, directory)

    analyze_single_file_partial = partial(analyze_single_file, directory = directory,
                                          dataset = dataset,
                                          plotting = plotting, plot_baseline = plot_baseline,
                                          ax = ax, fig = fig)

    with ProcessPoolExecutor(max_workers=multiprocessing.cpu_count()) as executor:
        results = list(executor.map(analyze_single_file_partial, files))

    for result in results:
        if result['success']:
            dataset.add_experiment(result['experiment_name'], result['data'])
        else:
            print(f"Failed to process {result['file']}: {result['error']}")

    return results
  
def main():
    #keywords = ['V', 'X', 'Y', 'Z', 'ZA', 'ZB', 'ZC', 'ZE', 'ZF', 'ZG', 'ZH']
    keywords = ['ZP-2-2']

    # experimental_dataset = ExperimentalDataset.load_from_hdf5('HTE_dataset.h5')

    experimental_dataset = ExperimentalDataset(overview_df=
                                               import_overview_excel('/Users/jacob/Documents/Water_Splitting/Projects/HTE_Photocatalysis/photocat-hte/data_analysis/overview/HTE-overview_250408.xlsx'))

    # analyze_single_file('results_MRG-059-ZC-3-3.csv', 'data_analysis', experimental_dataset)

    analyze_files_multiprocessing(keywords, '/Users/jacob/Documents/Water_Splitting/Projects/HTE_Photocatalysis/photocat-hte/data_analysis', 
                                  dataset = experimental_dataset)

    #experimental_dataset.save_to_hdf5('250408_HTE_dataset.h5')
    experimental_dataset.print_experiments()

if __name__ == '__main__':
    main()
    plt.show()