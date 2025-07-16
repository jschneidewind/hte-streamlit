import os
from concurrent.futures import ProcessPoolExecutor
import multiprocessing
from functools import partial
import traceback

from hte_streamlit.fit import fit_data
from hte_streamlit.experiments_database import TimeSeriesData, ExperimentMetadata, AnalysisMetadata, DataSets, ExperimentalData, get_experiment_color, import_data, get_experimental_metadata

def generate_list_of_files(keywords, directory):
    files = [file for file in os.listdir(f'{directory}/csv') if any(keyword in file for keyword in keywords)]
    return files

def analyze_single_file(file, 
                        directory, 
                        dataset, 
                        plotting = True, 
                        plot_baseline = True,
                        ax = None,
                        fig = None):
    """
    Analyze a single experimental data file and create a complete ExperimentalData object.
    
    This function processes raw experimental data from a CSV file, performs data fitting
    and analysis, extracts experimental metadata, and packages everything into a structured
    ExperimentalData object. It handles both successful analysis and error cases gracefully.
    
    Parameters
    ----------
    file : str
        Filename of the CSV data file to analyze. Expected format: 'prefix_experimentname.csv'
        where experimentname will be extracted for metadata lookup.
    directory : str
        Base directory path containing 'csv/' and 'png/' subdirectories for input data
        and output plots respectively.
    dataset : ExperimentalDataset
        Dataset object containing the overview DataFrame with experimental metadata
        for matching experiment names to their conditions.
    plotting : bool, default True
        Whether to generate and save analysis plots. If True, saves plots to
        '{directory}/png/{file}_fit.png'.
    plot_baseline : bool, default True
        Whether to include baseline visualization in the analysis plots.
    ax : matplotlib.axes.Axes, optional
        Matplotlib axes object for plotting. If None, a new figure is created.
    fig : matplotlib.figure.Figure, optional
        Matplotlib figure object for plotting. If None, a new figure is created.
    
    Returns
    -------
    dict
        Result dictionary with the following structure:
        
        On success:
        {
            'success': True,
            'experiment_name': str,
            'data': ExperimentalData
        }
        
        On failure:
        {
            'success': False,
            'file': str,
            'error': str (includes full traceback)
        }
    
    Notes
    -----
    The function performs the following operations:
    1. Extracts experiment name from filename using underscore and dot delimiters
    2. Loads raw data from CSV file in the specified directory
    3. Performs data fitting and analysis using the fit_data function
    4. Retrieves experimental metadata from the dataset's overview DataFrame
    5. Creates structured data classes (TimeSeriesData, ExperimentMetadata, etc.)
    6. Packages everything into an ExperimentalData object
    
    The experiment name extraction assumes filenames follow the pattern 'prefix_name.csv'
    where 'name' corresponds to entries in the dataset's overview DataFrame.
    
    Error handling includes full traceback capture for debugging purposes, and all
    exceptions are caught to prevent crashes during batch processing.
    
    Examples
    --------
    >>> # Analyze a single file with plotting
    >>> result = analyze_single_file('data_MRG-059-ZO-1-1.csv', '/path/to/data', 
    ...                             dataset, plotting=True)
    >>> if result['success']:
    ...     experiment_data = result['data']
    ...     print(f"Analyzed {result['experiment_name']}")
    ... else:
    ...     print(f"Failed: {result['error']}")
    >>> 
    >>> # Analyze without saving plots
    >>> result = analyze_single_file('data_MRG-059-ZO-1-2.csv', '/path/to/data',
    ...                             dataset, plotting=False)
    >>> 
    >>> # Use custom matplotlib objects
    >>> fig, ax = plt.subplots()
    >>> result = analyze_single_file('data_MRG-059-ZO-1-3.csv', '/path/to/data',
    ...                             dataset, ax=ax, fig=fig)
    """
    
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

def analyze_files_multiprocessing(keywords, 
                                  directory, 
                                  dataset, 
                                  plotting = True, 
                                  plot_baseline = True,
                                  ax = None, 
                                  fig = None): 
    """
    Analyze multiple experimental data files in parallel using multiprocessing.
    
    This function provides high-performance batch processing of experimental data files
    by leveraging all available CPU cores. It filters files based on keywords, analyzes
    each file in parallel, and automatically adds successful results to the dataset.
    
    Parameters
    ----------
    keywords : list of str
        List of keywords to filter files. Only files containing at least one of these
        keywords in their filename will be processed.
    directory : str
        Base directory path containing 'csv/' subdirectory with input data files
        and 'png/' subdirectory for output plots.
    dataset : ExperimentalDataset
        Dataset object that will be updated with successfully analyzed experiments.
        Must contain overview DataFrame with experimental metadata.
    plotting : bool, default True
        Whether to generate and save analysis plots for each file. If True, saves plots
        to '{directory}/png/{filename}_fit.png'.
    plot_baseline : bool, default True
        Whether to include baseline visualization in the analysis plots.
    ax : matplotlib.axes.Axes, optional
        Matplotlib axes object for plotting. Note: sharing axes across multiprocessing
        may cause issues; typically used for single-threaded analysis.
    fig : matplotlib.figure.Figure, optional
        Matplotlib figure object for plotting. Note: sharing figures across multiprocessing
        may cause issues; typically used for single-threaded analysis.
    
    Returns
    -------
    list of dict
        List of result dictionaries from analyze_single_file, one per processed file.
        Each dictionary contains either:
        
        On success:
        {
            'success': True,
            'experiment_name': str,
            'data': ExperimentalData
        }
        
        On failure:
        {
            'success': False,
            'file': str,
            'error': str (includes full traceback)
        }
    
    Notes
    -----
    The function performs the following operations:
    1. Generates a filtered list of files based on provided keywords
    2. Creates a partial function with fixed parameters for multiprocessing
    3. Distributes file analysis across all available CPU cores
    4. Collects results and automatically adds successful experiments to the dataset
    5. Reports failed analyses with detailed error information
    
    The multiprocessing approach provides significant performance improvements for
    large datasets but may consume substantial system resources. Each worker process
    will have its own copy of the dataset object for analysis.
    
    Error handling is robust - failed analyses don't interrupt the processing of
    other files, and detailed error information is preserved for debugging.
    
    Warnings
    --------
    When using matplotlib objects (ax, fig) with multiprocessing, be aware that
    sharing these objects across processes may cause unexpected behavior or errors.
    Consider using plotting=False for large batch jobs and generating plots separately.
    
    Examples
    --------
    >>> # Process all files containing specific keywords
    >>> keywords = ['MRG-059', 'ZO-1']
    >>> results = analyze_files_multiprocessing(keywords, '/path/to/data', dataset)
    >>> print(f"Processed {len(results)} files")
    >>> 
    >>> # Process without plotting for faster batch analysis
    >>> results = analyze_files_multiprocessing(['experiment'], '/path/to/data', 
    ...                                        dataset, plotting=False)
    >>> 
    >>> # Check results and count successes
    >>> successful = [r for r in results if r['success']]
    >>> failed = [r for r in results if not r['success']]
    >>> print(f"Success: {len(successful)}, Failed: {len(failed)}")
    """

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
