import math
import matplotlib.pyplot as plt
import numpy as np
from numpy.linalg import inv
from scipy.optimize import least_squares
from scipy.signal import savgol_filter
import pandas as pd

import hte_streamlit.rxn_ode_fitting as ode

def align_time(df):
    df["duration"] = (
        df["duration"] - df[df["status"] == "DEGASSING"]["duration"].values[0]
    )

def find_nearest(array, values):
    """
    Find the indices of the nearest values in the array to a list of target values.

    Parameters:
        array (numpy.ndarray): The input array.
        values (numpy.ndarray or scalar): The target values.

    Returns:
        list: A list of indices of the nearest values in the array to the target values.

    Note:
        If the input array has more than one dimension, the function flattens the first dimension.
        The function uses the `numpy.searchsorted` function to find the indices of the target values in the array.
        The function then checks if the index is not the last index in the array and if the difference between the target value and the previous value in the array is less than the difference between the target value and the current value in the array. If both conditions are true, the index of the previous value is returned, otherwise the index of the current value is returned.
    """

    if array.ndim != 1:
        array_1d = array[:, 0]
    else:
        array_1d = array

    values = np.atleast_1d(values)
    hits = []

    for i in range(len(values)):
        idx = np.searchsorted(array_1d, values[i], side="left")
        if idx > 0 and (
            idx == len(array_1d)
            or math.fabs(values[i] - array_1d[idx - 1])
            < math.fabs(values[i] - array_1d[idx])
        ):
            hits.append(idx - 1)
        else:
            hits.append(idx)

    return hits

def poly(a, x):
    y = a[0] * x**0

    for i in range(1, len(a)):
        y += a[i] * x**i

    return y

def logistic(m, spn, x, c = 1):
	return (c/(1 + np.exp(spn*(x - m))))

def LBC_WLS(x, y, m, spn, order_poly, w):
	'''Weighted Least Square Implementation of LBC with polynomial baseline. w is a vector of the same length as x specifying the weights for
	each residual'''

	X_raw = np.tile(x, (order_poly+1, 1))
	powers = np.arange(0, order_poly+1)
	X = np.power(X_raw.T, powers)

	logit_values = logistic(m, spn, x)
	X = np.c_[X, logit_values]

	W = np.diag(w)

	a = inv(np.dot(np.dot(X.T, W), X))
	a = np.dot(np.dot(a, X.T), W)
	coef = np.dot(a, y)

	return coef

def LBC_WLS_fitting(data, start, end, order_poly, weight = 1., plotting = False, filename = None):
    '''LBC fitting function using weighted least square implementation. Specifying weight to be a vector of the same length
    as x enables weighted least square, otherwise ordinary least square is performed. Order_Poly specifies the order
    of the polynomial used to describe the baseline.'''

    idx = find_nearest(data, (start, end))

    baseline = np.r_[data[:idx[0]], data[idx[1]:]]
    x = baseline[:,0]
    y = baseline[:,1]

    m = (data[:,0][idx[0]] + data[:,0][idx[1]]) / 2  # sigmoidal midpoint midway between pre- and post-signal intervals
    x_75 = (data[:,0][idx[1]] - m) / 2
    spn = np.log(1./9e6) / x_75    # logistic growth rate set so that curvature of logistic function does not interfere with baseline fitting

    if weight == 1.:
        weight = np.ones(len(x)) * weight

    p = LBC_WLS(x, y, m, spn, order_poly, weight)

    baseline = poly(p[:-1], data[:,0])
    lbc_fit = baseline + logistic(m, spn, data[:,0], c = p[-1])
    y_corrected = data[:,1] - baseline

    if plotting:
        fig, ax = plt.subplots()
        ax.plot(x, y, ".", markersize=2.0)
        ax.plot(data[:,0], baseline, linewidth = 1.0)
        ax.plot(data[:,0], lbc_fit, linewidth = 1.0)
        ax.plot(data[:,0], y_corrected, color = 'darkgreen')
        if filename is not None:
            fig.savefig(filename, dpi=400)
                    
    data_corr = np.c_[data[:,0], y_corrected]

    return data_corr, baseline, lbc_fit, data[:,0], y_corrected

def preprocess_data_LBC(data_df, offset, plotting, filename = None):
    """Pre-processing data by selecting relevant portion of data and then performing
    baseline correction based on pre-reaction phase.
    """

    align_time(data_df)

    # subset data to relevant statuses
    data_subset = data_df[data_df["status"].isin(["PREREACTION-BASELINE", "REACTION", "POSTREACTION-BASELINE"])]
    data_subset = data_subset[
        data_subset["command"].isin(["LAMP-ON", "FIRESTING-START", "LAMP-OFF"])
    ]

    start = data_subset[data_subset["status"] == "REACTION"]["duration"].values[0]
    end = data_subset[data_subset["status"] == "POSTREACTION-BASELINE"]["duration"].values[0]

    time = data_subset["duration"].values
    o2_data = data_subset["uM_1"].values

    data_corrected, baseline, lbc_fit, x_values, y_corrected = LBC_WLS_fitting(
        np.c_[time, o2_data], start, end, 4, plotting=plotting, filename = filename
    )

    rxn_subset = data_subset[data_subset["status"] == "REACTION"]

    rxn_start = find_nearest(time, rxn_subset["duration"].values[0] + offset)[0]
    rxn_end = find_nearest(time, rxn_subset["duration"].values[-1])[0]

    return data_subset, data_corrected, rxn_start, rxn_end, baseline, lbc_fit, x_values, y_corrected, time, o2_data


def plotting_fit_results(time_reaction, data_reaction, y_diff, y_fit, y_diff_smoothed, 
                         ax = None, fig = None, label = None, color = None):
    """Plotting of fit results."""

    if ax is None:
        fig, ax = plt.subplots()

    if color is None:
        random_value = np.random.rand()
        cmap = plt.get_cmap("plasma")
        color = cmap(random_value)

    ax2 = ax.twinx()
    ax2.plot(time_reaction[1:], y_diff, '.')
    ax2.plot(time_reaction[1:], np.diff(y_fit))
    ax2.plot(time_reaction[1:], y_diff_smoothed)
    ax2.set_ylabel('Reaction rate')
    
    ax.plot(time_reaction, data_reaction, ".", color=color, label = label)
    ax.plot(time_reaction, y_fit, color=color, label = label)
    ax.set_xlabel("Time / s")
    ax.set_ylabel(r"Oxygen / $\mu$mol/L")

    return fig

def fit_data(
    data_df: pd.DataFrame,
    filename: str = None,
    baseline_file_name: str = None,
    offset: int = 0,
    reaction_string=["A > B, k1", "B > C, k2", "C > D, k3"],
    bounds=[[0, 1], [0, 1], [0, 1]],
    idx_for_rate_constant = 0,
    idx_for_fitting = 2,
    savgol_window_factor = 5,
    savgol_poly_order = 3,
    plotting: bool = False,
    plot_baseline: bool = False,
    return_full: bool = False,
    ax = None,
    fig = None
):
    """
    Fitting data to arbitrary reaction model using 'rxn_ode_fitting' module.

    Reaction string is specified, in this case A >k1> B >k2> C, to model induction period due to O2 diffusion to sensor.
    Experimental data is fitted to concentration profile of C (idx_for_fitting = 2).
    k1 is optimized and k2 is fixed to 0.15 (bounds), k1 is returned (idx_for_rate_constant = 0).
    The bounds for k2 might have to be adjusted for other reactions, we will have to take a look at this with future data.
    return_full is for interfacing with other code for debugging.

    Args:
        data_df (pd.DataFrame):
            Pandas dataframe containing experimental data.
        filename (str):
            filename
        offset (int):
            Offset for fitting of experimental data (for example to skip induction period). However, using
            the offset seems to lower the quality of the fit.
        reaction_string (List[str]):
            Reaction string describing reaction sequence that is fitted to data.
        idx_for_rate_constant (int):
            idx to select rate constant from obtained array. Depending on reaction string, array has as many entries as
            there are k values. Default is that first rate constant is picked.
        idx_for_fitting (int):
            idx to select which part of model is fitted to experimental data. Default is that species "C" (idx = 2) is
            fitted to data.
        plotting (bool):
            Flag to control if plot is generated.
        plot_baseline (bool):
            Flag to control if baseline plot is generated
        return_full (bool):
            Flag to control if full output is returned (default = False).
    """

    if baseline_file_name is not None:
        baseline_file_name = filename.split('/')[-1].split('.')[0]
        baseline_file_name = f'data_analysis/baseline/{baseline_file_name}_baseline.png'

    data_subset, data_corrected, rxn_start, rxn_end, baseline, lbc_fit, x_values, y_corrected, time, o2_data = preprocess_data_LBC(
        data_df, offset=offset, filename = baseline_file_name, plotting = plot_baseline
    )

    time_reaction = (
        data_corrected[:, 0][rxn_start:rxn_end] - data_corrected[:, 0][rxn_start]
    )
    data_reaction = data_corrected[:, 1][rxn_start:rxn_end]

    idx = np.argmax(data_reaction)
    time_reaction = time_reaction[:idx]
    data_reaction = data_reaction[:idx]

    p, matrix, initial_state, _residual = ode.ode_fitting(
        data_reaction,
        time_reaction,
        reaction_string,
        idx=idx_for_fitting,
        bounds_arr=bounds,
    )

    rate_constant = p[idx_for_rate_constant]

    y_fit = ode.ode_matrix_fit_func(
        p, 
        initial_state, 
        time_reaction, 
        matrix, 
        ravel = False
    )

    y_fit_selection = y_fit[:,idx_for_fitting]

    y_diff_fit = np.diff(y_fit_selection) / np.diff(time_reaction)
    max_rate = np.amax(y_diff_fit)

    y_diff = np.diff(data_reaction) / np.diff(time_reaction)
    y_diff_smoothed = savgol_filter(y_diff, int(len(y_diff)/savgol_window_factor), savgol_poly_order)

    max_rate_ydiff = np.amax(y_diff_smoothed)

    if plotting is True:
        fig = plotting_fit_results(
            time_reaction, data_reaction, y_diff, y_fit_selection, y_diff_smoothed, ax = ax, fig = fig
        )
        if filename is not None:
            fig.savefig(filename, dpi=400)

    if return_full is True:

        metadata = {'p': p, 
                    'max_rate': max_rate,
                    'max_rate_ydiff': max_rate_ydiff,
                    'initial_state': initial_state, 
                    'matrix': str(matrix),
                    'rate_constant': rate_constant,
                    'rxn_start': rxn_start,
                    'rxn_end': rxn_end, 
                    'residual': _residual, 
                    'idx_for_fitting': idx_for_fitting
                    }
        
        dataframes = {'data_corrected': data_corrected
                      }

        time_series_data = {'time_reaction': time_reaction,
                            'data_reaction': data_reaction,
                            'y_fit': y_fit_selection,
                            'baseline_y': baseline,
                            'lbc_fit_y': lbc_fit,
                            'full_x_values': x_values,
                            'full_y_corrected': y_corrected,
                            'x_diff': time_reaction[1:],
                            'y_diff': y_diff,
                            'y_diff_smoothed': y_diff_smoothed,
                            'y_diff_fit': y_diff_fit,
                            'time_full': time,
                            'data_full': o2_data
                            }

        return metadata, dataframes, time_series_data
    
    return max_rate
