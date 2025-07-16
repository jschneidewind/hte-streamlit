import pandas as pd
import numpy as np
from scipy.stats import skewnorm
from scipy.optimize import least_squares, minimize
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
from matplotlib import rcParams
rcParams['font.family'] = 'sans-serif'
rcParams['font.sans-serif'] = ['Arial']
rcParams['font.size'] = 12
rcParams['mathtext.fontset'] = 'custom'
rcParams['mathtext.rm'] = 'Arial'
rcParams['mathtext.it'] = 'Arial:italic'
rcParams['mathtext.bf'] = 'Arial:bold'


from hte_streamlit.experiments_database import ExperimentalDataset

def poly(a, x):
    a = np.flip(a)
    y = a[0] * x**0

    for i in range(1, len(a)):
        y += a[i] * x**i

    return y

def logarithmic_model(p, x):
    """
    Logarithmic model of form: p[0] + p[1] * log(x)
    
    Parameters:
    -----------
    p : array-like
        Parameters [offset, scaling]
    x : array-like
        Input values (must be positive)
    """
    # Ensure x is positive for logarithm
    x_safe = np.maximum(x, 1e-10)
    return p[0] + p[1] * np.log(x_safe)

def exponential_model(p, x):
    """
    Exponential model of form: p[0] * exp(p[1] * x)
    
    Parameters:
    -----------
    p : array-like
        Parameters [amplitude, rate]
    x : array-like
        Input values
    """
    return p[0] * (1 - np.exp(-p[1] * x))

def residual_generic(p, x, y, function):
    y_fit = function(p, x)
    res = y - y_fit

    return res

def skewed_gaussian_model(p, x):
    return p[0] * skewnorm.pdf(x, p[3], loc=p[1], scale=p[2])

def combined_model(datapoints, functions, parameters, offset):


    result = np.zeros_like(datapoints[0])
    
    for data, function, parameter in zip(datapoints, functions, parameters):
        result += function(parameter, data)

    result -= offset
    result = np.maximum(result, 0)

    return result

def residual_combined_model(offset, datapoints, functions, parameters, y):

    y_fit = combined_model(datapoints, functions, parameters, offset)
    residuals = y - y_fit

    return np.sum(residuals**2)

def optimize_offset(df, functions, parameters, experiments,
                    columns = ['c([Ru(bpy(3]Cl2) [M]', 'c(Na2S2O8) [M]', 'pH [-]', 'mean_rate']):

    filtered_df = filter_data(df, experiments)
    selected_df = filtered_df[columns]
    data = selected_df.to_numpy()

    datapoints = [data[:, i] for i in range(len(data[0]))]
    rates = data[:,-1]

    initial_guess = 0

    p = minimize(residual_combined_model, initial_guess, 
                 args = (datapoints, functions, parameters, rates))

    return p.x[0]
    
def average_reproductions(df_cleaned, rate_column = 'max rate ydiff'):

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


def filter_data(df, experiment_list, prefix="MRG-059-"):
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


def plot_data(filtered_df, column_name, legend = True, ax = None, fig = None, axis_label = None, color = None):
    print(filtered_df['Power output [W/m^2]'].unique())

    if ax is None:
        fig, ax = plt.subplots(figsize = (10,6))
        fig.subplots_adjust(right = 0.8)

    cmap = plt.get_cmap('viridis')
    norm = mcolors.Normalize(vmin=0, vmax=len(filtered_df) - 1)

    counter = 0

    for index, row in filtered_df.iterrows():
        x = row[column_name]
        y = row['mean_rate']
        yerr = [[row['mean_rate'] - row['min_rate']], [row['max_rate'] - row['mean_rate']]]
      
        #color = cmap(norm(counter))

        ax.errorbar(x, y, yerr = yerr, fmt='o', ecolor=color, 
                 capsize=3, capthick=1, markersize=5,label = row['Experiment'], color = color)
        
        counter += 1
        
    ax.set_xlabel(axis_label, color = color)
    ax.set_ylabel(r'Max. rate / $\mu M(O_2) \, s^{-1}$')
    #ax.set_title(f'{column_name} vs. rate')

    if legend:
        ax.legend(title='Base Experiment', bbox_to_anchor=(1.02, 1.01), loc='upper left')

    return fig, ax

def construct_model(df_averaged, experiment_list, column_of_interest, model, 
                    plotting = False, save_fig = False, legend = True,
                    ax = None, fig = None, axis_label = None, color = None):

    df_filtered = filter_data(df_averaged, experiment_list)

    x = df_filtered[column_of_interest]
    x_full = np.linspace(df_filtered[column_of_interest].min(), df_filtered[column_of_interest].max(), 100)
    y = df_filtered['mean_rate']

    if model == skewed_gaussian_model:
        popt = least_squares(fun=residual_generic, x0 = [1/1e6, np.mean(x), np.std(x), 1/1e6], args=(x, y, skewed_gaussian_model)).x
        y_fit = skewed_gaussian_model(popt, x_full)

    elif model == poly:
        popt = np.polyfit(x, y, 2)
        y_fit = poly(popt, x_full)

    elif model == logarithmic_model:
        # Initial guess for logarithmic fit [offset, scaling]
        initial_guess = [np.min(y), (np.max(y) - np.min(y)) / (np.log(np.max(x)) - np.log(np.min(x) + 1e-10))]

        popt = least_squares(fun=residual_generic, x0=initial_guess, 
                            args=(x, y, logarithmic_model)).x
        y_fit = logarithmic_model(popt, x_full)

    elif model == exponential_model:
        # Good initial guesses for inverted exponential:
        # p[0] = slightly above max observed rate (asymptote)
        # p[1] = rough rate constant that gives good curvature
        max_rate = np.max(y) * 1.1  # 10% above max observed rate as asymptote guess
        rate_const = 1.0 / np.mean(x)  # Reasonable starting value for rate constant
        
        initial_guess = [max_rate, rate_const]
        popt = least_squares(fun=residual_generic, x0=initial_guess, 
                            args=(x, y, exponential_model)).x
        y_fit = exponential_model(popt, x_full)
    
    else:
        raise Exception("Model not found")

    if plotting:
        fig, ax = plot_data(df_filtered, column_of_interest, legend = False, ax = ax, fig = fig, axis_label = axis_label, color = color)
                            
        ax.plot(x_full, y_fit, color = color, label = 'Fit')

        if legend:
            ax.legend(bbox_to_anchor=(1.02, 1.01), loc='upper left')

        if save_fig:
            fig.savefig(f'/Users/jacob/Documents/Water_Splitting/HTE_Photocatalysis/{column_of_interest}_analyzed.pdf')

    return x, y, x_full, popt

class Model:
    def __init__(self, df_averaged, experiment_list, column_of_interest, model, 
                    plotting = False, save_fig = False, ax = None, fig = None,
                    legend = False, axis_label = None, color = 'black'):
        
        self.column_of_interest = column_of_interest

        if axis_label is None:
            self.axis_label = column_of_interest
        else:
            self.axis_label = axis_label

        self.model = model
        self.experiment_list = experiment_list
        self.df_averaged = df_averaged
        self.color = color

        self.x, self.y, self.x_full, self.popt = construct_model(df_averaged, 
                                                                 experiment_list, 
                                                                 column_of_interest, 
                                                                 model, 
                                                                 plotting = plotting, 
                                                                 save_fig = save_fig,
                                                                 ax = ax,
                                                                 fig = fig,
                                                                 legend = legend,
                                                                 axis_label = self.axis_label,
                                                                 color = self.color)

def plot_3D_model_fit(model_X_axis, model_Y_axis, other_models, 
                      other_models_values, heatmap = True,
                      ax = None, fig = None):

    x = model_X_axis.x_full
    y = model_Y_axis.x_full

    X, Y = np.meshgrid(x, y)
    datapoints = [X, Y]

    functions = [model_X_axis.model, model_Y_axis.model]
    parameters = [model_X_axis.popt, model_Y_axis.popt]
    names = [model_X_axis.column_of_interest, model_Y_axis.column_of_interest]
    experiments_XY = model_X_axis.experiment_list + model_Y_axis.experiment_list
    experiments = experiments_XY.copy()

    other_models_names = []

    for model, value in zip(other_models, other_models_values):
        W = np.ones_like(X) * value
        datapoints.append(W)
        functions.append(model.model)
        parameters.append(model.popt)
        other_models_names.append(model.column_of_interest)
        names.append(model.column_of_interest)
        experiments += model.experiment_list

    # drop duplicates in experiments
    experiments = list(set(experiments))
    names.append('mean_rate')
    
    offset = optimize_offset(model_X_axis.df_averaged, 
                             functions, 
                             parameters, 
                             experiments,
                             columns = names)

    Z = combined_model(datapoints, functions, parameters, offset)

    if heatmap:
        if ax is None:
            fig, ax = plt.subplots()

        heatmap = ax.pcolormesh(X, Y, Z, cmap='viridis', shading='auto')

        contour = ax.contour(X, Y, Z, colors='black', linewidths=0.5, alpha=0.7)
        ax.clabel(contour, inline=True, fontsize=12, fmt='%.1f')
        
        cbar = fig.colorbar(heatmap, ax=ax, label=r'Max. rate / $\mu M(O_2) \, s^{-1}$')
        ax.set_xlabel(model_X_axis.axis_label, color = model_X_axis.color)
        ax.set_ylabel(model_Y_axis.axis_label, color = model_Y_axis.color)
    

    else:
        fig = plt.figure()
        ax = fig.add_subplot(111, projection='3d')

        ax.plot_surface(X, Y, Z, cmap='viridis')

        ax.set_xlabel(model_X_axis.axis_label)
        ax.set_ylabel(model_Y_axis.axis_label)
        ax.set_zlabel(r'Rate / $\mu mol(O_2) \, s^{-1}$')

        filtered_data = filter_data(model_X_axis.df_averaged, experiments_XY)

        ax.scatter(
            filtered_data[model_X_axis.column_of_interest],
            filtered_data[model_Y_axis.column_of_interest],
            filtered_data['mean_rate'],
            c='b', marker='o')
        

def plot_2D_model_fit(model_X_axis, other_models, other_models_values, drop_ZA = True):

    x = model_X_axis.x_full

    datapoints = [x]
    functions = [model_X_axis.model]
    parameters = [model_X_axis.popt]
    names = [model_X_axis.column_of_interest]
    other_models_names = []

    for model, value in zip(other_models, other_models_values):
        y = np.ones_like(x) * value
        datapoints.append(y)
        functions.append(model.model)
        parameters.append(model.popt)
        names.append(model.column_of_interest)
        other_models_names.append(model.column_of_interest)

    names.append('mean_rate')
    
    offset = optimize_offset(model_X_axis.df_averaged, functions, parameters, columns = names)
    print(offset)

    Z = combined_model(datapoints, functions, parameters, offset)

    filtered_data = filter_data(model_X_axis.df_averaged, other_models_names, other_models_values, drop_ZA = drop_ZA)

    fig, ax = plot_data(filtered_data, model_X_axis.column_of_interest, legend = False)

    ax.plot(x, Z, color = 'green', label = 'Combined model fit')
    ax.legend(bbox_to_anchor=(1.02, 1.01), loc='upper left')


# def main():
    
#     df_cleaned = import_data('/Users/jacob/Documents/Water_Splitting/Projects/HTE_Photocatalysis/photocat-hte/data_analysis/overview/HTE-overview_240815_corrected.xlsx',
#                                 '/Users/jacob/Documents/Water_Splitting/Projects/HTE_Photocatalysis/photocat-hte/data_analysis/analyzed_csv/output.csv')
#     df_averaged = average_reproductions(df_cleaned)


#     Ru_model = Model(df_averaged, ['c(Na2S2O8) [M]', 'pH [-]'], [0.006, 9.6], 
#                      'c([Ru(bpy(3]Cl2) [M]',
#                      skewed_gaussian_model, plotting = False, save_fig=False)
    
#     Ox_model = Model(df_averaged, ['c([Ru(bpy(3]Cl2) [M]', 'pH [-]'], [0.00001, 9.6], 'c(Na2S2O8) [M]', 
#                     skewed_gaussian_model, plotting = False, save_fig=False)
    
#     pH_model = Model(df_averaged, ['c([Ru(bpy(3]Cl2) [M]', 'c(Na2S2O8) [M]'], [0.00001, 0.06], 'pH [-]', 
#                     poly, drop_ZA = True, plotting = False, save_fig=False)
    
#     plot_3D_model_fit(Ru_model, Ox_model, [pH_model], [9.6])
    #plot_3D_model_fit(Ru_model, pH_model, [Ox_model], [0.006])
    #plot_3D_model_fit(pH_model, Ox_model, [Ru_model], [0.00001])

    #plot_2D_model_fit(Ru_model, [Ox_model, pH_model], [0.006, 9.6])
    #plot_2D_model_fit(pH_model, [Ru_model, Ox_model], [0.00001, 0.006])
    #plot_2D_model_fit(Ox_model, [Ru_model, pH_model], [0.00001, 9.6])


def kinetic_function(Ru_conc, Ox_conc, k1, k2):

    rate = (k1 * Ru_conc * Ox_conc) / (1 + k2 * Ru_conc**2)

    return rate

    

def new_dataset():

    dataset = ExperimentalDataset.load_from_hdf5('/Users/jacob/Documents/Water_Splitting/Projects/HTE_Photocatalysis/HTE_Streamlit_App/data/250608_HTE.h5')
    df = dataset.overview_df

    df['c([Ru(bpy(3]Cl2) [M]'] *= 1e6 # convert to uM
    df['c(Na2S2O8) [M]'] *= 1e3 # convert to mM
    #df['Power output [W/m^2]'] *= 1e2 # correcting values for right W/m2

    df_averaged = average_reproductions(df)

    OX_experiments = ['ZN-10', 'ZN-9', 'ZN-8', 'ZO-1', 'ZO-9']
    Irradiation_experiments = ['ZO-1', 'ZO-3', 'ZN-14', 'ZN-13', 'ZN-11']
    Ru_experiments = ['ZO-1', 'ZO-8', 'ZO-2', 'ZN-7', 'ZN-4', 'ZN-3', 'ZN-2', 'ZN-1']
    pH_experiments = ['ZO-1', 'ZO-7', 'ZO-6', 'ZO-5', 'ZO-4']

    viridis = plt.cm.viridis
    colors = [viridis(i) for i in np.linspace(0, 0.85, 4)]

    fig, ax = plt.subplots(nrows = 2, ncols = 3, figsize=(9, 5))
    fig.subplots_adjust(left = 0.08, right = 0.96, top = 0.93, bottom = 0.12,
                        wspace = 0.4, hspace = 0.4)
    

    Ru_model = Model(df_averaged, Ru_experiments, 'c([Ru(bpy(3]Cl2) [M]', 
                     skewed_gaussian_model, plotting = True, save_fig=False,
                     ax = ax[0,0], fig = fig,
                     axis_label = r'$\mathrm{[Ru(bpy)_3]Cl_2}$ / $\mu\mathrm{M}$', color = colors[0])

    k1 = 0.00005
    k2 = 0.03

    Ox_conc = 6000
    Ru_conc = np.linspace(0, 100, 100)

    # Ru conc needs to be >1 to see effect ??? 

    # rate = kinetic_function(Ru_conc, Ox_conc, k1, k2)
    # ax[0,0].plot(Ru_conc, rate, color = colors[0], label = 'Kinetic model')


    OX_model = Model(df_averaged, OX_experiments, 'c(Na2S2O8) [M]', 
                     exponential_model, plotting = True, save_fig=False,
                     ax = ax[0,1], fig = fig,
                     axis_label = r'$\mathrm{Na_2S_2O_8}$ / $\mathrm{mM}$',
                     color = colors[1])

    Irradiation_model = Model(df_averaged, Irradiation_experiments, 'Power output [W/m^2]', 
                     poly, plotting = True, save_fig=False,
                     ax = ax[1,0], fig = fig,
                     axis_label = r'Irradiance / W $m^{-2}$', color = colors[2])
    
    pH_model = Model(df_averaged, pH_experiments, 'pH [-]', 
                     poly, plotting = True, save_fig=False,
                     ax = ax[1,1], fig = fig, axis_label = 'pH', color = colors[3])
    

    plot_3D_model_fit(Ru_model, OX_model, [Irradiation_model, pH_model], [6.637*1e2, 9.6], ax = ax[0,2], fig = fig)

    # plot_3D_model_fit(pH_model, Irradiation_model, [Ru_model, OX_model], [0.00001, 0.006], ax = ax[1,2], fig = fig)

    plot_3D_model_fit(Irradiation_model, pH_model, [Ru_model, OX_model], [10, 6], ax = ax[1,2], fig = fig)


    # Add labels A-F to the subplots
    labels = ['A', 'B', 'C', 'D', 'E', 'F']
    position = [[-0.25, 1.01], [-0.25, 1.01], [-0.45, 1.01],
                [-0.25, 1.01], [-0.25, 1.01], [-0.45, 1.01]]

    for i in range(2):
        for j in range(3):
            # Calculate index in the labels list
            idx = i*3 + j
            # Add text slightly above and to the left of each subplot
            ax[i, j].text(position[idx][0], position[idx][1], labels[idx], 
                         transform=ax[i, j].transAxes, 
                         fontsize=22, fontweight='bold')
    
    #fig.savefig('HTE_Data_Figure.png', dpi = 500)
            
            


if __name__ == '__main__':
    #main()
    #irradiation_power()
    new_dataset()
    plt.show()