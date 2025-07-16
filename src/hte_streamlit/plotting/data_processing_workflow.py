import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib import rcParams
rcParams['font.family'] = 'sans-serif'
rcParams['font.sans-serif'] = ['Arial']
rcParams['font.size'] = 12
rcParams['mathtext.fontset'] = 'custom'
rcParams['mathtext.rm'] = 'Arial'
rcParams['mathtext.it'] = 'Arial:italic'
rcParams['mathtext.bf'] = 'Arial:bold'

import matplotlib.patches as patches

from hte_streamlit.experiments_database import ExperimentalDataset, import_data
from hte_streamlit.fit import align_time


def main():
    data_df = import_data('data/Raw_Data/results_MRG-059-ZN-12-2.csv')
    dataset = ExperimentalDataset.load_from_hdf5('data/250616_HTE.h5')

    align_time(data_df)

    # subset data to relevant statuses
    data_subset = data_df[data_df["status"].isin(["DEGASSING", "PREREACTION-BASELINE", "REACTION", "POSTREACTION-BASELINE"])]
    data_subset = data_subset[
        data_subset["command"].isin(["LAMP-ON", "FIRESTING-START", "LAMP-OFF"])
    ]

    pre_baseline_start = data_subset[data_subset["status"] == "PREREACTION-BASELINE"]["duration"].values[0]
    reaction_start = data_subset[data_subset["status"] == "REACTION"]["duration"].values[0]
    reaction_end = data_subset[data_subset["status"] == "POSTREACTION-BASELINE"]["duration"].values[0]

    time = data_subset["duration"].values
    o2_data = data_subset["uM_1"].values

    exp = dataset.experiments['MRG-059-ZN-12-2']
    max_rate_idx = np.argmax(exp.time_series_data.y_diff_smoothed)
    
    # Create a figure with 3 subplots
    fig, ax = plt.subplots(nrows = 3, ncols = 1, figsize = (4, 8))
    fig.subplots_adjust(left = 0.2, right = 0.93, top = 0.97, bottom = 0.08, hspace=0.6)

    ax[0].plot(time, o2_data, 'o-', markersize=3, label = 'Raw data', color = 'grey')
    ax[0].axvline(x=pre_baseline_start, color='black', linestyle='--')
    ax[0].axvline(x=reaction_start, color='black', linestyle='--')
    ax[0].axvline(x=reaction_end, color='black', linestyle='--')
    ax[0].set_xlim(time.min(), time.max())
    ax[0].set_xlabel('Time / s')
    ax[0].set_ylabel(r'Oxygen / $\mathrm{\mu M(O_2)}$')

    tick_locations = [0, 1000, 3000]
    ax[0].set_xticks(tick_locations)

    xlabel = ax[0].xaxis.get_label()
    xlabel.set_position((0.2, 0))

    # Get the data limits from the x-axis
    x_min, x_max = ax[0].get_xlim()

    # Convert data coordinate to axes fraction (0-1 range)
    reaction_start_axes_frac = (reaction_start - x_min) / (x_max - x_min)
    reaction_end_axes_frac = (reaction_end - x_min) / (x_max - x_min)

    # Create a connection line from reaction_start on x-axis to left edge of plot
    con1 = patches.ConnectionPatch(
        xyA=(reaction_start_axes_frac,0),  # Start at reaction_start on x-axis
        xyB=(0,1),     # End at left edge, top of plot
        coordsA='axes fraction', coordsB='axes fraction',
        axesA=ax[0], axesB=ax[1],
        color='black', linestyle='--', linewidth=1.5
    )
    ax[0].add_patch(con1)
    
    con2 = patches.ConnectionPatch(
        xyA=(reaction_end_axes_frac,0),  # Start at reaction_start on x-axis
        xyB=(1,1),     # End at left edge, top of plot
        coordsA='axes fraction', coordsB='axes fraction',
        axesA=ax[0], axesB=ax[1],
        color='black', linestyle='--', linewidth=1.5
    )
    ax[0].add_patch(con2)

    ax[1].plot(exp.time_series_data.time_reaction, exp.time_series_data.data_reaction, 'o', markersize = 3, label = 'Concentration data', color = 'grey')
    ax[1].plot(exp.time_series_data.time_reaction, exp.time_series_data.y_fit, label = 'Kinetic fit', color = 'orange', linewidth = 1.5)
    ax[1].set_xlabel('Time / s')
    ax[1].set_ylabel(r'Oxygen / $\mathrm{\mu M(O_2)}$')
    ax[1].legend()

    ax[2].plot(exp.time_series_data.x_diff, exp.time_series_data.y_diff, 'o', markersize = 3, label = 'Rate data', color = 'grey')
    ax[2].plot(exp.time_series_data.x_diff, exp.time_series_data.y_diff_smoothed, label = 'Smoothing', color = 'black')
    ax[2].plot(exp.time_series_data.x_diff[max_rate_idx], exp.time_series_data.y_diff_smoothed[max_rate_idx], 'o', markersize = 10, label = 'Max. rate', color = 'darkred')
    ax[2].plot(exp.time_series_data.x_diff, exp.time_series_data.y_diff_fit, label = 'Kinetic fit', color = 'orange', linewidth = 1.5)
    ax[2].set_xlabel('Time / s')
    ax[2].set_ylabel(r'Rate / $\mathrm{\mu M(O_2) \, s^{-1}}$')
    ax[2].legend()

    fig.text(x=0.45, y=0.675, s='1. Baseline correction', 
         fontsize=12, fontweight='bold', 
         ha='left', va='center')
    fig.text(x=0.45, y=0.65, s='2. Optional kinetic fit', 
         fontsize=12, fontweight='bold', 
         ha='left', va='center')
    
    fig.text(x=0.12, y=0.335, s='3. Differentiation', 
         fontsize=12, fontweight='bold', 
         ha='left', va='center')
    fig.text(x=0.70, y=0.335, s='4. Smoothing', 
         fontsize=12, fontweight='bold', 
         ha='left', va='center')
    
    arrow = patches.FancyArrow(
        x=0.57, y=0.35,           # Start position
        dx=0, dy=-0.04,            # Direction vector
        width=0.08,               # Width of arrow
        head_width=0.15,          # Width of arrow head
        head_length=0.02,         # Length of arrow head
        length_includes_head=True,
        color='grey',
        transform=fig.transFigure  # Use figure coordinates (0-1)
    )
    fig.patches.append(arrow)     # Add to figure

    ax[0].text(-0.25, 0.95, 'A',transform=ax[0].transAxes,fontsize=22, fontweight='bold')
    ax[1].text(-0.25, 0.95, 'B',transform=ax[1].transAxes,fontsize=22, fontweight='bold')
    ax[2].text(-0.25, 0.95, 'C',transform=ax[2].transAxes,fontsize=22, fontweight='bold')

    ax[0].text(0.2, 0.75, 'I',transform=ax[0].transAxes,fontsize=18, fontweight='bold')
    ax[0].text(0.53, 0.75, 'II',transform=ax[0].transAxes,fontsize=18, fontweight='bold')
    ax[0].text(0.77, 0.75, 'III',transform=ax[0].transAxes,fontsize=18, fontweight='bold')
    ax[0].text(0.915, 0.75, 'IV',transform=ax[0].transAxes,fontsize=18, fontweight='bold')

    #fig.savefig('Figures/HTE_Data_Processing.png', dpi = 500)

    plt.show()


if __name__ == "__main__":
    main()

