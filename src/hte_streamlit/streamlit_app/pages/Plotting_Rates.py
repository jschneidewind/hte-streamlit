import streamlit as st
import pandas as pd
import plotly.graph_objects as go
import re

def get_experiment_group(exp_name):
    # Remove the last number and any trailing dash
    return re.sub(r'-?\d+$', '', exp_name)

def create_visualization(data, selected_outcome):
    # Group experiments
    data['group'] = data['Experiment'].apply(get_experiment_group)
    
    # Create figure
    fig = go.Figure()
    
    # Add vertical lines for each group
    for i, (group, group_data) in enumerate(data.groupby('group')):
        y_values = group_data[selected_outcome]
        
        # Add vertical line
        fig.add_trace(go.Scatter(
            x=[i] * len(y_values),
            y=y_values,
            mode='markers',
            name=group,
            text=group_data['Experiment'],  # Create array of experiment names
            hovertemplate=(
            "Experiment: %{text}<br>" +
            f"Value: %{{y:.3f}}<br>"
        )
        ))
        
        # Add error bars
        if len(y_values) > 1:
            fig.add_trace(go.Scatter(
                x=[i, i],
                y=[min(y_values), max(y_values)],
                mode='lines',
                showlegend=False,
                line=dict(color='gray', width=1),
                hoverinfo='skip'
            ))

    # Update layout
    fig.update_layout(
        title=f'{selected_outcome} by Experiment Group',
        xaxis_title='Experiment Group',
        yaxis_title=selected_outcome,
        xaxis=dict(
            tickmode='array',
            ticktext=list(data.groupby('group').groups.keys()),
            tickvals=list(range(len(data.groupby('group'))))
        ),
        showlegend=False,
        hovermode='closest'
    )
    
    return fig


st.set_page_config(
    page_title="HTE Data Visualization", 
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Title
st.title("Plotting Rates")

if st.session_state.experimental_dataset is not None:
    df = st.session_state.experimental_dataset.overview_df

    st.header("Experiment Analysis")
    
    # Create filter columns
    col1, col2, col3, col4, col5 = st.columns(5)
  
    with col1:
        # Filter by irradiation intensity concentration
        intensities = sorted(df['Power output [W/m^2]'].unique())
        selected_intensities = st.multiselect(
            'Select irradiation intensity [W/m^2]',
            intensities,
            default=list()
        )
    
    with col2:
        # Filter by Ru concentration
        ru_concentrations = sorted(df['c([Ru(bpy(3]Cl2) [M]'].unique())
        selected_ru = st.multiselect(
            'Select [Ru] Catalyst Concentrations [M]',
            ru_concentrations,
            default=list()
        )
    
    with col3:
        # Filter by persulfate concentration
        persulfate_concentrations = sorted(df['c(Na2S2O8) [M]'].unique())
        selected_persulfate = st.multiselect(
            'Select Persulfate Concentrations [M]',
            persulfate_concentrations,
            default=list()
        )
    
    with col4:
        # Filter by pH
        ph_values = sorted(df['pH [-]'].unique())
        selected_ph = st.multiselect(
            'Select pH Values',
            ph_values,
            default=list()
        )

    with col5:
        selected_outcome = st.radio(
            'Select analysis outcome',  # Label
            ['rate', 'max rate', 'max rate ydiff', 'rate constant']  # List of options
        )

    # Apply filters
    mask = (
        df['Power output [W/m^2]'].isin(selected_intensities) &
        df['c([Ru(bpy(3]Cl2) [M]'].isin(selected_ru) &
        df['c(Na2S2O8) [M]'].isin(selected_persulfate) &
        df['pH [-]'].isin(selected_ph)
    )
    filtered_df = df[mask]

    # Create and display visualization
    if not filtered_df.empty:
        fig = create_visualization(filtered_df, selected_outcome)
        st.plotly_chart(fig)
    else:
        st.warning('No data available for the selected filters.')

    st.dataframe(filtered_df, use_container_width=True)

else:
    st.info("Please upload a HDF5 file on the home page first.")

