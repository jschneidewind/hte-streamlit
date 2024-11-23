import streamlit as st
import os
import tempfile
import sys

from hte_streamlit.file_processing import analyze_single_file

st.set_page_config(
    page_title="HTE Data Upload and Download", 
    page_icon="📤",
    layout="wide",
    initial_sidebar_state="expanded"
)

def save_uploaded_files(uploaded_files, temp_dir):
    """Save uploaded files to temporary directory and return their paths"""
    saved_paths = []
    for uploaded_file in uploaded_files:
        # Create a temporary file path
        temp_path = os.path.join(temp_dir, uploaded_file.name)
        
        # Write the uploaded file to the temporary location
        with open(temp_path, 'wb') as f:
            f.write(uploaded_file.getvalue())
        
        saved_paths.append(temp_path)
    
    return saved_paths

def process_files(csv_files, overview_file, dataset):
    """Process uploaded CSV files using the existing analysis functions"""
    # Create temporary directory for processing
    with tempfile.TemporaryDirectory() as temp_dir:
        # Create necessary subdirectories
        os.makedirs(os.path.join(temp_dir, 'csv'), exist_ok=True)
        os.makedirs(os.path.join(temp_dir, 'png'), exist_ok=True)
        
        # Save uploaded files
        csv_paths = save_uploaded_files(csv_files, os.path.join(temp_dir, 'csv'))
        
        # Update overview if new file provided
        if overview_file is not None:
            dataset.update_metadata(overview_file)
        # else:
        #     st.warning("Please upload a new overview file.")
        #     return
        
        # Process each CSV file
        total_files = len(csv_paths)
        progress_bar = st.progress(0)
        status_text = st.empty()
        
        for i, file_path in enumerate(csv_paths):
            try:
                file_name = os.path.basename(file_path)
                status_text.text(f"Processing {file_name}...")
                
                # Analyze the file
                result = analyze_single_file(
                    file_name, 
                    temp_dir, 
                    dataset,
                    plotting = False,
                    plot_baseline = False
                )

                if result['success']:
                    dataset.add_experiment(result['experiment_name'], result['data'])
                    st.success(f"Successfully processed {file_name}")
                else:
                    st.error(f"Failed to process {file_name}: {result['error']}")
                
            except Exception as e:
                st.error(f"Error processing {file_name}: {str(e)}")
            
            # Update progress
            progress_bar.progress((i + 1) / total_files)
        
        status_text.text("Processing complete!")
        return dataset

def main():
    st.title("Data Upload and Download")
        
    if st.session_state.experimental_dataset is not None:
    
        dataset = st.session_state.experimental_dataset
        
        # File upload section
        st.header("Upload Files")
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.subheader("Overview File")
            uploaded_overview = st.file_uploader(
                    "Upload new overview Excel file",
                    type=['xlsx']
                )

        with col2:
            st.subheader("CSV Files")
            uploaded_csv_files = st.file_uploader(
                "Upload CSV files for analysis",
                type=['csv'],
                accept_multiple_files=True
            )
        
        # Process files when user clicks the button
        if st.button("Process Files"):
            if not uploaded_csv_files:
                st.warning("Please upload at least one CSV file.")
                return
            
            with st.spinner("Processing files..."):
                try:
                    # Process the files
                    updated_dataset = process_files(
                        uploaded_csv_files,
                        uploaded_overview,
                        dataset
                    )
                    
                    # Update session state
                    st.session_state.experimental_dataset = updated_dataset
                                        
                except Exception as e:
                    st.error(f"Error during processing: {str(e)}")
                                # Display updated overview

                # Save updated dataset
        if st.button("Save Updated Dataset"):
            with tempfile.NamedTemporaryFile(delete=False, suffix='.h5') as tmp_file:
                st.session_state.experimental_dataset.save_to_hdf5(tmp_file.name)
                st.download_button(
                    label="Download Updated Dataset",
                    data=open(tmp_file.name, 'rb').read(),
                    file_name="updated_dataset.h5",
                    mime="application/x-hdf5"
                )

        st.header("Analyzed Experiments")
        df = st.session_state.experimental_dataset.overview_df
        filtered_df = df[df['max rate ydiff'].notna()]
        st.dataframe(filtered_df, use_container_width=True)


        st.header("Full Dataset Overview")
        st.dataframe(st.session_state.experimental_dataset.overview_df, use_container_width=True)

    else:
        st.info("Please upload a HDF5 file on the home page first.")

if __name__ == "__main__":
    main()