import streamlit as st
import pandas as pd
import os
from pathlib import Path

# Import custom components
from components.dashboard import render_dashboard
from components.search_interface import render_search_interface
from components.data_overview import render_data_overview
from components.visualization import render_visualization
from utils.data_processor import DataProcessor
from utils.vector_search import VectorSearch

# Configure page
st.set_page_config(
    page_title="ClinGenome Navigator",
    page_icon="🧬",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Initialize session state
if 'data_processor' not in st.session_state:
    st.session_state.data_processor = None
if 'vector_search' not in st.session_state:
    st.session_state.vector_search = None
if 'data_loaded' not in st.session_state:
    st.session_state.data_loaded = False

def load_default_data():
    """Load the default CSV file if it exists"""
    default_file = "attached_assets/extended_clinicogenomic_synthetic_data.csv"
    if os.path.exists(default_file):
        try:
            data_processor = DataProcessor()
            df = data_processor.load_data(default_file)
            
            # Initialize vector search
            vector_search = VectorSearch()
            vector_search.build_index(df)
            
            st.session_state.data_processor = data_processor
            st.session_state.vector_search = vector_search
            st.session_state.data_loaded = True
            
            return True
        except Exception as e:
            st.error(f"Error loading default data: {str(e)}")
            return False
    return False

def main():
    # Header
    st.title("🧬 ClinGenome Navigator")
    st.markdown("**Advanced Clinical Genomics Research Platform**")
    
    # Sidebar navigation
    st.sidebar.title("Navigation")
    
    # Auto-load data if available
    if not st.session_state.data_loaded:
        with st.spinner("Loading clinical genomics dataset..."):
            if load_default_data():
                st.success("Dataset loaded successfully!")
                st.rerun()
    
    # Navigation options
    if st.session_state.data_loaded:
        page = st.sidebar.selectbox(
            "Select a view:",
            ["Dashboard", "Data Overview", "Semantic Search", "Advanced Analytics"]
        )
    else:
        page = "Data Upload"
    
    # File upload section (always available)
    st.sidebar.markdown("---")
    st.sidebar.subheader("Data Management")
    
    uploaded_file = st.sidebar.file_uploader(
        "Upload Clinical Data (CSV)",
        type=['csv'],
        help="Upload a CSV file containing clinical genomics data"
    )
    
    if uploaded_file is not None:
        try:
            with st.spinner("Processing uploaded data..."):
                data_processor = DataProcessor()
                df = data_processor.load_uploaded_data(uploaded_file)
                
                # Initialize vector search
                vector_search = VectorSearch()
                vector_search.build_index(df)
                
                st.session_state.data_processor = data_processor
                st.session_state.vector_search = vector_search
                st.session_state.data_loaded = True
                
            st.sidebar.success("Data uploaded successfully!")
            st.rerun()
            
        except Exception as e:
            st.sidebar.error(f"Error processing file: {str(e)}")
    
    # Main content area
    if not st.session_state.data_loaded:
        # Welcome screen
        st.markdown("""
        ## Welcome to ClinGenome Navigator
        
        **Professional Clinical Genomics Research Platform**
        
        This platform provides advanced semantic search and analytics capabilities for clinical genomics research.
        
        ### Key Features:
        - **Semantic Search**: Natural language queries across clinical and genomic data
        - **Vector Database**: Efficient similarity search using advanced embeddings
        - **Interactive Analytics**: Comprehensive data visualization and filtering
        - **Export Capabilities**: Download search results and insights
        
        ### Getting Started:
        1. The platform will automatically load the sample dataset containing 1,500 patient records
        2. Alternatively, upload your own clinical genomics CSV file using the sidebar
        3. Explore the data using the navigation menu once loaded
        
        ### Dataset Overview:
        Our sample dataset includes:
        - Patient demographics and clinical information
        - Genetic variant data (APOL1, NPHS1, NPHS2, WT1, UMOD, COL4A3)
        - Kidney function markers (eGFR, Creatinine)
        - Diagnosis and medication information
        - Clinical trial eligibility status
        """)
        
        if st.button("🔄 Try Loading Default Dataset", type="primary"):
            st.rerun()
    
    else:
        # Render selected page
        if page == "Dashboard":
            render_dashboard(st.session_state.data_processor, st.session_state.vector_search)
        elif page == "Data Overview":
            render_data_overview(st.session_state.data_processor)
        elif page == "Semantic Search":
            render_search_interface(st.session_state.data_processor, st.session_state.vector_search)
        elif page == "Advanced Analytics":
            render_visualization(st.session_state.data_processor)

if __name__ == "__main__":
    main()
