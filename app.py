import streamlit as st
import pandas as pd
import os
from pathlib import Path

# Import custom components
from components.dashboard import render_dashboard
from components.search_interface import render_search_interface
from components.data_overview import render_data_overview
from components.visualization import render_visualization
from components.intelligent_search import render_intelligent_search
from utils.data_processor import DataProcessor
from utils.vector_search import VectorSearch
from utils.llm_processor import LLMProcessor

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
if 'llm_processor' not in st.session_state:
    st.session_state.llm_processor = LLMProcessor()
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
            ["Intelligent Search Hub", "Dashboard", "Data Overview", "Advanced Analytics"]
        )
    else:
        page = "Intelligent Search Hub"
    
    # Main content area
    if not st.session_state.data_loaded:
        # Welcome screen with integrated search
        render_intelligent_search(None, None, st.session_state.llm_processor)
        
        st.markdown("---")
        st.markdown("""
        ## Welcome to ClinGenome Navigator
        
        **AI-Powered Clinical Genomics Research Platform**
        
        ### 🧠 Enhanced Features:
        - **Intelligent Search**: Ask questions in natural language and get AI-powered insights
        - **Clinical Analysis**: Extract patterns from clinical notes using advanced LLM processing
        - **RAG Search**: Retrieval-augmented generation for precise clinical research queries
        - **Smart Insights**: Automated identification of research opportunities and patient cohorts
        
        ### 📊 Dataset Overview:
        Your clinical genomics dataset includes:
        - **1,500 patient records** with comprehensive clinical and genetic data
        - **Genetic variants**: APOL1, NPHS1, NPHS2, WT1, UMOD, COL4A3 analysis
        - **Clinical metrics**: eGFR, Creatinine, diagnosis, and medication tracking
        - **Trial eligibility**: Patient suitability assessment for clinical studies
        - **Clinical notes**: Rich text data for AI-powered insights extraction
        """)
        
        if st.button("🚀 Load Dataset & Start Research", type="primary"):
            st.rerun()
    
    else:
        # Render selected page
        if page == "Intelligent Search Hub":
            render_intelligent_search(st.session_state.data_processor, st.session_state.vector_search, st.session_state.llm_processor)
        elif page == "Dashboard":
            render_dashboard(st.session_state.data_processor, st.session_state.vector_search)
        elif page == "Data Overview":
            render_data_overview(st.session_state.data_processor)
        elif page == "Advanced Analytics":
            render_visualization(st.session_state.data_processor)

if __name__ == "__main__":
    main()
