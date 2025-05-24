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
from utils.vector_search_new import VectorSearch
from utils.llm_processor import LLMProcessor

# Configure page
st.set_page_config(
    page_title="ClinGenome Navigator",
    page_icon="🧬",
    layout="wide",
    initial_sidebar_state="collapsed"
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
    # Static header ribbon
    st.markdown("""
    <div style="background: linear-gradient(90deg, #1f77b4 0%, #2e8b57 100%); padding: 1rem; margin: -1rem -1rem 2rem -1rem; border-radius: 0;">
        <h1 style="color: white; margin: 0; font-size: 2.5rem; font-weight: bold;">🧬 ClinGenome Navigator</h1>
        <p style="color: #e6f3ff; margin: 0.5rem 0 0 0; font-size: 1.2rem;">GenAI based Clinico Genomics Research Platform</p>
    </div>
    """, unsafe_allow_html=True)
    
    # Auto-load data if available
    if not st.session_state.data_loaded:
        with st.spinner("Loading clinical genomics dataset..."):
            if load_default_data():
                st.success("Dataset loaded successfully!")
                st.rerun()
    
    # Main content area
    if not st.session_state.data_loaded:
        # Welcome screen
        st.markdown("""
        ### 🚀 Welcome to Your Clinical Research Platform
        
        **Loading your comprehensive clinical genomics dataset...**
        
        **Your dataset includes:**
        - **1,500 patient records** with clinical and genetic data
        - **Genetic variants**: APOL1, NPHS1, NPHS2, WT1, UMOD, COL4A3 analysis
        - **Clinical metrics**: eGFR, Creatinine, diagnosis, and medication tracking
        - **Clinical notes**: Rich text data for AI-powered insights extraction
        """)
        
        if st.button("🚀 Load Dataset & Start Research", type="primary"):
            st.rerun()
    
    else:
        # Tab interface
        tab1, tab2 = st.tabs(["🧠 Intelligent Search Hub", "📊 Data Exploration"])
        
        with tab1:
            render_intelligent_search(st.session_state.data_processor, st.session_state.vector_search, st.session_state.llm_processor)
        
        with tab2:
            render_visualization(st.session_state.data_processor)

if __name__ == "__main__":
    main()
