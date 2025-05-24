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
    # Static header ribbon with GPT-4o logo
    st.markdown("""
    <div style="background: linear-gradient(90deg, #1f77b4 0%, #2e8b57 100%); padding: 1rem; margin: -1rem -1rem 2rem -1rem; border-radius: 0; display: flex; justify-content: space-between; align-items: center;">
        <div>
            <h1 style="color: white; margin: 0; font-size: 2.5rem; font-weight: bold;">🧬 ClinGenome Navigator</h1>
            <p style="color: #e6f3ff; margin: 0.5rem 0 0 0; font-size: 1.2rem;">GenAI based Clinico Genomics Research Platform</p>
        </div>
        <div style="display: flex; align-items: center; color: white;">
            <span style="font-size: 0.9rem; margin-right: 0.5rem;">Powered by</span>
            <div style="background: white; padding: 0.3rem 0.6rem; border-radius: 20px; display: flex; align-items: center;">
                <svg width="24" height="24" viewBox="0 0 100 100" style="margin-right: 0.4rem;">
                    <g fill="#000">
                        <path d="M20,30 Q30,15 50,25 Q70,15 80,30 Q85,50 70,65 Q50,75 30,65 Q15,50 20,30 Z" opacity="0.8"/>
                        <path d="M30,25 Q50,10 70,25 Q75,40 60,55 Q40,65 20,55 Q25,40 30,25 Z" opacity="0.6"/>
                        <path d="M35,35 Q50,20 65,35 Q70,50 55,65 Q40,70 25,55 Q30,40 35,35 Z" opacity="0.9"/>
                        <circle cx="50" cy="50" r="6" fill="white"/>
                    </g>
                </svg>
                <span style="color: #000; font-size: 0.9rem; font-weight: bold;">GPT-4o</span>
            </div>
        </div>
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
        # Tab interface with larger, more intuitive styling
        st.markdown("""
        <style>
        .stTabs [data-baseweb="tab-list"] {
            gap: 8px;
        }
        .stTabs [data-baseweb="tab"] {
            height: 60px;
            font-size: 18px !important;
            font-weight: bold !important;
            background-color: #f0f2f6;
            border-radius: 8px 8px 0 0;
            padding: 12px 24px;
        }
        .stTabs [aria-selected="true"] {
            background-color: #1f77b4 !important;
            color: white !important;
        }
        </style>
        """, unsafe_allow_html=True)
        
        tab1, tab2 = st.tabs(["🧠 Intelligent Search Hub", "📊 Data Exploration"])
        
        with tab1:
            render_intelligent_search(st.session_state.data_processor, st.session_state.vector_search, st.session_state.llm_processor)
        
        with tab2:
            render_visualization(st.session_state.data_processor)

if __name__ == "__main__":
    main()
