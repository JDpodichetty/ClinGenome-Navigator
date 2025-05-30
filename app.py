import streamlit as st
import pandas as pd
import os
from pathlib import Path

# Import custom components
from components.dashboard import render_dashboard
from components.data_overview import render_data_overview
from components.visualization import render_visualization
from components.intelligent_search_clean import render_intelligent_search_clean
from components.simple_knowledge_graph import render_simple_knowledge_graph
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
    default_file = "attached_assets/extended_clinicogenomic_synthetic_data - Corrected.csv"
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
            <h1 style="color: white; margin: 0; font-size: 2.5rem; font-weight: bold;">🧬 ClinGenome Navigator <span style="font-size: 1.2rem; font-weight: normal; color: #e6f3ff;">Demo</span></h1>
            <p style="color: #e6f3ff; margin: 0.5rem 0 0 0; font-size: 1.2rem;">GenAI based Clinico Genomics Research Platform</p>
        </div>
        <div style="display: flex; align-items: center; color: white;">
            <span style="font-size: 0.9rem; margin-right: 0.5rem;">Powered by</span>
            <div style="background: white; padding: 0.4rem 0.8rem; border-radius: 25px; display: flex; align-items: center; box-shadow: 0 2px 4px rgba(0,0,0,0.1);">
                <svg width="20" height="20" viewBox="0 0 24 24" style="margin-right: 0.5rem;" fill="none">
                    <path d="M22.2819 9.8211a5.9847 5.9847 0 0 0-.5157-4.9108 6.0462 6.0462 0 0 0-6.5098-2.9A6.0651 6.0651 0 0 0 4.9807 4.1818a5.9847 5.9847 0 0 0-3.9977 2.9 6.0462 6.0462 0 0 0 .7427 7.0966 5.98 5.98 0 0 0 .511 4.9107 6.051 6.051 0 0 0 6.5146 2.9001A5.9847 5.9847 0 0 0 13.2599 24a6.0557 6.0557 0 0 0 5.7718-4.2058 5.9894 5.9894 0 0 0 3.9977-2.9001 6.0557 6.0557 0 0 0-.7475-7.0729zm-9.022 12.6081a4.4755 4.4755 0 0 1-2.8764-1.0408l.1419-.0804 4.7783-2.7582a.7948.7948 0 0 0 .3927-.6813v-6.7369l2.02 1.1686a.071.071 0 0 1 .038.052v5.5826a4.504 4.504 0 0 1-4.4945 4.4944zm-9.6607-4.1254a4.4708 4.4708 0 0 1-.5346-3.0137l.142.0852 4.783 2.7582a.7712.7712 0 0 0 .7806 0l5.8428-3.3685v2.3324a.0804.0804 0 0 1-.0332.0615L9.74 19.9502a4.4992 4.4992 0 0 1-6.1408-1.6464zM2.3408 7.8956a4.485 4.485 0 0 1 2.3655-1.9728V11.6a.7664.7664 0 0 0 .3879.6765l5.8144 3.3543-2.0201 1.1685a.0757.0757 0 0 1-.071 0l-4.8303-2.7865A4.504 4.504 0 0 1 2.3408 7.872zm16.5963 3.8558L13.1038 8.364 15.1192 7.2a.0757.0757 0 0 1 .071 0l4.8303 2.7913a4.4944 4.4944 0 0 1-.6765 8.1042v-5.6772a.79.79 0 0 0-.407-.667zm2.0107-3.0231l-.142-.0852-4.7735-2.7818a.7759.7759 0 0 0-.7854 0L9.409 9.2297V6.8974a.0662.0662 0 0 1 .0284-.0615l4.8303-2.7866a4.4992 4.4992 0 0 1 6.6802 4.66zM8.3065 12.863l-2.02-1.1638a.0804.0804 0 0 1-.038-.0567V6.0742a4.4992 4.4992 0 0 1 7.3757-3.4537l-.142.0805L8.704 5.459a.7948.7948 0 0 0-.3927.6813zm1.0976-2.3654l2.602-1.4998 2.6069 1.4998v2.9994l-2.5974 1.4997-2.6067-1.4997Z" fill="#000"/>
                </svg>
                <span style="color: #000; font-size: 1rem; font-weight: bold; letter-spacing: -0.5px;">GPT-4o</span>
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
            font-size: 22px !important;
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
        
        tab1, tab2, tab3 = st.tabs(["🧠 Intelligent Search Hub", "📊 Data Exploration", "📈 Knowledge Graph"])
        
        with tab1:
            render_intelligent_search_clean(st.session_state.data_processor, st.session_state.vector_search, st.session_state.llm_processor)
        
        with tab2:
            render_visualization(st.session_state.data_processor)
        
        with tab3:
            render_simple_knowledge_graph(st.session_state.data_processor, st.session_state.vector_search, st.session_state.llm_processor)
    
    # Copyright footer
    st.markdown("---")
    st.markdown(
        "<div style='text-align: center; color: #666; font-size: 0.9rem; padding: 2rem 0;'>"
        "© 2025 Jagdeep Podichetty"
        "</div>", 
        unsafe_allow_html=True
    )

if __name__ == "__main__":
    main()
