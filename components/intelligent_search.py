import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from typing import Dict, List
import json

def render_intelligent_search(data_processor, vector_search, llm_processor):
    """Render the intelligent search hub with LLM-powered RAG capabilities"""
    
    st.header("🧠 Intelligent Search Hub")
    
    if data_processor is None or vector_search is None:
        st.warning("Dataset not loaded. Loading default clinical genomics data...")
        return
    
    df = data_processor.get_data()
    
    # Main search interface
    st.markdown("### 💬 Ask Questions About Your Clinical Data")
    
    col1, col2 = st.columns([4, 1])
    
    with col1:
        query = st.text_input(
            "Enter your research question:",
            placeholder="e.g., What patterns do you see in patients with APOL1 mutations and kidney dysfunction?",
            help="Ask natural language questions about the clinical data"
        )
    
    with col2:
        st.markdown("<br>", unsafe_allow_html=True)
        search_clicked = st.button("🔍 Analyze", type="primary", use_container_width=True)
    
    # Quick action buttons
    st.markdown("### ⚡ Quick Actions")
    
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        if st.button("🧬 Genetic Insights", use_container_width=True):
            query = "Analyze genetic variant patterns and their clinical significance in this dataset"
            search_clicked = True
    
    with col2:
        if st.button("🎯 Trial Opportunities", use_container_width=True):
            query = "Identify optimal patient populations for clinical trials based on this data"
            search_clicked = True
    
    with col3:
        if st.button("⚠️ High-Risk Patients", use_container_width=True):
            query = "Find high-risk patients who need immediate clinical attention"
            search_clicked = True
    
    with col4:
        if st.button("📊 Research Summary", use_container_width=True):
            query = "Generate a comprehensive research summary of this clinical dataset"
            search_clicked = True
    
    # Process query with LLM
    if search_clicked and query.strip():
        with st.spinner("🧠 Analyzing your query with advanced AI..."):
            try:
                # Create context from the dataset
                context_data = _create_dataset_context(df)
                
                # Process query with LLM
                llm_response = llm_processor.process_clinical_query(query, context_data)
                
                if "error" not in llm_response:
                    st.markdown("---")
                    
                    # Modern card-style layout for AI Analysis
                    st.markdown("""
                    <div style="background: linear-gradient(135deg, #667eea 0%, #764ba2 100%); padding: 1rem; border-radius: 10px; margin: 1rem 0;">
                        <h3 style="color: white; margin: 0; display: flex; align-items: center;">
                            🤖 AI Analysis Results
                        </h3>
                    </div>
                    """, unsafe_allow_html=True)
                    
                    # Summary Card
                    st.markdown("""
                    <div style="background: #f8f9fa; border-left: 4px solid #007bff; padding: 1rem; border-radius: 5px; margin: 1rem 0;">
                    """, unsafe_allow_html=True)
                    st.markdown("**📋 Executive Summary**")
                    st.write(llm_response.get("summary", "No summary available"))
                    st.markdown("</div>", unsafe_allow_html=True)
                    
                    # Three column layout for key information
                    col1, col2, col3 = st.columns(3)
                    
                    with col1:
                        st.markdown("""
                        <div style="background: #e3f2fd; border-radius: 8px; padding: 1rem; height: 300px; overflow-y: auto;">
                            <h4 style="color: #1976d2; margin-top: 0;">🔍 Key Insights</h4>
                        </div>
                        """, unsafe_allow_html=True)
                        insights = llm_response.get("key_insights", [])
                        for i, insight in enumerate(insights, 1):
                            st.markdown(f"**{i}.** {insight}")
                    
                    with col2:
                        st.markdown("""
                        <div style="background: #f3e5f5; border-radius: 8px; padding: 1rem; height: 300px; overflow-y: auto;">
                            <h4 style="color: #7b1fa2; margin-top: 0;">🏥 Clinical Significance</h4>
                        </div>
                        """, unsafe_allow_html=True)
                        st.write(llm_response.get("clinical_significance", "No clinical significance noted"))
                        
                        st.markdown("**👥 Patient Populations:**")
                        st.write(llm_response.get("patient_populations", "No specific populations identified"))
                    
                    with col3:
                        st.markdown("""
                        <div style="background: #fff3e0; border-radius: 8px; padding: 1rem; height: 300px; overflow-y: auto;">
                            <h4 style="color: #ef6c00; margin-top: 0;">🎯 Suggested Action (Caution: AI Generated)</h4>
                        </div>
                        """, unsafe_allow_html=True)
                        actions = llm_response.get("recommended_actions", [])
                        for action in actions:
                            st.markdown(f"• {action}")
                        
                        st.markdown("**📊 Data References:**")
                        references = llm_response.get("data_references", [])
                        for ref in references:
                            st.markdown(f"• {ref}")
                    
                    # Find relevant patients based on query
                    st.markdown("---")
                    st.markdown("### 🔍 Relevant Patients Found")
                    
                    # Perform vector search to find relevant patients
                    indices, scores = vector_search.search(query, top_k=10, similarity_threshold=0.1)
                    
                    if indices:
                        relevant_df = df.iloc[indices].copy()
                        relevant_df['relevance_score'] = [f"{score:.3f}" for score in scores]
                        
                        # Display results
                        display_columns = [
                            'PatientID', 'Age', 'Sex', 'Ethnicity', 'Diagnosis', 
                            'eGFR', 'APOL1_Variant', 'Eligible_For_Trial', 'relevance_score'
                        ]
                        available_columns = [col for col in display_columns if col in relevant_df.columns]
                        
                        st.dataframe(
                            relevant_df[available_columns],
                            use_container_width=True,
                            height=300
                        )
                        
                        # Analyze the cohort with LLM
                        cohort_analysis = llm_processor.analyze_patient_cohort(relevant_df, query)
                        
                        if "error" not in cohort_analysis:
                            st.markdown("#### 🧠 AI Cohort Analysis")
                            
                            col1, col2 = st.columns(2)
                            
                            with col1:
                                st.markdown("**Cohort Summary:**")
                                st.write(cohort_analysis.get("cohort_summary", "No summary available"))
                                
                                st.markdown("**Key Characteristics:**")
                                chars = cohort_analysis.get("key_characteristics", [])
                                for char in chars:
                                    st.markdown(f"• {char}")
                            
                            with col2:
                                st.markdown("**Clinical Trial Suitability:**")
                                st.write(cohort_analysis.get("trial_suitability", "No assessment available"))
                                
                                st.markdown("**Recommendations:**")
                                recs = cohort_analysis.get("recommendations", [])
                                for rec in recs:
                                    st.markdown(f"• {rec}")
                        

                    else:
                        st.warning("No specific patients found matching your query, but the analysis above provides insights about the overall dataset.")
                
                else:
                    st.error(f"Analysis error: {llm_response.get('error', 'Unknown error')}")
                    
            except Exception as e:
                st.error(f"Error processing query: {str(e)}")
    

    
    # Smart Query Suggestions
    if not query.strip():
        st.markdown("---")
        st.markdown("### 💡 AI-Generated Smart Queries")
        
        # Generate intelligent suggestions
        suggestions = llm_processor.generate_smart_query_suggestions(df)
        
        if suggestions:
            cols = st.columns(2)
            for i, suggestion in enumerate(suggestions[:8]):
                col = cols[i % 2]
                with col:
                    if st.button(f"💭 {suggestion[:60]}...", key=f"smart_suggestion_{i}", help=suggestion):
                        query = suggestion
                        search_clicked = True
                        st.rerun()

def _create_dataset_context(df: pd.DataFrame) -> str:
    """Create context summary of the dataset for LLM processing"""
    try:
        context_parts = []
        
        # Dataset overview
        context_parts.append(f"Dataset contains {len(df)} patients with clinical genomics data.")
        
        # Demographics summary
        if 'Age' in df.columns:
            context_parts.append(f"Age range: {df['Age'].min()}-{df['Age'].max()} years (mean: {df['Age'].mean():.1f})")
        
        if 'Sex' in df.columns:
            sex_dist = df['Sex'].value_counts()
            context_parts.append(f"Sex distribution: {sex_dist.to_dict()}")
        
        if 'Ethnicity' in df.columns:
            eth_dist = df['Ethnicity'].value_counts()
            context_parts.append(f"Ethnicity distribution: {eth_dist.to_dict()}")
        
        # Clinical data summary
        if 'Diagnosis' in df.columns:
            diag_dist = df['Diagnosis'].value_counts().head(5)
            context_parts.append(f"Top diagnoses: {diag_dist.to_dict()}")
        
        if 'eGFR' in df.columns:
            context_parts.append(f"eGFR range: {df['eGFR'].min():.1f}-{df['eGFR'].max():.1f} (mean: {df['eGFR'].mean():.1f})")
            severe_ckd = len(df[df['eGFR'] < 30])
            context_parts.append(f"Patients with severe kidney dysfunction (eGFR < 30): {severe_ckd}")
        
        if 'Eligible_For_Trial' in df.columns:
            trial_eligible = df['Eligible_For_Trial'].value_counts()
            context_parts.append(f"Clinical trial eligibility: {trial_eligible.to_dict()}")
        
        # Genetic variants summary
        genetic_cols = ['APOL1_Variant', 'APOL1', 'NPHS1', 'NPHS2', 'WT1', 'UMOD', 'COL4A3']
        genetic_data = []
        for col in genetic_cols:
            if col in df.columns:
                if col == 'APOL1_Variant':
                    variants = df[col].value_counts()
                    genetic_data.append(f"{col}: {variants.to_dict()}")
                else:
                    mutations = (df[col] == 'Mut').sum()
                    wild_type = (df[col] == 'WT').sum()
                    genetic_data.append(f"{col} mutations: {mutations}/{mutations + wild_type}")
        
        if genetic_data:
            context_parts.append("Genetic variants: " + "; ".join(genetic_data))
        
        # Sample clinical notes
        if 'Clinical_Notes' in df.columns:
            sample_notes = df['Clinical_Notes'].dropna().head(3).tolist()
            if sample_notes:
                context_parts.append(f"Sample clinical notes: {'; '.join(sample_notes)}")
        
        return "\n".join(context_parts)
        
    except Exception as e:
        return f"Error creating context: {str(e)}"