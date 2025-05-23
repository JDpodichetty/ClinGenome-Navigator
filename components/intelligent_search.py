import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from typing import Dict, List
import json

def render_intelligent_search(data_processor, vector_search, llm_processor):
    """Render the intelligent search hub with LLM-powered RAG capabilities"""
    
    st.header("🧠 Intelligent Search Hub")
    st.markdown("**AI-Powered Clinical Genomics Research Assistant**")
    
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
                    st.markdown("### 🎯 AI Analysis Results")
                    
                    # Display LLM response
                    col1, col2 = st.columns([2, 1])
                    
                    with col1:
                        st.markdown("#### 📋 Summary")
                        st.info(llm_response.get("summary", "No summary available"))
                        
                        st.markdown("#### 🔍 Key Insights")
                        insights = llm_response.get("key_insights", [])
                        for i, insight in enumerate(insights, 1):
                            st.markdown(f"**{i}.** {insight}")
                        
                        st.markdown("#### 🏥 Clinical Significance")
                        st.markdown(llm_response.get("clinical_significance", "No clinical significance noted"))
                    
                    with col2:
                        st.markdown("#### 👥 Patient Populations")
                        st.markdown(llm_response.get("patient_populations", "No specific populations identified"))
                        
                        st.markdown("#### 📝 Recommended Actions")
                        actions = llm_response.get("recommended_actions", [])
                        for action in actions:
                            st.markdown(f"• {action}")
                        
                        st.markdown("#### 📊 Data References")
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
                        
                        # Export functionality
                        col1, col2, col3 = st.columns([1, 1, 2])
                        
                        with col1:
                            if st.button("📥 Export Results"):
                                csv_data = data_processor.export_results(relevant_df, 'csv')
                                st.download_button(
                                    label="Download CSV",
                                    data=csv_data,
                                    file_name=f"ai_search_results_{len(relevant_df)}_patients.csv",
                                    mime="text/csv"
                                )
                        
                        with col2:
                            if st.button("📊 Export Analysis"):
                                analysis_text = f"Query: {query}\n\n"
                                analysis_text += f"Summary: {llm_response.get('summary', '')}\n\n"
                                analysis_text += "Key Insights:\n"
                                for insight in llm_response.get('key_insights', []):
                                    analysis_text += f"- {insight}\n"
                                
                                st.download_button(
                                    label="Download Analysis",
                                    data=analysis_text,
                                    file_name=f"ai_analysis_{len(relevant_df)}_patients.txt",
                                    mime="text/plain"
                                )
                    else:
                        st.warning("No specific patients found matching your query, but the analysis above provides insights about the overall dataset.")
                
                else:
                    st.error(f"Analysis error: {llm_response.get('error', 'Unknown error')}")
                    
            except Exception as e:
                st.error(f"Error processing query: {str(e)}")
    
    # Clinical Insights Extraction Section
    st.markdown("---")
    st.markdown("### 📝 Clinical Notes Analysis")
    
    col1, col2 = st.columns([3, 1])
    
    with col1:
        st.markdown("Extract insights from clinical notes using advanced AI analysis")
    
    with col2:
        if st.button("🔬 Analyze Clinical Notes", use_container_width=True):
            with st.spinner("🧠 Extracting insights from clinical notes..."):
                try:
                    # Get clinical notes
                    if 'Clinical_Notes' in df.columns:
                        clinical_notes = df['Clinical_Notes'].dropna().tolist()
                        
                        if clinical_notes:
                            # Extract insights using LLM
                            insights = llm_processor.extract_clinical_insights(clinical_notes)
                            
                            if "error" not in insights:
                                st.markdown("#### 🔬 Clinical Insights Extracted")
                                
                                col1, col2 = st.columns(2)
                                
                                with col1:
                                    st.markdown("**Overall Summary:**")
                                    st.info(insights.get("overall_summary", "No summary available"))
                                    
                                    st.markdown("**Disease Patterns:**")
                                    patterns = insights.get("disease_patterns", [])
                                    for pattern in patterns:
                                        st.markdown(f"• {pattern}")
                                    
                                    st.markdown("**Treatment Insights:**")
                                    treatments = insights.get("treatment_insights", [])
                                    for treatment in treatments:
                                        st.markdown(f"• {treatment}")
                                
                                with col2:
                                    st.markdown("**Risk Factors:**")
                                    risks = insights.get("risk_factors", [])
                                    for risk in risks:
                                        st.markdown(f"• {risk}")
                                    
                                    st.markdown("**Research Opportunities:**")
                                    opportunities = insights.get("research_opportunities", [])
                                    for opp in opportunities:
                                        st.markdown(f"• {opp}")
                                    
                                    urgent = insights.get("urgent_findings", [])
                                    if urgent:
                                        st.markdown("**⚠️ Urgent Findings:**")
                                        for finding in urgent:
                                            st.error(f"• {finding}")
                            else:
                                st.error(f"Error extracting insights: {insights.get('error', 'Unknown error')}")
                        else:
                            st.warning("No clinical notes found in the dataset")
                    else:
                        st.warning("No clinical notes column found in the dataset")
                        
                except Exception as e:
                    st.error(f"Error analyzing clinical notes: {str(e)}")
    
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