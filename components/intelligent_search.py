import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from typing import Dict, List
import json


def render_intelligent_search(data_processor, vector_search, llm_processor):
    """Render the intelligent search hub with LLM-powered RAG capabilities"""

    st.header("Intelligent Search Hub")

    if data_processor is None or vector_search is None:
        st.warning(
            "Dataset not loaded. Loading default clinical genomics data...")
        return

    df = data_processor.get_data()

    # Description of the tool
    st.markdown("""
    **ClinGenome Navigator** analyzes a synthetic clinico genomic data for kidney disease research with focus on chronic kidney disease (CKD) and genetic variants. 
    This platform combines patient demographics, genetic markers (APOL1, NPHS1, NPHS2), and clinical notes to generate insights and identify research opportunities for sponsors. 
    Use natural language queries to explore patterns, analyze cohorts, and extract insights from 1,500 patient records for pharmaceutical research and clinical trial planning.
    """)

    st.markdown("---")

    col1, col2 = st.columns([4, 1])

    with col1:
        query = st.text_input(
            "Enter your research question:",
            placeholder=
            "e.g., What patterns do you see in patients with APOL1 mutations and kidney dysfunction?",
            help="Ask natural language questions about the clinical data")

    with col2:
        st.markdown("<br>", unsafe_allow_html=True)
        search_clicked = st.button("🔍 Analyze",
                                   type="primary",
                                   use_container_width=True)

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
                llm_response = llm_processor.process_clinical_query(
                    query, context_data)

                if "error" not in llm_response:
                    st.markdown("---")

                    # Header for Analysis Results
                    st.markdown("### Analysis Results")

                    # Summary section
                    st.markdown("#### Executive Summary")
                    st.info(llm_response.get("summary",
                                             "No summary available"))

                    # Three column layout for key information in colored boxes
                    col1, col2, col3 = st.columns(3)

                    with col1:
                        # Blue container for Key Insights
                        st.markdown("""
                        <div style="background-color: #e8f4f8; padding: 1rem; border-radius: 8px; border-left: 4px solid #1f77b4; margin-bottom: 1rem;">
                            <h4 style="color: #1f77b4; margin: 0 0 0.5rem 0;">Key Insights</h4>
                        """, unsafe_allow_html=True)
                        
                        insights = llm_response.get("key_insights", [])
                        for i, insight in enumerate(insights, 1):
                            st.markdown(f"**{i}.** {insight}")
                        
                        st.markdown("</div>", unsafe_allow_html=True)

                    with col2:
                        # Purple container for Clinical Significance
                        st.markdown("""
                        <div style="background-color: #f3e8f8; padding: 1rem; border-radius: 8px; border-left: 4px solid #8e44ad; margin-bottom: 1rem;">
                            <h4 style="color: #8e44ad; margin: 0 0 0.5rem 0;">Clinical Significance</h4>
                        """, unsafe_allow_html=True)
                        
                        st.write(
                            llm_response.get("clinical_significance",
                                             "No clinical significance noted"))

                        st.markdown("**Patient Populations:**")
                        st.write(
                            llm_response.get(
                                "patient_populations",
                                "No specific populations identified"))
                        
                        st.markdown("</div>", unsafe_allow_html=True)

                    with col3:
                        # Yellow container for Suggested Actions
                        st.markdown("""
                        <div style="background-color: #fff3cd; padding: 1rem; border-radius: 8px; border-left: 4px solid #ffc107; margin-bottom: 1rem;">
                            <h4 style="color: #d97706; margin: 0 0 0.5rem 0;">Suggested Actions</h4>
                        """, unsafe_allow_html=True)
                        
                        actions = llm_response.get("recommended_actions", [])
                        for action in actions:
                            st.markdown(f"• {action}")

                        st.markdown("**Data References:**")
                        references = llm_response.get("data_references", [])
                        for ref in references:
                            st.markdown(f"• {ref}")
                        
                        st.markdown("</div>", unsafe_allow_html=True)

                    # Perform vector search to find relevant patients
                    indices, scores = vector_search.search(
                        query, top_k=1500, similarity_threshold=0.0)

                    if indices:
                        relevant_df = df.iloc[indices].copy()

                        # 1. MOVED: Cohort Analysis BEFORE Relevant Patients Found
                        cohort_analysis = llm_processor.analyze_patient_cohort(
                            relevant_df, query)

                        if "error" not in cohort_analysis:
                            st.markdown("---")
                            st.markdown("### 🧠 Cohort Analysis")

                            # Key metrics overview
                            col1, col2, col3, col4 = st.columns(4)
                            with col1:
                                st.metric("Total Patients", len(relevant_df))
                            with col2:
                                trial_eligible = len(relevant_df[relevant_df.get('Eligible_For_Trial') == 'Yes']) if 'Eligible_For_Trial' in relevant_df.columns else 0
                                st.metric("Trial Eligible", trial_eligible)
                            with col3:
                                avg_age = relevant_df['Age'].mean() if 'Age' in relevant_df.columns else 0
                                st.metric("Avg Age", f"{avg_age:.1f}")
                            with col4:
                                avg_egfr = relevant_df['eGFR'].mean() if 'eGFR' in relevant_df.columns else 0
                                st.metric("Avg eGFR", f"{avg_egfr:.1f}")

                            # 3. ADDED: Visualization/Summary Tables
                            
                            # Demographics Summary Table
                            col1, col2 = st.columns(2)
                            with col1:
                                st.markdown("#### Demographics Summary")
                                demo_summary = []
                                if 'Sex' in relevant_df.columns:
                                    sex_counts = relevant_df['Sex'].value_counts()
                                    for sex, count in sex_counts.items():
                                        demo_summary.append({"Category": "Sex", "Value": sex, "Count": count, "Percentage": f"{(count/len(relevant_df)*100):.1f}%"})
                                
                                if 'Ethnicity' in relevant_df.columns:
                                    eth_counts = relevant_df['Ethnicity'].value_counts().head(3)
                                    for eth, count in eth_counts.items():
                                        demo_summary.append({"Category": "Ethnicity", "Value": eth, "Count": count, "Percentage": f"{(count/len(relevant_df)*100):.1f}%"})
                                
                                if demo_summary:
                                    st.dataframe(pd.DataFrame(demo_summary), use_container_width=True, hide_index=True)
                            
                            with col2:
                                st.markdown("#### Clinical Characteristics")
                                clinical_summary = []
                                if 'Diagnosis' in relevant_df.columns:
                                    diag_counts = relevant_df['Diagnosis'].value_counts().head(4)
                                    for diag, count in diag_counts.items():
                                        clinical_summary.append({"Category": "Diagnosis", "Value": diag, "Count": count, "Percentage": f"{(count/len(relevant_df)*100):.1f}%"})
                                
                                if 'APOL1_Variant' in relevant_df.columns:
                                    high_risk_variants = ['G1/G1', 'G1/G2', 'G2/G2']
                                    high_risk_count = len(relevant_df[relevant_df['APOL1_Variant'].isin(high_risk_variants)])
                                    clinical_summary.append({"Category": "APOL1", "Value": "High-Risk Variants", "Count": high_risk_count, "Percentage": f"{(high_risk_count/len(relevant_df)*100):.1f}%"})
                                
                                if clinical_summary:
                                    st.dataframe(pd.DataFrame(clinical_summary), use_container_width=True, hide_index=True)
                            
                            # Clinical Risk Stratification - moved to appear BEFORE colored boxes
                            st.markdown("#### Risk Stratification")
                            risk_data = []
                            if 'eGFR' in relevant_df.columns:
                                severe_ckd = len(relevant_df[relevant_df['eGFR'] < 30])
                                moderate_ckd = len(relevant_df[(relevant_df['eGFR'] >= 30) & (relevant_df['eGFR'] < 60)])
                                mild_ckd = len(relevant_df[relevant_df['eGFR'] >= 60])
                                
                                risk_data = [
                                    {"Risk Level": "Severe (eGFR < 30)", "Patient Count": severe_ckd, "Percentage": f"{(severe_ckd/len(relevant_df)*100):.1f}%"},
                                    {"Risk Level": "Moderate (eGFR 30-59)", "Patient Count": moderate_ckd, "Percentage": f"{(moderate_ckd/len(relevant_df)*100):.1f}%"},
                                    {"Risk Level": "Mild (eGFR ≥ 60)", "Patient Count": mild_ckd, "Percentage": f"{(mild_ckd/len(relevant_df)*100):.1f}%"}
                                ]
                                st.dataframe(pd.DataFrame(risk_data), use_container_width=True, hide_index=True)

                            # AI Analysis Results in styled containers
                            col1, col2, col3 = st.columns(3)

                            with col1:
                                st.markdown("""
                                <div style="background-color: #e8f4f8; padding: 1rem; border-radius: 8px; border-left: 4px solid #1f77b4;">
                                    <h4 style="color: #1f77b4; margin: 0 0 0.5rem 0;">🔍 Key Insights</h4>
                                """, unsafe_allow_html=True)
                                
                                cohort_summary = cohort_analysis.get("cohort_summary", "No summary available")
                                st.write(cohort_summary)
                                
                                chars = cohort_analysis.get("key_characteristics", [])
                                if chars:
                                    st.markdown("**Key Characteristics:**")
                                    for char in chars[:3]:
                                        st.markdown(f"• {char}")
                                
                                st.markdown("</div>", unsafe_allow_html=True)

                            with col2:
                                st.markdown("""
                                <div style="background-color: #f3e8f8; padding: 1rem; border-radius: 8px; border-left: 4px solid #8e44ad;">
                                    <h4 style="color: #8e44ad; margin: 0 0 0.5rem 0;">🧬 Clinical Significance</h4>
                                """, unsafe_allow_html=True)
                                
                                trial_suitability = cohort_analysis.get("trial_suitability", "No assessment available")
                                st.write(trial_suitability)
                                
                                research_opps = cohort_analysis.get("research_opportunities", [])
                                if research_opps:
                                    st.markdown("**Research Opportunities:**")
                                    for opp in research_opps[:2]:
                                        st.markdown(f"• {opp}")
                                
                                st.markdown("</div>", unsafe_allow_html=True)

                            with col3:
                                st.markdown("""
                                <div style="background-color: #fff3cd; padding: 1rem; border-radius: 8px; border-left: 4px solid #ffc107;">
                                    <h4 style="color: #d97706; margin: 0 0 0.5rem 0;">⚠️ Suggested Action (Caution: AI Generated)</h4>
                                """, unsafe_allow_html=True)
                                
                                recommendations = cohort_analysis.get("recommendations", [])
                                if recommendations:
                                    for rec in recommendations[:3]:
                                        st.markdown(f"• {rec}")
                                else:
                                    st.write("Consider further analysis of this patient cohort for potential research opportunities.")
                                
                                st.markdown("</div>", unsafe_allow_html=True)

                        # 2. MOVED: Relevant Patients Found section comes AFTER Cohort Analysis
                        st.markdown("---")
                        st.markdown("### Relevant Patients Found")

                        # 2. MODIFIED: Include clinical notes and drop relevance score
                        display_columns = [
                            'PatientID', 'Age', 'Sex', 'Ethnicity',
                            'Diagnosis', 'eGFR', 'APOL1_Variant', 
                            'Clinical_Notes', 'Eligible_For_Trial'
                        ]
                        available_columns = [
                            col for col in display_columns
                            if col in relevant_df.columns
                        ]

                        st.dataframe(relevant_df[available_columns],
                                     use_container_width=True,
                                     height=300)

                    else:
                        st.warning(
                            "No specific patients found matching your query, but the analysis above provides insights about the overall dataset."
                        )

                else:
                    st.error(
                        f"Analysis error: {llm_response.get('error', 'Unknown error')}"
                    )

            except Exception as e:
                st.error(f"Error processing query: {str(e)}")


def _create_dataset_context(df: pd.DataFrame) -> str:
    """Create context summary of the dataset for LLM processing"""
    try:
        context_parts = []

        # Dataset overview
        context_parts.append(
            f"Dataset contains {len(df)} patients with clinical genomics data."
        )

        # Demographics summary
        if 'Age' in df.columns:
            context_parts.append(
                f"Age range: {df['Age'].min()}-{df['Age'].max()} years (mean: {df['Age'].mean():.1f})"
            )

        if 'Sex' in df.columns:
            sex_dist = df['Sex'].value_counts()
            context_parts.append(f"Sex distribution: {sex_dist.to_dict()}")

        if 'Ethnicity' in df.columns:
            eth_dist = df['Ethnicity'].value_counts()
            context_parts.append(
                f"Ethnicity distribution: {eth_dist.to_dict()}")

        # Clinical data summary
        if 'Diagnosis' in df.columns:
            diag_dist = df['Diagnosis'].value_counts().head(5)
            context_parts.append(f"Top diagnoses: {diag_dist.to_dict()}")

        if 'eGFR' in df.columns:
            context_parts.append(
                f"eGFR range: {df['eGFR'].min():.1f}-{df['eGFR'].max():.1f} (mean: {df['eGFR'].mean():.1f})"
            )
            severe_ckd = len(df[df['eGFR'] < 30])
            context_parts.append(
                f"Patients with severe kidney dysfunction (eGFR < 30): {severe_ckd}"
            )

        if 'Eligible_For_Trial' in df.columns:
            trial_eligible = df['Eligible_For_Trial'].value_counts()
            context_parts.append(
                f"Clinical trial eligibility: {trial_eligible.to_dict()}")

        # Genetic variants summary
        genetic_cols = [
            'APOL1_Variant', 'APOL1', 'NPHS1', 'NPHS2', 'WT1', 'UMOD', 'COL4A3'
        ]
        genetic_data = []
        for col in genetic_cols:
            if col in df.columns:
                if col == 'APOL1_Variant':
                    variants = df[col].value_counts()
                    genetic_data.append(f"{col}: {variants.to_dict()}")
                else:
                    mutations = (df[col] == 'Mut').sum()
                    wild_type = (df[col] == 'WT').sum()
                    genetic_data.append(
                        f"{col} mutations: {mutations}/{mutations + wild_type}"
                    )

        if genetic_data:
            context_parts.append("Genetic variants: " +
                                 "; ".join(genetic_data))

        # Sample clinical notes
        if 'Clinical_Notes' in df.columns:
            sample_notes = df['Clinical_Notes'].dropna().head(3).tolist()
            if sample_notes:
                context_parts.append(
                    f"Sample clinical notes: {'; '.join(sample_notes)}")

        return "\n".join(context_parts)

    except Exception as e:
        return f"Error creating context: {str(e)}"
