import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from typing import Dict, List
import json


def render_clinical_notes_analysis(df, llm_processor):
    """Render clinical notes analysis section"""
    st.header("📝 Clinical Notes Intelligence")
    st.markdown("*Deep analysis of clinical narratives and patient observations*")
    
    if 'Clinical_Notes' not in df.columns:
        st.error("No clinical notes data available in the dataset.")
        return
    
    # Clinical notes overview
    notes_data = df['Clinical_Notes'].dropna()
    total_notes = len(notes_data)
    
    st.markdown("### Clinical Notes Overview")
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.metric("Total Notes", total_notes)
    with col2:
        avg_length = notes_data.str.len().mean()
        st.metric("Avg. Note Length", f"{avg_length:.0f} chars")
    with col3:
        unique_notes = notes_data.nunique()
        st.metric("Unique Notes", unique_notes)
    with col4:
        completion_rate = (total_notes / len(df)) * 100
        st.metric("Documentation Rate", f"{completion_rate:.1f}%")
    
    # Analysis options
    st.markdown("### Analysis Options")
    analysis_type = st.selectbox(
        "Choose analysis type:",
        ["Treatment Patterns", "Disease Progression", "Risk Factors", "Medication Responses", "Trial Readiness"]
    )
    
    # Sample size selection
    sample_size = st.slider("Number of notes to analyze:", 10, min(100, total_notes), 50)
    
    if st.button("🔍 Analyze Clinical Notes", type="primary"):
        with st.spinner("Analyzing clinical narratives..."):
            # Sample clinical notes for analysis
            sample_notes = notes_data.sample(n=sample_size, random_state=42).tolist()
            
            # Get corresponding patient data for context
            sample_indices = notes_data.sample(n=sample_size, random_state=42).index
            sample_patients = df.loc[sample_indices]
            
            # Create patient context summary
            patient_context = {
                "total_patients": len(sample_patients),
                "age_range": f"{sample_patients['Age'].min()}-{sample_patients['Age'].max()}",
                "diagnoses": sample_patients['Diagnosis'].value_counts().head(5).to_dict(),
                "avg_egfr": round(sample_patients['eGFR'].mean(), 1)
            }
            
            # Perform LLM analysis
            insights = llm_processor.extract_clinical_insights(sample_notes, patient_context)
            
            if "error" not in insights:
                st.markdown("---")
                st.markdown("### 📊 Clinical Intelligence Results")
                
                # Executive summary
                st.markdown("#### Executive Summary")
                st.info(insights.get("overall_summary", "No summary available"))
                
                # Main insights in columns
                col1, col2 = st.columns(2)
                
                with col1:
                    st.markdown("#### 🔍 Disease Patterns")
                    patterns = insights.get("disease_patterns", [])
                    for i, pattern in enumerate(patterns[:5], 1):
                        st.markdown(f"**{i}.** {pattern}")
                    
                    st.markdown("#### 💊 Treatment Insights")
                    treatments = insights.get("treatment_insights", [])
                    for i, treatment in enumerate(treatments[:5], 1):
                        st.markdown(f"**{i}.** {treatment}")
                
                with col2:
                    st.markdown("#### ⚠️ Risk Factors")
                    risks = insights.get("risk_factors", [])
                    for i, risk in enumerate(risks[:5], 1):
                        st.markdown(f"**{i}.** {risk}")
                    
                    st.markdown("#### 🧪 Trial Considerations")
                    trials = insights.get("trial_considerations", [])
                    for i, trial in enumerate(trials[:5], 1):
                        st.markdown(f"**{i}.** {trial}")
                
                # Important findings
                if insights.get("urgent_findings"):
                    st.markdown("#### 🚨 Urgent Findings")
                    st.error("**Important clinical observations requiring attention:**")
                    for finding in insights.get("urgent_findings", []):
                        st.markdown(f"• {finding}")
                
                # Research opportunities
                if insights.get("research_opportunities"):
                    st.markdown("#### 🔬 Research Opportunities")
                    st.success("**Potential areas for pharmaceutical research:**")
                    for opportunity in insights.get("research_opportunities", []):
                        st.markdown(f"• {opportunity}")
                
                # Medication patterns
                if insights.get("medication_patterns"):
                    st.markdown("#### 💉 Medication Patterns")
                    for pattern in insights.get("medication_patterns", []):
                        st.markdown(f"• {pattern}")
            
            else:
                st.error(f"Analysis failed: {insights['error']}")
    
    # Quick insights from clinical notes data
    st.markdown("---")
    st.markdown("### 📈 Quick Clinical Notes Statistics")
    
    # Word frequency analysis
    if st.checkbox("Show Common Terms Analysis"):
        from collections import Counter
        import re
        
        # Extract common medical terms
        all_text = " ".join(notes_data.astype(str))
        words = re.findall(r'\b[a-zA-Z]{4,}\b', all_text.lower())
        
        # Filter for medical-relevant terms
        medical_terms = [word for word in words if len(word) > 4 and 
                        any(term in word for term in ['kidney', 'renal', 'ckd', 'egfr', 'creatinine', 
                                                     'protein', 'blood', 'pressure', 'medication', 
                                                     'treatment', 'therapy', 'patient', 'clinical'])]
        
        if medical_terms:
            term_counts = Counter(medical_terms).most_common(10)
            
            col1, col2 = st.columns(2)
            with col1:
                st.markdown("**Most Common Medical Terms:**")
                for term, count in term_counts:
                    st.markdown(f"• {term}: {count} mentions")
            
            with col2:
                # Create a simple bar chart
                terms, counts = zip(*term_counts)
                fig = px.bar(x=list(counts), y=list(terms), orientation='h',
                           title="Clinical Terms Frequency")
                fig.update_layout(height=400)
                st.plotly_chart(fig, use_container_width=True)


def render_intelligent_search(data_processor, vector_search, llm_processor):
    """Render the intelligent search hub with LLM-powered RAG capabilities"""

    st.header("Intelligent Search Hub")

    if data_processor is None or vector_search is None:
        st.warning(
            "Dataset not loaded. Loading default clinical genomics data...")
        return

    df = data_processor.get_data()
    
    # Check if Clinical Notes Analysis is requested
    if st.session_state.get('show_clinical_notes', False):
        render_clinical_notes_analysis(df, llm_processor)
        if st.button("← Back to Search"):
            st.session_state.show_clinical_notes = False
            st.rerun()
        return

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

    col1, col2, col3, col4, col5 = st.columns(5)

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

    with col5:
        if st.button("📝 Clinical Notes", use_container_width=True):
            st.session_state.show_clinical_notes = True
            st.rerun()

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

                    # Three column layout for key information
                    col1, col2, col3 = st.columns(3)

                    with col1:
                        st.markdown("#### Key Insights")
                        insights = llm_response.get("key_insights", [])
                        for i, insight in enumerate(insights, 1):
                            st.markdown(f"**{i}.** {insight}")

                    with col2:
                        st.markdown("#### Clinical Significance")
                        st.write(
                            llm_response.get("clinical_significance",
                                             "No clinical significance noted"))

                        st.markdown("**Patient Populations:**")
                        st.write(
                            llm_response.get(
                                "patient_populations",
                                "No specific populations identified"))

                    with col3:
                        st.markdown(
                            "#### Suggested Action (Caution: AI Generated)")
                        actions = llm_response.get("recommended_actions", [])
                        for action in actions:
                            st.markdown(f"• {action}")

                        st.markdown("**Data References:**")
                        references = llm_response.get("data_references", [])
                        for ref in references:
                            st.markdown(f"• {ref}")

                    # Find relevant patients based on query
                    st.markdown("---")
                    st.markdown("### Relevant Patients Found")

                    # Perform vector search to find relevant patients
                    indices, scores = vector_search.search(
                        query, top_k=1500, similarity_threshold=0.0)

                    if indices:
                        relevant_df = df.iloc[indices].copy()
                        relevant_df['relevance_score'] = [
                            f"{score:.3f}" for score in scores
                        ]

                        # Display results
                        display_columns = [
                            'PatientID', 'Age', 'Sex', 'Ethnicity',
                            'Diagnosis', 'eGFR', 'APOL1_Variant',
                            'Eligible_For_Trial', 'relevance_score'
                        ]
                        available_columns = [
                            col for col in display_columns
                            if col in relevant_df.columns
                        ]

                        st.dataframe(relevant_df[available_columns],
                                     use_container_width=True,
                                     height=300)

                        # Analyze the cohort with LLM
                        cohort_analysis = llm_processor.analyze_patient_cohort(
                            relevant_df, query)

                        if "error" not in cohort_analysis:
                            st.markdown("#### 🧠 AI Cohort Analysis")

                            col1, col2 = st.columns(2)

                            with col1:
                                st.markdown("**Cohort Summary:**")
                                st.write(
                                    cohort_analysis.get(
                                        "cohort_summary",
                                        "No summary available"))

                                st.markdown("**Key Characteristics:**")
                                chars = cohort_analysis.get(
                                    "key_characteristics", [])
                                for char in chars:
                                    st.markdown(f"• {char}")

                            with col2:
                                st.markdown("**Clinical Trial Suitability:**")
                                st.write(
                                    cohort_analysis.get(
                                        "trial_suitability",
                                        "No assessment available"))

                                st.markdown("**Recommendations:**")
                                recs = cohort_analysis.get(
                                    "recommendations", [])
                                for rec in recs:
                                    st.markdown(f"• {rec}")

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
