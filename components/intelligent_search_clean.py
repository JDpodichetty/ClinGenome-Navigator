import streamlit as st
import pandas as pd
from typing import Dict, List, Any
from utils.kg_cohort_analyzer import KGCohortAnalyzer


def render_intelligent_search_clean(data_processor, vector_search, llm_processor):
    """Clean implementation of intelligent search with knowledge graph backend"""
    
    st.markdown("# 🧠 Intelligent Search Hub")
    st.markdown("### Advanced AI-powered analysis of clinical genomics data")
    
    # Get data and enhanced knowledge graph from data processor
    df = data_processor.df
    enhanced_kg = data_processor.enhanced_kg
    
    # Search interface
    col1, col2 = st.columns([4, 1])
    
    with col1:
        query = st.text_input(
            "Enter your research question:",
            placeholder="e.g., insights on patients that have two or more mutations and low eGFR values",
            help="Ask natural language questions about the clinical data",
            key="search_query_clean")

    with col2:
        st.markdown("<br>", unsafe_allow_html=True)
        search_clicked = st.button("🔍 Analyze", type="primary", use_container_width=True)
    
    # Trigger search on Enter key press
    if query and query != st.session_state.get('last_search_query_clean', ''):
        search_clicked = True
        st.session_state.last_search_query_clean = query

    # Quick action buttons
    st.markdown("### ⚡ Quick Actions")
    col1, col2, col3, col4 = st.columns(4)

    with col1:
        if st.button("🧬 Genetic Insights", use_container_width=True, key="genetic_clean"):
            query = "Analyze genetic variant patterns and their clinical significance in this dataset"
            search_clicked = True

    with col2:
        if st.button("🎯 Trial Opportunities", use_container_width=True, key="trial_clean"):
            query = "Identify optimal patient populations for clinical trials based on this data"
            search_clicked = True

    with col3:
        if st.button("⚠️ High-Risk Patients", use_container_width=True, key="highrisk_clean"):
            query = "Find high-risk patients who need immediate clinical attention"
            search_clicked = True

    with col4:
        if st.button("📊 Research Summary", use_container_width=True, key="summary_clean"):
            query = "Generate a comprehensive research summary of this clinical dataset"
            search_clicked = True

    # Process query with knowledge graph backend
    if search_clicked and query.strip():
        with st.spinner("Analyzing your query with knowledge graph..."):
            try:
                # Process with knowledge graph
                kg_results = process_query_with_knowledge_graph(query, enhanced_kg, df)
                
                if kg_results:
                    display_knowledge_graph_results(kg_results, df, llm_processor, query)
                else:
                    st.warning("No specific patterns found for your query. Try refining your search terms.")
                    
            except Exception as e:
                st.error(f"Error processing query: {str(e)}")


def process_query_with_knowledge_graph(query: str, enhanced_kg, df: pd.DataFrame) -> Dict[str, Any]:
    """Process query using knowledge graph analysis"""
    
    query_lower = query.lower()
    kg_analyzer = KGCohortAnalyzer(enhanced_kg)
    
    # Multi-mutation + eGFR pattern detection
    mutation_patterns = ['two or more mutations', 'multiple mutations', 'multi mutation', '2+ mutations', 'more than one mutation', 'two mutations', 'several mutations']
    egfr_patterns = ['high egfr', 'elevated egfr', 'preserved kidney', 'good kidney function', 'egfr values', 'egfr levels', 'low egfr', 'reduced egfr', 'kidney dysfunction']
    
    has_mutation_pattern = any(phrase in query_lower for phrase in mutation_patterns)
    has_egfr_pattern = any(phrase in query_lower for phrase in egfr_patterns)
    has_combined_pattern = ('mutation' in query_lower and 'egfr' in query_lower and ('two' in query_lower or 'multiple' in query_lower or 'more' in query_lower))
    
    if has_mutation_pattern or has_egfr_pattern or has_combined_pattern:
        # Detect low vs high eGFR
        is_low_egfr = any(phrase in query_lower for phrase in ['low egfr', 'reduced egfr', 'kidney dysfunction', 'low'])
        
        if is_low_egfr:
            result = kg_analyzer.find_multi_mutation_low_egfr_cohort(45)
            result['analysis_type'] = 'multi_mutation_low_egfr'
        else:
            result = kg_analyzer.find_multi_mutation_high_egfr_cohort(60)
            result['analysis_type'] = 'multi_mutation_high_egfr'
        
        return result
    
    # High-risk genetic analysis
    elif any(phrase in query_lower for phrase in ['high risk', 'high-risk', 'genetic risk', 'apol1', 'gene mutation']):
        result = kg_analyzer.find_high_risk_genetic_cohort()
        result['analysis_type'] = 'high_risk_genetic'
        return result
    
    # Kidney dysfunction analysis
    elif any(phrase in query_lower for phrase in ['kidney dysfunction', 'kidney failure', 'egfr', 'creatinine', 'progression']):
        egfr_threshold = 30 if 'severe' in query_lower or 'failure' in query_lower else 45
        result = kg_analyzer.find_kidney_dysfunction_progression_cohort(egfr_threshold)
        result['analysis_type'] = 'kidney_dysfunction'
        return result
    
    # Genetic-lab correlation analysis
    elif any(phrase in query_lower for phrase in ['correlation', 'genetic lab', 'variant effect', 'mutation impact']):
        result = kg_analyzer.find_genetic_lab_correlation_cohort()
        result['analysis_type'] = 'genetic_lab_correlation'
        return result
    
    return None


def display_knowledge_graph_results(kg_results: Dict[str, Any], df: pd.DataFrame, llm_processor, query: str):
    """Display knowledge graph analysis results in rich format"""
    
    # Filter dataframe to relevant patients
    patient_ids = kg_results.get('patient_ids', [])
    if patient_ids:
        relevant_df = df[df['PatientID'].isin(patient_ids)].copy()
    else:
        relevant_df = df.iloc[0:0].copy()
    
    total_patients = kg_results.get('total_patients', 0)
    analysis_type = kg_results.get('analysis_type', 'unknown')
    
    # Create summary based on analysis type
    if analysis_type == 'multi_mutation_low_egfr':
        summary = f"Knowledge Graph Analysis identified {total_patients} patients with multiple gene mutations and reduced kidney function from the authentic clinical dataset. This represents a high-risk cohort showing expected functional decline from genetic burden."
        key_insights = [
            f"Found {total_patients} patients with both 2+ mutations AND low eGFR (<45)",
            f"Total patients with 2+ mutations: {kg_results.get('multi_mutation_total', 0)}",
            f"Total patients with low eGFR: {kg_results.get('low_egfr_total', 0)}",
            f"Average mutations per patient: {sum(kg_results.get('mutation_details', {}).values()) / len(kg_results.get('mutation_details', {})) if kg_results.get('mutation_details') else 0:.1f}",
            "Clinical significance: This cohort shows expected functional decline from genetic burden"
        ]
    elif analysis_type == 'multi_mutation_high_egfr':
        summary = f"Knowledge Graph Analysis identified {total_patients} patients with multiple gene mutations but preserved kidney function from the authentic clinical dataset. This represents a cohort with genetic risk but maintained renal function."
        key_insights = [
            f"Found {total_patients} patients with both 2+ mutations AND high eGFR (>60)",
            f"Total patients with 2+ mutations: {kg_results.get('multi_mutation_total', 0)}",
            f"Total patients with high eGFR: {kg_results.get('high_egfr_total', 0)}",
            f"Average mutations per patient: {sum(kg_results.get('mutation_details', {}).values()) / len(kg_results.get('mutation_details', {})) if kg_results.get('mutation_details') else 0:.1f}",
            "Clinical significance: This cohort shows preserved function despite genetic risk"
        ]
    elif analysis_type == 'high_risk_genetic':
        summary = f"Knowledge Graph Analysis identified {total_patients} high-risk patients based on genetic variants from the authentic clinical dataset."
        key_insights = [
            f"Found {total_patients} high-risk patients",
            f"APOL1 variant patients: {kg_results.get('apol1_patients', 0)}",
            f"Gene mutation patients: {kg_results.get('mutation_patients', 0)}",
            "High-risk classification based on genetic markers"
        ]
    else:
        summary = f"Knowledge Graph Analysis identified {total_patients} patients matching your criteria from the authentic clinical dataset."
        key_insights = [f"Found {total_patients} patients matching specific criteria"]
    
    # Display results
    st.markdown("---")
    st.markdown("### Analysis Results")
    
    # Executive Summary
    st.markdown("#### Executive Summary")
    st.info(summary)
    
    # Three column layout for key information
    col1, col2, col3 = st.columns(3)

    with col1:
        # Key Insights
        st.markdown("""
        <div style="background-color: #e8f4f8; padding: 1rem; border-radius: 8px; border-left: 4px solid #1f77b4; margin-bottom: 1rem;">
            <h4 style="color: #1f77b4; margin: 0 0 0.5rem 0;">Key Insights</h4>
        """, unsafe_allow_html=True)
        
        for i, insight in enumerate(key_insights, 1):
            st.markdown(f"**{i}.** {insight}")
        
        st.markdown("</div>", unsafe_allow_html=True)

    with col2:
        # Clinical Significance
        st.markdown("""
        <div style="background-color: #f3e8ff; padding: 1rem; border-radius: 8px; border-left: 4px solid #8b5cf6; margin-bottom: 1rem;">
            <h4 style="color: #7c3aed; margin: 0 0 0.5rem 0;">Clinical Significance</h4>
        """, unsafe_allow_html=True)
        
        if analysis_type == 'multi_mutation_low_egfr':
            st.markdown("**High-risk cohort with functional decline**")
            st.markdown("• Multiple kidney disease genes affected")
            st.markdown("• Reduced kidney function evident")
            st.markdown("• May require intervention consideration")
        elif analysis_type == 'multi_mutation_high_egfr':
            st.markdown("**Genetic risk with preserved function**")
            st.markdown("• Multiple mutations present")
            st.markdown("• Kidney function maintained")
            st.markdown("• Monitor for future decline")
        else:
            st.markdown("**Clinically relevant patient population**")
            st.markdown("• Specific genetic or clinical patterns")
            st.markdown("• Research opportunity identified")
        
        st.markdown("</div>", unsafe_allow_html=True)

    with col3:
        # Suggested Actions
        st.markdown("""
        <div style="background-color: #fff3cd; padding: 1rem; border-radius: 8px; border-left: 4px solid #ffc107; margin-bottom: 1rem;">
            <h4 style="color: #d97706; margin: 0 0 0.5rem 0;">Suggested Actions</h4>
        """, unsafe_allow_html=True)
        
        st.markdown("• Further genetic analysis")
        st.markdown("• Clinical trial screening")
        st.markdown("• Longitudinal monitoring")
        st.markdown("• Therapeutic intervention planning")
        
        st.markdown("</div>", unsafe_allow_html=True)

    # Display patient data tables and metrics if we have relevant patients
    if len(relevant_df) > 0:
        st.markdown("---")
        st.markdown("### Patient Cohort Analysis")
        
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
        
        # Demographics and clinical characteristics tables
        col1, col2 = st.columns(2)
        
        with col1:
            st.markdown("#### Demographics Summary")
            demo_summary = []
            if 'Sex' in relevant_df.columns:
                sex_counts = relevant_df['Sex'].value_counts()
                for sex, count in sex_counts.items():
                    demo_summary.append({"Category": "Sex", "Value": sex, "Count": count, "Percentage": f"{(count/len(relevant_df)*100):.1f}%"})
            
            if 'Ethnicity' in relevant_df.columns:
                eth_counts = relevant_df['Ethnicity'].value_counts()
                for eth, count in eth_counts.items():
                    demo_summary.append({"Category": "Ethnicity", "Value": eth, "Count": count, "Percentage": f"{(count/len(relevant_df)*100):.1f}%"})
            
            if demo_summary:
                st.dataframe(pd.DataFrame(demo_summary), use_container_width=True, hide_index=True)
        
        with col2:
            st.markdown("#### Clinical Characteristics")
            clinical_summary = []
            if 'Diagnosis' in relevant_df.columns:
                diag_counts = relevant_df['Diagnosis'].value_counts()
                for diag, count in diag_counts.items():
                    clinical_summary.append({"Category": "Diagnosis", "Value": diag, "Count": count, "Percentage": f"{(count/len(relevant_df)*100):.1f}%"})
            
            if 'APOL1_Variant' in relevant_df.columns:
                high_risk_variants = ['G1/G1', 'G1/G2', 'G2/G2']
                high_risk_count = len(relevant_df[relevant_df['APOL1_Variant'].isin(high_risk_variants)])
                clinical_summary.append({"Category": "APOL1", "Value": "High-Risk Variants", "Count": high_risk_count, "Percentage": f"{(high_risk_count/len(relevant_df)*100):.1f}%"})
            
            if clinical_summary:
                st.dataframe(pd.DataFrame(clinical_summary), use_container_width=True, hide_index=True)
        
        # Risk stratification
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
    else:
        st.warning("No patient data found for the specified criteria.")