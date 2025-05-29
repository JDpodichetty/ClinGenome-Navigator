"""
Knowledge Graph Enhanced Search Component
Combines traditional search with semantic graph relationships
"""
import streamlit as st
import pandas as pd
import plotly.graph_objects as go
import plotly.express as px
from typing import Dict, List
import json
from utils.knowledge_graph import ClinicalKnowledgeGraph

def render_knowledge_graph_search(data_processor, vector_search, llm_processor):
    """Render knowledge graph enhanced search interface"""
    
    if data_processor is None:
        st.warning("Please load data first to use knowledge graph search.")
        return
    
    df = data_processor.get_data()
    
    st.header("🔗 Knowledge Graph Search")
    st.markdown("""
    Enhanced search using medical knowledge graphs that understand relationships between 
    genetic variants, clinical conditions, and treatment pathways from your authentic clinical data.
    """)
    
    # Initialize knowledge graph in session state
    if 'knowledge_graph' not in st.session_state:
        with st.spinner("Building knowledge graph from clinical data..."):
            kg = ClinicalKnowledgeGraph()
            kg.build_knowledge_graph(df)
            st.session_state.knowledge_graph = kg
            st.success("Knowledge graph built successfully!")
    
    kg = st.session_state.knowledge_graph
    
    # Query interface
    col1, col2 = st.columns([3, 1])
    
    with col1:
        query_type = st.selectbox(
            "Search Type:",
            ["Semantic Pathway Search", "Patient Similarity", "Genetic Interactions", "Traditional Search"],
            help="Choose how to search through the clinical data relationships"
        )
    
    with col2:
        st.markdown("<br>", unsafe_allow_html=True)
        search_button = st.button("🔍 Search", type="primary", use_container_width=True)
    
    # Query parameters based on type
    parameters = {}
    
    if query_type == "Semantic Pathway Search":
        st.markdown("### 🛤️ Clinical Pathway Analysis")
        st.markdown("Find patients following specific clinical pathways combining genetics and conditions")
        
        col1, col2 = st.columns(2)
        with col1:
            genetic_filter = st.multiselect(
                "Genetic Components:",
                ["APOL1 variants", "Gene mutations", "Any genetic risk"],
                default=["APOL1 variants"]
            )
        with col2:
            clinical_filter = st.multiselect(
                "Clinical Components:",
                ["CKD staging", "Diabetic nephropathy", "Laboratory abnormalities"],
                default=["CKD staging"]
            )
        
        parameters = {
            "genetic_components": genetic_filter,
            "clinical_components": clinical_filter
        }
    
    elif query_type == "Patient Similarity":
        st.markdown("### 👥 Find Similar Patients")
        
        # Patient selection
        patient_ids = [f"patient_{pid}" for pid in df['PatientID'].head(20)]
        selected_patient = st.selectbox(
            "Reference Patient:",
            patient_ids,
            help="Find patients with similar knowledge graph patterns"
        )
        
        similarity_threshold = st.slider(
            "Minimum Similarity:",
            min_value=1, max_value=5, value=2,
            help="Minimum number of shared characteristics"
        )
        
        parameters = {
            "patient_id": selected_patient,
            "min_similarity": similarity_threshold
        }
    
    elif query_type == "Genetic Interactions":
        st.markdown("### 🧬 Multi-Gene Interaction Analysis")
        
        min_variants = st.slider(
            "Minimum Genetic Variants:",
            min_value=1, max_value=5, value=2,
            help="Find patients with multiple genetic risk factors"
        )
        
        parameters = {
            "min_variants": min_variants
        }
    
    else:  # Traditional Search
        st.markdown("### 🔍 Enhanced Traditional Search")
        search_query = st.text_input(
            "Search Query:",
            placeholder="e.g., patients with APOL1 G1/G1 and diabetic nephropathy"
        )
        parameters = {"query": search_query}
    
    # Execute search
    if search_button or (query_type == "Traditional Search" and search_query):
        with st.spinner("Searching knowledge graph..."):
            if query_type == "Traditional Search":
                # Use existing vector search
                if search_query:
                    indices, scores = vector_search.search(search_query, top_k=100)
                    results_df = df.iloc[indices]
                    
                    st.success(f"Found {len(results_df)} patients")
                    
                    # Display results with knowledge graph insights
                    display_enhanced_results(results_df, kg, llm_processor)
            else:
                # Use knowledge graph queries
                if query_type == "Semantic Pathway Search":
                    patient_ids = kg.query_knowledge_graph("patients_with_pathway", parameters)
                elif query_type == "Patient Similarity":
                    patient_ids = kg.query_knowledge_graph("similar_patients", parameters)
                else:  # Genetic Interactions
                    patient_ids = kg.query_knowledge_graph("genetic_interactions", parameters)
                
                if patient_ids:
                    # Extract actual patient IDs and get dataframe rows
                    actual_patient_ids = [pid.replace('patient_', '') for pid in patient_ids]
                    results_df = df[df['PatientID'].isin(actual_patient_ids)]
                    
                    st.success(f"Found {len(results_df)} patients with matching patterns")
                    
                    # Display knowledge graph results
                    display_knowledge_graph_results(results_df, kg, patient_ids, query_type)
                else:
                    st.info("No patients found matching the specified criteria")

def display_enhanced_results(results_df: pd.DataFrame, kg: ClinicalKnowledgeGraph, llm_processor):
    """Display search results enhanced with knowledge graph insights"""
    
    if results_df.empty:
        return
    
    # Results overview
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.metric("Total Patients", len(results_df))
    
    with col2:
        apol1_variants = results_df['APOL1_Variant'].value_counts().to_dict()
        dominant_variant = max(apol1_variants.items(), key=lambda x: x[1])[0] if apol1_variants else "N/A"
        st.metric("Dominant APOL1 Variant", dominant_variant)
    
    with col3:
        avg_egfr = results_df['eGFR'].mean()
        st.metric("Average eGFR", f"{avg_egfr:.1f}")
    
    # Knowledge graph insights for first few patients
    st.markdown("### 🔗 Knowledge Graph Insights")
    
    sample_patients = results_df.head(5)
    for _, patient in sample_patients.iterrows():
        patient_kg_id = f"patient_{patient['PatientID']}"
        summary = kg.get_patient_knowledge_summary(patient_kg_id)
        
        if summary:
            with st.expander(f"Patient {patient['PatientID']} - Knowledge Graph Analysis"):
                
                col1, col2 = st.columns(2)
                
                with col1:
                    st.markdown("**Genetic Profile:**")
                    if summary['genetic_variants']:
                        for variant in summary['genetic_variants']:
                            st.write(f"• {variant.get('gene', 'Unknown')} variant")
                    else:
                        st.write("No genetic variants detected")
                    
                    st.markdown("**Clinical Conditions:**")
                    if summary['clinical_conditions']:
                        for condition in summary['clinical_conditions']:
                            st.write(f"• {condition.get('name', 'Unknown condition')}")
                    else:
                        st.write("No conditions detected")
                
                with col2:
                    st.markdown("**Clinical Notes Entities:**")
                    if summary['clinical_notes_entities']:
                        entity_types = {}
                        for entity in summary['clinical_notes_entities']:
                            entity_type = entity.get('type', 'unknown')
                            if entity_type not in entity_types:
                                entity_types[entity_type] = []
                            entity_types[entity_type].append(entity.get('name', 'Unknown'))
                        
                        for entity_type, entities in entity_types.items():
                            st.write(f"**{entity_type.title()}:** {', '.join(entities[:3])}")
                    else:
                        st.write("No clinical note entities extracted")
    
    # Display results table
    st.markdown("### 📊 Patient Results")
    st.dataframe(results_df, use_container_width=True)

def display_knowledge_graph_results(results_df: pd.DataFrame, kg: ClinicalKnowledgeGraph, 
                                   patient_ids: List[str], query_type: str):
    """Display results from knowledge graph queries"""
    
    # Results summary
    st.markdown("### 📈 Knowledge Graph Analysis Results")
    
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.metric("Matched Patients", len(results_df))
    
    with col2:
        trial_eligible = (results_df['Eligible_For_Trial'] == 'Yes').sum()
        st.metric("Trial Eligible", trial_eligible)
    
    with col3:
        genetic_risk_count = sum(1 for pid in patient_ids 
                               if kg.graph.nodes[pid].get('type') == 'patient')
        st.metric("With Genetic Risk", genetic_risk_count)
    
    with col4:
        avg_age = results_df['Age'].mean()
        st.metric("Average Age", f"{avg_age:.1f}")
    
    # Query-specific insights
    if query_type == "Patient Similarity":
        st.markdown("### 👥 Similarity Analysis")
        st.info("Patients shown share similar knowledge graph patterns including genetic variants, clinical conditions, and treatment responses.")
    
    elif query_type == "Genetic Interactions":
        st.markdown("### 🧬 Multi-Gene Analysis")
        
        # Count genetic variants per patient
        genetic_counts = {}
        for _, patient in results_df.iterrows():
            count = 0
            gene_cols = ['NPHS1', 'NPHS2', 'WT1', 'COL4A3', 'UMOD']
            for gene in gene_cols:
                if patient.get(gene) == 'Mut':
                    count += 1
            if patient.get('APOL1_Variant') != 'G0/G0':
                count += 1
            genetic_counts[patient['PatientID']] = count
        
        # Visualization
        fig = px.histogram(
            x=list(genetic_counts.values()),
            title="Distribution of Genetic Variants per Patient",
            labels={'x': 'Number of Genetic Variants', 'y': 'Patient Count'}
        )
        st.plotly_chart(fig, use_container_width=True)
    
    elif query_type == "Semantic Pathway Search":
        st.markdown("### 🛤️ Clinical Pathway Analysis")
        
        # Pathway distribution
        pathway_data = []
        for _, patient in results_df.iterrows():
            pathway = []
            if patient.get('APOL1_Variant') != 'G0/G0':
                pathway.append("Genetic Risk")
            if pd.notna(patient.get('Diagnosis')):
                pathway.append(patient['Diagnosis'])
            if patient.get('eGFR', 100) < 60:
                pathway.append("Reduced Kidney Function")
            
            pathway_data.append(" → ".join(pathway))
        
        pathway_counts = pd.Series(pathway_data).value_counts()
        
        fig = px.bar(
            x=pathway_counts.values,
            y=pathway_counts.index,
            orientation='h',
            title="Common Clinical Pathways",
            labels={'x': 'Patient Count', 'y': 'Clinical Pathway'}
        )
        st.plotly_chart(fig, use_container_width=True)
    
    # Detailed results
    st.markdown("### 📋 Detailed Patient Data")
    st.dataframe(results_df, use_container_width=True)
    
    # Export options
    st.markdown("### 📤 Export Options")
    col1, col2 = st.columns(2)
    
    with col1:
        if st.button("Export Results as CSV"):
            csv = results_df.to_csv(index=False)
            st.download_button(
                label="Download CSV",
                data=csv,
                file_name=f"knowledge_graph_results_{query_type.lower().replace(' ', '_')}.csv",
                mime="text/csv"
            )
    
    with col2:
        if st.button("Export Knowledge Graph"):
            kg_json = kg.export_knowledge_graph(format='json')
            st.download_button(
                label="Download Knowledge Graph",
                data=kg_json,
                file_name="clinical_knowledge_graph.json",
                mime="application/json"
            )