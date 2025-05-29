"""
Enhanced Neo4j-style Knowledge Graph Search Component
Provides superior graph database capabilities for pharmaceutical research
"""
import streamlit as st
import pandas as pd
import plotly.graph_objects as go
import plotly.express as px
import networkx as nx
from typing import Dict, List
import json
import numpy as np
from utils.neo4j_knowledge_graph import Neo4jKnowledgeGraph

def render_neo4j_knowledge_graph_search(data_processor, vector_search, llm_processor):
    """Render enhanced Neo4j-style knowledge graph search interface"""
    
    if data_processor is None:
        st.warning("Please load data first to use knowledge graph search.")
        return
    
    df = data_processor.get_data()
    
    st.header("🔗 Advanced Knowledge Graph Search")
    st.markdown("""
    **Neo4j-powered knowledge graph system** for pharmaceutical research with superior 
    relationship analysis, clinical pathway discovery, and patient similarity matching.
    """)
    
    # Initialize Neo4j-style knowledge graph
    if 'neo4j_knowledge_graph' not in st.session_state:
        with st.spinner("Building advanced knowledge graph with clinical reasoning..."):
            neo4j_kg = Neo4jKnowledgeGraph(use_embedded=True)
            stats = neo4j_kg.build_knowledge_graph(df)
            st.session_state.neo4j_knowledge_graph = neo4j_kg
            st.session_state.kg_stats = stats
            st.success(f"Advanced knowledge graph built: {stats['nodes']:,} nodes, {stats['relationships']:,} relationships")
    
    neo4j_kg = st.session_state.neo4j_knowledge_graph
    stats = st.session_state.kg_stats
    
    # Knowledge Graph Overview
    st.markdown("### 📊 Knowledge Graph Statistics")
    
    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.metric("Total Nodes", f"{stats['nodes']:,}")
    with col2:
        st.metric("Total Relationships", f"{stats['relationships']:,}")
    with col3:
        graph_stats = neo4j_kg.get_graph_statistics()
        patients = graph_stats['node_counts_by_type'].get('patient', 0)
        st.metric("Patients Analyzed", f"{patients:,}")
    with col4:
        genetic_entities = (graph_stats['node_counts_by_type'].get('genetic_variant', 0) + 
                          graph_stats['node_counts_by_type'].get('gene_mutation', 0))
        st.metric("Genetic Entities", f"{genetic_entities:,}")
    
    # Enhanced visualization tabs
    tab1, tab2, tab3, tab4 = st.tabs([
        "🎯 Graph Analytics", 
        "🔍 Patient Explorer", 
        "🧬 Clinical Pathways", 
        "📊 Risk Analysis"
    ])
    
    with tab1:
        render_graph_analytics(neo4j_kg, graph_stats)
    
    with tab2:
        render_patient_explorer(neo4j_kg, df)
    
    with tab3:
        render_clinical_pathways(neo4j_kg, df)
    
    with tab4:
        render_risk_analysis(neo4j_kg, df)

def render_graph_analytics(neo4j_kg: Neo4jKnowledgeGraph, graph_stats: Dict):
    """Render comprehensive graph analytics"""
    
    st.markdown("#### 🎯 Network Analysis & Entity Distribution")
    
    col1, col2 = st.columns(2)
    
    with col1:
        # Node type distribution
        node_counts = graph_stats['node_counts_by_type']
        if node_counts:
            fig_nodes = px.pie(
                values=list(node_counts.values()),
                names=list(node_counts.keys()),
                title="Entity Types in Knowledge Graph",
                color_discrete_sequence=px.colors.qualitative.Set3
            )
            fig_nodes.update_traces(textposition='inside', textinfo='percent+label')
            st.plotly_chart(fig_nodes, use_container_width=True)
    
    with col2:
        # Relationship type distribution
        rel_counts = graph_stats['relationship_counts_by_type']
        if rel_counts:
            fig_rels = px.bar(
                x=list(rel_counts.values()),
                y=list(rel_counts.keys()),
                orientation='h',
                title="Relationship Types",
                labels={'x': 'Count', 'y': 'Relationship Type'},
                color=list(rel_counts.values()),
                color_continuous_scale='viridis'
            )
            st.plotly_chart(fig_rels, use_container_width=True)
    
    # Network metrics
    st.markdown("#### 📈 Advanced Network Metrics")
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        density = graph_stats.get('graph_density', 0)
        st.metric("Graph Density", f"{density:.4f}")
        st.caption("Measures how interconnected the network is")
    
    with col2:
        total_nodes = graph_stats.get('total_nodes', 0)
        total_rels = graph_stats.get('total_relationships', 0)
        avg_connections = (2 * total_rels) / total_nodes if total_nodes > 0 else 0
        st.metric("Avg Connections per Node", f"{avg_connections:.1f}")
        st.caption("Average number of relationships per entity")
    
    with col3:
        genetic_coverage = (node_counts.get('genetic_variant', 0) + 
                          node_counts.get('gene_mutation', 0)) / node_counts.get('patient', 1)
        st.metric("Genetic Coverage Ratio", f"{genetic_coverage:.1f}")
        st.caption("Genetic entities per patient")

def render_patient_explorer(neo4j_kg: Neo4jKnowledgeGraph, df: pd.DataFrame):
    """Interactive patient exploration with detailed analysis"""
    
    st.markdown("#### 🔍 Individual Patient Analysis")
    
    # Patient selection
    col1, col2 = st.columns([3, 1])
    
    with col1:
        # Get first 20 patients for selection
        sample_patients = df['PatientID'].head(20).tolist()
        selected_patient_id = st.selectbox(
            "Select Patient for Deep Analysis:",
            sample_patients,
            format_func=lambda x: f"Patient {x}",
            key="neo4j_patient_selector"
        )
    
    with col2:
        if st.button("🔍 Analyze Patient", type="primary", use_container_width=True):
            st.rerun()
    
    if selected_patient_id:
        patient_graph_id = f"patient_{selected_patient_id}"
        
        # Get comprehensive patient summary
        summary = neo4j_kg.get_patient_summary(patient_graph_id)
        
        if summary:
            # Patient overview
            patient_info = summary['patient_info']
            st.markdown(f"### 👤 Patient {selected_patient_id} - Comprehensive Profile")
            
            col1, col2, col3, col4 = st.columns(4)
            with col1:
                st.metric("Age", f"{patient_info.get('age', 'N/A')}")
            with col2:
                st.metric("Sex", patient_info.get('sex', 'N/A'))
            with col3:
                st.metric("Ethnicity", patient_info.get('ethnicity', 'N/A'))
            with col4:
                st.metric("Total Connections", summary['total_connections'])
            
            # Risk assessment
            risk_assessment = summary['risk_assessment']
            st.markdown("#### ⚠️ Clinical Risk Assessment")
            
            risk_level = risk_assessment['risk_level']
            risk_score = risk_assessment['risk_score']
            
            # Color-code risk level
            risk_colors = {
                'Very High': '🔴',
                'High': '🟠', 
                'Moderate': '🟡',
                'Low': '🟢'
            }
            
            st.markdown(f"**Risk Level:** {risk_colors.get(risk_level, '⚪')} {risk_level} (Score: {risk_score})")
            
            if risk_assessment['risk_factors']:
                st.markdown("**Risk Factors:**")
                for factor in risk_assessment['risk_factors']:
                    st.write(f"• {factor}")
            
            # Detailed entity breakdown
            st.markdown("#### 🔬 Connected Medical Entities")
            
            entities_by_type = summary['entities_by_type']
            
            for entity_type, entities in entities_by_type.items():
                if entities:
                    with st.expander(f"{entity_type.replace('_', ' ').title()} ({len(entities)} entities)"):
                        for entity in entities:
                            props = entity['properties']
                            st.markdown(f"**{entity['id']}**")
                            
                            # Display relevant properties
                            if 'gene' in props:
                                st.write(f"  - Gene: {props['gene']}")
                            if 'variant' in props:
                                st.write(f"  - Variant: {props['variant']}")
                            if 'risk_level' in props:
                                st.write(f"  - Risk Level: {props['risk_level']}")
                            if 'value' in props:
                                st.write(f"  - Value: {props['value']}")
                            if 'category' in props:
                                st.write(f"  - Category: {props['category']}")
                            if 'ckd_stage' in props:
                                st.write(f"  - CKD Stage: {props['ckd_stage']}")
                            
                            st.write(f"  - Relationship: {entity['relationship']}")
                            st.divider()
            
            # Find similar patients
            st.markdown("#### 👥 Similar Patients")
            
            similar_patients = neo4j_kg.find_similar_patients(patient_graph_id, similarity_threshold=2)
            
            if similar_patients:
                st.write(f"Found {len(similar_patients)} patients with similar clinical patterns:")
                
                for similar in similar_patients[:5]:  # Show top 5
                    sim_patient_id = similar['patient_id'].replace('patient_', '')
                    similarity_score = similar['similarity_score']
                    total_entities = similar['total_target_entities']
                    
                    similarity_pct = (similarity_score / total_entities) * 100
                    
                    st.write(f"• **Patient {sim_patient_id}** - {similarity_score}/{total_entities} shared entities ({similarity_pct:.1f}% similarity)")
            else:
                st.info("No patients found with similar clinical patterns")
        else:
            st.warning(f"No data found for Patient {selected_patient_id}")

def render_clinical_pathways(neo4j_kg: Neo4jKnowledgeGraph, df: pd.DataFrame):
    """Analyze clinical pathways and disease progression patterns"""
    
    st.markdown("#### 🧬 Clinical Pathway Discovery")
    
    st.markdown("""
    Advanced pathway analysis identifies patients following similar clinical trajectories 
    based on genetic risk factors, diagnoses, and laboratory progressions.
    """)
    
    # Pathway analysis controls
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("**Pathway Criteria:**")
        has_genetic_risk = st.checkbox("Include Genetic Risk Factors", value=True, key="pathway_genetic")
        has_diagnosis = st.checkbox("Include Clinical Diagnoses", value=True, key="pathway_diagnosis")
        include_egfr = st.checkbox("Include eGFR Threshold", value=False, key="pathway_egfr")
        
        if include_egfr:
            egfr_threshold = st.slider("eGFR Threshold:", 15, 90, 60, key="pathway_egfr_value")
        else:
            egfr_threshold = None
    
    with col2:
        if st.button("🔍 Discover Pathways", type="primary", use_container_width=True, key="pathway_search"):
            # Build pathway criteria
            criteria = {
                'has_genetic_risk': has_genetic_risk,
                'has_diagnosis': has_diagnosis
            }
            
            if egfr_threshold:
                criteria['egfr_threshold'] = egfr_threshold
            
            # Query patients matching pathway
            pathway_patients = neo4j_kg.query_patients_by_pathway(criteria)
            
            if pathway_patients:
                st.success(f"Found {len(pathway_patients)} patients following this clinical pathway")
                
                # Get actual patient data
                actual_patient_ids = [pid.replace('patient_', '') for pid in pathway_patients]
                pathway_df = df[df['PatientID'].isin(actual_patient_ids)]
                
                # Pathway analysis visualizations
                st.markdown("### 📊 Pathway Analysis Results")
                
                col1, col2 = st.columns(2)
                
                with col1:
                    # APOL1 distribution in pathway
                    apol1_dist = pathway_df['APOL1_Variant'].value_counts()
                    fig_apol1 = px.pie(
                        values=apol1_dist.values,
                        names=apol1_dist.index,
                        title="APOL1 Variants in Pathway"
                    )
                    st.plotly_chart(fig_apol1, use_container_width=True)
                
                with col2:
                    # eGFR distribution
                    fig_egfr = px.histogram(
                        pathway_df,
                        x='eGFR',
                        nbins=20,
                        title="eGFR Distribution in Pathway",
                        labels={'x': 'eGFR', 'y': 'Patient Count'}
                    )
                    st.plotly_chart(fig_egfr, use_container_width=True)
                
                # Clinical characteristics
                st.markdown("#### 🏥 Pathway Clinical Characteristics")
                
                col1, col2, col3, col4 = st.columns(4)
                
                with col1:
                    avg_age = pathway_df['Age'].mean()
                    st.metric("Average Age", f"{avg_age:.1f}")
                
                with col2:
                    trial_eligible = (pathway_df['Eligible_For_Trial'] == 'Yes').sum()
                    st.metric("Trial Eligible", f"{trial_eligible}/{len(pathway_df)}")
                
                with col3:
                    avg_egfr = pathway_df['eGFR'].mean()
                    st.metric("Average eGFR", f"{avg_egfr:.1f}")
                
                with col4:
                    high_risk_apol1 = pathway_df['APOL1_Variant'].isin(['G1/G1', 'G1/G2', 'G2/G2']).sum()
                    st.metric("High-risk APOL1", f"{high_risk_apol1}/{len(pathway_df)}")
                
                # Show sample patients
                st.markdown("#### 📋 Sample Patients in Pathway")
                st.dataframe(pathway_df.head(10), use_container_width=True)
                
            else:
                st.info("No patients found matching the specified pathway criteria")

def render_risk_analysis(neo4j_kg: Neo4jKnowledgeGraph, df: pd.DataFrame):
    """Advanced risk stratification using knowledge graph insights"""
    
    st.markdown("#### 📊 Advanced Risk Stratification Analysis")
    
    st.markdown("""
    Knowledge graph-powered risk analysis combining genetic variants, clinical markers, 
    and relationship patterns to identify high-risk patient populations.
    """)
    
    # Risk analysis for all patients
    if st.button("🔍 Analyze Population Risk", type="primary", use_container_width=True, key="risk_analysis"):
        
        with st.spinner("Performing comprehensive risk analysis..."):
            
            # Analyze risk for all patients
            risk_data = []
            sample_patients = df['PatientID'].head(50)  # Analyze first 50 for performance
            
            for patient_id in sample_patients:
                patient_graph_id = f"patient_{patient_id}"
                summary = neo4j_kg.get_patient_summary(patient_graph_id)
                
                if summary and 'risk_assessment' in summary:
                    risk_assessment = summary['risk_assessment']
                    patient_data = summary['patient_info']
                    
                    risk_data.append({
                        'PatientID': patient_id,
                        'RiskLevel': risk_assessment['risk_level'],
                        'RiskScore': risk_assessment['risk_score'],
                        'RiskFactors': len(risk_assessment['risk_factors']),
                        'Age': patient_data.get('age', 0),
                        'Sex': patient_data.get('sex', 'Unknown'),
                        'Ethnicity': patient_data.get('ethnicity', 'Unknown')
                    })
            
            if risk_data:
                risk_df = pd.DataFrame(risk_data)
                
                st.success(f"Risk analysis completed for {len(risk_df)} patients")
                
                # Risk level distribution
                col1, col2 = st.columns(2)
                
                with col1:
                    risk_counts = risk_df['RiskLevel'].value_counts()
                    fig_risk = px.pie(
                        values=risk_counts.values,
                        names=risk_counts.index,
                        title="Population Risk Distribution",
                        color_discrete_map={
                            'Very High': '#ff4444',
                            'High': '#ff8844', 
                            'Moderate': '#ffcc44',
                            'Low': '#44ff44'
                        }
                    )
                    st.plotly_chart(fig_risk, use_container_width=True)
                
                with col2:
                    # Risk score distribution
                    fig_score = px.histogram(
                        risk_df,
                        x='RiskScore',
                        nbins=10,
                        title="Risk Score Distribution",
                        labels={'x': 'Risk Score', 'y': 'Patient Count'}
                    )
                    st.plotly_chart(fig_score, use_container_width=True)
                
                # Risk by demographics
                st.markdown("#### 👥 Risk Stratification by Demographics")
                
                col1, col2 = st.columns(2)
                
                with col1:
                    # Risk by age groups
                    risk_df['AgeGroup'] = pd.cut(risk_df['Age'], 
                                               bins=[0, 30, 50, 70, 100], 
                                               labels=['<30', '30-50', '50-70', '70+'])
                    
                    age_risk = risk_df.groupby('AgeGroup')['RiskScore'].mean().reset_index()
                    
                    fig_age = px.bar(
                        age_risk,
                        x='AgeGroup',
                        y='RiskScore',
                        title="Average Risk Score by Age Group",
                        labels={'AgeGroup': 'Age Group', 'RiskScore': 'Average Risk Score'}
                    )
                    st.plotly_chart(fig_age, use_container_width=True)
                
                with col2:
                    # Risk by sex
                    sex_risk = risk_df.groupby('Sex')['RiskScore'].mean().reset_index()
                    
                    fig_sex = px.bar(
                        sex_risk,
                        x='Sex',
                        y='RiskScore',
                        title="Average Risk Score by Sex",
                        labels={'Sex': 'Sex', 'RiskScore': 'Average Risk Score'}
                    )
                    st.plotly_chart(fig_sex, use_container_width=True)
                
                # High-risk patients table
                st.markdown("#### ⚠️ High-Risk Patients Requiring Attention")
                
                high_risk_patients = risk_df[risk_df['RiskLevel'].isin(['Very High', 'High'])].sort_values('RiskScore', ascending=False)
                
                if not high_risk_patients.empty:
                    st.dataframe(high_risk_patients, use_container_width=True)
                    
                    st.markdown(f"**Summary:** {len(high_risk_patients)} patients identified as high-risk requiring immediate clinical attention.")
                else:
                    st.info("No high-risk patients identified in the analyzed population.")
            
            else:
                st.warning("No risk data available for analysis")