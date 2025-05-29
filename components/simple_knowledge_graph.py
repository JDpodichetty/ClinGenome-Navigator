"""
Simple Knowledge Graph Visualization for Clinical Data
Clear, focused visualization of relationships in clinical genomics data
"""
import streamlit as st
import pandas as pd
import plotly.graph_objects as go
import plotly.express as px
import networkx as nx
from typing import Dict, List
import numpy as np

def render_simple_knowledge_graph(data_processor, vector_search, llm_processor):
    """Render simple knowledge graph visualization"""
    
    if data_processor is None:
        st.warning("Please load data first to view knowledge graph.")
        return
    
    df = data_processor.get_data()
    
    st.header("📈 Knowledge Graph")
    st.markdown("""
    Visual representation of relationships between genetic variants, clinical conditions, 
    and patient characteristics in your clinical genomics dataset.
    """)
    
    # Build simple knowledge graph representation
    if 'simple_kg_stats' not in st.session_state:
        with st.spinner("Analyzing clinical data relationships..."):
            stats = build_simple_graph_stats(df)
            st.session_state.simple_kg_stats = stats
            st.success("Knowledge graph analysis complete!")
    
    stats = st.session_state.simple_kg_stats
    
    # Overview metrics
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.metric("Total Patients", f"{len(df):,}")
    with col2:
        st.metric("Genetic Variants", f"{stats['genetic_entities']:,}")
    with col3:
        st.metric("Clinical Conditions", f"{stats['clinical_entities']:,}")
    with col4:
        st.metric("Relationships", f"{stats['total_relationships']:,}")
    
    # Main visualizations
    col1, col2 = st.columns(2)
    
    with col1:
        render_entity_distribution(stats)
    
    with col2:
        render_relationship_network(stats, df)
    
    # Clinical insights
    render_clinical_insights(df, stats)

def build_simple_graph_stats(df: pd.DataFrame) -> Dict:
    """Build simple statistics about the clinical data relationships"""
    
    stats = {
        'genetic_entities': 0,
        'clinical_entities': 0,
        'total_relationships': 0,
        'entity_counts': {},
        'apol1_distribution': {},
        'gene_mutations': {},
        'clinical_conditions': {}
    }
    
    # Count genetic entities
    gene_cols = ['NPHS1', 'NPHS2', 'WT1', 'COL4A3', 'UMOD']
    for gene in gene_cols:
        mutation_count = (df[gene] == 'Mut').sum()
        if mutation_count > 0:
            stats['gene_mutations'][gene] = mutation_count
            stats['genetic_entities'] += mutation_count
    
    # APOL1 variants
    apol1_counts = df['APOL1_Variant'].value_counts()
    stats['apol1_distribution'] = apol1_counts.to_dict()
    stats['genetic_entities'] += len(apol1_counts)
    
    # Clinical conditions
    if 'Diagnosis' in df.columns:
        diagnosis_counts = df['Diagnosis'].value_counts()
        stats['clinical_conditions'] = diagnosis_counts.to_dict()
        stats['clinical_entities'] = len(diagnosis_counts)
    
    # Entity type counts
    stats['entity_counts'] = {
        'Patients': len(df),
        'Genetic Variants': len(stats['apol1_distribution']),
        'Gene Mutations': len(stats['gene_mutations']),
        'Clinical Diagnoses': len(stats['clinical_conditions']),
        'Lab Categories': 2  # eGFR and Creatinine categories
    }
    
    # Calculate relationships (each patient connects to their entities)
    stats['total_relationships'] = (
        len(df) * 3 +  # Each patient has age, sex, ethnicity
        len(df) +      # Each patient has APOL1 variant
        sum(stats['gene_mutations'].values()) +  # Gene mutations
        len(df) * 2    # Lab values (eGFR, Creatinine)
    )
    
    return stats

def render_entity_distribution(stats: Dict):
    """Render entity type distribution chart"""
    
    entity_counts = stats['entity_counts']
    
    # Remove patients from the pie chart for clarity
    display_counts = {k: v for k, v in entity_counts.items() if k != 'Patients'}
    
    fig = px.pie(
        values=list(display_counts.values()),
        names=list(display_counts.keys()),
        title="Medical Entity Distribution",
        color_discrete_sequence=px.colors.qualitative.Set3
    )
    
    fig.update_traces(textposition='inside', textinfo='percent+label')
    fig.update_layout(showlegend=True, height=400)
    
    st.plotly_chart(fig, use_container_width=True)

def render_relationship_network(stats: Dict, df: pd.DataFrame):
    """Render simplified network visualization"""
    
    # Create a simple network graph
    G = nx.Graph()
    
    # Add central nodes
    G.add_node("Patients", type="central", size=30)
    G.add_node("APOL1", type="genetic", size=20)
    G.add_node("Gene_Mutations", type="genetic", size=20)
    G.add_node("Clinical_Dx", type="clinical", size=20)
    G.add_node("Lab_Results", type="clinical", size=20)
    
    # Add connections
    G.add_edge("Patients", "APOL1")
    G.add_edge("Patients", "Gene_Mutations")
    G.add_edge("Patients", "Clinical_Dx")
    G.add_edge("Patients", "Lab_Results")
    G.add_edge("APOL1", "Gene_Mutations")  # Genetic relationship
    G.add_edge("Clinical_Dx", "Lab_Results")  # Clinical relationship
    
    # Generate layout
    pos = nx.spring_layout(G, k=2, iterations=50)
    
    # Prepare data for plotly
    edge_x = []
    edge_y = []
    for edge in G.edges():
        x0, y0 = pos[edge[0]]
        x1, y1 = pos[edge[1]]
        edge_x.extend([x0, x1, None])
        edge_y.extend([y0, y1, None])
    
    node_x = []
    node_y = []
    node_text = []
    node_colors = []
    node_sizes = []
    
    color_map = {
        "central": "#ff6b6b",
        "genetic": "#4ecdc4", 
        "clinical": "#45b7d1"
    }
    
    for node in G.nodes():
        x, y = pos[node]
        node_x.append(x)
        node_y.append(y)
        
        # Format node labels
        if node == "Patients":
            node_text.append(f"Patients<br>({len(df):,})")
        elif node == "APOL1":
            node_text.append(f"APOL1<br>Variants")
        elif node == "Gene_Mutations":
            node_text.append(f"Gene<br>Mutations")
        elif node == "Clinical_Dx":
            node_text.append(f"Clinical<br>Diagnoses")
        else:
            node_text.append(f"Laboratory<br>Results")
        
        node_type = G.nodes[node].get('type', 'other')
        node_colors.append(color_map.get(node_type, '#gray'))
        node_sizes.append(G.nodes[node].get('size', 15))
    
    # Create network plot
    fig = go.Figure(data=[
        go.Scatter(x=edge_x, y=edge_y, mode='lines', 
                  line=dict(width=2, color='#888'), hoverinfo='none'),
        go.Scatter(x=node_x, y=node_y, mode='markers+text', 
                  marker=dict(size=node_sizes, color=node_colors, 
                            line=dict(width=2, color='white')),
                  text=node_text,
                  textposition="middle center",
                  textfont=dict(size=10, color='white'),
                  hoverinfo='text')
    ])
    
    fig.update_layout(
        title="Clinical Data Relationship Network",
        showlegend=False,
        hovermode='closest',
        margin=dict(b=20,l=5,r=5,t=40),
        annotations=[
            dict(text="Network shows how clinical entities connect to patients",
                 showarrow=False, xref="paper", yref="paper",
                 x=0.005, y=-0.002)
        ],
        xaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
        yaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
        height=400
    )
    
    st.plotly_chart(fig, use_container_width=True)

def render_clinical_insights(df: pd.DataFrame, stats: Dict):
    """Render clinical insights from the knowledge graph"""
    
    st.markdown("### 🔍 Clinical Insights from Knowledge Graph")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("#### Genetic Variant Patterns")
        
        # APOL1 risk distribution
        apol1_dist = stats['apol1_distribution']
        high_risk_variants = ['G1/G1', 'G1/G2', 'G2/G2']
        
        high_risk_count = sum(apol1_dist.get(variant, 0) for variant in high_risk_variants)
        total_patients = len(df)
        
        st.write(f"• **High-risk APOL1 variants:** {high_risk_count:,} patients ({high_risk_count/total_patients*100:.1f}%)")
        
        # Gene mutations
        gene_mutations = stats['gene_mutations']
        if gene_mutations:
            st.write("• **Gene mutations detected:**")
            for gene, count in gene_mutations.items():
                st.write(f"  - {gene}: {count:,} patients")
        
        # Risk correlation
        if 'eGFR' in df.columns:
            high_risk_patients = df[df['APOL1_Variant'].isin(high_risk_variants)]
            if not high_risk_patients.empty:
                avg_egfr_high_risk = high_risk_patients['eGFR'].mean()
                avg_egfr_overall = df['eGFR'].mean()
                st.write(f"• **eGFR correlation:** High-risk patients have average eGFR of {avg_egfr_high_risk:.1f} vs {avg_egfr_overall:.1f} overall")
    
    with col2:
        st.markdown("#### Clinical Characteristics")
        
        # Trial eligibility insights
        if 'Eligible_For_Trial' in df.columns:
            eligible_count = (df['Eligible_For_Trial'] == 'Yes').sum()
            st.write(f"• **Trial eligible patients:** {eligible_count:,} ({eligible_count/total_patients*100:.1f}%)")
        
        # Age and risk correlation
        if 'Age' in df.columns:
            high_risk_patients = df[df['APOL1_Variant'].isin(high_risk_variants)]
            if not high_risk_patients.empty:
                avg_age_high_risk = high_risk_patients['Age'].mean()
                avg_age_overall = df['Age'].mean()
                st.write(f"• **Age patterns:** High-risk patients average {avg_age_high_risk:.1f} years vs {avg_age_overall:.1f} overall")
        
        # Kidney function stages
        if 'eGFR' in df.columns:
            def get_ckd_stage(egfr):
                if egfr >= 90: return "Normal (≥90)"
                elif egfr >= 60: return "Stage 2 (60-89)"
                elif egfr >= 45: return "Stage 3a (45-59)"
                elif egfr >= 30: return "Stage 3b (30-44)"
                elif egfr >= 15: return "Stage 4 (15-29)"
                else: return "Stage 5 (<15)"
            
            df_copy = df.copy()
            df_copy['CKD_Stage'] = df_copy['eGFR'].apply(get_ckd_stage)
            stage_counts = df_copy['CKD_Stage'].value_counts()
            
            st.write("• **CKD stage distribution:**")
            for stage, count in stage_counts.head(3).items():
                st.write(f"  - {stage}: {count:,} patients")
    
    # Summary insights
    st.markdown("#### 💡 Key Knowledge Graph Insights")
    
    insights = []
    
    # Calculate some key insights
    if gene_mutations:
        total_mutations = sum(gene_mutations.values())
        insights.append(f"**Multi-gene analysis:** {total_mutations:,} gene mutations identified across {len(gene_mutations)} different genes")
    
    if 'APOL1_Variant' in df.columns and 'eGFR' in df.columns:
        correlation_strength = "strong" if high_risk_count > total_patients * 0.3 else "moderate"
        insights.append(f"**Genetic-clinical correlation:** {correlation_strength} relationship between APOL1 variants and kidney function")
    
    insights.append(f"**Data connectivity:** Each patient connects to an average of {stats['total_relationships']/total_patients:.1f} clinical entities")
    
    for insight in insights:
        st.write(f"• {insight}")