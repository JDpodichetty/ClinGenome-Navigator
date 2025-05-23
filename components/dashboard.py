import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from typing import Dict, Any

def render_dashboard(data_processor, vector_search):
    """Render the main dashboard with key metrics and insights"""
    
    st.header("📊 Clinical Genomics Dashboard")
    
    if data_processor is None:
        st.warning("No data loaded. Please upload a dataset first.")
        return
    
    df = data_processor.get_data()
    metadata = data_processor.get_metadata()
    
    # Key metrics row
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.metric(
            label="Total Patients",
            value=f"{metadata['total_patients']:,}",
            help="Total number of patients in the dataset"
        )
    
    with col2:
        trial_eligible = df['Eligible_For_Trial'].value_counts().get('Yes', 0)
        eligibility_rate = (trial_eligible / len(df)) * 100
        st.metric(
            label="Trial Eligible",
            value=f"{trial_eligible:,}",
            delta=f"{eligibility_rate:.1f}%",
            help="Number and percentage of patients eligible for clinical trials"
        )
    
    with col3:
        if 'eGFR' in df.columns:
            avg_egfr = df['eGFR'].mean()
            st.metric(
                label="Avg eGFR",
                value=f"{avg_egfr:.1f}",
                help="Average estimated Glomerular Filtration Rate"
            )
        else:
            st.metric(label="Avg eGFR", value="N/A")
    
    with col4:
        unique_diagnoses = df['Diagnosis'].nunique()
        st.metric(
            label="Diagnoses",
            value=f"{unique_diagnoses}",
            help="Number of unique diagnoses in the dataset"
        )
    
    st.markdown("---")
    
    # Demographics overview
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("Demographics Distribution")
        
        # Age distribution
        if 'Age' in df.columns:
            fig_age = px.histogram(
                df, 
                x='Age', 
                nbins=20,
                title="Age Distribution",
                labels={'Age': 'Age (years)', 'count': 'Number of Patients'}
            )
            fig_age.update_layout(height=300)
            st.plotly_chart(fig_age, use_container_width=True)
        
        # Sex distribution
        if 'Sex' in df.columns:
            sex_counts = df['Sex'].value_counts()
            fig_sex = px.pie(
                values=sex_counts.values,
                names=sex_counts.index,
                title="Sex Distribution"
            )
            fig_sex.update_layout(height=300)
            st.plotly_chart(fig_sex, use_container_width=True)
    
    with col2:
        st.subheader("Clinical Overview")
        
        # Diagnosis distribution
        if 'Diagnosis' in df.columns:
            diag_counts = df['Diagnosis'].value_counts().head(6)
            fig_diag = px.bar(
                x=diag_counts.values,
                y=diag_counts.index,
                orientation='h',
                title="Top Diagnoses",
                labels={'x': 'Number of Patients', 'y': 'Diagnosis'}
            )
            fig_diag.update_layout(height=300)
            st.plotly_chart(fig_diag, use_container_width=True)
        
        # Ethnicity distribution
        if 'Ethnicity' in df.columns:
            eth_counts = df['Ethnicity'].value_counts()
            fig_eth = px.bar(
                x=eth_counts.index,
                y=eth_counts.values,
                title="Ethnicity Distribution",
                labels={'x': 'Ethnicity', 'y': 'Number of Patients'}
            )
            fig_eth.update_layout(height=300)
            st.plotly_chart(fig_eth, use_container_width=True)
    
    st.markdown("---")
    
    # Clinical metrics
    st.subheader("Clinical Metrics Analysis")
    
    col1, col2 = st.columns(2)
    
    with col1:
        # eGFR distribution
        if 'eGFR' in df.columns:
            fig_egfr = px.histogram(
                df,
                x='eGFR',
                nbins=30,
                title="eGFR Distribution",
                labels={'eGFR': 'eGFR (mL/min/1.73m²)', 'count': 'Number of Patients'}
            )
            
            # Add vertical lines for CKD stages
            fig_egfr.add_vline(x=90, line_dash="dash", line_color="orange", 
                              annotation_text="Stage 1/2 boundary")
            fig_egfr.add_vline(x=60, line_dash="dash", line_color="red", 
                              annotation_text="Stage 2/3 boundary")
            fig_egfr.add_vline(x=30, line_dash="dash", line_color="darkred", 
                              annotation_text="Stage 3/4 boundary")
            fig_egfr.add_vline(x=15, line_dash="dash", line_color="black", 
                              annotation_text="Stage 4/5 boundary")
            
            st.plotly_chart(fig_egfr, use_container_width=True)
    
    with col2:
        # Creatinine vs eGFR scatter
        if 'eGFR' in df.columns and 'Creatinine' in df.columns:
            fig_scatter = px.scatter(
                df,
                x='eGFR',
                y='Creatinine',
                color='Sex',
                title="eGFR vs Creatinine",
                labels={
                    'eGFR': 'eGFR (mL/min/1.73m²)',
                    'Creatinine': 'Creatinine (mg/dL)'
                }
            )
            st.plotly_chart(fig_scatter, use_container_width=True)
    
    st.markdown("---")
    
    # Genetic variants analysis
    st.subheader("Genetic Variants Analysis")
    
    genetic_columns = ['APOL1', 'NPHS1', 'NPHS2', 'WT1', 'UMOD', 'COL4A3']
    available_genetic = [col for col in genetic_columns if col in df.columns]
    
    if available_genetic:
        # Create mutation frequency chart
        mutation_data = []
        for gene in available_genetic:
            mut_count = (df[gene] == 'Mut').sum()
            wt_count = (df[gene] == 'WT').sum()
            total = mut_count + wt_count
            if total > 0:
                mut_freq = (mut_count / total) * 100
                mutation_data.append({
                    'Gene': gene,
                    'Mutation Frequency (%)': mut_freq,
                    'Mutations': mut_count,
                    'Wild Type': wt_count
                })
        
        if mutation_data:
            mutation_df = pd.DataFrame(mutation_data)
            
            col1, col2 = st.columns(2)
            
            with col1:
                fig_mut_freq = px.bar(
                    mutation_df,
                    x='Gene',
                    y='Mutation Frequency (%)',
                    title="Mutation Frequency by Gene",
                    labels={'Mutation Frequency (%)': 'Mutation Frequency (%)'}
                )
                st.plotly_chart(fig_mut_freq, use_container_width=True)
            
            with col2:
                # APOL1 variant distribution if available
                if 'APOL1_Variant' in df.columns:
                    apol1_counts = df['APOL1_Variant'].value_counts()
                    fig_apol1 = px.pie(
                        values=apol1_counts.values,
                        names=apol1_counts.index,
                        title="APOL1 Variant Distribution"
                    )
                    st.plotly_chart(fig_apol1, use_container_width=True)
                else:
                    # Show genetic variant heatmap
                    genetic_matrix = df[available_genetic].apply(lambda x: (x == 'Mut').astype(int))
                    correlation_matrix = genetic_matrix.corr()
                    
                    fig_heatmap = px.imshow(
                        correlation_matrix,
                        title="Genetic Variant Co-occurrence",
                        labels=dict(color="Correlation"),
                        aspect="auto"
                    )
                    st.plotly_chart(fig_heatmap, use_container_width=True)
    
    st.markdown("---")
    
    # Quick insights
    st.subheader("📈 Key Insights")
    
    insights_col1, insights_col2 = st.columns(2)
    
    with insights_col1:
        st.markdown("**Clinical Insights:**")
        
        # High-risk patients
        if 'eGFR' in df.columns:
            high_risk = df[df['eGFR'] < 30]
            st.info(f"🔴 {len(high_risk)} patients with severe kidney dysfunction (eGFR < 30)")
        
        # Trial eligibility by diagnosis
        if 'Diagnosis' in df.columns and 'Eligible_For_Trial' in df.columns:
            eligible_by_diag = df[df['Eligible_For_Trial'] == 'Yes']['Diagnosis'].value_counts().head(3)
            st.success(f"📊 Top trial-eligible diagnoses: {', '.join(eligible_by_diag.index.tolist())}")
    
    with insights_col2:
        st.markdown("**Genetic Insights:**")
        
        # High-risk genetic combinations
        if available_genetic:
            # Count patients with multiple mutations
            df_genetic = df[available_genetic].copy()
            df_genetic['mutation_count'] = (df_genetic == 'Mut').sum(axis=1)
            multiple_mutations = (df_genetic['mutation_count'] >= 2).sum()
            st.warning(f"⚠️ {multiple_mutations} patients with multiple genetic mutations")
        
        # APOL1 high-risk variants
        if 'APOL1_Variant' in df.columns:
            high_risk_apol1 = df[df['APOL1_Variant'].isin(['G1/G2', 'G2/G2', 'G1/G1'])].shape[0]
            st.error(f"🧬 {high_risk_apol1} patients with high-risk APOL1 variants")
    
    # Search suggestions
    st.markdown("---")
    st.subheader("💡 Try These Searches")
    
    if vector_search:
        suggestions = vector_search.suggest_queries()
        
        suggestion_cols = st.columns(2)
        for i, suggestion in enumerate(suggestions[:6]):
            col = suggestion_cols[i % 2]
            with col:
                if st.button(f"🔍 {suggestion}", key=f"suggestion_{i}"):
                    st.session_state.suggested_query = suggestion
                    st.session_state.show_search = True
                    st.rerun()
    
    # Quick statistics table
    st.markdown("---")
    st.subheader("📋 Dataset Summary")
    
    summary_data = {
        'Metric': ['Total Patients', 'Average Age', 'Male/Female Ratio', 'Trial Eligible Rate'],
        'Value': [
            f"{len(df):,}",
            f"{df['Age'].mean():.1f} years" if 'Age' in df.columns else "N/A",
            f"{len(df[df['Sex']=='M'])}/{len(df[df['Sex']=='F'])}" if 'Sex' in df.columns else "N/A",
            f"{(df['Eligible_For_Trial']=='Yes').mean()*100:.1f}%" if 'Eligible_For_Trial' in df.columns else "N/A"
        ]
    }
    
    summary_df = pd.DataFrame(summary_data)
    st.dataframe(summary_df, use_container_width=True, hide_index=True)
