import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import numpy as np
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
import seaborn as sns

def render_visualization(data_processor):
    """Render data exploration interface with comprehensive analytics"""
    
    st.header("📊 Data Exploration")
    
    # Enhanced tab styling for Data Exploration
    st.markdown("""
    <style>
    .stTabs [data-baseweb="tab"] {
        font-size: 20px !important;
        font-weight: bold !important;
        padding: 14px 20px !important;
        height: 55px !important;
    }
    </style>
    """, unsafe_allow_html=True)
    
    if data_processor is None:
        st.warning("No data loaded. Please load data first.")
        return
    
    df = data_processor.get_data()
    
    if df.empty:
        st.warning("Dataset is empty.")
        return
    
    # Clean tabbed interface for different analyses
    tab1, tab2, tab3, tab4 = st.tabs([
        "👥 Demographics", 
        "🧬 Genetic Analysis", 
        "🏥 Clinical Metrics", 
        "📈 Advanced Analytics"
    ])
    
    with tab1:
        render_demographics_analysis(df)
    
    with tab2:
        render_genetic_analysis(df)
    
    with tab3:
        render_clinical_metrics_analysis(df)
    
    with tab4:
        # Sub-tabs for advanced analytics
        subtab1, subtab2, subtab3, subtab4 = st.tabs([
            "🔗 Correlations", 
            "📊 PCA Analysis", 
            "⚠️ Risk Stratification", 
            "💊 Treatment & Trials"
        ])
        
        with subtab1:
            render_correlation_analysis(df)
        
        with subtab2:
            render_pca_analysis(df)
        
        with subtab3:
            render_risk_stratification(df)
        
        with subtab4:
            col1, col2 = st.columns(2)
            with col1:
                st.markdown("### 💊 Medication Analysis")
                render_medication_analysis(df)
            with col2:
                st.markdown("### 🧪 Trial Eligibility")
                render_trial_eligibility_analysis(df)

def render_demographics_analysis(df):
    """Render demographics analysis visualizations"""
    
    st.subheader("👥 Patient Demographics Analysis")
    
    col1, col2 = st.columns(2)
    
    with col1:
        # Age distribution by sex
        if 'Age' in df.columns and 'Sex' in df.columns:
            fig_age_sex = px.histogram(
                df, 
                x='Age', 
                color='Sex',
                nbins=20,
                title="Age Distribution by Sex",
                barmode='overlay',
                opacity=0.7
            )
            fig_age_sex.update_layout(height=400)
            st.plotly_chart(fig_age_sex, use_container_width=True)
        
        # Ethnicity distribution
        if 'Ethnicity' in df.columns:
            ethnicity_counts = df['Ethnicity'].value_counts()
            fig_ethnicity = px.pie(
                values=ethnicity_counts.values,
                names=ethnicity_counts.index,
                title="Ethnicity Distribution"
            )
            fig_ethnicity.update_layout(height=400)
            st.plotly_chart(fig_ethnicity, use_container_width=True)
    
    with col2:
        # Age vs eGFR by ethnicity
        if all(col in df.columns for col in ['Age', 'eGFR', 'Ethnicity']):
            fig_age_egfr = px.scatter(
                df,
                x='Age',
                y='eGFR',
                color='Ethnicity',
                title="Age vs eGFR by Ethnicity",
                hover_data=['Sex', 'Diagnosis']
            )
            fig_age_egfr.update_layout(height=400)
            st.plotly_chart(fig_age_egfr, use_container_width=True)
        
        # Sex distribution by diagnosis
        if 'Sex' in df.columns and 'Diagnosis' in df.columns:
            sex_diag_crosstab = pd.crosstab(df['Diagnosis'], df['Sex'])
            fig_sex_diag = px.bar(
                sex_diag_crosstab,
                title="Sex Distribution by Diagnosis",
                barmode='group'
            )
            fig_sex_diag.update_layout(height=400)
            st.plotly_chart(fig_sex_diag, use_container_width=True)
    
    # Detailed demographics table
    st.markdown("### 📋 Demographics Summary Table")
    
    if all(col in df.columns for col in ['Age', 'Sex', 'Ethnicity']):
        demo_summary = df.groupby(['Sex', 'Ethnicity']).agg({
            'Age': ['count', 'mean', 'std'],
            'eGFR': ['mean', 'std'] if 'eGFR' in df.columns else ['count']
        }).round(2)
        
        # Flatten column names
        demo_summary.columns = ['_'.join(col).strip() for col in demo_summary.columns]
        demo_summary = demo_summary.reset_index()
        
        st.dataframe(demo_summary, use_container_width=True)

def render_clinical_metrics_analysis(df):
    """Render clinical metrics analysis"""
    
    st.subheader("🏥 Clinical Metrics Analysis")
    
    # eGFR and Creatinine analysis
    if 'eGFR' in df.columns and 'Creatinine' in df.columns:
        col1, col2 = st.columns(2)
        
        with col1:
            # eGFR distribution with CKD staging
            fig_egfr = px.histogram(
                df,
                x='eGFR',
                nbins=30,
                title="eGFR Distribution with CKD Stages"
            )
            
            # Add CKD stage boundaries
            colors = ['red', 'orange', 'yellow', 'lightgreen']
            stages = [15, 30, 60, 90]
            stage_names = ['Stage 5', 'Stage 4', 'Stage 3', 'Stage 2']
            
            for i, (stage, color, name) in enumerate(zip(stages, colors, stage_names)):
                fig_egfr.add_vline(
                    x=stage, 
                    line_dash="dash", 
                    line_color=color,
                    annotation_text=name
                )
            
            st.plotly_chart(fig_egfr, use_container_width=True)
        
        with col2:
            # eGFR vs Creatinine scatter with diagnosis
            fig_scatter = px.scatter(
                df,
                x='Creatinine',
                y='eGFR',
                color='Diagnosis' if 'Diagnosis' in df.columns else None,
                title="eGFR vs Creatinine by Diagnosis",
                hover_data=['Age', 'Sex'] if all(col in df.columns for col in ['Age', 'Sex']) else None
            )
            st.plotly_chart(fig_scatter, use_container_width=True)
    
    # CKD staging analysis
    if 'eGFR' in df.columns:
        st.markdown("### 🎯 CKD Staging Analysis")
        
        # Create CKD stages
        def get_ckd_stage(egfr):
            if egfr >= 90:
                return "Stage 1-2 (eGFR ≥90)"
            elif egfr >= 60:
                return "Stage 3a (eGFR 60-89)"
            elif egfr >= 45:
                return "Stage 3b (eGFR 45-59)"
            elif egfr >= 30:
                return "Stage 4 (eGFR 30-44)"
            elif egfr >= 15:
                return "Stage 4 (eGFR 15-29)"
            else:
                return "Stage 5 (eGFR <15)"
        
        df_with_stages = df.copy()
        df_with_stages['CKD_Stage'] = df_with_stages['eGFR'].apply(get_ckd_stage)
        
        col1, col2 = st.columns(2)
        
        with col1:
            # CKD stage distribution
            stage_counts = df_with_stages['CKD_Stage'].value_counts()
            fig_stages = px.bar(
                x=stage_counts.index,
                y=stage_counts.values,
                title="CKD Stage Distribution",
                labels={'x': 'CKD Stage', 'y': 'Number of Patients'}
            )
            st.plotly_chart(fig_stages, use_container_width=True)
        
        with col2:
            # Age distribution by CKD stage
            fig_age_stage = px.box(
                df_with_stages,
                x='CKD_Stage',
                y='Age',
                title="Age Distribution by CKD Stage"
            )
            fig_age_stage.update_xaxes(tickangle=45)
            st.plotly_chart(fig_age_stage, use_container_width=True)
    
    # Clinical metrics summary
    st.markdown("### 📊 Clinical Metrics Summary")
    
    clinical_cols = ['eGFR', 'Creatinine']
    available_clinical = [col for col in clinical_cols if col in df.columns]
    
    if available_clinical:
        summary_stats = df[available_clinical].describe().T
        summary_stats['IQR'] = summary_stats['75%'] - summary_stats['25%']
        
        st.dataframe(summary_stats, use_container_width=True)

def render_genetic_analysis(df):
    """Render genetic variants analysis"""
    
    st.subheader("🧬 Genetic Variants Analysis")
    
    genetic_columns = ['APOL1', 'NPHS1', 'NPHS2', 'WT1', 'UMOD', 'COL4A3']
    available_genetic = [col for col in genetic_columns if col in df.columns]
    
    if not available_genetic:
        st.warning("No genetic variant data available")
        return
    
    # Mutation frequency analysis
    col1, col2 = st.columns(2)
    
    with col1:
        # Mutation frequencies
        mutation_data = []
        for gene in available_genetic:
            total_valid = df[gene].isin(['Mut', 'WT']).sum()
            mutation_count = (df[gene] == 'Mut').sum()
            if total_valid > 0:
                mutation_freq = (mutation_count / total_valid) * 100
                mutation_data.append({
                    'Gene': gene,
                    'Mutation_Frequency': mutation_freq,
                    'Mutations': mutation_count,
                    'Total': total_valid
                })
        
        if mutation_data:
            mutation_df = pd.DataFrame(mutation_data)
            fig_mut_freq = px.bar(
                mutation_df,
                x='Gene',
                y='Mutation_Frequency',
                title="Mutation Frequency by Gene (%)",
                text='Mutations'
            )
            fig_mut_freq.update_traces(texttemplate='%{text}', textposition='outside')
            st.plotly_chart(fig_mut_freq, use_container_width=True)
    
    with col2:
        # APOL1 variant analysis
        if 'APOL1_Variant' in df.columns:
            st.markdown("**APOL1 Risk Categories:**")
            st.markdown("• **High Risk:** G1/G1, G1/G2, G2/G2 (kidney disease susceptibility)")
            st.markdown("• **Intermediate Risk:** G0/G1, G0/G2 (one risk allele)")
            st.markdown("• **Low Risk:** G0/G0 (reference, no risk alleles)")
            
            apol1_counts = df['APOL1_Variant'].value_counts()
            
            # Define risk categories
            high_risk = ['G1/G1', 'G1/G2', 'G2/G2']
            intermediate_risk = ['G0/G1', 'G0/G2']
            low_risk = ['G0/G0']
            
            risk_data = []
            for variant, count in apol1_counts.items():
                if variant in high_risk:
                    risk = 'High Risk'
                elif variant in intermediate_risk:
                    risk = 'Intermediate Risk'
                else:
                    risk = 'Low Risk'
                risk_data.append({'Variant': variant, 'Count': count, 'Risk': risk})
            
            risk_df = pd.DataFrame(risk_data)
            fig_apol1 = px.bar(
                risk_df,
                x='Variant',
                y='Count',
                color='Risk',
                title="APOL1 Variant Distribution by Risk Category"
            )
            st.plotly_chart(fig_apol1, use_container_width=True)
    
    # Genetic co-occurrence analysis
    st.markdown("### 🔗 Genetic Variant Co-occurrence")
    
    if len(available_genetic) > 1:
        # Create binary matrix for mutations
        genetic_binary = df[available_genetic].apply(lambda x: (x == 'Mut').astype(int))
        
        # Calculate correlation matrix
        correlation_matrix = genetic_binary.corr()
        
        # Create heatmap
        fig_heatmap = px.imshow(
            correlation_matrix,
            title="Genetic Variant Correlation Matrix",
            labels=dict(color="Correlation"),
            aspect="auto",
            color_continuous_scale="RdBu"
        )
        st.plotly_chart(fig_heatmap, use_container_width=True)
        
        # Multiple mutations analysis
        genetic_binary['total_mutations'] = genetic_binary.sum(axis=1)
        
        col1, col2 = st.columns(2)
        
        with col1:
            # Distribution of mutation counts
            mutation_count_dist = genetic_binary['total_mutations'].value_counts().sort_index()
            fig_mut_count = px.bar(
                x=mutation_count_dist.index,
                y=mutation_count_dist.values,
                title="Distribution of Multiple Mutations",
                labels={'x': 'Number of Mutations', 'y': 'Number of Patients'}
            )
            st.plotly_chart(fig_mut_count, use_container_width=True)
        
        with col2:
            # Gene interaction summary
            st.markdown("### 🧬 Gene Interaction Summary")
            
            # Show mutation co-occurrence
            if len(available_genetic) > 1:
                cooccurrence_data = []
                for i, gene1 in enumerate(available_genetic):
                    for gene2 in available_genetic[i+1:]:
                        both_mut = ((df[gene1] == 'Mut') & (df[gene2] == 'Mut')).sum()
                        total_valid = ((df[gene1].isin(['Mut', 'WT'])) & 
                                     (df[gene2].isin(['Mut', 'WT']))).sum()
                        if total_valid > 0:
                            cooccurrence_rate = (both_mut / total_valid) * 100
                            cooccurrence_data.append({
                                'Gene Pair': f"{gene1} + {gene2}",
                                'Co-occurrence Rate (%)': cooccurrence_rate,
                                'Patients': both_mut
                            })
                
                if cooccurrence_data:
                    cooccurrence_df = pd.DataFrame(cooccurrence_data)
                    cooccurrence_df = cooccurrence_df.sort_values('Co-occurrence Rate (%)', ascending=False)
                    st.dataframe(cooccurrence_df, use_container_width=True)

def render_correlation_analysis(df):
    """Render correlation analysis"""
    
    st.subheader("🔗 Correlation Analysis")
    
    # Select numeric columns
    numeric_columns = df.select_dtypes(include=[np.number]).columns.tolist()
    
    if len(numeric_columns) < 2:
        st.warning("Insufficient numeric columns for correlation analysis")
        return
    
    # Column selection
    selected_columns = st.multiselect(
        "Select columns for correlation analysis:",
        options=numeric_columns,
        default=numeric_columns[:6] if len(numeric_columns) >= 6 else numeric_columns
    )
    
    if len(selected_columns) < 2:
        st.warning("Please select at least 2 columns")
        return
    
    # Calculate correlation matrix
    correlation_matrix = df[selected_columns].corr()
    
    col1, col2 = st.columns(2)
    
    with col1:
        # Correlation heatmap
        fig_corr = px.imshow(
            correlation_matrix,
            title="Correlation Matrix",
            labels=dict(color="Correlation"),
            aspect="auto",
            color_continuous_scale="RdBu",
            zmin=-1,
            zmax=1
        )
        st.plotly_chart(fig_corr, use_container_width=True)
    
    with col2:
        # Strongest correlations
        st.markdown("### 🔍 Strongest Correlations")
        
        # Get correlation pairs
        corr_pairs = []
        for i in range(len(correlation_matrix.columns)):
            for j in range(i+1, len(correlation_matrix.columns)):
                col1_name = correlation_matrix.columns[i]
                col2_name = correlation_matrix.columns[j]
                corr_value = correlation_matrix.iloc[i, j]
                corr_pairs.append({
                    'Variable 1': col1_name,
                    'Variable 2': col2_name,
                    'Correlation': corr_value,
                    'Abs Correlation': abs(corr_value)
                })
        
        corr_df = pd.DataFrame(corr_pairs)
        corr_df = corr_df.sort_values('Abs Correlation', ascending=False)
        
        # Display top correlations
        top_correlations = corr_df.head(10)
        for _, row in top_correlations.iterrows():
            corr_strength = "Strong" if abs(row['Correlation']) > 0.7 else "Moderate" if abs(row['Correlation']) > 0.3 else "Weak"
            corr_direction = "Positive" if row['Correlation'] > 0 else "Negative"
            
            st.write(f"**{row['Variable 1']} ↔ {row['Variable 2']}**")
            st.write(f"Correlation: {row['Correlation']:.3f} ({corr_strength} {corr_direction})")
            st.write("---")
    
    # Scatter plot matrix
    if len(selected_columns) <= 5:  # Limit for readability
        st.markdown("### 📊 Scatter Plot Matrix")
        
        fig_scatter_matrix = px.scatter_matrix(
            df[selected_columns],
            title="Scatter Plot Matrix"
        )
        fig_scatter_matrix.update_layout(height=600)
        st.plotly_chart(fig_scatter_matrix, use_container_width=True)

def render_pca_analysis(df):
    """Render Principal Component Analysis"""
    
    st.subheader("📊 Principal Component Analysis")
    
    # Select numeric columns for PCA
    numeric_columns = df.select_dtypes(include=[np.number]).columns.tolist()
    
    if len(numeric_columns) < 2:
        st.warning("Insufficient numeric columns for PCA")
        return
    
    # Column selection
    selected_columns = st.multiselect(
        "Select columns for PCA:",
        options=numeric_columns,
        default=numeric_columns[:5] if len(numeric_columns) >= 5 else numeric_columns
    )
    
    if len(selected_columns) < 2:
        st.warning("Please select at least 2 columns")
        return
    
    # Prepare data
    pca_data = df[selected_columns].dropna()
    
    if len(pca_data) == 0:
        st.warning("No complete cases available for PCA")
        return
    
    # Standardize data
    scaler = StandardScaler()
    scaled_data = scaler.fit_transform(pca_data)
    
    # Perform PCA
    pca = PCA()
    pca_result = pca.fit_transform(scaled_data)
    
    # Create PCA DataFrame
    pca_df = pd.DataFrame(
        pca_result[:, :min(3, pca_result.shape[1])],
        columns=[f'PC{i+1}' for i in range(min(3, pca_result.shape[1]))]
    )
    
    # Add original data for coloring
    for col in ['Diagnosis', 'Sex', 'Ethnicity']:
        if col in df.columns:
            pca_df[col] = df.loc[pca_data.index, col].values
            break
    
    col1, col2 = st.columns(2)
    
    with col1:
        # Explained variance
        explained_variance = pca.explained_variance_ratio_
        cumulative_variance = explained_variance.cumsum()
        
        fig_variance = go.Figure()
        fig_variance.add_trace(go.Bar(
            x=[f'PC{i+1}' for i in range(len(explained_variance))],
            y=explained_variance,
            name='Individual',
            marker_color='lightblue'
        ))
        fig_variance.add_trace(go.Scatter(
            x=[f'PC{i+1}' for i in range(len(cumulative_variance))],
            y=cumulative_variance,
            mode='lines+markers',
            name='Cumulative',
            yaxis='y2',
            marker_color='red'
        ))
        
        fig_variance.update_layout(
            title="PCA Explained Variance",
            xaxis_title="Principal Components",
            yaxis=dict(title="Explained Variance Ratio", side="left"),
            yaxis2=dict(title="Cumulative Explained Variance", side="right", overlaying="y"),
            legend=dict(x=0.7, y=0.9)
        )
        
        st.plotly_chart(fig_variance, use_container_width=True)
    
    with col2:
        # PCA scatter plot
        color_col = None
        for col in ['Diagnosis', 'Sex', 'Ethnicity']:
            if col in pca_df.columns:
                color_col = col
                break
        
        if pca_df.shape[1] >= 2:
            fig_pca = px.scatter(
                pca_df,
                x='PC1',
                y='PC2',
                color=color_col,
                title=f"PCA Plot (PC1 vs PC2)",
                hover_data=['PC3'] if 'PC3' in pca_df.columns else None
            )
            st.plotly_chart(fig_pca, use_container_width=True)
    
    # Component loadings
    st.markdown("### 📊 Component Loadings")
    
    # Create loadings DataFrame
    loadings_data = []
    for i, component in enumerate(pca.components_[:3]):  # First 3 components
        for j, loading in enumerate(component):
            loadings_data.append({
                'Component': f'PC{i+1}',
                'Variable': selected_columns[j],
                'Loading': loading
            })
    
    loadings_df = pd.DataFrame(loadings_data)
    
    # Heatmap of loadings
    loadings_pivot = loadings_df.pivot(index='Variable', columns='Component', values='Loading')
    
    fig_loadings = px.imshow(
        loadings_pivot,
        title="PCA Component Loadings",
        labels=dict(color="Loading"),
        aspect="auto",
        color_continuous_scale="RdBu"
    )
    st.plotly_chart(fig_loadings, use_container_width=True)
    
    # Summary statistics
    st.markdown("### 📋 PCA Summary")
    
    summary_data = {
        'Component': [f'PC{i+1}' for i in range(len(explained_variance[:5]))],
        'Explained Variance': [f'{var:.3f}' for var in explained_variance[:5]],
        'Cumulative Variance': [f'{cum:.3f}' for cum in cumulative_variance[:5]]
    }
    
    summary_df = pd.DataFrame(summary_data)
    st.dataframe(summary_df, use_container_width=True)

def render_risk_stratification(df):
    """Render risk stratification analysis"""
    
    st.subheader("⚠️ Risk Stratification Analysis")
    
    # Risk calculation methodology explanation
    st.markdown("""
    #### 📋 Risk Score Methodology
    
    **How Risk Scores Are Calculated:**
    
    **Individual Risk Factors:**
    - **eGFR Risk:** Low (≥60), Moderate (30-59), High (<30)
    - **APOL1 Risk:** Low (G0/G0, G0/G1, G0/G2), High (G1/G1, G1/G2, G2/G2)
    - **Age Risk:** Low (≤45), Moderate (46-65), High (>65)
    
    **Combined Risk Score:**
    - High Risk = 3 points, Moderate Risk = 2 points, Low Risk = 1 point
    - **Final Categories:** Very High (≥75% max score), High (≥67%), Moderate (≥50%), Low (<50%)
    
    ---
    """)
    
    # Define risk factors
    risk_factors = []
    
    # eGFR-based risk
    if 'eGFR' in df.columns:
        df['eGFR_Risk'] = df['eGFR'].apply(lambda x: 
            'High' if x < 30 else 'Moderate' if x < 60 else 'Low'
        )
        risk_factors.append('eGFR_Risk')
    
    # APOL1-based risk
    if 'APOL1_Variant' in df.columns:
        high_risk_variants = ['G1/G1', 'G1/G2', 'G2/G2']
        df['APOL1_Risk'] = df['APOL1_Variant'].apply(lambda x: 
            'High' if x in high_risk_variants else 'Low'
        )
        risk_factors.append('APOL1_Risk')
    
    # Age-based risk
    if 'Age' in df.columns:
        df['Age_Risk'] = df['Age'].apply(lambda x: 
            'High' if x > 65 else 'Moderate' if x > 45 else 'Low'
        )
        risk_factors.append('Age_Risk')
    
    if not risk_factors:
        st.warning("Insufficient data for risk stratification")
        return
    
    # Risk distribution analysis
    col1, col2 = st.columns(2)
    
    with col1:
        # Individual risk factor distributions
        for risk_factor in risk_factors:
            risk_counts = df[risk_factor].value_counts()
            fig_risk = px.pie(
                values=risk_counts.values,
                names=risk_counts.index,
                title=f"{risk_factor.replace('_', ' ')} Distribution"
            )
            st.plotly_chart(fig_risk, use_container_width=True)
    
    with col2:
        # Combined risk analysis
        if len(risk_factors) >= 2:
            # Create combined risk score
            risk_scores = []
            for _, row in df.iterrows():
                score = 0
                for factor in risk_factors:
                    if row[factor] == 'High':
                        score += 3
                    elif row[factor] == 'Moderate':
                        score += 2
                    else:
                        score += 1
                risk_scores.append(score)
            
            df['Combined_Risk_Score'] = risk_scores
            df['Combined_Risk_Category'] = df['Combined_Risk_Score'].apply(lambda x:
                'Very High' if x >= len(risk_factors) * 2.5 else
                'High' if x >= len(risk_factors) * 2 else
                'Moderate' if x >= len(risk_factors) * 1.5 else 'Low'
            )
            
            # Combined risk distribution
            combined_risk_counts = df['Combined_Risk_Category'].value_counts()
            fig_combined = px.bar(
                x=combined_risk_counts.index,
                y=combined_risk_counts.values,
                title="Combined Risk Category Distribution"
            )
            st.plotly_chart(fig_combined, use_container_width=True)
    
    # Risk by demographics
    st.markdown("### 👥 Risk by Demographics")
    
    demo_cols = ['Sex', 'Ethnicity']
    available_demo = [col for col in demo_cols if col in df.columns]
    
    if available_demo and 'Combined_Risk_Category' in df.columns:
        for demo_col in available_demo:
            risk_demo_crosstab = pd.crosstab(df[demo_col], df['Combined_Risk_Category'])
            risk_demo_pct = risk_demo_crosstab.div(risk_demo_crosstab.sum(axis=1), axis=0) * 100
            
            fig_risk_demo = px.bar(
                risk_demo_pct,
                title=f"Risk Distribution by {demo_col} (%)",
                barmode='stack'
            )
            st.plotly_chart(fig_risk_demo, use_container_width=True)

def render_medication_analysis(df):
    """Render medication analysis"""
    
    st.subheader("💊 Medication Analysis")
    
    if 'Medications' not in df.columns:
        st.warning("No medication data available")
        return
    
    # Parse medications
    all_medications = []
    for med_string in df['Medications'].dropna():
        meds = [med.strip() for med in str(med_string).split(',')]
        all_medications.extend(meds)
    
    med_counts = pd.Series(all_medications).value_counts()
    
    col1, col2 = st.columns(2)
    
    with col1:
        # Most common medications
        top_meds = med_counts.head(10)
        fig_meds = px.bar(
            x=top_meds.values,
            y=top_meds.index,
            orientation='h',
            title="Most Common Medications"
        )
        st.plotly_chart(fig_meds, use_container_width=True)
    
    with col2:
        # Medication classes
        med_classes = {
            'ACE Inhibitors': ['ACE Inhibitors'],
            'ARBs': ['ARBs'],
            'Diuretics': ['Diuretics'],
            'Calcium Channel Blockers': ['Calcium Channel Blockers'],
            'Statins': ['Statins'],
            'Insulin': ['Insulin'],
            'Erythropoietin': ['Erythropoietin']
        }
        
        class_counts = {}
        for class_name, class_meds in med_classes.items():
            count = 0
            for med in class_meds:
                count += med_counts.get(med, 0)
            class_counts[class_name] = count
        
        fig_classes = px.pie(
            values=list(class_counts.values()),
            names=list(class_counts.keys()),
            title="Medication Classes Distribution"
        )
        st.plotly_chart(fig_classes, use_container_width=True)
    
    # Polypharmacy analysis
    st.markdown("### 💊 Polypharmacy Analysis")
    
    # Count medications per patient
    med_counts_per_patient = []
    for med_string in df['Medications'].fillna(''):
        if med_string:
            med_count = len([med.strip() for med in str(med_string).split(',')])
        else:
            med_count = 0
        med_counts_per_patient.append(med_count)
    
    df['Medication_Count'] = med_counts_per_patient
    
    col1, col2 = st.columns(2)
    
    with col1:
        # Distribution of medication counts
        fig_poly = px.histogram(
            df,
            x='Medication_Count',
            title="Number of Medications per Patient",
            nbins=max(df['Medication_Count']) + 1
        )
        st.plotly_chart(fig_poly, use_container_width=True)
    
    with col2:
        # Medication count by diagnosis
        if 'Diagnosis' in df.columns:
            fig_med_diag = px.box(
                df,
                x='Diagnosis',
                y='Medication_Count',
                title="Medication Count by Diagnosis"
            )
            fig_med_diag.update_xaxes(tickangle=45)
            st.plotly_chart(fig_med_diag, use_container_width=True)

def render_trial_eligibility_analysis(df):
    """Render clinical trial eligibility analysis"""
    
    st.subheader("🔬 Clinical Trial Eligibility Analysis")
    
    if 'Eligible_For_Trial' not in df.columns:
        st.warning("No trial eligibility data available")
        return
    
    # Overall eligibility
    eligibility_counts = df['Eligible_For_Trial'].value_counts()
    eligibility_rate = (eligibility_counts.get('Yes', 0) / len(df)) * 100
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.metric("Total Patients", len(df))
    with col2:
        st.metric("Eligible Patients", eligibility_counts.get('Yes', 0))
    with col3:
        st.metric("Eligibility Rate", f"{eligibility_rate:.1f}%")
    
    # Eligibility by demographics
    demo_cols = ['Sex', 'Ethnicity', 'Diagnosis']
    available_demo = [col for col in demo_cols if col in df.columns]
    
    for demo_col in available_demo:
        eligibility_demo = pd.crosstab(df[demo_col], df['Eligible_For_Trial'])
        eligibility_demo_pct = eligibility_demo.div(eligibility_demo.sum(axis=1), axis=0) * 100
        
        fig_elig_demo = px.bar(
            eligibility_demo_pct,
            title=f"Trial Eligibility by {demo_col} (%)",
            barmode='stack'
        )
        st.plotly_chart(fig_elig_demo, use_container_width=True)
    
    # Eligibility by clinical metrics
    if 'eGFR' in df.columns:
        st.markdown("### 🏥 Eligibility by Clinical Metrics")
        
        fig_egfr_elig = px.box(
            df,
            x='Eligible_For_Trial',
            y='eGFR',
            title="eGFR Distribution by Trial Eligibility"
        )
        st.plotly_chart(fig_egfr_elig, use_container_width=True)
        
        # eGFR thresholds analysis
        egfr_thresholds = [15, 30, 45, 60, 90]
        threshold_data = []
        
        for threshold in egfr_thresholds:
            below_threshold = df[df['eGFR'] < threshold]
            if len(below_threshold) > 0:
                eligible_pct = (below_threshold['Eligible_For_Trial'] == 'Yes').mean() * 100
                threshold_data.append({
                    'eGFR Threshold': f'<{threshold}',
                    'Patients': len(below_threshold),
                    'Eligibility Rate (%)': eligible_pct
                })
        
        if threshold_data:
            threshold_df = pd.DataFrame(threshold_data)
            st.dataframe(threshold_df, use_container_width=True)
