import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots

def render_data_overview(data_processor):
    """Render comprehensive data overview and documentation"""
    
    st.header("📋 Dataset Overview")
    
    if data_processor is None:
        st.warning("No data loaded. Please upload a dataset first.")
        return
    
    df = data_processor.get_data()
    metadata = data_processor.get_metadata()
    
    # Dataset summary
    st.markdown("## 📊 Dataset Summary")
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.metric("Total Records", f"{len(df):,}")
        st.metric("Data Columns", len(df.columns))
    
    with col2:
        missing_data = df.isnull().sum().sum()
        completeness = ((df.size - missing_data) / df.size) * 100
        st.metric("Data Completeness", f"{completeness:.1f}%")
        st.metric("Missing Values", f"{missing_data:,}")
    
    with col3:
        numeric_cols = df.select_dtypes(include=['number']).columns
        st.metric("Numeric Columns", len(numeric_cols))
        categorical_cols = df.select_dtypes(include=['object']).columns
        st.metric("Text Columns", len(categorical_cols))
    
    # Data schema
    st.markdown("---")
    st.markdown("## 🏗️ Data Schema")
    
    # Create schema table
    schema_data = []
    for col in df.columns:
        dtype = str(df[col].dtype)
        null_count = df[col].isnull().sum()
        null_pct = (null_count / len(df)) * 100
        unique_count = df[col].nunique()
        
        schema_data.append({
            'Column': col,
            'Data Type': dtype,
            'Unique Values': unique_count,
            'Missing Count': null_count,
            'Missing %': f"{null_pct:.1f}%",
            'Sample Values': ', '.join(str(x) for x in df[col].dropna().unique()[:3])
        })
    
    schema_df = pd.DataFrame(schema_data)
    st.dataframe(schema_df, use_container_width=True, height=400)
    
    # Field descriptions
    st.markdown("---")
    st.markdown("## 📝 Field Descriptions")
    
    field_descriptions = {
        'PatientID': 'Unique identifier for each patient',
        'Age': 'Patient age in years',
        'Sex': 'Patient biological sex (M/F)',
        'Ethnicity': 'Patient ethnicity (African American, Caucasian, Hispanic, Asian, Other)',
        'APOL1_Variant': 'APOL1 genetic variant status (G0/G0, G0/G1, G1/G1, G1/G2, G2/G2)',
        'eGFR': 'Estimated Glomerular Filtration Rate (mL/min/1.73m²)',
        'Creatinine': 'Serum creatinine level (mg/dL)',
        'Diagnosis': 'Primary kidney-related diagnosis',
        'Medications': 'Current medications (comma-separated)',
        'Eligible_For_Trial': 'Clinical trial eligibility status (Yes/No)',
        'Clinical_Notes': 'Free-text clinical observations',
        'APOL1': 'APOL1 gene mutation status (Mut/WT)',
        'NPHS1': 'NPHS1 gene mutation status (Mut/WT)',
        'NPHS2': 'NPHS2 gene mutation status (Mut/WT)',
        'WT1': 'WT1 gene mutation status (Mut/WT)',
        'UMOD': 'UMOD gene mutation status (Mut/WT)',
        'COL4A3': 'COL4A3 gene mutation status (Mut/WT)'
    }
    
    # Display field descriptions in expandable sections
    for field, description in field_descriptions.items():
        if field in df.columns:
            with st.expander(f"📋 {field}"):
                st.write(f"**Description:** {description}")
                
                # Show basic statistics
                if df[field].dtype in ['int64', 'float64']:
                    col1, col2 = st.columns(2)
                    with col1:
                        st.write("**Statistics:**")
                        st.write(f"• Mean: {df[field].mean():.2f}")
                        st.write(f"• Median: {df[field].median():.2f}")
                        st.write(f"• Min: {df[field].min():.2f}")
                        st.write(f"• Max: {df[field].max():.2f}")
                    with col2:
                        # Mini histogram
                        fig = px.histogram(df, x=field, title=f"{field} Distribution")
                        fig.update_layout(height=200, margin=dict(l=0, r=0, t=30, b=0))
                        st.plotly_chart(fig, use_container_width=True)
                else:
                    col1, col2 = st.columns(2)
                    with col1:
                        st.write("**Value Counts:**")
                        value_counts = df[field].value_counts().head(5)
                        for value, count in value_counts.items():
                            pct = (count / len(df)) * 100
                            st.write(f"• {value}: {count} ({pct:.1f}%)")
                    with col2:
                        # Mini bar chart
                        value_counts = df[field].value_counts().head(10)
                        fig = px.bar(x=value_counts.index, y=value_counts.values, 
                                   title=f"{field} Distribution")
                        fig.update_layout(height=200, margin=dict(l=0, r=0, t=30, b=0))
                        st.plotly_chart(fig, use_container_width=True)
    
    # Clinical metrics guide
    st.markdown("---")
    st.markdown("## 🏥 Clinical Reference Guide")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("### eGFR (Estimated Glomerular Filtration Rate)")
        st.markdown("""
        **Normal Range:** >90 mL/min/1.73m²
        
        **CKD Staging:**
        - **Stage 1:** eGFR ≥90 (with kidney damage)
        - **Stage 2:** eGFR 60-89 (mild decrease)
        - **Stage 3a:** eGFR 45-59 (moderate decrease)
        - **Stage 3b:** eGFR 30-44 (moderate decrease)
        - **Stage 4:** eGFR 15-29 (severe decrease)
        - **Stage 5:** eGFR <15 (kidney failure)
        """)
        
        st.markdown("### Creatinine")
        st.markdown("""
        **Normal Range:**
        - Men: 0.7-1.3 mg/dL
        - Women: 0.6-1.1 mg/dL
        
        **Clinical Significance:**
        Higher levels indicate reduced kidney function
        """)
    
    with col2:
        st.markdown("### APOL1 Risk Variants")
        st.markdown("""
        **Risk Categories:**
        - **G0/G0:** Low risk (reference)
        - **G0/G1, G0/G2:** Intermediate risk
        - **G1/G1, G1/G2, G2/G2:** High risk
        
        **Clinical Impact:**
        High-risk variants increase susceptibility to 
        kidney disease in individuals of African ancestry
        """)
        
        st.markdown("### Genetic Markers")
        st.markdown("""
        **Key Genes:**
        - **NPHS1:** Nephrin gene (congenital nephrotic syndrome)
        - **NPHS2:** Podocin gene (steroid-resistant nephrotic syndrome)
        - **WT1:** Wilms tumor 1 (Denys-Drash syndrome)
        - **UMOD:** Uromodulin (autosomal dominant tubulointerstitial kidney disease)
        - **COL4A3:** Collagen IV alpha 3 (Alport syndrome)
        """)
    
    # Data quality assessment
    st.markdown("---")
    st.markdown("## 🔍 Data Quality Assessment")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("### Missing Data Analysis")
        missing_data = df.isnull().sum().sort_values(ascending=False)
        missing_pct = (missing_data / len(df)) * 100
        
        missing_df = pd.DataFrame({
            'Column': missing_data.index,
            'Missing Count': missing_data.values,
            'Missing %': missing_pct.values
        })
        missing_df = missing_df[missing_df['Missing Count'] > 0]
        
        if len(missing_df) > 0:
            fig_missing = px.bar(
                missing_df,
                x='Missing %',
                y='Column',
                orientation='h',
                title="Missing Data by Column"
            )
            st.plotly_chart(fig_missing, use_container_width=True)
        else:
            st.success("✅ No missing data found!")
    
    with col2:
        st.markdown("### Data Distribution Quality")
        
        # Check for potential data quality issues
        quality_issues = []
        
        # Age outliers
        if 'Age' in df.columns:
            age_outliers = ((df['Age'] < 0) | (df['Age'] > 120)).sum()
            if age_outliers > 0:
                quality_issues.append(f"⚠️ {age_outliers} potential age outliers")
        
        # eGFR outliers
        if 'eGFR' in df.columns:
            egfr_outliers = ((df['eGFR'] < 0) | (df['eGFR'] > 200)).sum()
            if egfr_outliers > 0:
                quality_issues.append(f"⚠️ {egfr_outliers} potential eGFR outliers")
        
        # Duplicate PatientIDs
        if 'PatientID' in df.columns:
            duplicates = df['PatientID'].duplicated().sum()
            if duplicates > 0:
                quality_issues.append(f"⚠️ {duplicates} duplicate Patient IDs")
        
        if quality_issues:
            st.markdown("**Potential Issues:**")
            for issue in quality_issues:
                st.write(issue)
        else:
            st.success("✅ No major data quality issues detected!")
    
    # Sample data preview
    st.markdown("---")
    st.markdown("## 👀 Sample Data Preview")
    
    # Random sample of rows
    sample_size = min(10, len(df))
    sample_df = df.sample(n=sample_size, random_state=42)
    
    st.dataframe(sample_df, use_container_width=True, height=400)
    
    # Export data description
    st.markdown("---")
    st.markdown("## 📤 Export Options")
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        if st.button("📊 Export Schema"):
            schema_csv = schema_df.to_csv(index=False)
            st.download_button(
                label="Download Schema (CSV)",
                data=schema_csv,
                file_name="dataset_schema.csv",
                mime="text/csv"
            )
    
    with col2:
        if st.button("📈 Export Statistics"):
            stats_df = df.describe(include='all').T
            stats_csv = stats_df.to_csv()
            st.download_button(
                label="Download Statistics (CSV)",
                data=stats_csv,
                file_name="dataset_statistics.csv",
                mime="text/csv"
            )
    
    with col3:
        if st.button("📋 Export Sample"):
            sample_csv = sample_df.to_csv(index=False)
            st.download_button(
                label="Download Sample (CSV)",
                data=sample_csv,
                file_name="dataset_sample.csv",
                mime="text/csv"
            )
    
    # Quick search tips
    st.markdown("---")
    st.markdown("## 💡 Search Tips")
    
    search_tips = [
        "Use natural language: 'Find patients with kidney disease and diabetes'",
        "Specify demographics: 'Young African American patients with genetic mutations'",
        "Include clinical metrics: 'Patients with eGFR below 30'",
        "Search by medications: 'Patients taking ACE inhibitors and diuretics'",
        "Filter by trial eligibility: 'Trial eligible patients with nephrotic syndrome'",
        "Combine multiple criteria: 'Hispanic patients with APOL1 mutations and high creatinine'"
    ]
    
    for tip in search_tips:
        st.info(f"💡 {tip}")
