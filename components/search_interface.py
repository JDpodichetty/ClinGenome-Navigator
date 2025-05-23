import streamlit as st
import pandas as pd
import plotly.express as px
from typing import List, Dict, Tuple
from utils.query_processor import QueryProcessor

def render_search_interface(data_processor, vector_search):
    """Render the semantic search interface"""
    
    st.header("🔍 Semantic Search")
    st.markdown("Use natural language to search through clinical genomics data")
    
    if data_processor is None or vector_search is None:
        st.warning("Vector search not initialized. Please load data first.")
        return
    
    df = data_processor.get_data()
    query_processor = QueryProcessor()
    
    # Initialize session state for search
    if 'search_results' not in st.session_state:
        st.session_state.search_results = None
    if 'last_query' not in st.session_state:
        st.session_state.last_query = ""
    
    # Check for suggested query from dashboard
    initial_query = ""
    if hasattr(st.session_state, 'suggested_query'):
        initial_query = st.session_state.suggested_query
        delattr(st.session_state, 'suggested_query')
    
    # Search interface
    col1, col2 = st.columns([3, 1])
    
    with col1:
        query = st.text_input(
            "Enter your search query:",
            value=initial_query,
            placeholder="e.g., Find patients with diabetic nephropathy and APOL1 mutations",
            help="Use natural language to describe what you're looking for"
        )
    
    with col2:
        st.markdown("<br>", unsafe_allow_html=True)  # Add spacing
        search_clicked = st.button("🔍 Search", type="primary", use_container_width=True)
    
    # Advanced search options
    with st.expander("⚙️ Advanced Search Options"):
        col1, col2, col3 = st.columns(3)
        
        with col1:
            top_k = st.slider("Number of results", min_value=5, max_value=50, value=10)
            similarity_threshold = st.slider("Similarity threshold", min_value=0.0, max_value=1.0, value=0.0, step=0.1)
        
        with col2:
            enhance_query = st.checkbox("Enhance query with medical terms", value=True)
            show_similarity_scores = st.checkbox("Show similarity scores", value=False)
        
        with col3:
            export_format = st.selectbox("Export format", ["CSV", "JSON"])
    
    # Example queries
    st.markdown("### 💡 Example Queries")
    example_queries = [
        "Find African American patients with high-risk APOL1 variants",
        "Show patients with eGFR below 30 and multiple medications",
        "Young patients with genetic mutations affecting kidney function",
        "Patients with nephrotic syndrome eligible for clinical trials",
        "Hispanic patients with diabetic nephropathy and hypertension"
    ]
    
    cols = st.columns(len(example_queries))
    for i, example in enumerate(example_queries):
        with cols[i]:
            if st.button(f"Try: {example[:30]}...", key=f"example_{i}", help=example):
                query = example
                search_clicked = True
    
    # Perform search
    if search_clicked and query.strip():
        with st.spinner("Searching..."):
            try:
                # Enhance query if requested
                search_query = query_processor.enhance_query(query) if enhance_query else query
                
                # Perform vector search
                indices, scores = vector_search.search(
                    search_query, 
                    top_k=top_k, 
                    similarity_threshold=similarity_threshold
                )
                
                if indices:
                    # Get results dataframe
                    results_df = df.iloc[indices].copy()
                    results_df['similarity_score'] = scores
                    
                    # Store results in session state
                    st.session_state.search_results = {
                        'query': query,
                        'enhanced_query': search_query,
                        'results_df': results_df,
                        'indices': indices,
                        'scores': scores
                    }
                    st.session_state.last_query = query
                    
                    st.success(f"Found {len(indices)} matching patients")
                else:
                    st.warning("No patients found matching your query. Try adjusting the similarity threshold or using different terms.")
                    st.session_state.search_results = None
                    
            except Exception as e:
                st.error(f"Search error: {str(e)}")
                st.session_state.search_results = None
    
    # Display search results
    if st.session_state.search_results:
        results_data = st.session_state.search_results
        results_df = results_data['results_df']
        
        st.markdown("---")
        st.subheader(f"🎯 Search Results for: '{results_data['query']}'")
        
        # Search summary
        search_summary = query_processor.generate_search_summary(
            results_data['query'], 
            len(results_df)
        )
        st.info(search_summary)
        
        # Results insights
        insights = vector_search.get_query_insights(results_data['query'], results_data['indices'], df)
        
        if insights.get('total_results', 0) > 0:
            # Display insights
            col1, col2, col3 = st.columns(3)
            
            with col1:
                st.markdown("**Demographics:**")
                if insights['demographics']['age_range']['mean']:
                    st.metric("Average Age", f"{insights['demographics']['age_range']['mean']:.1f}")
                
                if insights['demographics']['sex_distribution']:
                    sex_dist = insights['demographics']['sex_distribution']
                    st.write("Sex Distribution:")
                    for sex, count in sex_dist.items():
                        st.write(f"• {sex}: {count}")
            
            with col2:
                st.markdown("**Clinical:**")
                if insights['clinical']['egfr_stats'].get('mean'):
                    st.metric("Average eGFR", f"{insights['clinical']['egfr_stats']['mean']:.1f}")
                
                if insights['clinical']['trial_eligibility']:
                    trial_dist = insights['clinical']['trial_eligibility']
                    st.write("Trial Eligibility:")
                    for status, count in trial_dist.items():
                        st.write(f"• {status}: {count}")
            
            with col3:
                st.markdown("**Diagnoses:**")
                if insights['clinical']['diagnoses']:
                    top_diagnoses = list(insights['clinical']['diagnoses'].items())[:3]
                    for diagnosis, count in top_diagnoses:
                        st.write(f"• {diagnosis}: {count}")
        
        # Filters for results
        st.markdown("### 🔧 Filter Results")
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            if 'Diagnosis' in results_df.columns:
                diagnosis_filter = st.multiselect(
                    "Diagnoses",
                    options=sorted(results_df['Diagnosis'].unique()),
                    default=[]
                )
        
        with col2:
            if 'Sex' in results_df.columns:
                sex_filter = st.multiselect(
                    "Sex",
                    options=sorted(results_df['Sex'].unique()),
                    default=[]
                )
        
        with col3:
            if 'Ethnicity' in results_df.columns:
                ethnicity_filter = st.multiselect(
                    "Ethnicity",
                    options=sorted(results_df['Ethnicity'].unique()),
                    default=[]
                )
        
        with col4:
            if 'Eligible_For_Trial' in results_df.columns:
                trial_filter = st.selectbox(
                    "Trial Eligibility",
                    options=["All", "Yes", "No"],
                    index=0
                )
        
        # Apply filters
        filtered_df = results_df.copy()
        
        if 'diagnosis_filter' in locals() and diagnosis_filter:
            filtered_df = filtered_df[filtered_df['Diagnosis'].isin(diagnosis_filter)]
        if 'sex_filter' in locals() and sex_filter:
            filtered_df = filtered_df[filtered_df['Sex'].isin(sex_filter)]
        if 'ethnicity_filter' in locals() and ethnicity_filter:
            filtered_df = filtered_df[filtered_df['Ethnicity'].isin(ethnicity_filter)]
        if trial_filter != "All":
            filtered_df = filtered_df[filtered_df['Eligible_For_Trial'] == trial_filter]
        
        # Display filtered results
        st.markdown(f"### 📊 Results ({len(filtered_df)} patients)")
        
        # Results visualization
        if len(filtered_df) > 0:
            # Chart options
            chart_col1, chart_col2 = st.columns(2)
            
            with chart_col1:
                if 'eGFR' in filtered_df.columns and 'Age' in filtered_df.columns:
                    fig_scatter = px.scatter(
                        filtered_df,
                        x='Age',
                        y='eGFR',
                        color='Diagnosis',
                        title="Age vs eGFR by Diagnosis",
                        hover_data=['PatientID', 'Sex', 'Ethnicity']
                    )
                    st.plotly_chart(fig_scatter, use_container_width=True)
            
            with chart_col2:
                if 'Diagnosis' in filtered_df.columns:
                    diag_counts = filtered_df['Diagnosis'].value_counts()
                    fig_bar = px.bar(
                        x=diag_counts.values,
                        y=diag_counts.index,
                        orientation='h',
                        title="Diagnosis Distribution in Results"
                    )
                    st.plotly_chart(fig_bar, use_container_width=True)
        
        # Results table
        display_columns = [
            'PatientID', 'Age', 'Sex', 'Ethnicity', 'Diagnosis', 
            'eGFR', 'Creatinine', 'APOL1_Variant', 'Eligible_For_Trial'
        ]
        if show_similarity_scores:
            display_columns.append('similarity_score')
        
        # Filter columns that exist in the dataframe
        available_columns = [col for col in display_columns if col in filtered_df.columns]
        
        display_df = filtered_df[available_columns].copy()
        
        # Format similarity scores
        if 'similarity_score' in display_df.columns:
            display_df['similarity_score'] = display_df['similarity_score'].round(3)
        
        st.dataframe(
            display_df,
            use_container_width=True,
            height=400
        )
        
        # Export functionality
        col1, col2, col3 = st.columns([1, 1, 2])
        
        with col1:
            if st.button("📥 Export Results"):
                try:
                    if export_format == "CSV":
                        csv_data = data_processor.export_results(filtered_df, 'csv')
                        st.download_button(
                            label="Download CSV",
                            data=csv_data,
                            file_name=f"search_results_{len(filtered_df)}_patients.csv",
                            mime="text/csv"
                        )
                    else:
                        json_data = data_processor.export_results(filtered_df, 'json')
                        st.download_button(
                            label="Download JSON",
                            data=json_data,
                            file_name=f"search_results_{len(filtered_df)}_patients.json",
                            mime="application/json"
                        )
                except Exception as e:
                    st.error(f"Export error: {str(e)}")
        
        with col2:
            if st.button("🔄 New Search"):
                st.session_state.search_results = None
                st.session_state.last_query = ""
                st.rerun()
        
        # Similar patients section
        if len(filtered_df) > 0:
            st.markdown("---")
            st.markdown("### 👥 Find Similar Patients")
            
            patient_options = [
                f"{row['PatientID']} - {row.get('Age', 'N/A')} yr {row.get('Sex', 'N/A')} - {row.get('Diagnosis', 'N/A')}"
                for _, row in filtered_df.head(10).iterrows()
            ]
            
            selected_patient = st.selectbox(
                "Select a patient to find similar cases:",
                options=range(len(patient_options)),
                format_func=lambda x: patient_options[x]
            )
            
            if st.button("Find Similar Patients"):
                try:
                    patient_idx = filtered_df.iloc[selected_patient].name
                    similar_indices, similar_scores = vector_search.get_similar_patients(patient_idx, top_k=5)
                    
                    if similar_indices:
                        similar_df = df.iloc[similar_indices].copy()
                        similar_df['similarity_score'] = similar_scores
                        
                        st.markdown("#### 🔍 Similar Patients Found:")
                        st.dataframe(
                            similar_df[available_columns + ['similarity_score']],
                            use_container_width=True
                        )
                    else:
                        st.warning("No similar patients found.")
                        
                except Exception as e:
                    st.error(f"Error finding similar patients: {str(e)}")
    
    # Query suggestions and tips
    if not st.session_state.search_results and query.strip():
        suggestions = query_processor.suggest_query_improvements(query)
        if suggestions:
            st.markdown("### 💭 Query Suggestions")
            for suggestion in suggestions:
                st.info(f"💡 {suggestion}")
