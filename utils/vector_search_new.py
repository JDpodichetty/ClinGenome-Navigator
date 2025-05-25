import numpy as np
import pandas as pd
from typing import List, Tuple, Dict, Optional
import re
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

class VectorSearch:
    """Handles semantic search using TF-IDF vectorization for accurate clinical data search"""

    def __init__(self, model_name: str = 'tfidf'):
        """Initialize the search system"""
        self.model_name = model_name
        self.vectorizer = TfidfVectorizer(
            max_features=10000,
            stop_words='english',
            ngram_range=(1, 3),
            min_df=1,
            max_df=0.95,
            lowercase=True
        )
        self.tfidf_matrix = None
        self.texts = None
        self.df = None

    def build_index(self, df: pd.DataFrame, texts: Optional[List[str]] = None):
        """Build search index from the provided data"""
        if df.empty:
            raise ValueError("DataFrame is empty")
        
        self.df = df.copy()
        
        if texts is None:
            self.texts = self._create_texts_from_df(df)
        else:
            self.texts = texts
        
        if self.texts:
            # Build TF-IDF matrix
            self.tfidf_matrix = self.vectorizer.fit_transform(self.texts)
            print(f"Built search index with {len(self.texts)} documents")
        else:
            raise ValueError("No texts available for indexing")

    def _create_texts_from_df(self, df: pd.DataFrame) -> List[str]:
        """Create searchable text representations from dataframe"""
        texts = []
        
        for _, row in df.iterrows():
            text_parts = []
            
            # Patient demographics
            if 'Age' in df.columns:
                text_parts.append(f"Patient age {row.get('Age', 'unknown')}")
            
            if 'Sex' in df.columns:
                sex = row.get('Sex', '')
                if sex == 'M':
                    text_parts.append("Male patient")
                elif sex == 'F':
                    text_parts.append("Female patient")
            
            if 'Ethnicity' in df.columns:
                ethnicity = row.get('Ethnicity', '')
                if ethnicity:
                    text_parts.append(f"{ethnicity} ethnicity")
            
            # Clinical information
            if 'Diagnosis' in df.columns:
                diagnosis = row.get('Diagnosis', '')
                if diagnosis:
                    text_parts.append(f"diagnosed with {diagnosis}")
            
            if 'eGFR' in df.columns:
                egfr = row.get('eGFR', 0)
                if egfr:
                    text_parts.append(f"eGFR {egfr}")
                    if egfr < 30:
                        text_parts.append("severe kidney dysfunction")
                    elif egfr < 60:
                        text_parts.append("moderate kidney dysfunction")
                    else:
                        text_parts.append("normal kidney function")
            
            if 'Creatinine' in df.columns:
                creatinine = row.get('Creatinine', 0)
                if creatinine:
                    text_parts.append(f"creatinine {creatinine}")
            
            # Genetic information
            if 'APOL1_Variant' in df.columns:
                apol1_variant = row.get('APOL1_Variant', '')
                if apol1_variant:
                    text_parts.append(f"APOL1 variant {apol1_variant}")
                    if apol1_variant in ['G1/G2', 'G1/G1', 'G2/G2']:
                        text_parts.append("high-risk APOL1 mutation")
            
            # Individual genetic markers
            genetic_cols = ['APOL1', 'NPHS1', 'NPHS2', 'WT1', 'UMOD', 'COL4A3']
            for col in genetic_cols:
                if col in df.columns:
                    value = row.get(col, '')
                    if value == 'Mut':
                        text_parts.append(f"{col} mutation")
                    elif value == 'WT':
                        text_parts.append(f"{col} wild type")
            
            # Medications
            if 'Medications' in df.columns:
                medications = row.get('Medications', '')
                if medications and medications != 'None':
                    text_parts.append(f"medications {medications}")
            
            # Trial eligibility
            if 'Eligible_For_Trial' in df.columns:
                eligible = row.get('Eligible_For_Trial', '')
                if eligible == 'Yes':
                    text_parts.append("eligible for clinical trials")
                elif eligible == 'No':
                    text_parts.append("not eligible for clinical trials")
            
            # Clinical notes
            if 'Clinical_Notes' in df.columns:
                notes = row.get('Clinical_Notes', '')
                if notes:
                    text_parts.append(notes)
            
            texts.append(" ".join(text_parts))
        
        return texts

    def search(self, query: str, top_k: int = 10, similarity_threshold: float = 0.0) -> Tuple[List[int], List[float]]:
        """Perform semantic search with enhanced filtering"""
        if self.tfidf_matrix is None:
            raise ValueError("Index not built. Call build_index() first.")
        
        # Apply numerical filters first
        filtered_indices = self._apply_numerical_filters(query)
        
        if filtered_indices is not None and len(filtered_indices) == 0:
            return [], []
        
        # Transform query using the same vectorizer
        query_vector = self.vectorizer.transform([query.lower()])
        
        # Calculate similarities
        if filtered_indices is not None:
            # Only calculate similarities for filtered indices
            subset_matrix = self.tfidf_matrix[filtered_indices]
            similarities = cosine_similarity(query_vector, subset_matrix).flatten()
            
            # Create (similarity, original_index) pairs
            sim_idx_pairs = [(similarities[i], filtered_indices[i]) for i in range(len(filtered_indices))]
        else:
            # Calculate similarities for all documents
            similarities = cosine_similarity(query_vector, self.tfidf_matrix).flatten()
            sim_idx_pairs = [(similarities[i], i) for i in range(len(similarities))]
        
        # Filter by similarity threshold and sort
        filtered_pairs = [(sim, idx) for sim, idx in sim_idx_pairs if sim >= similarity_threshold]
        filtered_pairs.sort(reverse=True, key=lambda x: x[0])
        
        # Extract top_k results
        top_pairs = filtered_pairs[:top_k]
        indices = [idx for _, idx in top_pairs]
        scores = [sim for sim, _ in top_pairs]
        
        return indices, scores

    def _apply_numerical_filters(self, query: str) -> Optional[List[int]]:
        """Apply direct, simple filtering - no complex regex patterns"""
        if self.df is None:
            return None
        
        query_lower = query.lower()
        filtered_df = self.df.copy()
        original_size = len(filtered_df)
        
        # High-risk patients (direct keyword matching)
        if any(keyword in query_lower for keyword in ['high risk', 'high-risk', 'immediate clinical attention', 'need immediate']):
            if 'APOL1_Variant' in filtered_df.columns and 'eGFR' in filtered_df.columns and 'Diagnosis' in filtered_df.columns:
                # High-risk APOL1 variants OR severe kidney dysfunction OR advanced CKD
                high_risk_apol1 = filtered_df['APOL1_Variant'].isin(['G1/G1', 'G1/G2', 'G2/G2'])
                severe_kidney = filtered_df['eGFR'] < 30
                advanced_ckd = filtered_df['Diagnosis'].isin(['CKD Stage 4', 'CKD Stage 5'])
                filtered_df = filtered_df[high_risk_apol1 | severe_kidney | advanced_ckd]
        
        # Universal numerical filtering system
        import re
        numerical_cols = {'Age': 2, 'eGFR': 5, 'Creatinine': 0.2}  # column: margin for approximate searches
        
        for col_name, margin in numerical_cols.items():
            if col_name in filtered_df.columns:
                col_lower = col_name.lower()
                
                # Above/greater than patterns
                above_patterns = [
                    rf'(?:above|over|greater than)\s+{col_lower}\s*(\d+(?:\.\d+)?)',
                    rf'{col_lower}\s+(?:above|over|greater than)\s*(\d+(?:\.\d+)?)',
                    rf'(?:above|over)\s+{col_lower}\s+(\d+(?:\.\d+)?)',
                    rf'{col_lower}\s*>\s*(\d+(?:\.\d+)?)'
                ]
                
                # Below/less than patterns
                below_patterns = [
                    rf'(?:below|under|less than)\s+{col_lower}\s*(\d+(?:\.\d+)?)',
                    rf'{col_lower}\s+(?:below|under|less than)\s*(\d+(?:\.\d+)?)',
                    rf'(?:below|under)\s+{col_lower}\s+(\d+(?:\.\d+)?)',
                    rf'{col_lower}\s*<\s*(\d+(?:\.\d+)?)'
                ]
                
                # Near/about patterns
                near_patterns = [
                    rf'{col_lower}\s*(?:near|about|around|approximately)\s*(\d+(?:\.\d+)?)',
                    rf'(?:near|about|around|approximately)\s+{col_lower}\s*(\d+(?:\.\d+)?)',
                    rf'close\s+to.*?{col_lower}.*?(\d+(?:\.\d+)?)',
                    rf'value\s+of\s+{col_lower}\s+of\s*(\d+(?:\.\d+)?)',
                    rf'{col_lower}\s+(?:close\s+to|near)\s*(\d+(?:\.\d+)?)',
                    rf'around\s+the\s+{col_lower}\s+of\s+(\d+(?:\.\d+)?)'
                ]
                
                # Check for matches and apply filters
                threshold = None
                filter_type = None
                
                # Check above patterns
                for pattern in above_patterns:
                    match = re.search(pattern, query_lower)
                    if match:
                        threshold = float(match.group(1))
                        filter_type = 'above'
                        break
                
                # Check below patterns if no above match
                if not threshold:
                    for pattern in below_patterns:
                        match = re.search(pattern, query_lower)
                        if match:
                            threshold = float(match.group(1))
                            filter_type = 'below'
                            break
                
                # Check near patterns if no above/below match
                if not threshold:
                    for pattern in near_patterns:
                        match = re.search(pattern, query_lower)
                        if match:
                            threshold = float(match.group(1))
                            filter_type = 'near'
                            break
                
                # Apply the appropriate filter
                if threshold is not None:
                    if filter_type == 'above':
                        filtered_df = filtered_df[filtered_df[col_name] > threshold]
                    elif filter_type == 'below':
                        filtered_df = filtered_df[filtered_df[col_name] < threshold]
                    elif filter_type == 'near':
                        filtered_df = filtered_df[
                            (filtered_df[col_name] >= threshold - margin) & 
                            (filtered_df[col_name] <= threshold + margin)
                        ]
                    break  # Exit loop once a filter is applied
        
        # Sex filtering
        if 'male' in query_lower and 'female' not in query_lower:
            if 'Sex' in filtered_df.columns:
                filtered_df = filtered_df[filtered_df['Sex'] == 'Male']
        elif 'female' in query_lower and 'male' not in query_lower:
            if 'Sex' in filtered_df.columns:
                filtered_df = filtered_df[filtered_df['Sex'] == 'Female']
        
        # Ethnicity filtering
        if 'caucasian' in query_lower or 'white' in query_lower:
            if 'Ethnicity' in filtered_df.columns:
                filtered_df = filtered_df[filtered_df['Ethnicity'] == 'Caucasian']
        elif 'african american' in query_lower or 'black' in query_lower:
            if 'Ethnicity' in filtered_df.columns:
                filtered_df = filtered_df[filtered_df['Ethnicity'] == 'African American']
        elif 'hispanic' in query_lower or 'latino' in query_lower:
            if 'Ethnicity' in filtered_df.columns:
                filtered_df = filtered_df[filtered_df['Ethnicity'] == 'Hispanic']
        elif 'asian' in query_lower:
            if 'Ethnicity' in filtered_df.columns:
                filtered_df = filtered_df[filtered_df['Ethnicity'] == 'Asian']
        
        # Trial eligibility
        if 'trial eligible' in query_lower or 'eligible for trial' in query_lower:
            if 'Eligible_For_Trial' in filtered_df.columns:
                filtered_df = filtered_df[filtered_df['Eligible_For_Trial'] == 'Yes']
        
        # Specific APOL1 variants
        if 'g2/g2' in query_lower:
            if 'APOL1_Variant' in filtered_df.columns:
                filtered_df = filtered_df[filtered_df['APOL1_Variant'] == 'G2/G2']
        
        # Return indices only if filtering actually occurred
        return filtered_df.index.tolist() if len(filtered_df) < original_size else None

    def get_similar_patients(self, patient_idx: int, top_k: int = 5) -> Tuple[List[int], List[float]]:
        """Find similar patients to a given patient"""
        if self.tfidf_matrix is None:
            raise ValueError("Index not built. Call build_index() first.")
        
        if patient_idx >= len(self.texts):
            raise ValueError(f"Patient index {patient_idx} out of range")
        
        # Get the patient's vector
        patient_vector = self.tfidf_matrix[patient_idx]
        
        # Calculate similarities with all other patients
        similarities = cosine_similarity(patient_vector, self.tfidf_matrix).flatten()
        
        # Get top similar patients (excluding the patient itself)
        sim_indices = np.argsort(similarities)[::-1]
        similar_indices = [idx for idx in sim_indices if idx != patient_idx][:top_k]
        similar_scores = [similarities[idx] for idx in similar_indices]
        
        return similar_indices, similar_scores

    def suggest_queries(self) -> List[str]:
        """Suggest example queries for users"""
        return [
            "Find patients above age 50 with kidney disease",
            "Show me African American patients with APOL1 mutations",
            "Patients with eGFR below 30",
            "Female patients eligible for clinical trials", 
            "Patients with high-risk genetic variants",
            "Male patients with CKD Stage 4",
            "Patients under age 40 with normal kidney function",
            "Hispanic patients with NPHS2 mutations"
        ]

    def get_query_insights(self, query: str, results_indices: List[int], df: pd.DataFrame) -> Dict:
        """Generate insights about search results"""
        if not results_indices:
            return {"total_results": 0}
        
        results_df = df.iloc[results_indices]
        
        insights = {
            "total_results": len(results_indices),
            "demographics": {},
            "clinical": {},
            "genetic": {}
        }
        
        # Demographics insights
        if 'Age' in results_df.columns:
            insights["demographics"]["age_range"] = {
                "min": int(results_df['Age'].min()),
                "max": int(results_df['Age'].max()),
                "mean": round(results_df['Age'].mean(), 1)
            }
        
        if 'Sex' in results_df.columns:
            sex_counts = results_df['Sex'].value_counts()
            insights["demographics"]["sex_distribution"] = sex_counts.to_dict()
        
        if 'Ethnicity' in results_df.columns:
            ethnicity_counts = results_df['Ethnicity'].value_counts()
            insights["demographics"]["ethnicity_distribution"] = ethnicity_counts.to_dict()
        
        # Clinical insights
        if 'Diagnosis' in results_df.columns:
            diagnosis_counts = results_df['Diagnosis'].value_counts()
            insights["clinical"]["diagnoses"] = diagnosis_counts.to_dict()
        
        if 'eGFR' in results_df.columns:
            insights["clinical"]["egfr_stats"] = {
                "mean": round(results_df['eGFR'].mean(), 1),
                "min": round(results_df['eGFR'].min(), 1),
                "max": round(results_df['eGFR'].max(), 1)
            }
        
        if 'Eligible_For_Trial' in results_df.columns:
            trial_counts = results_df['Eligible_For_Trial'].value_counts()
            insights["clinical"]["trial_eligibility"] = trial_counts.to_dict()
        
        # Genetic insights
        if 'APOL1_Variant' in results_df.columns:
            apol1_counts = results_df['APOL1_Variant'].value_counts()
            insights["genetic"]["apol1_variants"] = apol1_counts.to_dict()
        
        return insights