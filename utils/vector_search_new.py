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
        
        # Build TF-IDF matrix
        self.tfidf_matrix = self.vectorizer.fit_transform(self.texts)
        print(f"Built search index with {len(self.texts)} documents")

    def _create_texts_from_df(self, df: pd.DataFrame) -> List[str]:
        """Create searchable text representations from dataframe"""
        texts = []
        
        for _, row in df.iterrows():
            text_parts = []
            
            # Demographics
            if pd.notna(row.get('Age')):
                text_parts.append(f"age {row['Age']}")
            if pd.notna(row.get('Sex')):
                text_parts.append(f"{row['Sex']} patient")
            if pd.notna(row.get('Ethnicity')):
                text_parts.append(f"{row['Ethnicity']} ethnicity")
            
            # Genetic variants
            if pd.notna(row.get('APOL1_Variant')):
                text_parts.append(f"APOL1 {row['APOL1_Variant']}")
            
            # Gene mutations
            gene_cols = ['NPHS1', 'NPHS2', 'WT1', 'COL4A3', 'UMOD']
            for gene in gene_cols:
                if pd.notna(row.get(gene)):
                    if row[gene] == 'Mut':
                        text_parts.append(f"{gene} mutation")
                    else:
                        text_parts.append(f"{gene} wild-type")
            
            # Clinical data
            if pd.notna(row.get('eGFR')):
                text_parts.append(f"eGFR {row['eGFR']}")
            if pd.notna(row.get('Creatinine')):
                text_parts.append(f"creatinine {row['Creatinine']}")
            if pd.notna(row.get('Diagnosis')):
                text_parts.append(f"diagnosis {row['Diagnosis']}")
            if pd.notna(row.get('Medications')):
                text_parts.append(f"medications {row['Medications']}")
            if pd.notna(row.get('Eligible_For_Trial')):
                text_parts.append(f"trial eligible {row['Eligible_For_Trial']}")
            if pd.notna(row.get('Clinical_Notes')):
                text_parts.append(f"notes {row['Clinical_Notes']}")
            
            texts.append(" ".join(text_parts))
        
        return texts

    def search(self, query: str, top_k: int = 10, similarity_threshold: float = 0.0) -> Tuple[List[int], List[float]]:
        """Perform semantic search with enhanced filtering"""
        if self.tfidf_matrix is None:
            raise ValueError("Index not built. Call build_index first.")
        
        query_lower = query.lower()
        filtered_df = self.df.copy()
        
        # Apply direct filtering first
        filtered_indices = self._apply_numerical_filters(query_lower)
        
        if filtered_indices is not None:
            return filtered_indices, [1.0] * len(filtered_indices)
        
        # Fallback to TF-IDF semantic search
        query_vector = self.vectorizer.transform([query])
        similarities = cosine_similarity(query_vector, self.tfidf_matrix)[0]
        
        # Get top results above threshold
        valid_indices = np.where(similarities >= similarity_threshold)[0]
        sorted_indices = valid_indices[np.argsort(similarities[valid_indices])[::-1]]
        
        top_indices = sorted_indices[:top_k].tolist()
        top_scores = similarities[top_indices].tolist()
        
        return top_indices, top_scores

    def _apply_numerical_filters(self, query: str) -> Optional[List[int]]:
        """Apply direct, simple filtering - no complex regex patterns"""
        filtered_df = self.df.copy()
        
        # CKD Stage patterns
        if 'ckd stage 3' in query:
            if 'Diagnosis' in filtered_df.columns:
                filtered_df = filtered_df[filtered_df['Diagnosis'] == 'CKD Stage 3']
        
        elif 'ckd stage 4' in query:
            if 'Diagnosis' in filtered_df.columns:
                filtered_df = filtered_df[filtered_df['Diagnosis'] == 'CKD Stage 4']
        
        elif 'ckd stage 5' in query:
            if 'Diagnosis' in filtered_df.columns:
                filtered_df = filtered_df[filtered_df['Diagnosis'] == 'CKD Stage 5']
        
        # APOL1 mutation patterns
        elif 'apol1 mutations' in query or 'patients with apol1' in query:
            if 'APOL1_Variant' in filtered_df.columns:
                # APOL1 mutations = any variant except G0/G0 (wild-type)
                apol1_filtered = filtered_df[filtered_df['APOL1_Variant'] != 'G0/G0']
                filtered_df = apol1_filtered
        
        # Wild-type APOL1 patterns
        elif 'wild-type apol1' in query or 'wt apol1' in query or 'apol1 g0/g0' in query:
            if 'APOL1_Variant' in filtered_df.columns:
                filtered_df = filtered_df[filtered_df['APOL1_Variant'] == 'G0/G0']
        
        # Specific APOL1 variants
        elif 'g1/g1' in query:
            if 'APOL1_Variant' in filtered_df.columns:
                apol1_filtered = filtered_df[filtered_df['APOL1_Variant'] == 'G1/G1']
                filtered_df = apol1_filtered
        
        elif 'g1/g2' in query:
            if 'APOL1_Variant' in filtered_df.columns:
                apol1_filtered = filtered_df[filtered_df['APOL1_Variant'] == 'G1/G2']
                filtered_df = apol1_filtered
        
        elif 'g2/g2' in query:
            if 'APOL1_Variant' in filtered_df.columns:
                filtered_df = filtered_df[filtered_df['APOL1_Variant'] == 'G2/G2']
        
        # Gene mutation patterns
        elif 'nphs1 mutations' in query:
            if 'NPHS1' in filtered_df.columns:
                filtered_df = filtered_df[filtered_df['NPHS1'] == 'Mut']
        
        elif 'nphs2 mutations' in query:
            if 'NPHS2' in filtered_df.columns:
                filtered_df = filtered_df[filtered_df['NPHS2'] == 'Mut']
        
        elif 'wt1 mutations' in query:
            if 'WT1' in filtered_df.columns:
                filtered_df = filtered_df[filtered_df['WT1'] == 'Mut']
        
        elif 'col4a3 mutations' in query:
            if 'COL4A3' in filtered_df.columns:
                filtered_df = filtered_df[filtered_df['COL4A3'] == 'Mut']
        
        elif 'umod mutations' in query:
            if 'UMOD' in filtered_df.columns:
                filtered_df = filtered_df[filtered_df['UMOD'] == 'Mut']
        
        # Clinical condition patterns
        elif 'diabetic nephropathy' in query:
            if 'Diagnosis' in filtered_df.columns:
                filtered_df = filtered_df[filtered_df['Diagnosis'] == 'Diabetic Nephropathy']
        
        elif 'nephrotic syndrome' in query:
            if 'Diagnosis' in filtered_df.columns:
                filtered_df = filtered_df[filtered_df['Diagnosis'] == 'Nephrotic Syndrome']
        
        # Laboratory value patterns with "between X and Y" support
        elif 'egfr' in query and 'eGFR' in filtered_df.columns:
            # Handle both "eGFR 60-90" and "eGFR between 60 and 90"
            egfr_pattern = r'egfr\s+(?:between\s+)?(\d+)(?:[-–]\s*|(?:\s+and\s+))(\d+)'
            match = re.search(egfr_pattern, query)
            if match:
                min_val, max_val = int(match.group(1)), int(match.group(2))
                egfr_filtered = filtered_df[(filtered_df['eGFR'] >= min_val) & (filtered_df['eGFR'] <= max_val)]
                filtered_df = egfr_filtered
            else:
                # Single value patterns
                if 'egfr below' in query or 'egfr under' in query or 'egfr <' in query:
                    egfr_match = re.search(r'(?:below|under|<)\s*(\d+)', query)
                    if egfr_match:
                        threshold = int(egfr_match.group(1))
                        filtered_df = filtered_df[filtered_df['eGFR'] < threshold]
                elif 'egfr above' in query or 'egfr over' in query or 'egfr >' in query:
                    egfr_match = re.search(r'(?:above|over|>)\s*(\d+)', query)
                    if egfr_match:
                        threshold = int(egfr_match.group(1))
                        filtered_df = filtered_df[filtered_df['eGFR'] > threshold]
        
        # Creatinine patterns with "between X and Y" support
        elif 'creatinine' in query and 'Creatinine' in filtered_df.columns:
            # Handle both "creatinine 2-3" and "creatinine between 2 and 3"
            creat_pattern = r'creatinine\s+(?:between\s+)?(\d+(?:\.\d+)?)(?:[-–]\s*|(?:\s+and\s+))(\d+(?:\.\d+)?)'
            match = re.search(creat_pattern, query)
            if match:
                min_val, max_val = float(match.group(1)), float(match.group(2))
                filtered_df = filtered_df[(filtered_df['Creatinine'] >= min_val) & (filtered_df['Creatinine'] <= max_val)]
            else:
                # Single value patterns
                if 'creatinine above' in query or 'creatinine over' in query or 'creatinine >' in query:
                    creat_match = re.search(r'(?:above|over|>)\s*(\d+(?:\.\d+)?)', query)
                    if creat_match:
                        threshold = float(creat_match.group(1))
                        filtered_df = filtered_df[filtered_df['Creatinine'] > threshold]
                elif 'creatinine below' in query or 'creatinine under' in query or 'creatinine <' in query:
                    creat_match = re.search(r'(?:below|under|<)\s*(\d+(?:\.\d+)?)', query)
                    if creat_match:
                        threshold = float(creat_match.group(1))
                        filtered_df = filtered_df[filtered_df['Creatinine'] < threshold]
        
        # Age patterns
        elif 'over' in query and 'age' in query or 'above' in query and 'age' in query:
            age_match = re.search(r'(?:over|above)\s*(?:age\s*)?(\d+)', query)
            if age_match and 'Age' in filtered_df.columns:
                threshold = int(age_match.group(1))
                filtered_df = filtered_df[filtered_df['Age'] > threshold]
        
        elif 'under' in query and 'age' in query or 'below' in query and 'age' in query:
            age_match = re.search(r'(?:under|below)\s*(?:age\s*)?(\d+)', query)
            if age_match and 'Age' in filtered_df.columns:
                threshold = int(age_match.group(1))
                filtered_df = filtered_df[filtered_df['Age'] < threshold]
        
        # Demographic patterns
        elif 'african american' in query:
            if 'Ethnicity' in filtered_df.columns:
                filtered_df = filtered_df[filtered_df['Ethnicity'] == 'African American']
        
        elif 'hispanic' in query:
            if 'Ethnicity' in filtered_df.columns:
                filtered_df = filtered_df[filtered_df['Ethnicity'] == 'Hispanic']
        
        elif 'asian' in query:
            if 'Ethnicity' in filtered_df.columns:
                filtered_df = filtered_df[filtered_df['Ethnicity'] == 'Asian']
        
        elif 'caucasian' in query or 'white' in query:
            if 'Ethnicity' in filtered_df.columns:
                filtered_df = filtered_df[filtered_df['Ethnicity'] == 'Caucasian']
        
        elif 'female' in query or 'females' in query:
            if 'Sex' in filtered_df.columns:
                filtered_df = filtered_df[filtered_df['Sex'] == 'F']
        
        elif 'male' in query or 'males' in query:
            if 'Sex' in filtered_df.columns:
                filtered_df = filtered_df[filtered_df['Sex'] == 'M']
        
        # Trial eligibility
        elif 'trial eligible' in query or 'eligible for trials' in query:
            if 'Eligible_For_Trial' in filtered_df.columns:
                filtered_df = filtered_df[filtered_df['Eligible_For_Trial'] == 'Yes']
        
        elif 'not eligible' in query or 'ineligible' in query:
            if 'Eligible_For_Trial' in filtered_df.columns:
                filtered_df = filtered_df[filtered_df['Eligible_For_Trial'] == 'No']
        
        # No mutation patterns
        elif 'no mutations' in query or 'no gene mutations' in query:
            gene_cols = ['NPHS1', 'NPHS2', 'WT1', 'COL4A3', 'UMOD']
            if all(col in filtered_df.columns for col in gene_cols):
                filtered_df = filtered_df[(filtered_df[gene_cols] == 'WT').all(axis=1)]
        
        # At least one mutation
        elif 'at least one' in query and 'mutation' in query:
            gene_cols = ['NPHS1', 'NPHS2', 'WT1', 'COL4A3', 'UMOD']
            if all(col in filtered_df.columns for col in gene_cols):
                filtered_df = filtered_df[(filtered_df[gene_cols] == 'Mut').any(axis=1)]
        
        # Complex patterns using "and"
        elif ' and ' in query:
            return self._handle_complex_query(query, filtered_df)
        
        else:
            # No specific pattern matched, return None to use semantic search
            return None
        
        if len(filtered_df) < len(self.df):
            return filtered_df.index.tolist()
        else:
            return None

    def _handle_complex_query(self, query: str, df: pd.DataFrame) -> Optional[List[int]]:
        """Handle complex queries with multiple conditions"""
        query_lower = query.lower()
        filtered_df = df.copy()
        
        # APOL1 and clinical conditions
        if 'apol1' in query_lower and 'kidney dysfunction' in query_lower:
            if 'APOL1_Variant' in filtered_df.columns:
                # APOL1 mutations (any variant except G0/G0)
                apol1_filtered = filtered_df[filtered_df['APOL1_Variant'] != 'G0/G0']
                
                # Kidney dysfunction criteria
                kidney_dysfunction = (
                    (filtered_df['eGFR'] < 45) |
                    (filtered_df['Creatinine'] > 2.0) |
                    (filtered_df['Diagnosis'].str.contains('CKD Stage [3-5]', na=False))
                )
                
                filtered_df = apol1_filtered[kidney_dysfunction]
        
        # Gene combinations
        elif 'nphs2 and wt1' in query_lower:
            if 'NPHS2' in filtered_df.columns and 'WT1' in filtered_df.columns:
                filtered_df = filtered_df[
                    (filtered_df['NPHS2'] == 'Mut') & 
                    (filtered_df['WT1'] == 'Mut')
                ]
        
        elif 'col4a3 and umod' in query_lower:
            if 'COL4A3' in filtered_df.columns and 'UMOD' in filtered_df.columns:
                filtered_df = filtered_df[
                    (filtered_df['COL4A3'] == 'Mut') & 
                    (filtered_df['UMOD'] == 'Mut')
                ]
        
        # Demographics and genetics
        elif 'african american' in query_lower and 'g1/g1' in query_lower:
            if 'Ethnicity' in filtered_df.columns and 'APOL1_Variant' in filtered_df.columns:
                filtered_df = filtered_df[
                    (filtered_df['Ethnicity'] == 'African American') & 
                    (filtered_df['APOL1_Variant'] == 'G1/G1')
                ]
        
        elif 'hispanic' in query_lower and 'ckd' in query_lower:
            if 'Ethnicity' in filtered_df.columns and 'Diagnosis' in filtered_df.columns:
                filtered_df = filtered_df[
                    (filtered_df['Ethnicity'] == 'Hispanic') & 
                    (filtered_df['Diagnosis'].str.contains('CKD', na=False))
                ]
        
        else:
            return None
        
        return filtered_df.index.tolist()

    def get_similar_patients(self, patient_idx: int, top_k: int = 5) -> Tuple[List[int], List[float]]:
        """Find similar patients to a given patient"""
        if self.tfidf_matrix is None:
            raise ValueError("Index not built. Call build_index first.")
        
        if patient_idx >= len(self.texts):
            raise ValueError(f"Patient index {patient_idx} out of range")
        
        patient_vector = self.tfidf_matrix[patient_idx]
        similarities = cosine_similarity(patient_vector, self.tfidf_matrix)[0]
        
        # Exclude the patient itself
        similarities[patient_idx] = -1
        
        # Get top similar patients
        top_indices = np.argsort(similarities)[::-1][:top_k]
        top_scores = similarities[top_indices]
        
        return top_indices.tolist(), top_scores.tolist()

    def suggest_queries(self) -> List[str]:
        """Suggest example queries for users"""
        return [
            "Find patients with APOL1 G1/G1 mutations",
            "Show me patients with diabetic nephropathy",
            "Patients with eGFR below 45",
            "African American patients with kidney disease",
            "Patients eligible for clinical trials",
            "Find patients with multiple gene mutations",
            "Show patients with high creatinine levels",
            "Young patients with genetic variants",
            "Patients with CKD Stage 3 or 4",
            "Hispanic females with APOL1 variants"
        ]

    def get_query_insights(self, query: str, results_indices: List[int], df: pd.DataFrame) -> Dict:
        """Generate insights about search results"""
        if not results_indices:
            return {"total_results": 0, "message": "No patients found matching the criteria"}
        
        results_df = df.iloc[results_indices]
        
        insights = {
            "total_results": len(results_indices),
            "demographics": {
                "age_range": f"{results_df['Age'].min():.0f}-{results_df['Age'].max():.0f} years",
                "sex_distribution": results_df['Sex'].value_counts().to_dict(),
                "ethnicity_distribution": results_df['Ethnicity'].value_counts().to_dict()
            },
            "clinical": {
                "egfr_range": f"{results_df['eGFR'].min():.1f}-{results_df['eGFR'].max():.1f}",
                "creatinine_range": f"{results_df['Creatinine'].min():.1f}-{results_df['Creatinine'].max():.1f}",
                "diagnoses": results_df['Diagnosis'].value_counts().to_dict(),
                "trial_eligible": results_df['Eligible_For_Trial'].value_counts().to_dict()
            }
        }
        
        return insights