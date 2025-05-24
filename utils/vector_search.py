import numpy as np
import pandas as pd
from typing import List, Tuple, Dict, Optional
import streamlit as st
from sentence_transformers import SentenceTransformer

class VectorSearch:
    """Handles semantic search using sentence transformers"""

    def __init__(self, model_name: str = 'all-MiniLM-L6-v2'):
        """Initialize the search system with sentence transformer model"""
        self.model = SentenceTransformer(model_name)
        self.embeddings = None
        self.texts = []
        self.df = None

    def build_index(self, df: pd.DataFrame, texts: Optional[List[str]] = None):
        """Build search index from the provided data"""
        try:
            self.df = df

            if texts is None:
                self.texts = self._create_texts_from_df(df)
            else:
                self.texts = texts

            # Generate embeddings for all texts
            self.embeddings = self.model.encode(self.texts, convert_to_tensor=True)
            st.success(f"Search index built successfully with {len(self.texts)} records")

        except Exception as e:
            st.error(f"Error building search index: {str(e)}")
            raise e

    def _create_texts_from_df(self, df: pd.DataFrame) -> List[str]:
        """Create searchable text representations from dataframe"""
        texts = []

        for _, row in df.iterrows():
            text_parts = []

            # Patient demographics
            if pd.notna(row.get('Age')):
                text_parts.append(f"Patient age {row['Age']}")
            if pd.notna(row.get('Sex')):
                text_parts.append(f"{row['Sex']} patient")
            if pd.notna(row.get('Ethnicity')):
                text_parts.append(f"{row['Ethnicity']} ethnicity")

            # Clinical information
            if pd.notna(row.get('Diagnosis')):
                text_parts.append(f"diagnosed with {row['Diagnosis']}")

            # Kidney function
            if pd.notna(row.get('eGFR')):
                egfr_val = float(row['eGFR'])
                if egfr_val < 15:
                    kidney_status = "severe kidney dysfunction"
                elif egfr_val < 30:
                    kidney_status = "moderate to severe kidney dysfunction"
                elif egfr_val < 60:
                    kidney_status = "mild to moderate kidney dysfunction"
                elif egfr_val < 90:
                    kidney_status = "mild kidney dysfunction"
                else:
                    kidney_status = "normal kidney function"
                text_parts.append(f"eGFR {egfr_val} indicating {kidney_status}")

            if pd.notna(row.get('Creatinine')):
                text_parts.append(f"creatinine level {row['Creatinine']}")

            # Medications
            if pd.notna(row.get('Medications')):
                meds = str(row['Medications']).replace(',', ' and')
                text_parts.append(f"treated with {meds}")

            # Genetic variants
            genetic_variants = []
            variant_cols = ['APOL1_Variant', 'APOL1', 'NPHS1', 'NPHS2', 'WT1', 'UMOD', 'COL4A3']
            for col in variant_cols:
                if col in row and pd.notna(row[col]):
                    if row[col] == 'Mut':
                        genetic_variants.append(f"{col} mutation")
                    elif row[col] == 'WT':
                        genetic_variants.append(f"{col} wild type")
                    else:
                        genetic_variants.append(f"{col} variant {row[col]}")

            if genetic_variants:
                text_parts.append("genetic profile: " + ", ".join(genetic_variants))

            # Clinical trial eligibility
            if pd.notna(row.get('Eligible_For_Trial')):
                eligibility = "eligible" if row['Eligible_For_Trial'] == 'Yes' else "not eligible"
                text_parts.append(f"{eligibility} for clinical trials")

            # Clinical notes
            if pd.notna(row.get('Clinical_Notes')):
                text_parts.append(f"clinical notes: {row['Clinical_Notes']}")

            texts.append(". ".join(text_parts))

        return texts

    def search(self, query: str, top_k: int = 10, similarity_threshold: float = 0.0) -> Tuple[List[int], List[float]]:
        """Perform semantic search using sentence transformers"""
        if self.embeddings is None or self.df is None:
            raise ValueError("Search index not built. Please build index first.")

        try:
            # Get filtered indices based on numerical/categorical filters
            filtered_indices = self._apply_numerical_filters(query)
            filtered_embeddings = self.embeddings if filtered_indices is None else self.embeddings[filtered_indices]

            # Encode query and compute similarities
            query_embedding = self.model.encode(query, convert_to_tensor=True)
            similarities = np.dot(filtered_embeddings, query_embedding) / (
                np.linalg.norm(filtered_embeddings, axis=1) * np.linalg.norm(query_embedding)
            )

            # Get top-k results above threshold
            if filtered_indices is not None:
                indices = filtered_indices[similarities >= similarity_threshold]
                scores = similarities[similarities >= similarity_threshold]
            else:
                indices = np.where(similarities >= similarity_threshold)[0]
                scores = similarities[similarities >= similarity_threshold]

            # Sort by similarity
            sorted_idx = np.argsort(scores)[::-1][:top_k]
            return indices[sorted_idx].tolist(), scores[sorted_idx].tolist()

        except Exception as e:
            st.error(f"Error during search: {str(e)}")
            return [], []

    def get_similar_patients(self, patient_idx: int, top_k: int = 5) -> Tuple[List[int], List[float]]:
        """Find similar patients using sentence transformer embeddings"""
        if self.embeddings is None or patient_idx >= len(self.embeddings):
            return [], []

        try:
            # Get patient embedding
            patient_embedding = self.embeddings[patient_idx]

            # Calculate similarities
            similarities = np.dot(self.embeddings, patient_embedding) / (
                np.linalg.norm(self.embeddings, axis=1) * np.linalg.norm(patient_embedding)
            )

            # Get top-k most similar patients (excluding the patient itself)
            similarities[patient_idx] = -1
            top_indices = np.argsort(similarities)[::-1][:top_k]

            return top_indices.tolist(), similarities[top_indices].tolist()

        except Exception as e:
            st.error(f"Error finding similar patients: {str(e)}")
            return [], []

    def _apply_numerical_filters(self, query: str) -> Optional[np.ndarray]:
        """Apply numerical filters based on query text"""
        if self.df is None:
            return None
        
        query_lower = query.lower()
        filtered_df = self.df.copy()
        has_filters = False
        
        # Age filters
        if 'age' in query_lower:
            # Pattern for "above age X", "over X years", "age > X", etc.
            age_patterns = [
                r'above.*?age.*?(\d+)',
                r'over.*?(\d+).*?years?',
                r'age.*?(?:>|above|over).*?(\d+)',
                r'(?:>|above|over).*?age.*?(\d+)',
                r'age.*?(\d+).*?(?:and|or).*?(?:above|over|older)',
                r'patients.*?(?:above|over).*?(\d+)'
            ]
            
            for pattern in age_patterns:
                matches = re.findall(pattern, query_lower)
                if matches:
                    min_age = int(matches[0])
                    if 'Age' in filtered_df.columns:
                        filtered_df = filtered_df[filtered_df['Age'] > min_age]
                        has_filters = True
                    break
            
            # Pattern for "below age X", "under X years", "age < X", etc.
            age_patterns_max = [
                r'below.*?age.*?(\d+)',
                r'under.*?(\d+).*?years?',
                r'age.*?(?:<|below|under).*?(\d+)',
                r'(?:<|below|under).*?age.*?(\d+)',
                r'younger.*?than.*?(\d+)'
            ]
            
            for pattern in age_patterns_max:
                matches = re.findall(pattern, query_lower)
                if matches:
                    max_age = int(matches[0])
                    if 'Age' in filtered_df.columns:
                        filtered_df = filtered_df[filtered_df['Age'] < max_age]
                        has_filters = True
                    break
        
        # eGFR filters
        if 'egfr' in query_lower:
            # Pattern for eGFR thresholds
            egfr_patterns = [
                r'egfr.*?(?:<|below|under).*?(\d+)',
                r'egfr.*?(?:>|above|over).*?(\d+)',
                r'(?:<|below|under).*?(\d+).*?egfr',
                r'(?:>|above|over).*?(\d+).*?egfr'
            ]
            
            for pattern in egfr_patterns:
                matches = re.findall(pattern, query_lower)
                if matches:
                    threshold = int(matches[0])
                    if 'eGFR' in filtered_df.columns:
                        if any(word in pattern for word in ['<', 'below', 'under']):
                            filtered_df = filtered_df[filtered_df['eGFR'] < threshold]
                        else:
                            filtered_df = filtered_df[filtered_df['eGFR'] > threshold]
                        has_filters = True
                    break
        
        # Sex filters
        if any(word in query_lower for word in ['male', 'female', 'men', 'women']):
            if 'Sex' in filtered_df.columns:
                if any(word in query_lower for word in ['male', 'men']):
                    filtered_df = filtered_df[filtered_df['Sex'] == 'M']
                    has_filters = True
                elif any(word in query_lower for word in ['female', 'women']):
                    filtered_df = filtered_df[filtered_df['Sex'] == 'F']
                    has_filters = True
        
        # Ethnicity filters
        ethnicity_terms = {
            'african american': 'African American',
            'black': 'African American',
            'caucasian': 'Caucasian',
            'white': 'Caucasian',
            'hispanic': 'Hispanic',
            'latino': 'Hispanic',
            'asian': 'Asian'
        }
        
        for term, ethnicity in ethnicity_terms.items():
            if term in query_lower and 'Ethnicity' in filtered_df.columns:
                filtered_df = filtered_df[filtered_df['Ethnicity'] == ethnicity]
                has_filters = True
                break
        
        # Trial eligibility filters
        if 'eligible' in query_lower or 'trial' in query_lower:
            if 'Eligible_For_Trial' in filtered_df.columns:
                if 'not eligible' in query_lower or 'ineligible' in query_lower:
                    filtered_df = filtered_df[filtered_df['Eligible_For_Trial'] == 'No']
                    has_filters = True
                elif 'eligible' in query_lower:
                    filtered_df = filtered_df[filtered_df['Eligible_For_Trial'] == 'Yes']
                    has_filters = True
        
        if has_filters and len(filtered_df) > 0:
            return np.array(filtered_df.index.tolist())
        
        return None

    def suggest_queries(self) -> List[str]:
        """Suggest example queries for users"""
        return [
            "Find patients with diabetic nephropathy and high creatinine",
            "Show patients with APOL1 mutations and kidney dysfunction",
            "Patients eligible for clinical trials with nephrotic syndrome",
            "Young patients with genetic mutations affecting kidney function",
            "Hispanic patients with CKD and hypertension medications",
            "Patients with eGFR below 30 and multiple medications",
            "Find similar cases to chronic kidney disease stage 4",
            "Patients with focal segmental glomerulosclerosis",
            "African American patients with high risk genetic variants",
            "Show patients with normal kidney function but genetic risk factors"
        ]

    def get_query_insights(self, query: str, results_indices: List[int], df: pd.DataFrame) -> Dict:
        """Generate insights about search results"""
        if not results_indices:
            return {"message": "No results found for this query"}

        results_df = df.iloc[results_indices]

        insights = {
            "total_results": len(results_indices),
            "demographics": {
                "age_range": {
                    "min": int(results_df['Age'].min()) if 'Age' in results_df.columns else None,
                    "max": int(results_df['Age'].max()) if 'Age' in results_df.columns else None,
                    "mean": round(results_df['Age'].mean(), 1) if 'Age' in results_df.columns else None
                },
                "sex_distribution": results_df['Sex'].value_counts().to_dict() if 'Sex' in results_df.columns else {},
                "ethnicity_distribution": results_df['Ethnicity'].value_counts().to_dict() if 'Ethnicity' in results_df.columns else {}
            },
            "clinical": {
                "diagnoses": results_df['Diagnosis'].value_counts().to_dict() if 'Diagnosis' in results_df.columns else {},
                "trial_eligibility": results_df['Eligible_For_Trial'].value_counts().to_dict() if 'Eligible_For_Trial' in results_df.columns else {},
                "egfr_stats": {
                    "mean": round(results_df['eGFR'].mean(), 1) if 'eGFR' in results_df.columns else None,
                    "median": round(results_df['eGFR'].median(), 1) if 'eGFR' in results_df.columns else None
                } if 'eGFR' in results_df.columns else {}
            }
        }

        return insights