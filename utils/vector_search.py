import numpy as np
import pandas as pd
from typing import List, Tuple, Dict, Optional
import streamlit as st
import re
from difflib import SequenceMatcher

class VectorSearch:
    """Handles semantic search using text-based similarity matching"""
    
    def __init__(self, model_name: str = 'text-search'):
        """Initialize the search system"""
        self.model_name = model_name
        self.texts = []
        self.df = None
        
    def build_index(self, df: pd.DataFrame, texts: Optional[List[str]] = None):
        """Build search index from the provided data"""
        try:
            # Store dataframe
            self.df = df
            
            # Use provided texts or create from dataframe
            if texts is None:
                self.texts = self._create_texts_from_df(df)
            else:
                self.texts = texts
            
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
        """Perform semantic search using text similarity"""
        if not self.texts:
            raise ValueError("Search index not built. Please build index first.")
        
        try:
            # Normalize query
            query_lower = query.lower()
            
            # Calculate similarity scores
            scores = []
            for i, text in enumerate(self.texts):
                score = self._calculate_similarity(query_lower, text.lower())
                scores.append((i, score))
            
            # Sort by score
            scores.sort(key=lambda x: x[1], reverse=True)
            
            # Filter by threshold and limit results
            valid_results = []
            valid_scores = []
            
            for idx, score in scores[:top_k]:
                if score >= similarity_threshold:
                    valid_results.append(idx)
                    valid_scores.append(score)
            
            return valid_results, valid_scores
            
        except Exception as e:
            st.error(f"Error during search: {str(e)}")
            return [], []
    
    def _calculate_similarity(self, query: str, text: str) -> float:
        """Calculate similarity between query and text"""
        # Simple keyword matching with weights
        score = 0.0
        query_words = set(query.split())
        text_words = set(text.split())
        
        # Exact keyword matches
        matches = query_words.intersection(text_words)
        score += len(matches) * 2
        
        # Partial matches using sequence matching
        for q_word in query_words:
            for t_word in text_words:
                if len(q_word) > 3 and len(t_word) > 3:
                    similarity = SequenceMatcher(None, q_word, t_word).ratio()
                    if similarity > 0.7:
                        score += similarity
        
        # Normalize by query length
        if len(query_words) > 0:
            score = score / len(query_words)
        
        return min(score, 1.0)  # Cap at 1.0
    
    def get_similar_patients(self, patient_idx: int, top_k: int = 5) -> Tuple[List[int], List[float]]:
        """Find similar patients to a given patient"""
        if not self.texts or patient_idx >= len(self.texts):
            return [], []
        
        try:
            # Use the patient's text as query
            patient_text = self.texts[patient_idx]
            
            # Calculate similarities
            scores = []
            for i, text in enumerate(self.texts):
                if i != patient_idx:  # Exclude the patient itself
                    score = self._calculate_similarity(patient_text.lower(), text.lower())
                    scores.append((i, score))
            
            # Sort and return top matches
            scores.sort(key=lambda x: x[1], reverse=True)
            
            indices = [idx for idx, score in scores[:top_k]]
            similarities = [score for idx, score in scores[:top_k]]
            
            return indices, similarities
            
        except Exception as e:
            st.error(f"Error finding similar patients: {str(e)}")
            return [], []
    
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