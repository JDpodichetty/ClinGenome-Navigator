import pandas as pd
import numpy as np
import streamlit as st
from typing import Dict, List, Tuple, Optional
import io
from utils.enhanced_knowledge_graph import EnhancedKnowledgeGraph

class DataProcessor:
    """Handles data loading, processing, and preparation for the clinical genomics platform"""
    
    def __init__(self):
        self.df = None
        self.metadata = {}
        self.processed_texts = []
        self.enhanced_kg = None
    
    def load_data(self, file_path: str) -> pd.DataFrame:
        """Load data from CSV file"""
        try:
            self.df = pd.read_csv(file_path)
            self._process_data()
            return self.df
        except Exception as e:
            raise Exception(f"Error loading data: {str(e)}")
    
    def load_uploaded_data(self, uploaded_file) -> pd.DataFrame:
        """Load data from uploaded file"""
        try:
            # Read the uploaded file
            bytes_data = uploaded_file.read()
            self.df = pd.read_csv(io.StringIO(bytes_data.decode('utf-8')))
            self._process_data()
            return self.df
        except Exception as e:
            raise Exception(f"Error processing uploaded file: {str(e)}")
    
    def _process_data(self):
        """Process and clean the loaded data"""
        if self.df is None:
            return
        
        # Basic data cleaning
        self.df = self.df.dropna(subset=['PatientID'])
        
        # Generate metadata
        self._generate_metadata()
        
        # Create searchable text representations
        self._create_searchable_texts()
        
        # Initialize enhanced knowledge graph
        self._initialize_enhanced_kg()
    
    def _generate_metadata(self):
        """Generate metadata about the dataset"""
        self.metadata = {
            'total_patients': len(self.df),
            'columns': list(self.df.columns),
            'genetic_variants': self._get_genetic_variants(),
            'diagnoses': self._get_unique_values('Diagnosis'),
            'medications': self._get_unique_medications(),
            'demographics': self._get_demographics_summary(),
            'clinical_metrics': self._get_clinical_metrics()
        }
    
    def _get_genetic_variants(self) -> Dict[str, List]:
        """Extract information about genetic variants"""
        variant_columns = ['APOL1', 'NPHS1', 'NPHS2', 'WT1', 'UMOD', 'COL4A3']
        variants = {}
        
        for col in variant_columns:
            if col in self.df.columns:
                variants[col] = self.df[col].value_counts().to_dict()
        
        # APOL1 variant analysis
        if 'APOL1_Variant' in self.df.columns:
            variants['APOL1_Variant'] = self.df['APOL1_Variant'].value_counts().to_dict()
        
        return variants
    
    def _get_unique_values(self, column: str) -> List[str]:
        """Get unique values for a column"""
        if column in self.df.columns:
            return sorted(self.df[column].dropna().unique().tolist())
        return []
    
    def _get_unique_medications(self) -> List[str]:
        """Extract unique medications from the medications column"""
        if 'Medications' not in self.df.columns:
            return []
        
        all_meds = set()
        for med_string in self.df['Medications'].dropna():
            # Split by comma and clean
            meds = [med.strip() for med in str(med_string).split(',')]
            all_meds.update(meds)
        
        return sorted(list(all_meds))
    
    def _get_demographics_summary(self) -> Dict:
        """Generate demographics summary"""
        demographics = {}
        
        if 'Age' in self.df.columns:
            demographics['age_stats'] = {
                'mean': self.df['Age'].mean(),
                'median': self.df['Age'].median(),
                'min': self.df['Age'].min(),
                'max': self.df['Age'].max()
            }
        
        if 'Sex' in self.df.columns:
            demographics['sex_distribution'] = self.df['Sex'].value_counts().to_dict()
        
        if 'Ethnicity' in self.df.columns:
            demographics['ethnicity_distribution'] = self.df['Ethnicity'].value_counts().to_dict()
        
        return demographics
    
    def _get_clinical_metrics(self) -> Dict:
        """Generate clinical metrics summary"""
        metrics = {}
        
        clinical_columns = ['eGFR', 'Creatinine']
        for col in clinical_columns:
            if col in self.df.columns:
                metrics[col] = {
                    'mean': self.df[col].mean(),
                    'median': self.df[col].median(),
                    'std': self.df[col].std(),
                    'min': self.df[col].min(),
                    'max': self.df[col].max()
                }
        
        return metrics
    
    def _create_searchable_texts(self):
        """Create searchable text representations for each patient"""
        self.processed_texts = []
        
        for _, row in self.df.iterrows():
            # Combine relevant fields into searchable text
            text_parts = []
            
            # Demographics
            if pd.notna(row.get('Age')):
                text_parts.append(f"Age: {row['Age']} years old")
            if pd.notna(row.get('Sex')):
                text_parts.append(f"Sex: {row['Sex']}")
            if pd.notna(row.get('Ethnicity')):
                text_parts.append(f"Ethnicity: {row['Ethnicity']}")
            
            # Clinical information
            if pd.notna(row.get('Diagnosis')):
                text_parts.append(f"Diagnosis: {row['Diagnosis']}")
            if pd.notna(row.get('eGFR')):
                text_parts.append(f"eGFR: {row['eGFR']}")
            if pd.notna(row.get('Creatinine')):
                text_parts.append(f"Creatinine: {row['Creatinine']}")
            
            # Medications
            if pd.notna(row.get('Medications')):
                text_parts.append(f"Medications: {row['Medications']}")
            
            # Genetic variants
            genetic_info = []
            genetic_columns = ['APOL1_Variant', 'APOL1', 'NPHS1', 'NPHS2', 'WT1', 'UMOD', 'COL4A3']
            for col in genetic_columns:
                if col in row and pd.notna(row[col]):
                    genetic_info.append(f"{col}: {row[col]}")
            if genetic_info:
                text_parts.append("Genetic variants: " + ", ".join(genetic_info))
            
            # Clinical notes
            if pd.notna(row.get('Clinical_Notes')):
                text_parts.append(f"Clinical notes: {row['Clinical_Notes']}")
            
            # Trial eligibility
            if pd.notna(row.get('Eligible_For_Trial')):
                text_parts.append(f"Trial eligible: {row['Eligible_For_Trial']}")
            
            self.processed_texts.append(" | ".join(text_parts))
    
    def _initialize_enhanced_kg(self):
        """Initialize the enhanced knowledge graph with the loaded data"""
        if self.df is not None:
            self.enhanced_kg = EnhancedKnowledgeGraph()
            self.enhanced_kg.build_knowledge_graph(self.df)
    
    def get_data(self) -> pd.DataFrame:
        """Return the processed dataframe"""
        return self.df
    
    def get_metadata(self) -> Dict:
        """Return dataset metadata"""
        return self.metadata
    
    def get_searchable_texts(self) -> List[str]:
        """Return searchable text representations"""
        return self.processed_texts
    
    def filter_data(self, filters: Dict) -> pd.DataFrame:
        """Apply filters to the data"""
        if self.df is None:
            return pd.DataFrame()
        
        filtered_df = self.df.copy()
        
        # Age range filter
        if 'age_range' in filters and filters['age_range']:
            min_age, max_age = filters['age_range']
            filtered_df = filtered_df[
                (filtered_df['Age'] >= min_age) & 
                (filtered_df['Age'] <= max_age)
            ]
        
        # Sex filter
        if 'sex' in filters and filters['sex']:
            filtered_df = filtered_df[filtered_df['Sex'].isin(filters['sex'])]
        
        # Ethnicity filter
        if 'ethnicity' in filters and filters['ethnicity']:
            filtered_df = filtered_df[filtered_df['Ethnicity'].isin(filters['ethnicity'])]
        
        # Diagnosis filter
        if 'diagnosis' in filters and filters['diagnosis']:
            filtered_df = filtered_df[filtered_df['Diagnosis'].isin(filters['diagnosis'])]
        
        # eGFR range filter
        if 'egfr_range' in filters and filters['egfr_range']:
            min_egfr, max_egfr = filters['egfr_range']
            filtered_df = filtered_df[
                (filtered_df['eGFR'] >= min_egfr) & 
                (filtered_df['eGFR'] <= max_egfr)
            ]
        
        # Trial eligibility filter
        if 'trial_eligible' in filters and filters['trial_eligible'] is not None:
            filtered_df = filtered_df[filtered_df['Eligible_For_Trial'] == filters['trial_eligible']]
        
        return filtered_df
    
    def export_results(self, results_df: pd.DataFrame, format: str = 'csv') -> bytes:
        """Export search results in specified format"""
        if format == 'csv':
            return results_df.to_csv(index=False).encode('utf-8')
        elif format == 'json':
            return results_df.to_json(orient='records', indent=2).encode('utf-8')
        else:
            raise ValueError(f"Unsupported export format: {format}")
