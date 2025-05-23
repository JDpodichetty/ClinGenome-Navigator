import re
import pandas as pd
from typing import Dict, List, Tuple, Optional
import streamlit as st

class QueryProcessor:
    """Processes and enhances natural language queries for better search results"""
    
    def __init__(self):
        self.clinical_terms = self._load_clinical_terms()
        self.genetic_terms = self._load_genetic_terms()
        self.medication_terms = self._load_medication_terms()
    
    def _load_clinical_terms(self) -> Dict[str, List[str]]:
        """Load clinical terminology mappings"""
        return {
            'kidney_disease': [
                'chronic kidney disease', 'ckd', 'kidney dysfunction', 
                'renal disease', 'nephropathy', 'kidney failure',
                'end stage renal disease', 'esrd'
            ],
            'diabetes': [
                'diabetic nephropathy', 'diabetes', 'diabetic kidney disease',
                'type 1 diabetes', 'type 2 diabetes'
            ],
            'nephrotic_syndrome': [
                'nephrotic syndrome', 'proteinuria', 'protein in urine',
                'nephrosis', 'minimal change disease'
            ],
            'fsgs': [
                'focal segmental glomerulosclerosis', 'fsgs', 
                'glomerulosclerosis', 'segmental sclerosis'
            ],
            'hypertension': [
                'high blood pressure', 'hypertension', 'htn',
                'elevated blood pressure'
            ]
        }
    
    def _load_genetic_terms(self) -> Dict[str, List[str]]:
        """Load genetic terminology mappings"""
        return {
            'apol1': [
                'apol1', 'apolipoprotein l1', 'g1 variant', 'g2 variant',
                'g0 variant', 'apol1 mutation', 'apol1 risk variant'
            ],
            'nphs1': [
                'nphs1', 'nephrin', 'nphs1 mutation', 'nephrin gene'
            ],
            'nphs2': [
                'nphs2', 'podocin', 'nphs2 mutation', 'podocin gene'
            ],
            'wt1': [
                'wt1', 'wilms tumor 1', 'wt1 mutation', 'wilms tumor gene'
            ],
            'umod': [
                'umod', 'uromodulin', 'umod mutation', 'uromodulin gene'
            ],
            'col4a3': [
                'col4a3', 'collagen type iv alpha 3', 'col4a3 mutation'
            ]
        }
    
    def _load_medication_terms(self) -> Dict[str, List[str]]:
        """Load medication terminology mappings"""
        return {
            'ace_inhibitors': [
                'ace inhibitors', 'ace inhibitor', 'acei', 'lisinopril',
                'enalapril', 'captopril', 'ramipril'
            ],
            'arbs': [
                'arbs', 'arb', 'angiotensin receptor blockers',
                'losartan', 'valsartan', 'irbesartan', 'olmesartan'
            ],
            'diuretics': [
                'diuretics', 'diuretic', 'water pills', 'furosemide',
                'hydrochlorothiazide', 'spironolactone'
            ],
            'calcium_channel_blockers': [
                'calcium channel blockers', 'ccb', 'amlodipine',
                'nifedipine', 'diltiazem', 'verapamil'
            ],
            'statins': [
                'statins', 'statin', 'atorvastatin', 'simvastatin',
                'rosuvastatin', 'pravastatin'
            ],
            'insulin': [
                'insulin', 'insulin therapy', 'diabetes medication'
            ],
            'erythropoietin': [
                'erythropoietin', 'epo', 'epoetin', 'anemia treatment'
            ]
        }
    
    def enhance_query(self, query: str) -> str:
        """Enhance the query with clinical terminology"""
        enhanced_query = query.lower()
        
        # Expand clinical terms
        for category, terms in self.clinical_terms.items():
            for term in terms:
                if term in enhanced_query:
                    # Add related terms to improve search
                    related_terms = [t for t in terms if t != term]
                    if related_terms:
                        enhanced_query += f" {' '.join(related_terms[:2])}"
        
        # Expand genetic terms
        for category, terms in self.genetic_terms.items():
            for term in terms:
                if term in enhanced_query:
                    enhanced_query += f" {category} genetic variant mutation"
        
        # Expand medication terms
        for category, terms in self.medication_terms.items():
            for term in terms:
                if term in enhanced_query:
                    enhanced_query += f" {category.replace('_', ' ')}"
        
        return enhanced_query
    
    def extract_filters(self, query: str) -> Dict:
        """Extract potential filters from the query"""
        filters = {}
        query_lower = query.lower()
        
        # Age filters
        age_pattern = r'age\s*(?:>|>=|above|over)\s*(\d+)|(\d+)\s*(?:years?\s*old|yo)'
        age_matches = re.findall(age_pattern, query_lower)
        if age_matches:
            for match in age_matches:
                age = int(match[0]) if match[0] else int(match[1])
                filters['min_age'] = age
        
        age_pattern_max = r'age\s*(?:<|<=|below|under)\s*(\d+)|under\s*(\d+)\s*years?'
        age_matches_max = re.findall(age_pattern_max, query_lower)
        if age_matches_max:
            for match in age_matches_max:
                age = int(match[0]) if match[0] else int(match[1])
                filters['max_age'] = age
        
        # Gender filters
        if any(term in query_lower for term in ['male', 'men', 'man']):
            filters['sex'] = ['M']
        elif any(term in query_lower for term in ['female', 'women', 'woman']):
            filters['sex'] = ['F']
        
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
            if term in query_lower:
                filters['ethnicity'] = [ethnicity]
                break
        
        # eGFR filters
        egfr_pattern = r'egfr\s*(?:<|<=|below|under)\s*(\d+)'
        egfr_matches = re.findall(egfr_pattern, query_lower)
        if egfr_matches:
            filters['max_egfr'] = int(egfr_matches[0])
        
        egfr_pattern_min = r'egfr\s*(?:>|>=|above|over)\s*(\d+)'
        egfr_matches_min = re.findall(egfr_pattern_min, query_lower)
        if egfr_matches_min:
            filters['min_egfr'] = int(egfr_matches_min[0])
        
        # Trial eligibility
        if any(term in query_lower for term in ['eligible', 'trial eligible', 'clinical trial']):
            filters['trial_eligible'] = True
        elif any(term in query_lower for term in ['not eligible', 'ineligible']):
            filters['trial_eligible'] = False
        
        return filters
    
    def suggest_query_improvements(self, query: str) -> List[str]:
        """Suggest improvements to the user's query"""
        suggestions = []
        query_lower = query.lower()
        
        # Suggest adding specificity
        if len(query.split()) < 3:
            suggestions.append("Try adding more specific terms like age range, ethnicity, or specific conditions")
        
        # Suggest genetic terms
        if 'genetic' in query_lower or 'mutation' in query_lower:
            suggestions.append("Specify genetic variants like APOL1, NPHS1, NPHS2, WT1, UMOD, or COL4A3")
        
        # Suggest clinical terms
        if 'kidney' in query_lower:
            suggestions.append("Be more specific: chronic kidney disease, diabetic nephropathy, or nephrotic syndrome")
        
        # Suggest adding demographics
        if not any(demo in query_lower for demo in ['male', 'female', 'age', 'african', 'caucasian', 'hispanic', 'asian']):
            suggestions.append("Consider adding demographic filters like age range, sex, or ethnicity")
        
        # Suggest adding clinical metrics
        if not any(metric in query_lower for metric in ['egfr', 'creatinine', 'stage']):
            suggestions.append("Include clinical metrics like eGFR levels or CKD stages")
        
        return suggestions[:3]  # Return top 3 suggestions
    
    def parse_complex_query(self, query: str) -> Dict:
        """Parse complex queries with multiple conditions"""
        parsed = {
            'main_terms': [],
            'conditions': [],
            'demographics': [],
            'genetics': [],
            'medications': [],
            'metrics': []
        }
        
        query_lower = query.lower()
        words = query_lower.split()
        
        # Extract main medical terms
        medical_terms = [
            'diabetes', 'hypertension', 'nephropathy', 'kidney', 'renal',
            'proteinuria', 'fsgs', 'nephrotic', 'ckd'
        ]
        
        for term in medical_terms:
            if term in query_lower:
                parsed['main_terms'].append(term)
        
        # Extract demographic information
        demo_patterns = {
            'age': r'age\s*(\d+)',
            'male': r'\b(?:male|men|man)\b',
            'female': r'\b(?:female|women|woman)\b'
        }
        
        for demo, pattern in demo_patterns.items():
            if re.search(pattern, query_lower):
                parsed['demographics'].append(demo)
        
        # Extract genetic terms
        for gene_category, terms in self.genetic_terms.items():
            for term in terms:
                if term in query_lower:
                    parsed['genetics'].append(gene_category)
                    break
        
        # Extract medication terms
        for med_category, terms in self.medication_terms.items():
            for term in terms:
                if term in query_lower:
                    parsed['medications'].append(med_category)
                    break
        
        return parsed
    
    def generate_search_summary(self, query: str, num_results: int) -> str:
        """Generate a summary of the search performed"""
        enhanced_query = self.enhance_query(query)
        filters = self.extract_filters(query)
        
        summary_parts = [f"Searched for: '{query}'"]
        
        if filters:
            filter_descriptions = []
            if 'min_age' in filters:
                filter_descriptions.append(f"age ≥ {filters['min_age']}")
            if 'max_age' in filters:
                filter_descriptions.append(f"age ≤ {filters['max_age']}")
            if 'sex' in filters:
                filter_descriptions.append(f"sex: {', '.join(filters['sex'])}")
            if 'ethnicity' in filters:
                filter_descriptions.append(f"ethnicity: {', '.join(filters['ethnicity'])}")
            if 'min_egfr' in filters:
                filter_descriptions.append(f"eGFR ≥ {filters['min_egfr']}")
            if 'max_egfr' in filters:
                filter_descriptions.append(f"eGFR ≤ {filters['max_egfr']}")
            if 'trial_eligible' in filters:
                eligibility = "eligible" if filters['trial_eligible'] else "not eligible"
                filter_descriptions.append(f"trial {eligibility}")
            
            if filter_descriptions:
                summary_parts.append(f"Filters applied: {', '.join(filter_descriptions)}")
        
        summary_parts.append(f"Found {num_results} matching patients")
        
        return " | ".join(summary_parts)
