"""
Query Normalizer - Systematic approach to handle all query variations
"""
import re
import pandas as pd
from typing import Dict, List, Tuple, Optional

class QueryNormalizer:
    """Centralized system to normalize and standardize query patterns"""
    
    def __init__(self):
        self.range_patterns = self._build_range_patterns()
        self.comparison_patterns = self._build_comparison_patterns()
        self.medication_patterns = self._build_medication_patterns()
        self.clinical_patterns = self._build_clinical_patterns()
    
    def _build_range_patterns(self) -> List[str]:
        """Build comprehensive range pattern variations"""
        return [
            # "between X and Y" formats
            r'(\w+)\s+between\s+(\d+(?:\.\d+)?)\s+and\s+(\d+(?:\.\d+)?)',
            r'between\s+(\d+(?:\.\d+)?)\s+and\s+(\d+(?:\.\d+)?)\s+(\w+)',
            
            # "X-Y" or "X to Y" formats  
            r'(\w+)\s+(\d+(?:\.\d+)?)[-–]\s*(\d+(?:\.\d+)?)',
            r'(\w+)\s+(\d+(?:\.\d+)?)\s+to\s+(\d+(?:\.\d+)?)',
            r'(\w+)\s+from\s+(\d+(?:\.\d+)?)\s+to\s+(\d+(?:\.\d+)?)',
            
            # "in the range of X-Y"
            r'(\w+)\s+in\s+the\s+range\s+of\s+(\d+(?:\.\d+)?)[-–]\s*(\d+(?:\.\d+)?)',
            r'(\w+)\s+ranging\s+from\s+(\d+(?:\.\d+)?)\s+to\s+(\d+(?:\.\d+)?)',
        ]
    
    def _build_comparison_patterns(self) -> Dict[str, List[str]]:
        """Build comparison operator patterns"""
        return {
            'above': [
                r'(\w+)\s+(?:above|over|greater than|more than|exceeding)\s+(\d+(?:\.\d+)?)',
                r'(\w+)\s*>\s*(\d+(?:\.\d+)?)',
                r'(?:above|over|greater than)\s+(\d+(?:\.\d+)?)\s+(\w+)',
            ],
            'below': [
                r'(\w+)\s+(?:below|under|less than|lower than)\s+(\d+(?:\.\d+)?)',
                r'(\w+)\s*<\s*(\d+(?:\.\d+)?)',
                r'(?:below|under|less than)\s+(\d+(?:\.\d+)?)\s+(\w+)',
            ],
            'equal': [
                r'(\w+)\s+(?:equals?|is)\s+(\d+(?:\.\d+)?)',
                r'(\w+)\s*=\s*(\d+(?:\.\d+)?)',
            ]
        }
    
    def _build_medication_patterns(self) -> Dict[str, List[str]]:
        """Build medication name variations"""
        return {
            'ace_inhibitors': [
                'ace inhibitors?', 'ace-inhibitors?', 'angiotensin converting enzyme inhibitors?',
                'lisinopril', 'enalapril', 'captopril', 'ramipril'
            ],
            'calcium_channel_blockers': [
                'calcium channel blockers?', 'ccb', 'amlodipine', 'nifedipine', 
                'diltiazem', 'verapamil'
            ],
            'diuretics': [
                'diuretics?', 'water pills?', 'furosemide', 'hydrochlorothiazide',
                'spironolactone', 'lasix'
            ],
            'statins': [
                'statins?', 'atorvastatin', 'simvastatin', 'lovastatin', 'pravastatin'
            ],
            'erythropoietin': [
                'erythropoietin', 'epo', 'epoetin', 'darbepoetin'
            ]
        }
    
    def _build_clinical_patterns(self) -> Dict[str, List[str]]:
        """Build clinical condition variations"""
        return {
            'diabetic_nephropathy': [
                'diabetic nephropathy', 'diabetic kidney disease', 'dkd',
                'diabetes with kidney', 'diabetic renal'
            ],
            'nephrotic_syndrome': [
                'nephrotic syndrome', 'nephrosis', 'proteinuria syndrome'
            ],
            'glomerulonephritis': [
                'glomerulonephritis', 'gn', 'glomerular nephritis'
            ],
            'normal_function': [
                'normal kidney function', 'normal renal function', 'healthy kidneys',
                'good kidney function', 'preserved kidney function'
            ],
            'kidney_dysfunction': [
                'kidney dysfunction', 'renal dysfunction', 'impaired kidney',
                'reduced kidney function', 'declining kidney function'
            ]
        }
    
    def normalize_query(self, query: str) -> Dict:
        """
        Normalize query into structured components
        Returns: {
            'ranges': [{'field': 'eGFR', 'min': 60, 'max': 90}],
            'comparisons': [{'field': 'creatinine', 'operator': 'above', 'value': 3.0}],
            'medications': ['ace_inhibitors', 'diuretics'],
            'conditions': ['diabetic_nephropathy'],
            'original_query': query
        }
        """
        query_lower = query.lower()
        normalized = {
            'ranges': [],
            'comparisons': [],
            'medications': [],
            'conditions': [],
            'original_query': query
        }
        
        # Extract ranges
        for pattern in self.range_patterns:
            matches = re.finditer(pattern, query_lower)
            for match in matches:
                groups = match.groups()
                if len(groups) == 3:
                    field = self._normalize_field_name(groups[0])
                    min_val = float(groups[1])
                    max_val = float(groups[2])
                    normalized['ranges'].append({
                        'field': field,
                        'min': min_val,
                        'max': max_val
                    })
        
        # Extract comparisons
        for operator, patterns in self.comparison_patterns.items():
            for pattern in patterns:
                matches = re.finditer(pattern, query_lower)
                for match in matches:
                    groups = match.groups()
                    if len(groups) == 2:
                        field = self._normalize_field_name(groups[0])
                        value = float(groups[1])
                        normalized['comparisons'].append({
                            'field': field,
                            'operator': operator,
                            'value': value
                        })
        
        # Extract medications
        for med_category, variations in self.medication_patterns.items():
            for variation in variations:
                if re.search(variation, query_lower):
                    if med_category not in normalized['medications']:
                        normalized['medications'].append(med_category)
        
        # Extract conditions
        for condition, variations in self.clinical_patterns.items():
            for variation in variations:
                if re.search(variation, query_lower):
                    if condition not in normalized['conditions']:
                        normalized['conditions'].append(condition)
        
        return normalized
    
    def _normalize_field_name(self, field: str) -> str:
        """Normalize field names to match dataset columns"""
        field_mappings = {
            'egfr': 'eGFR',
            'creatinine': 'Creatinine',
            'age': 'Age',
            'bun': 'BUN'
        }
        return field_mappings.get(field.lower(), field)
    
    def build_filter_from_normalized(self, normalized: Dict, df) -> 'pd.DataFrame':
        """Build pandas filter from normalized query components"""
        filtered_df = df.copy()
        
        # Apply range filters
        for range_filter in normalized['ranges']:
            field = range_filter['field']
            if field in filtered_df.columns:
                min_val, max_val = range_filter['min'], range_filter['max']
                filtered_df = filtered_df[
                    (filtered_df[field] >= min_val) & 
                    (filtered_df[field] <= max_val)
                ]
        
        # Apply comparison filters
        for comp_filter in normalized['comparisons']:
            field = comp_filter['field']
            if field in filtered_df.columns:
                operator = comp_filter['operator']
                value = comp_filter['value']
                
                if operator == 'above':
                    filtered_df = filtered_df[filtered_df[field] > value]
                elif operator == 'below':
                    filtered_df = filtered_df[filtered_df[field] < value]
                elif operator == 'equal':
                    filtered_df = filtered_df[filtered_df[field] == value]
        
        # Apply medication filters (would need medication columns in dataset)
        # This is where we'd add medication logic when available
        
        # Apply condition filters
        for condition in normalized['conditions']:
            if condition == 'diabetic_nephropathy' and 'Diagnosis' in filtered_df.columns:
                filtered_df = filtered_df[filtered_df['Diagnosis'] == 'Diabetic Nephropathy']
            elif condition == 'nephrotic_syndrome' and 'Diagnosis' in filtered_df.columns:
                filtered_df = filtered_df[filtered_df['Diagnosis'] == 'Nephrotic Syndrome']
            # Add more condition mappings as needed
        
        return filtered_df