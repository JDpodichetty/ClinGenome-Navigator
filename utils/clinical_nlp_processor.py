"""
Clinical NLP Processor for extracting structured information from clinical notes
Processes authentic clinical data to build comprehensive knowledge graphs
"""
import pandas as pd
import re
from typing import Dict, List, Tuple, Set
import json

class ClinicalNLPProcessor:
    """Advanced clinical note processing for knowledge graph construction"""
    
    def __init__(self):
        self.medical_entities = self._build_medical_ontology()
        self.lab_patterns = self._build_lab_patterns()
        self.medication_patterns = self._build_medication_patterns()
        self.condition_patterns = self._build_condition_patterns()
    
    def _build_medical_ontology(self) -> Dict[str, Dict]:
        """Build comprehensive medical entity recognition patterns"""
        return {
            'symptoms': {
                'patterns': [
                    'proteinuria', 'edema', 'swelling', 'fatigue', 'dyspnea', 'shortness of breath',
                    'chest pain', 'nausea', 'vomiting', 'dizziness', 'headache', 'weakness',
                    'joint pain', 'muscle weakness', 'weight gain', 'weight loss', 'fever',
                    'hypertension', 'elevated blood pressure', 'palpitations', 'syncope'
                ],
                'severity_indicators': ['mild', 'moderate', 'severe', 'marked', 'significant']
            },
            'conditions': {
                'patterns': [
                    'diabetic nephropathy', 'nephrotic syndrome', 'ckd', 'chronic kidney disease',
                    'end stage renal disease', 'esrd', 'glomerulonephritis', 'focal segmental glomerulosclerosis',
                    'fsgs', 'minimal change disease', 'membranous nephropathy', 'iga nephropathy',
                    'hypertension', 'diabetes mellitus', 'cardiovascular disease', 'heart failure',
                    'coronary artery disease', 'stroke', 'peripheral artery disease'
                ],
                'stages': ['stage 1', 'stage 2', 'stage 3', 'stage 4', 'stage 5']
            },
            'procedures': {
                'patterns': [
                    'hemodialysis', 'peritoneal dialysis', 'kidney transplant', 'renal biopsy',
                    'ultrasound', 'ct scan', 'mri', 'echocardiogram', 'stress test',
                    'cardiac catheterization', 'angiography', 'fistula creation'
                ],
                'frequency': ['daily', 'weekly', 'monthly', 'as needed', 'prn']
            }
        }
    
    def _build_lab_patterns(self) -> Dict[str, Dict]:
        """Build laboratory value extraction patterns"""
        return {
            'egfr': {
                'patterns': [
                    r'egfr[:\s]*(\d+(?:\.\d+)?)',
                    r'estimated gfr[:\s]*(\d+(?:\.\d+)?)',
                    r'glomerular filtration rate[:\s]*(\d+(?:\.\d+)?)'
                ],
                'units': ['ml/min/1.73m2', 'ml/min'],
                'normal_range': (90, 120)
            },
            'creatinine': {
                'patterns': [
                    r'creatinine[:\s]*(\d+(?:\.\d+)?)',
                    r'serum creatinine[:\s]*(\d+(?:\.\d+)?)',
                    r'cr[:\s]*(\d+(?:\.\d+)?)'
                ],
                'units': ['mg/dl', 'umol/l'],
                'normal_range': (0.6, 1.2)
            },
            'bun': {
                'patterns': [
                    r'bun[:\s]*(\d+(?:\.\d+)?)',
                    r'blood urea nitrogen[:\s]*(\d+(?:\.\d+)?)',
                    r'urea[:\s]*(\d+(?:\.\d+)?)'
                ],
                'units': ['mg/dl'],
                'normal_range': (7, 20)
            },
            'albumin': {
                'patterns': [
                    r'albumin[:\s]*(\d+(?:\.\d+)?)',
                    r'serum albumin[:\s]*(\d+(?:\.\d+)?)'
                ],
                'units': ['g/dl'],
                'normal_range': (3.5, 5.0)
            },
            'hemoglobin': {
                'patterns': [
                    r'hemoglobin[:\s]*(\d+(?:\.\d+)?)',
                    r'hgb[:\s]*(\d+(?:\.\d+)?)',
                    r'hb[:\s]*(\d+(?:\.\d+)?)'
                ],
                'units': ['g/dl'],
                'normal_range': (12.0, 16.0)
            }
        }
    
    def _build_medication_patterns(self) -> Dict[str, Dict]:
        """Build medication extraction patterns"""
        return {
            'ace_inhibitors': {
                'patterns': ['lisinopril', 'enalapril', 'captopril', 'ramipril', 'ace inhibitor'],
                'class': 'antihypertensive'
            },
            'arbs': {
                'patterns': ['losartan', 'valsartan', 'irbesartan', 'candesartan', 'arb'],
                'class': 'antihypertensive'
            },
            'diuretics': {
                'patterns': ['furosemide', 'hydrochlorothiazide', 'spironolactone', 'diuretic'],
                'class': 'diuretic'
            },
            'statins': {
                'patterns': ['atorvastatin', 'simvastatin', 'rosuvastatin', 'statin'],
                'class': 'lipid_lowering'
            },
            'diabetes_meds': {
                'patterns': ['metformin', 'insulin', 'glipizide', 'glyburide'],
                'class': 'antidiabetic'
            }
        }
    
    def _build_condition_patterns(self) -> Dict[str, Dict]:
        """Build condition severity and progression patterns"""
        return {
            'progression_indicators': [
                'worsening', 'improving', 'stable', 'deteriorating', 'progressing',
                'advancing', 'declining', 'recovering', 'resolved'
            ],
            'severity_modifiers': [
                'mild', 'moderate', 'severe', 'critical', 'acute', 'chronic',
                'end-stage', 'advanced', 'early', 'late'
            ],
            'temporal_indicators': [
                'recent', 'longstanding', 'new onset', 'established', 'chronic',
                'acute', 'subacute', 'persistent'
            ]
        }
    
    def extract_clinical_entities(self, clinical_notes: str, patient_id: str) -> Dict[str, List]:
        """Extract structured entities from clinical notes"""
        notes_lower = clinical_notes.lower()
        entities = {
            'symptoms': [],
            'conditions': [],
            'procedures': [],
            'medications': [],
            'lab_values': [],
            'clinical_assessments': []
        }
        
        # Extract symptoms
        entities['symptoms'] = self._extract_symptoms(notes_lower, clinical_notes)
        
        # Extract conditions with severity
        entities['conditions'] = self._extract_conditions(notes_lower, clinical_notes)
        
        # Extract procedures
        entities['procedures'] = self._extract_procedures(notes_lower, clinical_notes)
        
        # Extract medications
        entities['medications'] = self._extract_medications(notes_lower, clinical_notes)
        
        # Extract laboratory values
        entities['lab_values'] = self._extract_lab_values(notes_lower, clinical_notes)
        
        # Extract clinical assessments
        entities['clinical_assessments'] = self._extract_clinical_assessments(clinical_notes)
        
        return entities
    
    def _extract_symptoms(self, notes_lower: str, original_notes: str) -> List[Dict]:
        """Extract symptoms with context and severity"""
        symptoms = []
        
        for symptom in self.medical_entities['symptoms']['patterns']:
            if symptom in notes_lower:
                context = self._extract_context(original_notes, symptom)
                severity = self._extract_severity(context)
                
                symptoms.append({
                    'name': symptom,
                    'context': context,
                    'severity': severity,
                    'type': 'symptom'
                })
        
        return symptoms
    
    def _extract_conditions(self, notes_lower: str, original_notes: str) -> List[Dict]:
        """Extract medical conditions with staging and severity"""
        conditions = []
        
        for condition in self.medical_entities['conditions']['patterns']:
            if condition in notes_lower:
                context = self._extract_context(original_notes, condition)
                stage = self._extract_stage(context)
                severity = self._extract_severity(context)
                progression = self._extract_progression(context)
                
                conditions.append({
                    'name': condition,
                    'context': context,
                    'stage': stage,
                    'severity': severity,
                    'progression': progression,
                    'type': 'condition'
                })
        
        return conditions
    
    def _extract_procedures(self, notes_lower: str, original_notes: str) -> List[Dict]:
        """Extract medical procedures with frequency"""
        procedures = []
        
        for procedure in self.medical_entities['procedures']['patterns']:
            if procedure in notes_lower:
                context = self._extract_context(original_notes, procedure)
                frequency = self._extract_frequency(context)
                
                procedures.append({
                    'name': procedure,
                    'context': context,
                    'frequency': frequency,
                    'type': 'procedure'
                })
        
        return procedures
    
    def _extract_medications(self, notes_lower: str, original_notes: str) -> List[Dict]:
        """Extract medications with dosage and class"""
        medications = []
        
        for med_class, med_info in self.medication_patterns.items():
            for medication in med_info['patterns']:
                if medication in notes_lower:
                    context = self._extract_context(original_notes, medication)
                    dosage = self._extract_dosage(context)
                    
                    medications.append({
                        'name': medication,
                        'class': med_info['class'],
                        'context': context,
                        'dosage': dosage,
                        'type': 'medication'
                    })
        
        return medications
    
    def _extract_lab_values(self, notes_lower: str, original_notes: str) -> List[Dict]:
        """Extract laboratory values with interpretation"""
        lab_values = []
        
        for lab_name, lab_info in self.lab_patterns.items():
            for pattern in lab_info['patterns']:
                matches = re.findall(pattern, notes_lower)
                for match in matches:
                    try:
                        value = float(match)
                        interpretation = self._interpret_lab_value(lab_name, value, lab_info)
                        context = self._extract_context(original_notes, f"{lab_name} {value}")
                        
                        lab_values.append({
                            'name': lab_name,
                            'value': value,
                            'interpretation': interpretation,
                            'context': context,
                            'normal_range': lab_info.get('normal_range'),
                            'type': 'lab_value'
                        })
                    except ValueError:
                        continue
        
        return lab_values
    
    def _extract_clinical_assessments(self, notes: str) -> List[Dict]:
        """Extract clinical assessments and impressions"""
        assessments = []
        
        # Look for assessment sections
        assessment_markers = ['assessment:', 'impression:', 'plan:', 'clinical impression:']
        
        for marker in assessment_markers:
            if marker in notes.lower():
                # Extract text after the marker
                start_idx = notes.lower().find(marker)
                # Find next section or end of notes
                end_markers = ['plan:', 'recommendations:', '\n\n']
                end_idx = len(notes)
                
                for end_marker in end_markers:
                    temp_idx = notes.lower().find(end_marker, start_idx + len(marker))
                    if temp_idx != -1 and temp_idx < end_idx:
                        end_idx = temp_idx
                
                assessment_text = notes[start_idx:end_idx].strip()
                
                if len(assessment_text) > len(marker):
                    assessments.append({
                        'type': 'clinical_assessment',
                        'category': marker.replace(':', ''),
                        'text': assessment_text,
                        'extracted_conditions': self._extract_assessment_conditions(assessment_text)
                    })
        
        return assessments
    
    def _extract_context(self, text: str, entity: str, window: int = 100) -> str:
        """Extract context around an entity mention"""
        entity_lower = entity.lower()
        text_lower = text.lower()
        
        idx = text_lower.find(entity_lower)
        if idx == -1:
            return ""
        
        start = max(0, idx - window)
        end = min(len(text), idx + len(entity) + window)
        
        return text[start:end].strip()
    
    def _extract_severity(self, context: str) -> str:
        """Extract severity indicators from context"""
        context_lower = context.lower()
        
        for severity in self.medical_entities['symptoms']['severity_indicators']:
            if severity in context_lower:
                return severity
        
        for severity in self.condition_patterns['severity_modifiers']:
            if severity in context_lower:
                return severity
        
        return 'unspecified'
    
    def _extract_stage(self, context: str) -> str:
        """Extract disease staging from context"""
        context_lower = context.lower()
        
        for stage in self.medical_entities['conditions']['stages']:
            if stage in context_lower:
                return stage
        
        # Look for CKD stages specifically
        ckd_stage_pattern = r'stage (\d+[ab]?)'
        match = re.search(ckd_stage_pattern, context_lower)
        if match:
            return f"stage {match.group(1)}"
        
        return 'unspecified'
    
    def _extract_progression(self, context: str) -> str:
        """Extract disease progression indicators"""
        context_lower = context.lower()
        
        for indicator in self.condition_patterns['progression_indicators']:
            if indicator in context_lower:
                return indicator
        
        return 'stable'
    
    def _extract_frequency(self, context: str) -> str:
        """Extract procedure or treatment frequency"""
        context_lower = context.lower()
        
        for freq in self.medical_entities['procedures']['frequency']:
            if freq in context_lower:
                return freq
        
        # Look for specific frequency patterns
        freq_patterns = [
            r'(\d+)\s*times?\s*per?\s*(day|week|month)',
            r'every\s*(\d+)\s*(days?|weeks?|months?)',
            r'(\d+)x\s*(daily|weekly|monthly)'
        ]
        
        for pattern in freq_patterns:
            match = re.search(pattern, context_lower)
            if match:
                return match.group(0)
        
        return 'unspecified'
    
    def _extract_dosage(self, context: str) -> str:
        """Extract medication dosage information"""
        dosage_patterns = [
            r'(\d+(?:\.\d+)?)\s*(mg|g|ml|units?)',
            r'(\d+(?:\.\d+)?)\s*(mg|g|ml)\s*(daily|twice daily|bid|tid|qid)',
            r'(\d+(?:\.\d+)?)\s*mg/day'
        ]
        
        for pattern in dosage_patterns:
            match = re.search(pattern, context.lower())
            if match:
                return match.group(0)
        
        return 'unspecified'
    
    def _interpret_lab_value(self, lab_name: str, value: float, lab_info: Dict) -> str:
        """Interpret laboratory values based on normal ranges"""
        normal_range = lab_info.get('normal_range')
        if not normal_range:
            return 'normal_range_unknown'
        
        min_normal, max_normal = normal_range
        
        if value < min_normal:
            return 'below_normal'
        elif value > max_normal:
            return 'above_normal'
        else:
            return 'normal'
    
    def _extract_assessment_conditions(self, assessment_text: str) -> List[str]:
        """Extract specific conditions mentioned in clinical assessments"""
        conditions = []
        
        for condition in self.medical_entities['conditions']['patterns']:
            if condition in assessment_text.lower():
                conditions.append(condition)
        
        return conditions
    
    def build_entity_relationships(self, entities: Dict[str, List], patient_data: Dict) -> List[Dict]:
        """Build relationships between extracted entities"""
        relationships = []
        
        # Medication-condition relationships
        for medication in entities['medications']:
            for condition in entities['conditions']:
                if self._is_medication_for_condition(medication['name'], condition['name']):
                    relationships.append({
                        'source': medication['name'],
                        'target': condition['name'],
                        'relationship': 'treats',
                        'type': 'medication_condition'
                    })
        
        # Lab value-condition relationships
        for lab in entities['lab_values']:
            for condition in entities['conditions']:
                if self._is_lab_relevant_to_condition(lab['name'], condition['name']):
                    relationships.append({
                        'source': lab['name'],
                        'target': condition['name'],
                        'relationship': 'monitors',
                        'type': 'lab_condition'
                    })
        
        # Symptom-condition relationships
        for symptom in entities['symptoms']:
            for condition in entities['conditions']:
                if self._is_symptom_of_condition(symptom['name'], condition['name']):
                    relationships.append({
                        'source': symptom['name'],
                        'target': condition['name'],
                        'relationship': 'symptom_of',
                        'type': 'symptom_condition'
                    })
        
        return relationships
    
    def _is_medication_for_condition(self, medication: str, condition: str) -> bool:
        """Determine if medication treats condition"""
        # ACE inhibitors for kidney disease and hypertension
        ace_inhibitors = ['lisinopril', 'enalapril', 'captopril', 'ramipril']
        kidney_conditions = ['ckd', 'chronic kidney disease', 'diabetic nephropathy']
        
        if medication in ace_inhibitors and any(kc in condition for kc in kidney_conditions):
            return True
        
        if medication in ace_inhibitors and 'hypertension' in condition:
            return True
        
        # Diuretics for heart failure and edema
        diuretics = ['furosemide', 'hydrochlorothiazide']
        if medication in diuretics and ('heart failure' in condition or 'edema' in condition):
            return True
        
        return False
    
    def _is_lab_relevant_to_condition(self, lab: str, condition: str) -> bool:
        """Determine if lab value monitors condition"""
        kidney_labs = ['egfr', 'creatinine', 'bun']
        kidney_conditions = ['ckd', 'chronic kidney disease', 'diabetic nephropathy', 'renal']
        
        if lab in kidney_labs and any(kc in condition for kc in kidney_conditions):
            return True
        
        return False
    
    def _is_symptom_of_condition(self, symptom: str, condition: str) -> bool:
        """Determine if symptom is related to condition"""
        kidney_symptoms = ['edema', 'swelling', 'fatigue', 'proteinuria']
        kidney_conditions = ['ckd', 'chronic kidney disease', 'diabetic nephropathy']
        
        if symptom in kidney_symptoms and any(kc in condition for kc in kidney_conditions):
            return True
        
        return False