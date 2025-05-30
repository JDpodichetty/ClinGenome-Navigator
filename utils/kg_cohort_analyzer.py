"""
Knowledge Graph-Based Cohort Analysis
Uses relationship traversal for sophisticated patient cohort identification
"""
import pandas as pd
from typing import Dict, List, Set, Any
from utils.enhanced_knowledge_graph import EnhancedKnowledgeGraph

class KGCohortAnalyzer:
    """Advanced cohort analysis using knowledge graph relationship traversal"""
    
    def __init__(self, enhanced_kg: EnhancedKnowledgeGraph):
        self.kg = enhanced_kg
        self.cursor = enhanced_kg.connection.cursor()
    
    def find_patients_by_relationship_path(self, start_entity_type: str, start_entity_name: str, 
                                         relationship_type: str, target_entity_type: str) -> List[str]:
        """Find patients by traversing specific relationship paths in the knowledge graph"""
        
        # Find all patients with the starting entity
        self.cursor.execute('''
            SELECT DISTINCT patient_id, id 
            FROM clinical_entities 
            WHERE entity_type = ? AND entity_name = ?
        ''', (start_entity_type, start_entity_name))
        
        start_entities = self.cursor.fetchall()
        patient_ids = set()
        
        for patient_id, entity_id in start_entities:
            # Find related entities through relationships
            self.cursor.execute('''
                SELECT ce.patient_id 
                FROM entity_relationships er
                JOIN clinical_entities ce ON er.target_entity_id = ce.id
                WHERE er.source_entity_id = ? 
                AND er.relationship_type = ?
                AND ce.entity_type = ?
            ''', (entity_id, relationship_type, target_entity_type))
            
            related_patients = [row[0] for row in self.cursor.fetchall()]
            if related_patients:
                patient_ids.add(patient_id)
        
        return list(patient_ids)
    
    def find_high_risk_genetic_cohort(self) -> Dict[str, Any]:
        """Find patients with high-risk genetic profiles using relationship analysis"""
        
        # Find patients with APOL1 high-risk variants that influence kidney function
        apol1_patients = self.find_patients_by_relationship_path(
            'genetic_variant', 'APOL1', 'influences_kidney_function', 'lab_result'
        )
        
        # Find patients with gene mutations that cause kidney dysfunction
        kidney_genes = ['NPHS1', 'NPHS2', 'WT1', 'COL4A3', 'UMOD']
        mutation_patients = set()
        
        for gene in kidney_genes:
            gene_patients = self.find_patients_by_relationship_path(
                'gene_mutation', gene, 'causes_kidney_dysfunction', 'lab_result'
            )
            mutation_patients.update(gene_patients)
        
        # Combine cohorts
        combined_cohort = set(apol1_patients) | mutation_patients
        
        return {
            'cohort_type': 'high_risk_genetic',
            'patient_ids': list(combined_cohort),
            'apol1_patients': len(apol1_patients),
            'mutation_patients': len(mutation_patients),
            'total_patients': len(combined_cohort)
        }
    
    def find_kidney_dysfunction_progression_cohort(self, egfr_threshold: float = 45) -> Dict[str, Any]:
        """Find patients showing kidney dysfunction progression using lab correlations"""
        
        # Find patients with eGFR-creatinine inverse correlations (kidney dysfunction pattern)
        self.cursor.execute('''
            SELECT DISTINCT ce1.patient_id
            FROM clinical_entities ce1
            JOIN entity_relationships er ON ce1.id = er.source_entity_id
            JOIN clinical_entities ce2 ON er.target_entity_id = ce2.id
            WHERE ce1.entity_name = 'eGFR' 
            AND ce2.entity_name = 'Creatinine'
            AND er.relationship_type = 'inversely_correlates_with'
            AND CAST(ce1.entity_value AS REAL) < ?
        ''', (egfr_threshold,))
        
        dysfunction_patients = [row[0] for row in self.cursor.fetchall()]
        
        # Find patients with symptoms indicating kidney dysfunction
        symptom_patients = self.find_patients_by_relationship_path(
            'symptoms', 'proteinuria', 'indicates_dysfunction_of', 'lab_result'
        )
        
        # Find patients with edema symptoms
        edema_patients = self.find_patients_by_relationship_path(
            'symptoms', 'edema', 'indicates_dysfunction_of', 'lab_result'
        )
        
        # Combine all dysfunction indicators
        all_dysfunction = set(dysfunction_patients) | set(symptom_patients) | set(edema_patients)
        
        return {
            'cohort_type': 'kidney_dysfunction_progression',
            'patient_ids': list(all_dysfunction),
            'lab_dysfunction': len(dysfunction_patients),
            'symptom_based': len(symptom_patients),
            'edema_based': len(edema_patients),
            'total_patients': len(all_dysfunction)
        }
    
    def find_genetic_lab_correlation_cohort(self) -> Dict[str, Any]:
        """Find patients where genetic variants correlate with specific lab abnormalities"""
        
        # Find APOL1 variants that correlate with creatinine levels
        self.cursor.execute('''
            SELECT DISTINCT ce1.patient_id
            FROM clinical_entities ce1
            JOIN entity_relationships er ON ce1.id = er.source_entity_id
            JOIN clinical_entities ce2 ON er.target_entity_id = ce2.id
            WHERE ce1.entity_name = 'APOL1'
            AND ce2.entity_name = 'Creatinine' 
            AND er.relationship_type = 'correlates_with'
            AND ce1.entity_value IN ('G1/G1', 'G1/G2', 'G2/G2')
        ''', ())
        
        genetic_lab_patients = [row[0] for row in self.cursor.fetchall()]
        
        # Find gene mutations that elevate creatinine
        self.cursor.execute('''
            SELECT DISTINCT ce1.patient_id
            FROM clinical_entities ce1
            JOIN entity_relationships er ON ce1.id = er.source_entity_id
            JOIN clinical_entities ce2 ON er.target_entity_id = ce2.id
            WHERE ce1.entity_type = 'gene_mutation'
            AND ce2.entity_name = 'Creatinine'
            AND er.relationship_type = 'elevates'
        ''', ())
        
        mutation_creat_patients = [row[0] for row in self.cursor.fetchall()]
        
        combined_patients = set(genetic_lab_patients) | set(mutation_creat_patients)
        
        return {
            'cohort_type': 'genetic_lab_correlation',
            'patient_ids': list(combined_patients),
            'apol1_creatinine': len(genetic_lab_patients),
            'mutation_creatinine': len(mutation_creat_patients),
            'total_patients': len(combined_patients)
        }
    
    def find_condition_predisposition_cohort(self) -> Dict[str, Any]:
        """Find patients with genetic predisposition to specific conditions"""
        
        # Find APOL1 variants that predispose to kidney conditions
        self.cursor.execute('''
            SELECT DISTINCT ce2.patient_id
            FROM clinical_entities ce1
            JOIN entity_relationships er ON ce1.id = er.source_entity_id
            JOIN clinical_entities ce2 ON er.target_entity_id = ce2.id
            WHERE ce1.entity_name = 'APOL1'
            AND er.relationship_type = 'predisposes_to'
            AND ce2.entity_type = 'conditions'
        ''', ())
        
        predisposition_patients = [row[0] for row in self.cursor.fetchall()]
        
        return {
            'cohort_type': 'genetic_predisposition',
            'patient_ids': predisposition_patients,
            'total_patients': len(predisposition_patients)
        }
    
    def find_complex_phenotype_cohort(self, criteria: Dict[str, Any]) -> Dict[str, Any]:
        """Find patients matching complex phenotypic patterns using multiple relationship types"""
        
        candidate_patients = set()
        criteria_results = {}
        
        # Genetic criteria via relationships
        if criteria.get('genetic_risk'):
            genetic_cohort = self.find_high_risk_genetic_cohort()
            candidate_patients.update(genetic_cohort['patient_ids'])
            criteria_results['genetic'] = genetic_cohort['total_patients']
        
        # Kidney function criteria via relationships
        if criteria.get('kidney_dysfunction'):
            kidney_cohort = self.find_kidney_dysfunction_progression_cohort(
                criteria.get('egfr_threshold', 45)
            )
            if not candidate_patients:
                candidate_patients.update(kidney_cohort['patient_ids'])
            else:
                candidate_patients &= set(kidney_cohort['patient_ids'])
            criteria_results['kidney_dysfunction'] = kidney_cohort['total_patients']
        
        # Genetic-lab correlation criteria
        if criteria.get('genetic_lab_correlation'):
            correlation_cohort = self.find_genetic_lab_correlation_cohort()
            if not candidate_patients:
                candidate_patients.update(correlation_cohort['patient_ids'])
            else:
                candidate_patients &= set(correlation_cohort['patient_ids'])
            criteria_results['genetic_lab'] = correlation_cohort['total_patients']
        
        return {
            'cohort_type': 'complex_phenotype',
            'patient_ids': list(candidate_patients),
            'criteria_breakdown': criteria_results,
            'total_patients': len(candidate_patients)
        }
    
    def analyze_relationship_strength_cohort(self, min_strength: float = 0.8) -> Dict[str, Any]:
        """Find patients involved in high-strength relationships"""
        
        self.cursor.execute('''
            SELECT DISTINCT ce.patient_id
            FROM clinical_entities ce
            JOIN entity_relationships er ON (ce.id = er.source_entity_id OR ce.id = er.target_entity_id)
            WHERE er.strength >= ?
        ''', (min_strength,))
        
        high_strength_patients = [row[0] for row in self.cursor.fetchall()]
        
        return {
            'cohort_type': 'high_strength_relationships',
            'patient_ids': high_strength_patients,
            'total_patients': len(high_strength_patients),
            'min_strength_threshold': min_strength
        }
    
    def get_cohort_relationship_summary(self, patient_ids: List[str]) -> Dict[str, Any]:
        """Analyze the relationship patterns within a cohort"""
        
        if not patient_ids:
            return {}
        
        placeholders = ','.join(['?' for _ in patient_ids])
        
        # Count relationship types in the cohort
        self.cursor.execute(f'''
            SELECT er.relationship_type, COUNT(*) as count
            FROM entity_relationships er
            JOIN clinical_entities ce1 ON er.source_entity_id = ce1.id
            JOIN clinical_entities ce2 ON er.target_entity_id = ce2.id
            WHERE ce1.patient_id IN ({placeholders}) 
            OR ce2.patient_id IN ({placeholders})
            GROUP BY er.relationship_type
            ORDER BY count DESC
        ''', patient_ids + patient_ids)
        
        relationship_counts = dict(self.cursor.fetchall())
        
        # Count entity types in the cohort
        self.cursor.execute(f'''
            SELECT entity_type, COUNT(*) as count
            FROM clinical_entities
            WHERE patient_id IN ({placeholders})
            GROUP BY entity_type
            ORDER BY count DESC
        ''', patient_ids)
        
        entity_counts = dict(self.cursor.fetchall())
        
        return {
            'relationship_types': relationship_counts,
            'entity_types': entity_counts,
            'total_relationships': sum(relationship_counts.values()),
            'total_entities': sum(entity_counts.values())
        }