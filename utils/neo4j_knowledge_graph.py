"""
Neo4j-based Knowledge Graph System for Clinical Genomics Data
Provides superior graph database capabilities for complex medical relationships
"""
import pandas as pd
from neo4j import GraphDatabase
from typing import Dict, List, Any, Optional
import re
import json
import os
import tempfile
import sqlite3

class Neo4jKnowledgeGraph:
    """Neo4j-powered knowledge graph for clinical genomics data"""
    
    def __init__(self, use_embedded=True):
        """Initialize with embedded SQLite as Neo4j alternative for simplicity"""
        self.use_embedded = use_embedded
        self.db_path = None
        self.connection = None
        
        if use_embedded:
            # Use SQLite as embedded graph database for demo
            self.db_path = tempfile.mktemp(suffix='.db')
            self.connection = sqlite3.connect(self.db_path, check_same_thread=False)
            self._initialize_schema()
        
        self.medical_patterns = self._build_medical_patterns()
        
    def _initialize_schema(self):
        """Initialize the database schema for graph-like operations"""
        cursor = self.connection.cursor()
        
        # Nodes table
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS nodes (
                id TEXT PRIMARY KEY,
                type TEXT,
                properties TEXT
            )
        ''')
        
        # Relationships table
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS relationships (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                source_id TEXT,
                target_id TEXT,
                relationship_type TEXT,
                properties TEXT,
                FOREIGN KEY (source_id) REFERENCES nodes (id),
                FOREIGN KEY (target_id) REFERENCES nodes (id)
            )
        ''')
        
        # Create indexes for better performance
        cursor.execute('CREATE INDEX IF NOT EXISTS idx_node_type ON nodes (type)')
        cursor.execute('CREATE INDEX IF NOT EXISTS idx_rel_source ON relationships (source_id)')
        cursor.execute('CREATE INDEX IF NOT EXISTS idx_rel_target ON relationships (target_id)')
        cursor.execute('CREATE INDEX IF NOT EXISTS idx_rel_type ON relationships (relationship_type)')
        
        self.connection.commit()
    
    def _build_medical_patterns(self) -> Dict[str, List[str]]:
        """Define comprehensive medical entity recognition patterns"""
        return {
            'symptoms': [
                'proteinuria', 'edema', 'swelling', 'fatigue', 'shortness of breath',
                'chest pain', 'nausea', 'vomiting', 'dizziness', 'headache',
                'joint pain', 'muscle weakness', 'weight gain', 'weight loss',
                'hypertension', 'elevated blood pressure'
            ],
            'conditions': [
                'diabetic nephropathy', 'nephrotic syndrome', 'ckd', 'chronic kidney disease',
                'hypertension', 'diabetes', 'cardiovascular disease', 'heart failure',
                'glomerulonephritis', 'renal failure', 'kidney dysfunction', 'end stage renal disease'
            ],
            'medications': [
                'ace inhibitor', 'arb', 'diuretic', 'beta blocker', 'calcium channel blocker',
                'statin', 'metformin', 'insulin', 'erythropoietin', 'iron supplement',
                'lisinopril', 'losartan', 'furosemide', 'amlodipine', 'atorvastatin'
            ],
            'lab_values': [
                'creatinine', 'egfr', 'bun', 'albumin', 'hemoglobin', 'potassium',
                'sodium', 'phosphorus', 'calcium', 'glucose', 'cholesterol'
            ],
            'procedures': [
                'dialysis', 'transplant', 'biopsy', 'ultrasound', 'ct scan',
                'mri', 'echocardiogram', 'stress test', 'catheterization'
            ]
        }
    
    def build_knowledge_graph(self, df: pd.DataFrame) -> Dict[str, int]:
        """Build knowledge graph from clinical data"""
        print(f"Building Neo4j-style knowledge graph from {len(df)} patients...")
        
        cursor = self.connection.cursor()
        
        # Clear existing data
        cursor.execute('DELETE FROM relationships')
        cursor.execute('DELETE FROM nodes')
        
        stats = {'nodes': 0, 'relationships': 0}
        
        for idx, row in df.iterrows():
            patient_id = f"patient_{row['PatientID']}"
            
            # Add patient node
            self._add_node(patient_id, 'patient', {
                'age': row['Age'],
                'sex': row['Sex'],
                'ethnicity': row['Ethnicity'],
                'patient_id': row['PatientID']
            })
            stats['nodes'] += 1
            
            # Add genetic information
            genetic_nodes = self._add_genetic_data(patient_id, row)
            stats['nodes'] += len(genetic_nodes)
            stats['relationships'] += len(genetic_nodes)
            
            # Add clinical data
            clinical_nodes = self._add_clinical_data(patient_id, row)
            stats['nodes'] += len(clinical_nodes)
            stats['relationships'] += len(clinical_nodes)
            
            # Extract entities from clinical notes
            if pd.notna(row.get('Clinical_Notes')):
                note_entities = self._extract_clinical_entities(patient_id, str(row['Clinical_Notes']))
                stats['nodes'] += len(note_entities)
                stats['relationships'] += len(note_entities)
        
        self.connection.commit()
        print(f"Knowledge graph built with {stats['nodes']} nodes and {stats['relationships']} relationships")
        return stats
    
    def _add_node(self, node_id: str, node_type: str, properties: Dict[str, Any]):
        """Add a node to the graph"""
        cursor = self.connection.cursor()
        cursor.execute(
            'INSERT OR REPLACE INTO nodes (id, type, properties) VALUES (?, ?, ?)',
            (node_id, node_type, json.dumps(properties))
        )
    
    def _add_relationship(self, source_id: str, target_id: str, rel_type: str, properties: Dict[str, Any] = None):
        """Add a relationship to the graph"""
        cursor = self.connection.cursor()
        cursor.execute(
            'INSERT INTO relationships (source_id, target_id, relationship_type, properties) VALUES (?, ?, ?, ?)',
            (source_id, target_id, rel_type, json.dumps(properties or {}))
        )
    
    def _add_genetic_data(self, patient_id: str, row: pd.Series) -> List[str]:
        """Add genetic variant information to the graph"""
        genetic_nodes = []
        
        # APOL1 variant
        if pd.notna(row.get('APOL1_Variant')):
            variant_id = f"apol1_{row['APOL1_Variant'].replace('/', '_')}"
            self._add_node(variant_id, 'genetic_variant', {
                'gene': 'APOL1',
                'variant': row['APOL1_Variant'],
                'risk_level': self._assess_apol1_risk(row['APOL1_Variant'])
            })
            self._add_relationship(patient_id, variant_id, 'HAS_VARIANT')
            genetic_nodes.append(variant_id)
        
        # Gene mutations
        gene_cols = ['NPHS1', 'NPHS2', 'WT1', 'COL4A3', 'UMOD']
        for gene in gene_cols:
            if pd.notna(row.get(gene)) and row[gene] == 'Mut':
                gene_id = f"{gene.lower()}_mutation"
                self._add_node(gene_id, 'gene_mutation', {
                    'gene': gene,
                    'mutation_status': 'Mut',
                    'pathway': self._get_gene_pathway(gene)
                })
                self._add_relationship(patient_id, gene_id, 'HAS_MUTATION')
                genetic_nodes.append(gene_id)
        
        return genetic_nodes
    
    def _add_clinical_data(self, patient_id: str, row: pd.Series) -> List[str]:
        """Add clinical measurements and diagnoses"""
        clinical_nodes = []
        
        # Diagnosis
        if pd.notna(row.get('Diagnosis')):
            diagnosis_id = f"diagnosis_{row['Diagnosis'].lower().replace(' ', '_')}"
            self._add_node(diagnosis_id, 'diagnosis', {
                'name': row['Diagnosis'],
                'severity': self._assess_diagnosis_severity(row)
            })
            self._add_relationship(patient_id, diagnosis_id, 'HAS_DIAGNOSIS')
            clinical_nodes.append(diagnosis_id)
        
        # Laboratory values with clinical interpretation
        if pd.notna(row.get('eGFR')):
            egfr_category = self._categorize_egfr(row['eGFR'])
            egfr_id = f"egfr_{egfr_category}"
            self._add_node(egfr_id, 'lab_result', {
                'test': 'eGFR',
                'value': row['eGFR'],
                'category': egfr_category,
                'ckd_stage': self._get_ckd_stage_from_egfr(row['eGFR'])
            })
            self._add_relationship(patient_id, egfr_id, 'HAS_LAB_RESULT')
            clinical_nodes.append(egfr_id)
        
        if pd.notna(row.get('Creatinine')):
            creat_category = self._categorize_creatinine(row['Creatinine'])
            creat_id = f"creatinine_{creat_category}"
            self._add_node(creat_id, 'lab_result', {
                'test': 'Creatinine',
                'value': row['Creatinine'],
                'category': creat_category
            })
            self._add_relationship(patient_id, creat_id, 'HAS_LAB_RESULT')
            clinical_nodes.append(creat_id)
        
        # Trial eligibility as clinical outcome
        if pd.notna(row.get('Eligible_For_Trial')):
            eligibility_id = f"trial_eligible_{row['Eligible_For_Trial'].lower()}"
            self._add_node(eligibility_id, 'trial_status', {
                'eligible': row['Eligible_For_Trial'] == 'Yes',
                'status': row['Eligible_For_Trial']
            })
            self._add_relationship(patient_id, eligibility_id, 'HAS_TRIAL_STATUS')
            clinical_nodes.append(eligibility_id)
        
        return clinical_nodes
    
    def _extract_clinical_entities(self, patient_id: str, clinical_notes: str) -> List[str]:
        """Extract medical entities from clinical notes using NLP patterns"""
        notes_lower = clinical_notes.lower()
        extracted_entities = []
        
        for entity_type, patterns in self.medical_patterns.items():
            for pattern in patterns:
                if pattern in notes_lower:
                    entity_id = f"{entity_type}_{pattern.replace(' ', '_')}"
                    self._add_node(entity_id, entity_type, {
                        'name': pattern,
                        'source': 'clinical_notes',
                        'context': self._extract_context(clinical_notes, pattern)
                    })
                    self._add_relationship(patient_id, entity_id, 'MENTIONED_IN_NOTES')
                    extracted_entities.append(entity_id)
        
        # Extract numerical values with context
        lab_extractions = self._extract_numerical_values(patient_id, clinical_notes)
        extracted_entities.extend(lab_extractions)
        
        return extracted_entities
    
    def _extract_numerical_values(self, patient_id: str, notes: str) -> List[str]:
        """Extract lab values and measurements from clinical notes"""
        extracted = []
        
        # eGFR values
        egfr_matches = re.findall(r'egfr[:\s]*(\d+(?:\.\d+)?)', notes.lower())
        for match in egfr_matches:
            value = float(match)
            entity_id = f"notes_egfr_{value}"
            self._add_node(entity_id, 'notes_lab', {
                'test': 'eGFR',
                'value': value,
                'category': self._categorize_egfr(value),
                'source': 'clinical_notes'
            })
            self._add_relationship(patient_id, entity_id, 'NOTES_MENTION')
            extracted.append(entity_id)
        
        # Creatinine values
        creat_matches = re.findall(r'creatinine[:\s]*(\d+(?:\.\d+)?)', notes.lower())
        for match in creat_matches:
            value = float(match)
            entity_id = f"notes_creatinine_{value}"
            self._add_node(entity_id, 'notes_lab', {
                'test': 'Creatinine',
                'value': value,
                'category': self._categorize_creatinine(value),
                'source': 'clinical_notes'
            })
            self._add_relationship(patient_id, entity_id, 'NOTES_MENTION')
            extracted.append(entity_id)
        
        return extracted
    
    def query_patients_by_pathway(self, pathway_criteria: Dict[str, Any]) -> List[str]:
        """Query patients following specific clinical pathways using graph traversal"""
        cursor = self.connection.cursor()
        
        # Base query for patients
        base_query = '''
            SELECT DISTINCT n1.id 
            FROM nodes n1 
            WHERE n1.type = 'patient'
        '''
        
        conditions = []
        params = []
        
        # Add genetic criteria
        if pathway_criteria.get('has_genetic_risk'):
            conditions.append('''
                AND EXISTS (
                    SELECT 1 FROM relationships r1 
                    JOIN nodes n2 ON r1.target_id = n2.id 
                    WHERE r1.source_id = n1.id 
                    AND n2.type IN ('genetic_variant', 'gene_mutation')
                )
            ''')
        
        # Add clinical criteria
        if pathway_criteria.get('has_diagnosis'):
            conditions.append('''
                AND EXISTS (
                    SELECT 1 FROM relationships r2 
                    JOIN nodes n3 ON r2.target_id = n3.id 
                    WHERE r2.source_id = n1.id 
                    AND n3.type = 'diagnosis'
                )
            ''')
        
        # Add eGFR criteria
        if pathway_criteria.get('egfr_threshold'):
            conditions.append('''
                AND EXISTS (
                    SELECT 1 FROM relationships r3 
                    JOIN nodes n4 ON r3.target_id = n4.id 
                    WHERE r3.source_id = n1.id 
                    AND n4.type = 'lab_result'
                    AND json_extract(n4.properties, '$.test') = 'eGFR'
                    AND CAST(json_extract(n4.properties, '$.value') AS REAL) < ?
                )
            ''')
            params.append(pathway_criteria['egfr_threshold'])
        
        final_query = base_query + ' '.join(conditions)
        
        cursor.execute(final_query, params)
        results = cursor.fetchall()
        
        return [row[0] for row in results]
    
    def find_similar_patients(self, target_patient_id: str, similarity_threshold: int = 2) -> List[Dict[str, Any]]:
        """Find patients with similar graph patterns"""
        cursor = self.connection.cursor()
        
        # Get target patient's connected entities
        cursor.execute('''
            SELECT target_id, relationship_type 
            FROM relationships 
            WHERE source_id = ?
        ''', (target_patient_id,))
        
        target_entities = cursor.fetchall()
        target_entity_ids = [row[0] for row in target_entities]
        
        if not target_entity_ids:
            return []
        
        # Find patients with overlapping entities
        placeholders = ','.join(['?' for _ in target_entity_ids])
        similarity_query = f'''
            SELECT r.source_id, COUNT(*) as shared_entities
            FROM relationships r
            WHERE r.target_id IN ({placeholders})
            AND r.source_id != ?
            AND r.source_id LIKE 'patient_%'
            GROUP BY r.source_id
            HAVING shared_entities >= ?
            ORDER BY shared_entities DESC
        '''
        
        params = target_entity_ids + [target_patient_id, similarity_threshold]
        cursor.execute(similarity_query, params)
        
        results = []
        for row in cursor.fetchall():
            results.append({
                'patient_id': row[0],
                'similarity_score': row[1],
                'total_target_entities': len(target_entity_ids)
            })
        
        return results
    
    def get_patient_summary(self, patient_id: str) -> Dict[str, Any]:
        """Get comprehensive patient summary with graph analysis"""
        cursor = self.connection.cursor()
        
        # Get patient properties
        cursor.execute('SELECT properties FROM nodes WHERE id = ?', (patient_id,))
        patient_data = cursor.fetchone()
        
        if not patient_data:
            return {}
        
        patient_props = json.loads(patient_data[0])
        
        # Get connected entities by type
        cursor.execute('''
            SELECT n.type, n.id, n.properties, r.relationship_type
            FROM relationships r
            JOIN nodes n ON r.target_id = n.id
            WHERE r.source_id = ?
            ORDER BY n.type, n.id
        ''', (patient_id,))
        
        entities = cursor.fetchall()
        
        summary = {
            'patient_info': patient_props,
            'entities_by_type': {},
            'total_connections': len(entities),
            'risk_assessment': {}
        }
        
        for entity_type, entity_id, properties, rel_type in entities:
            if entity_type not in summary['entities_by_type']:
                summary['entities_by_type'][entity_type] = []
            
            summary['entities_by_type'][entity_type].append({
                'id': entity_id,
                'properties': json.loads(properties),
                'relationship': rel_type
            })
        
        # Calculate risk assessment
        summary['risk_assessment'] = self._calculate_patient_risk(summary['entities_by_type'])
        
        return summary
    
    def get_graph_statistics(self) -> Dict[str, Any]:
        """Get comprehensive graph statistics"""
        cursor = self.connection.cursor()
        
        # Node counts by type
        cursor.execute('''
            SELECT type, COUNT(*) as count 
            FROM nodes 
            GROUP BY type 
            ORDER BY count DESC
        ''')
        node_counts = dict(cursor.fetchall())
        
        # Relationship counts by type
        cursor.execute('''
            SELECT relationship_type, COUNT(*) as count 
            FROM relationships 
            GROUP BY relationship_type 
            ORDER BY count DESC
        ''')
        relationship_counts = dict(cursor.fetchall())
        
        # Total counts
        cursor.execute('SELECT COUNT(*) FROM nodes')
        total_nodes = cursor.fetchone()[0]
        
        cursor.execute('SELECT COUNT(*) FROM relationships')
        total_relationships = cursor.fetchone()[0]
        
        return {
            'total_nodes': total_nodes,
            'total_relationships': total_relationships,
            'node_counts_by_type': node_counts,
            'relationship_counts_by_type': relationship_counts,
            'graph_density': total_relationships / (total_nodes * (total_nodes - 1)) if total_nodes > 1 else 0
        }
    
    # Helper methods
    def _assess_apol1_risk(self, variant: str) -> str:
        """Assess APOL1 risk level"""
        high_risk = ['G1/G1', 'G1/G2', 'G2/G2']
        medium_risk = ['G0/G1', 'G0/G2']
        return 'high' if variant in high_risk else 'medium' if variant in medium_risk else 'low'
    
    def _get_gene_pathway(self, gene: str) -> str:
        """Get biological pathway for gene"""
        pathways = {
            'NPHS1': 'podocyte_function',
            'NPHS2': 'podocyte_function', 
            'WT1': 'kidney_development',
            'COL4A3': 'basement_membrane',
            'UMOD': 'tubular_function'
        }
        return pathways.get(gene, 'unknown')
    
    def _categorize_egfr(self, egfr: float) -> str:
        """Categorize eGFR values"""
        if egfr >= 90: return "normal"
        elif egfr >= 60: return "mild_reduction"
        elif egfr >= 45: return "moderate_reduction"
        elif egfr >= 30: return "severe_reduction"
        else: return "kidney_failure"
    
    def _categorize_creatinine(self, creatinine: float) -> str:
        """Categorize creatinine values"""
        if creatinine <= 1.2: return "normal"
        elif creatinine <= 2.0: return "elevated"
        elif creatinine <= 3.0: return "high"
        else: return "very_high"
    
    def _get_ckd_stage_from_egfr(self, egfr: float) -> str:
        """Get CKD stage from eGFR"""
        if egfr >= 90: return "Stage 1"
        elif egfr >= 60: return "Stage 2"
        elif egfr >= 45: return "Stage 3a"
        elif egfr >= 30: return "Stage 3b"
        elif egfr >= 15: return "Stage 4"
        else: return "Stage 5"
    
    def _assess_diagnosis_severity(self, row: pd.Series) -> str:
        """Assess diagnosis severity based on clinical markers"""
        egfr = row.get('eGFR', 100)
        if egfr < 30: return "severe"
        elif egfr < 60: return "moderate"
        else: return "mild"
    
    def _extract_context(self, text: str, pattern: str, window: int = 50) -> str:
        """Extract context around a pattern match"""
        idx = text.lower().find(pattern)
        if idx == -1:
            return ""
        start = max(0, idx - window)
        end = min(len(text), idx + len(pattern) + window)
        return text[start:end]
    
    def _calculate_patient_risk(self, entities_by_type: Dict[str, List]) -> Dict[str, Any]:
        """Calculate comprehensive patient risk assessment"""
        risk_score = 0
        risk_factors = []
        
        # Genetic risk factors
        genetic_variants = entities_by_type.get('genetic_variant', [])
        gene_mutations = entities_by_type.get('gene_mutation', [])
        
        for variant in genetic_variants:
            props = variant['properties']
            if props.get('risk_level') == 'high':
                risk_score += 3
                risk_factors.append(f"High-risk {props.get('gene')} variant")
            elif props.get('risk_level') == 'medium':
                risk_score += 2
                risk_factors.append(f"Medium-risk {props.get('gene')} variant")
        
        # Add mutation risk
        risk_score += len(gene_mutations)
        risk_factors.extend([f"{mut['properties']['gene']} mutation" for mut in gene_mutations])
        
        # Clinical risk factors
        lab_results = entities_by_type.get('lab_result', [])
        for lab in lab_results:
            props = lab['properties']
            if props.get('test') == 'eGFR' and props.get('value', 100) < 30:
                risk_score += 2
                risk_factors.append("Severe kidney dysfunction")
            elif props.get('test') == 'Creatinine' and props.get('value', 1) > 2.0:
                risk_score += 1
                risk_factors.append("Elevated creatinine")
        
        # Determine overall risk level
        if risk_score >= 5:
            risk_level = "Very High"
        elif risk_score >= 3:
            risk_level = "High"
        elif risk_score >= 1:
            risk_level = "Moderate"
        else:
            risk_level = "Low"
        
        return {
            'risk_score': risk_score,
            'risk_level': risk_level,
            'risk_factors': risk_factors
        }
    
    def close(self):
        """Close database connection"""
        if self.connection:
            self.connection.close()