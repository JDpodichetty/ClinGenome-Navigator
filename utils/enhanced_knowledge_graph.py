"""
Enhanced Knowledge Graph System with Clinical NLP Processing
Extracts structured information from clinical notes for improved querying and cohort analysis
"""
import pandas as pd
import sqlite3
import json
import tempfile
from typing import Dict, List, Any, Tuple
from .clinical_nlp_processor import ClinicalNLPProcessor

class EnhancedKnowledgeGraph:
    """Enhanced knowledge graph with deep clinical note processing"""
    
    def __init__(self):
        # Create in-memory SQLite database for graph operations
        self.db_path = ":memory:"
        self.connection = sqlite3.connect(self.db_path, check_same_thread=False)
        self.nlp_processor = ClinicalNLPProcessor()
        self._initialize_schema()
        
    def _initialize_schema(self):
        """Initialize enhanced database schema for clinical knowledge graph"""
        cursor = self.connection.cursor()
        
        # Patients table
        cursor.execute('''
            CREATE TABLE patients (
                patient_id TEXT PRIMARY KEY,
                age INTEGER,
                sex TEXT,
                ethnicity TEXT,
                apol1_variant TEXT,
                egfr REAL,
                creatinine REAL,
                diagnosis TEXT,
                trial_eligible TEXT
            )
        ''')
        
        # Clinical entities extracted from notes
        cursor.execute('''
            CREATE TABLE clinical_entities (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                patient_id TEXT,
                entity_type TEXT,
                entity_name TEXT,
                entity_value TEXT,
                context TEXT,
                severity TEXT,
                stage TEXT,
                interpretation TEXT,
                FOREIGN KEY (patient_id) REFERENCES patients (patient_id)
            )
        ''')
        
        # Relationships between entities
        cursor.execute('''
            CREATE TABLE entity_relationships (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                source_entity_id INTEGER,
                target_entity_id INTEGER,
                relationship_type TEXT,
                strength REAL,
                context TEXT,
                FOREIGN KEY (source_entity_id) REFERENCES clinical_entities (id),
                FOREIGN KEY (target_entity_id) REFERENCES clinical_entities (id)
            )
        ''')
        
        # Cohort definitions
        cursor.execute('''
            CREATE TABLE cohorts (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                cohort_name TEXT,
                criteria TEXT,
                patient_count INTEGER,
                created_date TEXT
            )
        ''')
        
        # Cohort membership
        cursor.execute('''
            CREATE TABLE cohort_patients (
                cohort_id INTEGER,
                patient_id TEXT,
                FOREIGN KEY (cohort_id) REFERENCES cohorts (id),
                FOREIGN KEY (patient_id) REFERENCES patients (patient_id)
            )
        ''')
        
        # Create indexes for performance
        cursor.execute('CREATE INDEX idx_entity_patient ON clinical_entities (patient_id)')
        cursor.execute('CREATE INDEX idx_entity_type ON clinical_entities (entity_type)')
        cursor.execute('CREATE INDEX idx_entity_name ON clinical_entities (entity_name)')
        cursor.execute('CREATE INDEX idx_relationship_type ON entity_relationships (relationship_type)')
        
        self.connection.commit()
    
    def build_knowledge_graph(self, df: pd.DataFrame) -> Dict[str, int]:
        """Build enhanced knowledge graph from clinical data with NLP processing"""
        print(f"Building enhanced knowledge graph from {len(df)} patients...")
        
        cursor = self.connection.cursor()
        stats = {'patients': 0, 'entities': 0, 'relationships': 0}
        
        for idx, row in df.iterrows():
            patient_id = str(row['PatientID'])
            
            # Insert patient data
            cursor.execute('''
                INSERT OR REPLACE INTO patients 
                (patient_id, age, sex, ethnicity, apol1_variant, egfr, creatinine, diagnosis, trial_eligible)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
            ''', (
                patient_id, row['Age'], row['Sex'], row['Ethnicity'],
                row.get('APOL1_Variant'), row.get('eGFR'), row.get('Creatinine'),
                row.get('Diagnosis'), row.get('Eligible_For_Trial')
            ))
            stats['patients'] += 1
            
            # Process clinical notes if available
            if pd.notna(row.get('Clinical_Notes')):
                clinical_notes = str(row['Clinical_Notes'])
                entities = self.nlp_processor.extract_clinical_entities(clinical_notes, patient_id)
                
                # Insert extracted entities
                entity_ids = []
                for entity_type, entity_list in entities.items():
                    for entity in entity_list:
                        cursor.execute('''
                            INSERT INTO clinical_entities 
                            (patient_id, entity_type, entity_name, entity_value, context, severity, stage, interpretation)
                            VALUES (?, ?, ?, ?, ?, ?, ?, ?)
                        ''', (
                            patient_id, entity_type, entity['name'],
                            entity.get('value', ''), entity.get('context', ''),
                            entity.get('severity', ''), entity.get('stage', ''),
                            entity.get('interpretation', '')
                        ))
                        entity_ids.append(cursor.lastrowid)
                        stats['entities'] += 1
                
                # Build relationships between entities
                relationships = self.nlp_processor.build_entity_relationships(entities, row.to_dict())
                for rel in relationships:
                    # Find entity IDs for relationship
                    source_id = self._find_entity_id(patient_id, rel['source'])
                    target_id = self._find_entity_id(patient_id, rel['target'])
                    
                    if source_id and target_id:
                        cursor.execute('''
                            INSERT INTO entity_relationships 
                            (source_entity_id, target_entity_id, relationship_type, strength, context)
                            VALUES (?, ?, ?, ?, ?)
                        ''', (source_id, target_id, rel['relationship'], 1.0, rel.get('context', '')))
                        stats['relationships'] += 1
            
            # Add genetic entities
            self._add_genetic_entities(cursor, patient_id, row)
            
            # Add lab result entities
            self._add_lab_entities(cursor, patient_id, row)
        
        self.connection.commit()
        print(f"Enhanced knowledge graph built: {stats['patients']} patients, {stats['entities']} entities, {stats['relationships']} relationships")
        return stats
    
    def _find_entity_id(self, patient_id: str, entity_name: str) -> int:
        """Find entity ID for relationship building"""
        cursor = self.connection.cursor()
        cursor.execute('''
            SELECT id FROM clinical_entities 
            WHERE patient_id = ? AND entity_name = ?
            ORDER BY id DESC LIMIT 1
        ''', (patient_id, entity_name))
        
        result = cursor.fetchone()
        return result[0] if result else None
    
    def _add_genetic_entities(self, cursor, patient_id: str, row: pd.Series):
        """Add genetic variant entities"""
        # APOL1 variant
        if pd.notna(row.get('APOL1_Variant')):
            risk_level = self._assess_apol1_risk(row['APOL1_Variant'])
            cursor.execute('''
                INSERT INTO clinical_entities 
                (patient_id, entity_type, entity_name, entity_value, interpretation)
                VALUES (?, ?, ?, ?, ?)
            ''', (patient_id, 'genetic_variant', 'APOL1', row['APOL1_Variant'], risk_level))
        
        # Gene mutations
        gene_cols = ['NPHS1', 'NPHS2', 'WT1', 'COL4A3', 'UMOD']
        for gene in gene_cols:
            if pd.notna(row.get(gene)) and row[gene] == 'Mut':
                cursor.execute('''
                    INSERT INTO clinical_entities 
                    (patient_id, entity_type, entity_name, entity_value, interpretation)
                    VALUES (?, ?, ?, ?, ?)
                ''', (patient_id, 'gene_mutation', gene, 'Mut', 'pathogenic_mutation'))
    
    def _add_lab_entities(self, cursor, patient_id: str, row: pd.Series):
        """Add laboratory result entities with clinical interpretation"""
        # eGFR
        if pd.notna(row.get('eGFR')):
            egfr_value = float(row['eGFR'])
            interpretation = self._interpret_egfr(egfr_value)
            ckd_stage = self._get_ckd_stage(egfr_value)
            
            cursor.execute('''
                INSERT INTO clinical_entities 
                (patient_id, entity_type, entity_name, entity_value, interpretation, stage)
                VALUES (?, ?, ?, ?, ?, ?)
            ''', (patient_id, 'lab_result', 'eGFR', str(egfr_value), interpretation, ckd_stage))
        
        # Creatinine
        if pd.notna(row.get('Creatinine')):
            creat_value = float(row['Creatinine'])
            interpretation = self._interpret_creatinine(creat_value)
            
            cursor.execute('''
                INSERT INTO clinical_entities 
                (patient_id, entity_type, entity_name, entity_value, interpretation)
                VALUES (?, ?, ?, ?, ?)
            ''', (patient_id, 'lab_result', 'Creatinine', str(creat_value), interpretation))
    
    def query_cohort_by_criteria(self, criteria: Dict[str, Any]) -> List[str]:
        """Query patients based on complex clinical criteria"""
        cursor = self.connection.cursor()
        
        # Build dynamic query based on criteria
        where_conditions = []
        params = []
        
        # Basic demographic criteria
        if criteria.get('min_age'):
            where_conditions.append('p.age >= ?')
            params.append(criteria['min_age'])
        
        if criteria.get('max_age'):
            where_conditions.append('p.age <= ?')
            params.append(criteria['max_age'])
        
        if criteria.get('sex'):
            where_conditions.append('p.sex = ?')
            params.append(criteria['sex'])
        
        # Genetic criteria
        if criteria.get('apol1_risk_level'):
            apol1_variants = self._get_apol1_variants_by_risk(criteria['apol1_risk_level'])
            placeholders = ','.join(['?' for _ in apol1_variants])
            where_conditions.append(f'p.apol1_variant IN ({placeholders})')
            params.extend(apol1_variants)
        
        # Lab value criteria
        if criteria.get('egfr_min'):
            where_conditions.append('p.egfr >= ?')
            params.append(criteria['egfr_min'])
        
        if criteria.get('egfr_max'):
            where_conditions.append('p.egfr <= ?')
            params.append(criteria['egfr_max'])
        
        # Build base query
        base_query = 'SELECT DISTINCT p.patient_id FROM patients p'
        
        # Add entity-based criteria
        joins = []
        if criteria.get('has_conditions') or criteria.get('has_symptoms') or criteria.get('has_medications'):
            joins.append('LEFT JOIN clinical_entities ce ON p.patient_id = ce.patient_id')
            
            entity_conditions = []
            if criteria.get('has_conditions'):
                for condition in criteria['has_conditions']:
                    entity_conditions.append('(ce.entity_type = "conditions" AND ce.entity_name LIKE ?)')
                    params.append(f'%{condition}%')
            
            if criteria.get('has_symptoms'):
                for symptom in criteria['has_symptoms']:
                    entity_conditions.append('(ce.entity_type = "symptoms" AND ce.entity_name LIKE ?)')
                    params.append(f'%{symptom}%')
            
            if criteria.get('has_medications'):
                for medication in criteria['has_medications']:
                    entity_conditions.append('(ce.entity_type = "medications" AND ce.entity_name LIKE ?)')
                    params.append(f'%{medication}%')
            
            if entity_conditions:
                where_conditions.append(f'({" OR ".join(entity_conditions)})')
        
        # Construct final query
        final_query = base_query
        if joins:
            final_query += ' ' + ' '.join(joins)
        if where_conditions:
            final_query += ' WHERE ' + ' AND '.join(where_conditions)
        
        cursor.execute(final_query, params)
        results = cursor.fetchall()
        
        return [row[0] for row in results]
    
    def analyze_cohort_characteristics(self, patient_ids: List[str]) -> Dict[str, Any]:
        """Analyze characteristics of a patient cohort"""
        if not patient_ids:
            return {}
        
        cursor = self.connection.cursor()
        placeholders = ','.join(['?' for _ in patient_ids])
        
        # Basic demographics
        cursor.execute(f'''
            SELECT 
                AVG(age) as avg_age,
                COUNT(CASE WHEN sex = 'Male' THEN 1 END) as male_count,
                COUNT(CASE WHEN sex = 'Female' THEN 1 END) as female_count,
                AVG(egfr) as avg_egfr,
                AVG(creatinine) as avg_creatinine,
                COUNT(CASE WHEN trial_eligible = 'Yes' THEN 1 END) as trial_eligible_count
            FROM patients 
            WHERE patient_id IN ({placeholders})
        ''', patient_ids)
        
        demographics = cursor.fetchone()
        
        # APOL1 variant distribution
        cursor.execute(f'''
            SELECT apol1_variant, COUNT(*) as count
            FROM patients 
            WHERE patient_id IN ({placeholders})
            GROUP BY apol1_variant
        ''', patient_ids)
        
        apol1_distribution = dict(cursor.fetchall())
        
        # Clinical entity analysis
        cursor.execute(f'''
            SELECT entity_type, COUNT(*) as count
            FROM clinical_entities 
            WHERE patient_id IN ({placeholders})
            GROUP BY entity_type
        ''', patient_ids)
        
        entity_counts = dict(cursor.fetchall())
        
        # Most common conditions
        cursor.execute(f'''
            SELECT entity_name, COUNT(*) as count
            FROM clinical_entities 
            WHERE patient_id IN ({placeholders}) AND entity_type = 'conditions'
            GROUP BY entity_name
            ORDER BY count DESC
            LIMIT 10
        ''', patient_ids)
        
        common_conditions = dict(cursor.fetchall())
        
        # Risk stratification
        high_risk_count = len([pid for pid in patient_ids if self._is_high_risk_patient(pid)])
        
        return {
            'cohort_size': len(patient_ids),
            'demographics': {
                'average_age': round(demographics[0], 1) if demographics[0] else 0,
                'male_count': demographics[1],
                'female_count': demographics[2],
                'average_egfr': round(demographics[3], 1) if demographics[3] else 0,
                'average_creatinine': round(demographics[4], 2) if demographics[4] else 0,
                'trial_eligible_count': demographics[5]
            },
            'genetic_profile': {
                'apol1_distribution': apol1_distribution,
                'high_risk_genetic_count': self._count_high_risk_genetic(patient_ids)
            },
            'clinical_entities': entity_counts,
            'common_conditions': common_conditions,
            'risk_assessment': {
                'high_risk_patients': high_risk_count,
                'risk_percentage': round((high_risk_count / len(patient_ids)) * 100, 1)
            }
        }
    
    def find_similar_cohorts(self, target_patient_ids: List[str], similarity_threshold: float = 0.7) -> List[Dict]:
        """Find cohorts with similar clinical patterns"""
        cursor = self.connection.cursor()
        
        # Get characteristic entities for target cohort
        placeholders = ','.join(['?' for _ in target_patient_ids])
        cursor.execute(f'''
            SELECT entity_type, entity_name, COUNT(*) as frequency
            FROM clinical_entities 
            WHERE patient_id IN ({placeholders})
            GROUP BY entity_type, entity_name
            ORDER BY frequency DESC
        ''', target_patient_ids)
        
        target_characteristics = cursor.fetchall()
        
        # Find patients with similar entity patterns
        similar_patients = []
        
        for entity_type, entity_name, frequency in target_characteristics[:10]:  # Top 10 characteristics
            cursor.execute('''
                SELECT patient_id, COUNT(*) as matches
                FROM clinical_entities 
                WHERE entity_type = ? AND entity_name = ?
                AND patient_id NOT IN ({})
                GROUP BY patient_id
                HAVING matches > 0
            '''.format(placeholders), [entity_type, entity_name] + target_patient_ids)
            
            similar_patients.extend([row[0] for row in cursor.fetchall()])
        
        # Group similar patients and calculate similarity scores
        from collections import Counter
        patient_similarity = Counter(similar_patients)
        
        similar_cohorts = []
        for patient_id, match_count in patient_similarity.most_common(50):
            similarity_score = match_count / len(target_characteristics[:10])
            if similarity_score >= similarity_threshold:
                similar_cohorts.append({
                    'patient_id': patient_id,
                    'similarity_score': similarity_score,
                    'matching_characteristics': match_count
                })
        
        return similar_cohorts
    
    def get_clinical_pathways(self, condition: str) -> Dict[str, Any]:
        """Analyze clinical pathways for a specific condition"""
        cursor = self.connection.cursor()
        
        # Find patients with the condition
        cursor.execute('''
            SELECT DISTINCT patient_id 
            FROM clinical_entities 
            WHERE entity_type = 'conditions' AND entity_name LIKE ?
        ''', (f'%{condition}%',))
        
        condition_patients = [row[0] for row in cursor.fetchall()]
        
        if not condition_patients:
            return {'message': f'No patients found with condition: {condition}'}
        
        # Analyze treatment patterns
        placeholders = ','.join(['?' for _ in condition_patients])
        cursor.execute(f'''
            SELECT entity_name, COUNT(*) as frequency
            FROM clinical_entities 
            WHERE patient_id IN ({placeholders}) AND entity_type = 'medications'
            GROUP BY entity_name
            ORDER BY frequency DESC
        ''', condition_patients)
        
        common_treatments = dict(cursor.fetchall())
        
        # Analyze progression markers
        cursor.execute(f'''
            SELECT entity_name, AVG(CAST(entity_value AS REAL)) as avg_value
            FROM clinical_entities 
            WHERE patient_id IN ({placeholders}) AND entity_type = 'lab_result'
            AND entity_value != ''
            GROUP BY entity_name
        ''', condition_patients)
        
        lab_patterns = dict(cursor.fetchall())
        
        return {
            'condition': condition,
            'patient_count': len(condition_patients),
            'common_treatments': common_treatments,
            'lab_value_patterns': lab_patterns,
            'cohort_characteristics': self.analyze_cohort_characteristics(condition_patients)
        }
    
    def _assess_apol1_risk(self, variant: str) -> str:
        """Assess APOL1 variant risk level"""
        high_risk = ['G1/G1', 'G1/G2', 'G2/G2']
        medium_risk = ['G0/G1', 'G0/G2']
        return 'high' if variant in high_risk else 'medium' if variant in medium_risk else 'low'
    
    def _get_apol1_variants_by_risk(self, risk_level: str) -> List[str]:
        """Get APOL1 variants by risk level"""
        if risk_level == 'high':
            return ['G1/G1', 'G1/G2', 'G2/G2']
        elif risk_level == 'medium':
            return ['G0/G1', 'G0/G2']
        else:
            return ['G0/G0']
    
    def _interpret_egfr(self, egfr: float) -> str:
        """Interpret eGFR value"""
        if egfr >= 90:
            return 'normal'
        elif egfr >= 60:
            return 'mild_reduction'
        elif egfr >= 45:
            return 'moderate_reduction'
        elif egfr >= 30:
            return 'severe_reduction'
        else:
            return 'kidney_failure'
    
    def _interpret_creatinine(self, creatinine: float) -> str:
        """Interpret creatinine value"""
        if creatinine <= 1.2:
            return 'normal'
        elif creatinine <= 2.0:
            return 'elevated'
        else:
            return 'high'
    
    def _get_ckd_stage(self, egfr: float) -> str:
        """Get CKD stage from eGFR"""
        if egfr >= 90:
            return 'Stage_1'
        elif egfr >= 60:
            return 'Stage_2'
        elif egfr >= 45:
            return 'Stage_3a'
        elif egfr >= 30:
            return 'Stage_3b'
        elif egfr >= 15:
            return 'Stage_4'
        else:
            return 'Stage_5'
    
    def _is_high_risk_patient(self, patient_id: str) -> bool:
        """Determine if patient is high risk based on multiple factors"""
        cursor = self.connection.cursor()
        
        # Check APOL1 risk
        cursor.execute('SELECT apol1_variant FROM patients WHERE patient_id = ?', (patient_id,))
        result = cursor.fetchone()
        if result and result[0] in ['G1/G1', 'G1/G2', 'G2/G2']:
            return True
        
        # Check eGFR
        cursor.execute('SELECT egfr FROM patients WHERE patient_id = ?', (patient_id,))
        result = cursor.fetchone()
        if result and result[0] < 30:
            return True
        
        return False
    
    def _count_high_risk_genetic(self, patient_ids: List[str]) -> int:
        """Count patients with high-risk genetic profiles"""
        cursor = self.connection.cursor()
        placeholders = ','.join(['?' for _ in patient_ids])
        
        cursor.execute(f'''
            SELECT COUNT(*) 
            FROM patients 
            WHERE patient_id IN ({placeholders}) 
            AND apol1_variant IN ('G1/G1', 'G1/G2', 'G2/G2')
        ''', patient_ids)
        
        return cursor.fetchone()[0]
    
    def get_graph_statistics(self) -> Dict[str, Any]:
        """Get comprehensive knowledge graph statistics"""
        cursor = self.connection.cursor()
        
        # Patient count
        cursor.execute('SELECT COUNT(*) FROM patients')
        total_patients = cursor.fetchone()[0]
        
        # Entity counts by type
        cursor.execute('''
            SELECT entity_type, COUNT(*) 
            FROM clinical_entities 
            GROUP BY entity_type
        ''')
        entity_counts = dict(cursor.fetchall())
        
        # Relationship counts
        cursor.execute('SELECT COUNT(*) FROM entity_relationships')
        total_relationships = cursor.fetchone()[0]
        
        return {
            'total_patients': total_patients,
            'total_entities': sum(entity_counts.values()),
            'total_relationships': total_relationships,
            'entity_counts_by_type': entity_counts
        }