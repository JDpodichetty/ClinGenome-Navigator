"""
Knowledge Graph System for Clinical Genomics Data
Extracts medical entities from clinical notes and creates semantic relationships
"""
import pandas as pd
import networkx as nx
from rdflib import Graph, Namespace, URIRef, Literal, RDF, RDFS
from typing import Dict, List, Tuple, Set
import re
import json

class ClinicalKnowledgeGraph:
    """Builds and queries knowledge graphs from clinical genomics data"""
    
    def __init__(self):
        self.graph = nx.DiGraph()
        self.rdf_graph = Graph()
        self.medical_entities = {}
        self.relationships = []
        
        # Define medical ontology namespaces
        self.PATIENT = Namespace('http://clingenome.org/patient/')
        self.GENE = Namespace('http://clingenome.org/gene/')
        self.CONDITION = Namespace('http://clingenome.org/condition/')
        self.MEDICATION = Namespace('http://clingenome.org/medication/')
        self.LAB = Namespace('http://clingenome.org/lab/')
        
        # Bind namespaces to RDF graph
        self.rdf_graph.bind("patient", self.PATIENT)
        self.rdf_graph.bind("gene", self.GENE)
        self.rdf_graph.bind("condition", self.CONDITION)
        self.rdf_graph.bind("medication", self.MEDICATION)
        self.rdf_graph.bind("lab", self.LAB)
        
        # Medical terminology patterns
        self.medical_patterns = self._build_medical_patterns()
    
    def _build_medical_patterns(self) -> Dict[str, List[str]]:
        """Define medical entity recognition patterns"""
        return {
            'symptoms': [
                'proteinuria', 'edema', 'swelling', 'fatigue', 'shortness of breath',
                'chest pain', 'nausea', 'vomiting', 'dizziness', 'headache',
                'joint pain', 'muscle weakness', 'weight gain', 'weight loss'
            ],
            'conditions': [
                'diabetic nephropathy', 'nephrotic syndrome', 'ckd', 'chronic kidney disease',
                'hypertension', 'diabetes', 'cardiovascular disease', 'heart failure',
                'glomerulonephritis', 'renal failure', 'kidney dysfunction'
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
    
    def build_knowledge_graph(self, df: pd.DataFrame) -> None:
        """Build knowledge graph from clinical data"""
        print(f"Building knowledge graph from {len(df)} patients...")
        
        for idx, row in df.iterrows():
            patient_id = f"patient_{row['PatientID']}"
            
            # Add patient node
            self.graph.add_node(patient_id, 
                               type='patient',
                               age=row['Age'],
                               sex=row['Sex'],
                               ethnicity=row['Ethnicity'])
            
            # Add genetic information
            self._add_genetic_data(patient_id, row)
            
            # Add clinical data
            self._add_clinical_data(patient_id, row)
            
            # Extract entities from clinical notes
            if pd.notna(row.get('Clinical_Notes')):
                self._extract_clinical_entities(patient_id, row['Clinical_Notes'])
            
            # Add to RDF graph
            self._add_to_rdf(patient_id, row)
        
        print(f"Knowledge graph built with {self.graph.number_of_nodes()} nodes and {self.graph.number_of_edges()} relationships")
    
    def _add_genetic_data(self, patient_id: str, row: pd.Series) -> None:
        """Add genetic variant information to the graph"""
        # APOL1 variant
        if pd.notna(row.get('APOL1_Variant')):
            variant_id = f"apol1_{row['APOL1_Variant'].replace('/', '_')}"
            self.graph.add_node(variant_id, type='genetic_variant', gene='APOL1')
            self.graph.add_edge(patient_id, variant_id, relationship='has_variant')
        
        # Gene mutations
        gene_cols = ['NPHS1', 'NPHS2', 'WT1', 'COL4A3', 'UMOD']
        for gene in gene_cols:
            if pd.notna(row.get(gene)) and row[gene] == 'Mut':
                gene_id = f"{gene.lower()}_mutation"
                self.graph.add_node(gene_id, type='gene_mutation', gene=gene)
                self.graph.add_edge(patient_id, gene_id, relationship='has_mutation')
    
    def _add_clinical_data(self, patient_id: str, row: pd.Series) -> None:
        """Add clinical measurements and diagnoses"""
        # Diagnosis
        if pd.notna(row.get('Diagnosis')):
            diagnosis_id = row['Diagnosis'].lower().replace(' ', '_')
            self.graph.add_node(diagnosis_id, type='diagnosis', name=row['Diagnosis'])
            self.graph.add_edge(patient_id, diagnosis_id, relationship='has_diagnosis')
        
        # Laboratory values
        if pd.notna(row.get('eGFR')):
            egfr_category = self._categorize_egfr(row['eGFR'])
            egfr_id = f"egfr_{egfr_category}"
            self.graph.add_node(egfr_id, type='lab_category', test='eGFR', value=row['eGFR'])
            self.graph.add_edge(patient_id, egfr_id, relationship='has_lab_result')
        
        if pd.notna(row.get('Creatinine')):
            creat_category = self._categorize_creatinine(row['Creatinine'])
            creat_id = f"creatinine_{creat_category}"
            self.graph.add_node(creat_id, type='lab_category', test='Creatinine', value=row['Creatinine'])
            self.graph.add_edge(patient_id, creat_id, relationship='has_lab_result')
        
        # Trial eligibility
        if pd.notna(row.get('Eligible_For_Trial')):
            eligibility_id = f"trial_eligible_{row['Eligible_For_Trial'].lower()}"
            self.graph.add_node(eligibility_id, type='trial_status')
            self.graph.add_edge(patient_id, eligibility_id, relationship='has_trial_status')
    
    def _extract_clinical_entities(self, patient_id: str, clinical_notes: str) -> None:
        """Extract medical entities from clinical notes using pattern matching"""
        notes_lower = clinical_notes.lower()
        
        for entity_type, patterns in self.medical_patterns.items():
            for pattern in patterns:
                if pattern in notes_lower:
                    entity_id = f"{entity_type}_{pattern.replace(' ', '_')}"
                    self.graph.add_node(entity_id, type=entity_type, name=pattern)
                    self.graph.add_edge(patient_id, entity_id, relationship='mentioned_in_notes')
        
        # Extract numerical values from notes
        self._extract_numerical_values(patient_id, clinical_notes)
    
    def _extract_numerical_values(self, patient_id: str, notes: str) -> None:
        """Extract lab values and measurements from clinical notes"""
        # eGFR values
        egfr_matches = re.findall(r'egfr[:\s]*(\d+(?:\.\d+)?)', notes.lower())
        for match in egfr_matches:
            value = float(match)
            category = self._categorize_egfr(value)
            entity_id = f"notes_egfr_{category}"
            self.graph.add_node(entity_id, type='notes_lab', test='eGFR', value=value)
            self.graph.add_edge(patient_id, entity_id, relationship='notes_mention')
        
        # Creatinine values
        creat_matches = re.findall(r'creatinine[:\s]*(\d+(?:\.\d+)?)', notes.lower())
        for match in creat_matches:
            value = float(match)
            category = self._categorize_creatinine(value)
            entity_id = f"notes_creatinine_{category}"
            self.graph.add_node(entity_id, type='notes_lab', test='Creatinine', value=value)
            self.graph.add_edge(patient_id, entity_id, relationship='notes_mention')
    
    def _categorize_egfr(self, egfr: float) -> str:
        """Categorize eGFR values for knowledge graph"""
        if egfr >= 90:
            return "normal"
        elif egfr >= 60:
            return "mild_reduction"
        elif egfr >= 45:
            return "moderate_reduction"
        elif egfr >= 30:
            return "severe_reduction"
        else:
            return "kidney_failure"
    
    def _categorize_creatinine(self, creatinine: float) -> str:
        """Categorize creatinine values"""
        if creatinine <= 1.2:
            return "normal"
        elif creatinine <= 2.0:
            return "elevated"
        elif creatinine <= 3.0:
            return "high"
        else:
            return "very_high"
    
    def _add_to_rdf(self, patient_id: str, row: pd.Series) -> None:
        """Add patient data to RDF graph for semantic queries"""
        patient_uri = URIRef(self.PATIENT[patient_id])
        
        # Patient demographics
        self.rdf_graph.add((patient_uri, RDF.type, self.PATIENT.Patient))
        self.rdf_graph.add((patient_uri, self.PATIENT.age, Literal(row['Age'])))
        self.rdf_graph.add((patient_uri, self.PATIENT.sex, Literal(row['Sex'])))
        self.rdf_graph.add((patient_uri, self.PATIENT.ethnicity, Literal(row['Ethnicity'])))
        
        # Genetic variants
        if pd.notna(row.get('APOL1_Variant')):
            variant_uri = URIRef(self.GENE[f"apol1_{row['APOL1_Variant']}"])
            self.rdf_graph.add((patient_uri, self.GENE.hasVariant, variant_uri))
        
        # Clinical diagnosis
        if pd.notna(row.get('Diagnosis')):
            condition_uri = URIRef(self.CONDITION[row['Diagnosis'].replace(' ', '_')])
            self.rdf_graph.add((patient_uri, self.CONDITION.hasDiagnosis, condition_uri))
    
    def query_knowledge_graph(self, query_type: str, parameters: Dict) -> List[str]:
        """Query the knowledge graph for complex relationships"""
        if query_type == "patients_with_pathway":
            return self._find_patients_with_pathway(parameters)
        elif query_type == "similar_patients":
            return self._find_similar_patients(parameters)
        elif query_type == "genetic_interactions":
            return self._find_genetic_interactions(parameters)
        else:
            return []
    
    def _find_patients_with_pathway(self, parameters: Dict) -> List[str]:
        """Find patients following a specific clinical pathway"""
        patients = []
        
        for node in self.graph.nodes():
            if self.graph.nodes[node].get('type') == 'patient':
                # Check if patient has required pathway elements
                neighbors = list(self.graph.neighbors(node))
                
                has_genetic_risk = any(
                    self.graph.nodes[n].get('type') == 'genetic_variant' 
                    for n in neighbors
                )
                
                has_clinical_condition = any(
                    self.graph.nodes[n].get('type') == 'diagnosis'
                    for n in neighbors
                )
                
                if has_genetic_risk and has_clinical_condition:
                    patients.append(node)
        
        return patients
    
    def _find_similar_patients(self, parameters: Dict) -> List[str]:
        """Find patients with similar knowledge graph patterns"""
        target_patient = parameters.get('patient_id')
        if not target_patient or target_patient not in self.graph:
            return []
        
        target_neighbors = set(self.graph.neighbors(target_patient))
        similar_patients = []
        
        for node in self.graph.nodes():
            if (self.graph.nodes[node].get('type') == 'patient' and 
                node != target_patient):
                
                node_neighbors = set(self.graph.neighbors(node))
                similarity = len(target_neighbors.intersection(node_neighbors))
                
                if similarity >= parameters.get('min_similarity', 2):
                    similar_patients.append((node, similarity))
        
        # Sort by similarity
        similar_patients.sort(key=lambda x: x[1], reverse=True)
        return [patient for patient, _ in similar_patients]
    
    def _find_genetic_interactions(self, parameters: Dict) -> List[str]:
        """Find patients with specific genetic interaction patterns"""
        patients_with_interactions = []
        
        for node in self.graph.nodes():
            if self.graph.nodes[node].get('type') == 'patient':
                genetic_variants = [
                    n for n in self.graph.neighbors(node)
                    if self.graph.nodes[n].get('type') in ['genetic_variant', 'gene_mutation']
                ]
                
                if len(genetic_variants) >= parameters.get('min_variants', 2):
                    patients_with_interactions.append(node)
        
        return patients_with_interactions
    
    def get_patient_knowledge_summary(self, patient_id: str) -> Dict:
        """Get comprehensive knowledge graph summary for a patient"""
        if patient_id not in self.graph:
            return {}
        
        neighbors = list(self.graph.neighbors(patient_id))
        summary = {
            'genetic_variants': [],
            'clinical_conditions': [],
            'lab_results': [],
            'clinical_notes_entities': [],
            'relationships': []
        }
        
        for neighbor in neighbors:
            node_data = self.graph.nodes[neighbor]
            node_type = node_data.get('type')
            
            if node_type == 'genetic_variant':
                summary['genetic_variants'].append(node_data)
            elif node_type == 'diagnosis':
                summary['clinical_conditions'].append(node_data)
            elif node_type in ['lab_category', 'notes_lab']:
                summary['lab_results'].append(node_data)
            elif node_type in ['symptoms', 'medications', 'procedures']:
                summary['clinical_notes_entities'].append(node_data)
            
            # Add relationship information
            edge_data = self.graph.edges[patient_id, neighbor]
            summary['relationships'].append({
                'target': neighbor,
                'relationship': edge_data.get('relationship'),
                'target_type': node_type
            })
        
        return summary
    
    def export_knowledge_graph(self, format: str = 'json') -> str:
        """Export the knowledge graph in various formats"""
        if format == 'json':
            return self._export_as_json()
        elif format == 'rdf':
            return self.rdf_graph.serialize(format='turtle')
        else:
            return ""
    
    def _export_as_json(self) -> str:
        """Export graph as JSON for visualization"""
        data = {
            'nodes': [],
            'edges': []
        }
        
        for node, attrs in self.graph.nodes(data=True):
            data['nodes'].append({
                'id': node,
                'type': attrs.get('type', 'unknown'),
                'attributes': attrs
            })
        
        for source, target, attrs in self.graph.edges(data=True):
            data['edges'].append({
                'source': source,
                'target': target,
                'relationship': attrs.get('relationship', 'connected')
            })
        
        return json.dumps(data, indent=2)