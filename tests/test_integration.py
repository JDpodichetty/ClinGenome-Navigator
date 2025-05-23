import unittest
import pandas as pd
import tempfile
import os
import sys
sys.path.append('..')

from utils.data_processor import DataProcessor
from utils.vector_search import VectorSearch
from utils.query_processor import QueryProcessor


class TestIntegration(unittest.TestCase):
    """Integration tests for ClinGenome Navigator components"""
    
    def setUp(self):
        """Set up test fixtures for integration testing."""
        # Create comprehensive test data matching real clinical genomics structure
        self.clinical_data = {
            'PatientID': ['CG001', 'CG002', 'CG003', 'CG004', 'CG005', 'CG006'],
            'Age': [34, 45, 29, 62, 38, 51],
            'Sex': ['M', 'F', 'M', 'F', 'M', 'F'],
            'Ethnicity': ['African American', 'Caucasian', 'Hispanic', 'Asian', 'African American', 'Caucasian'],
            'APOL1_Variant': ['G1/G2', 'G0/G0', 'G0/G1', 'G2/G2', 'G1/G1', 'G0/G0'],
            'eGFR': [28.5, 85.2, 95.1, 22.8, 45.3, 78.9],
            'Creatinine': [3.2, 1.1, 0.9, 3.8, 2.1, 1.3],
            'Diagnosis': ['CKD Stage 4', 'CKD Stage 2', 'Normal', 'CKD Stage 4', 'CKD Stage 3', 'CKD Stage 2'],
            'Medications': ['ACE Inhibitors, Diuretics', 'ACE Inhibitors', 'None', 'Erythropoietin, Diuretics', 'ARBs', 'Statins'],
            'Eligible_For_Trial': ['Yes', 'Yes', 'No', 'Yes', 'Yes', 'No'],
            'Clinical_Notes': [
                'Progressive CKD with proteinuria, family history positive',
                'Stable kidney function, regular monitoring required',
                'Healthy patient, routine screening',
                'Advanced CKD, considering dialysis options',
                'Moderate kidney dysfunction, medication adjustment needed',
                'Mild kidney impairment, lifestyle modifications recommended'
            ],
            'APOL1': ['Mut', 'WT', 'WT', 'Mut', 'Mut', 'WT'],
            'NPHS1': ['WT', 'WT', 'Mut', 'WT', 'WT', 'WT'],
            'NPHS2': ['Mut', 'WT', 'WT', 'Mut', 'WT', 'WT'],
            'WT1': ['WT', 'WT', 'WT', 'WT', 'Mut', 'WT'],
            'UMOD': ['WT', 'Mut', 'WT', 'WT', 'WT', 'Mut'],
            'COL4A3': ['WT', 'WT', 'Mut', 'WT', 'WT', 'WT']
        }
        self.test_df = pd.DataFrame(self.clinical_data)
        
        # Create temporary CSV file
        self.temp_file = tempfile.NamedTemporaryFile(mode='w', suffix='.csv', delete=False)
        self.test_df.to_csv(self.temp_file.name, index=False)
        self.temp_file.close()
        
        # Initialize components
        self.data_processor = DataProcessor()
        self.vector_search = VectorSearch()
        self.query_processor = QueryProcessor()
    
    def tearDown(self):
        """Clean up after tests."""
        if os.path.exists(self.temp_file.name):
            os.unlink(self.temp_file.name)
    
    def test_complete_data_pipeline(self):
        """Test the complete data processing pipeline"""
        # Step 1: Load data
        df = self.data_processor.load_data(self.temp_file.name)
        self.assertEqual(len(df), 6)
        
        # Step 2: Build search index
        self.vector_search.build_index(df)
        self.assertEqual(len(self.vector_search.texts), 6)
        
        # Step 3: Verify metadata generation
        metadata = self.data_processor.get_metadata()
        self.assertEqual(metadata['total_patients'], 6)
        self.assertIn('genetic_variants', metadata)
    
    def test_age_based_search_accuracy(self):
        """Test age-based search queries for accuracy"""
        # Load data and build index
        df = self.data_processor.load_data(self.temp_file.name)
        self.vector_search.build_index(df)
        
        # Test "above age 40" query
        indices, scores = self.vector_search.search("patients above age 40", top_k=10)
        
        self.assertGreater(len(indices), 0)
        filtered_df = df.iloc[indices]
        
        # Verify all returned patients are above 40
        ages = filtered_df['Age'].tolist()
        for age in ages:
            self.assertGreater(age, 40, f"Patient age {age} should be above 40")
        
        # Should return patients aged 45, 62, 51 (3 patients)
        expected_ages = [45, 62, 51]
        self.assertEqual(len(ages), 3)
        self.assertEqual(sorted(ages), sorted(expected_ages))
    
    def test_genetic_variant_search(self):
        """Test search for specific genetic variants"""
        # Load data and build index
        df = self.data_processor.load_data(self.temp_file.name)
        self.vector_search.build_index(df)
        
        # Test APOL1 high-risk variant search
        indices, scores = self.vector_search.search("patients with APOL1 G1/G2 variants", top_k=10)
        
        self.assertGreater(len(indices), 0)
        filtered_df = df.iloc[indices]
        
        # Should include patients with high-risk APOL1 variants
        high_risk_variants = ['G1/G2', 'G2/G2', 'G1/G1']
        apol1_variants = filtered_df['APOL1_Variant'].tolist()
        
        for variant in apol1_variants:
            self.assertIn(variant, high_risk_variants)
    
    def test_clinical_trial_eligibility_search(self):
        """Test search for trial-eligible patients"""
        # Load data and build index
        df = self.data_processor.load_data(self.temp_file.name)
        self.vector_search.build_index(df)
        
        # Test trial eligibility search
        indices, scores = self.vector_search.search("patients eligible for clinical trials", top_k=10)
        
        self.assertGreater(len(indices), 0)
        filtered_df = df.iloc[indices]
        
        # All returned patients should be trial eligible
        eligibility = filtered_df['Eligible_For_Trial'].tolist()
        for status in eligibility:
            self.assertEqual(status, 'Yes')
        
        # Should return 4 eligible patients
        self.assertEqual(len(eligibility), 4)
    
    def test_egfr_threshold_search(self):
        """Test eGFR-based filtering accuracy"""
        # Load data and build index
        df = self.data_processor.load_data(self.temp_file.name)
        self.vector_search.build_index(df)
        
        # Test "eGFR below 30" search
        indices, scores = self.vector_search.search("patients with eGFR below 30", top_k=10)
        
        self.assertGreater(len(indices), 0)
        filtered_df = df.iloc[indices]
        
        # All returned patients should have eGFR < 30
        egfr_values = filtered_df['eGFR'].tolist()
        for egfr in egfr_values:
            self.assertLess(egfr, 30, f"eGFR {egfr} should be below 30")
        
        # Should return patients with eGFR 28.5 and 22.8
        expected_egfrs = [28.5, 22.8]
        self.assertEqual(sorted(egfr_values), sorted(expected_egfrs))
    
    def test_combined_filters_search(self):
        """Test search with multiple combined filters"""
        # Load data and build index
        df = self.data_processor.load_data(self.temp_file.name)
        self.vector_search.build_index(df)
        
        # Test "African American patients above age 30"
        indices, scores = self.vector_search.search("African American patients above age 30", top_k=10)
        
        if len(indices) > 0:
            filtered_df = df.iloc[indices]
            
            # Check ethnicity filter
            ethnicities = filtered_df['Ethnicity'].tolist()
            for ethnicity in ethnicities:
                self.assertEqual(ethnicity, 'African American')
            
            # Check age filter
            ages = filtered_df['Age'].tolist()
            for age in ages:
                self.assertGreater(age, 30)
    
    def test_query_enhancement(self):
        """Test query enhancement functionality"""
        # Test clinical term enhancement
        enhanced_query = self.query_processor.enhance_query("kidney disease patients")
        
        self.assertIsInstance(enhanced_query, str)
        self.assertIn("kidney", enhanced_query.lower())
        
        # Test genetic term enhancement
        enhanced_genetic = self.query_processor.enhance_query("APOL1 mutations")
        self.assertIn("apol1", enhanced_genetic.lower())
    
    def test_filter_extraction(self):
        """Test filter extraction from natural language queries"""
        # Test age filter extraction
        filters = self.query_processor.extract_filters("patients above age 35")
        self.assertIn('min_age', filters)
        self.assertEqual(filters['min_age'], 35)
        
        # Test sex filter extraction
        sex_filters = self.query_processor.extract_filters("female patients")
        self.assertIn('sex', sex_filters)
        self.assertEqual(sex_filters['sex'], ['F'])
        
        # Test ethnicity filter extraction
        eth_filters = self.query_processor.extract_filters("African American patients")
        self.assertIn('ethnicity', eth_filters)
        self.assertEqual(eth_filters['ethnicity'], ['African American'])
    
    def test_data_export_integration(self):
        """Test data export functionality with search results"""
        # Load data and perform search
        df = self.data_processor.load_data(self.temp_file.name)
        self.vector_search.build_index(df)
        
        indices, scores = self.vector_search.search("CKD patients", top_k=5)
        results_df = df.iloc[indices]
        
        # Test CSV export
        csv_data = self.data_processor.export_results(results_df, 'csv')
        self.assertIsInstance(csv_data, bytes)
        
        csv_string = csv_data.decode('utf-8')
        self.assertIn('PatientID', csv_string)
        self.assertIn('CG001', csv_string)  # Should include patient IDs
        
        # Test JSON export
        json_data = self.data_processor.export_results(results_df, 'json')
        self.assertIsInstance(json_data, bytes)
        
        json_string = json_data.decode('utf-8')
        self.assertIn('PatientID', json_string)
    
    def test_search_insights_generation(self):
        """Test generation of search insights"""
        # Load data and perform search
        df = self.data_processor.load_data(self.temp_file.name)
        self.vector_search.build_index(df)
        
        indices, scores = self.vector_search.search("kidney disease patients", top_k=5)
        insights = self.vector_search.get_query_insights("kidney disease patients", indices, df)
        
        self.assertIn('total_results', insights)
        self.assertIn('demographics', insights)
        self.assertIn('clinical', insights)
        
        # Check demographics insights
        demographics = insights['demographics']
        self.assertIn('age_range', demographics)
        self.assertIn('sex_distribution', demographics)
        
        # Check clinical insights
        clinical = insights['clinical']
        self.assertIn('diagnoses', clinical)
        self.assertIn('trial_eligibility', clinical)
    
    def test_similar_patients_functionality(self):
        """Test finding similar patients"""
        # Load data and build index
        df = self.data_processor.load_data(self.temp_file.name)
        self.vector_search.build_index(df)
        
        # Find patients similar to first patient (index 0)
        similar_indices, similar_scores = self.vector_search.get_similar_patients(0, top_k=3)
        
        # Should not include the original patient
        self.assertNotIn(0, similar_indices)
        
        # Should return valid indices
        for idx in similar_indices:
            self.assertGreaterEqual(idx, 0)
            self.assertLess(idx, len(df))
        
        # Scores should be valid
        for score in similar_scores:
            self.assertGreaterEqual(score, 0)
            self.assertLessEqual(score, 1.0)
    
    def test_comprehensive_metadata_accuracy(self):
        """Test accuracy of generated metadata"""
        # Load data
        df = self.data_processor.load_data(self.temp_file.name)
        metadata = self.data_processor.get_metadata()
        
        # Verify patient count
        self.assertEqual(metadata['total_patients'], 6)
        
        # Verify genetic variants metadata
        genetic_variants = metadata['genetic_variants']
        self.assertIn('APOL1', genetic_variants)
        
        # Check APOL1 mutation counts
        apol1_counts = genetic_variants['APOL1']
        self.assertEqual(apol1_counts['Mut'], 3)  # CG001, CG004, CG005
        self.assertEqual(apol1_counts['WT'], 3)   # CG002, CG003, CG006
        
        # Verify demographics
        demographics = metadata['demographics']
        age_stats = demographics['age_stats']
        self.assertEqual(age_stats['min'], 29)
        self.assertEqual(age_stats['max'], 62)
        
        # Verify diagnosis distribution
        diagnoses = metadata['diagnoses']
        self.assertEqual(diagnoses.count('CKD Stage 4'), 2)
        self.assertEqual(diagnoses.count('CKD Stage 2'), 2)


if __name__ == '__main__':
    unittest.main()