import unittest
import pandas as pd
import numpy as np
import tempfile
import os
from io import StringIO
import sys
sys.path.append('..')

from utils.data_processor import DataProcessor


class TestDataProcessor(unittest.TestCase):
    """Test cases for DataProcessor class"""
    
    def setUp(self):
        """Set up test fixtures before each test method."""
        self.data_processor = DataProcessor()
        
        # Create sample test data
        self.sample_data = {
            'PatientID': ['P001', 'P002', 'P003', 'P004', 'P005'],
            'Age': [35, 42, 28, 67, 23],
            'Sex': ['M', 'F', 'M', 'F', 'M'],
            'Ethnicity': ['Caucasian', 'African American', 'Hispanic', 'Asian', 'Caucasian'],
            'APOL1_Variant': ['G0/G0', 'G1/G2', 'G0/G1', 'G2/G2', 'G0/G0'],
            'eGFR': [89.5, 25.3, 105.2, 15.8, 92.1],
            'Creatinine': [1.2, 3.8, 0.9, 4.5, 1.0],
            'Diagnosis': ['CKD Stage 2', 'CKD Stage 4', 'Normal', 'CKD Stage 5', 'Normal'],
            'Medications': ['ACE Inhibitors', 'Diuretics, ACE Inhibitors', 'None', 'Erythropoietin, Diuretics', 'None'],
            'Eligible_For_Trial': ['Yes', 'Yes', 'No', 'Yes', 'No'],
            'Clinical_Notes': ['Patient stable', 'Rapid progression noted', 'Regular checkup', 'Requires dialysis', 'Healthy patient'],
            'APOL1': ['WT', 'Mut', 'WT', 'Mut', 'WT'],
            'NPHS1': ['WT', 'WT', 'Mut', 'WT', 'WT'],
            'NPHS2': ['WT', 'Mut', 'WT', 'Mut', 'WT'],
            'WT1': ['WT', 'WT', 'WT', 'WT', 'Mut'],
            'UMOD': ['WT', 'Mut', 'WT', 'WT', 'WT'],
            'COL4A3': ['WT', 'WT', 'Mut', 'WT', 'WT']
        }
        self.sample_df = pd.DataFrame(self.sample_data)
        
        # Create temporary CSV file
        self.temp_file = tempfile.NamedTemporaryFile(mode='w', suffix='.csv', delete=False)
        self.sample_df.to_csv(self.temp_file.name, index=False)
        self.temp_file.close()
    
    def tearDown(self):
        """Clean up after each test method."""
        if os.path.exists(self.temp_file.name):
            os.unlink(self.temp_file.name)
    
    def test_load_data_success(self):
        """Test successful data loading from CSV file"""
        result_df = self.data_processor.load_data(self.temp_file.name)
        
        self.assertIsInstance(result_df, pd.DataFrame)
        self.assertEqual(len(result_df), 5)
        self.assertEqual(list(result_df.columns), list(self.sample_data.keys()))
        self.assertTrue(self.data_processor.data_loaded)
    
    def test_load_data_file_not_found(self):
        """Test error handling when file doesn't exist"""
        with self.assertRaises(Exception):
            self.data_processor.load_data('nonexistent_file.csv')
    
    def test_metadata_generation(self):
        """Test metadata generation after data loading"""
        self.data_processor.load_data(self.temp_file.name)
        metadata = self.data_processor.get_metadata()
        
        self.assertEqual(metadata['total_patients'], 5)
        self.assertIn('genetic_variants', metadata)
        self.assertIn('diagnoses', metadata)
        self.assertIn('demographics', metadata)
        self.assertIn('clinical_metrics', metadata)
    
    def test_genetic_variants_analysis(self):
        """Test genetic variants analysis"""
        self.data_processor.load_data(self.temp_file.name)
        metadata = self.data_processor.get_metadata()
        
        genetic_variants = metadata['genetic_variants']
        self.assertIn('APOL1', genetic_variants)
        self.assertIn('NPHS1', genetic_variants)
        self.assertIn('APOL1_Variant', genetic_variants)
        
        # Check APOL1 mutation counts
        apol1_counts = genetic_variants['APOL1']
        self.assertEqual(apol1_counts['WT'], 3)
        self.assertEqual(apol1_counts['Mut'], 2)
    
    def test_demographics_summary(self):
        """Test demographics summary generation"""
        self.data_processor.load_data(self.temp_file.name)
        metadata = self.data_processor.get_metadata()
        
        demographics = metadata['demographics']
        self.assertIn('age_stats', demographics)
        self.assertIn('sex_distribution', demographics)
        self.assertIn('ethnicity_distribution', demographics)
        
        # Check age statistics
        age_stats = demographics['age_stats']
        self.assertEqual(age_stats['min'], 23)
        self.assertEqual(age_stats['max'], 67)
        self.assertAlmostEqual(age_stats['mean'], 39.0, places=1)
    
    def test_clinical_metrics(self):
        """Test clinical metrics calculation"""
        self.data_processor.load_data(self.temp_file.name)
        metadata = self.data_processor.get_metadata()
        
        clinical_metrics = metadata['clinical_metrics']
        self.assertIn('eGFR', clinical_metrics)
        self.assertIn('Creatinine', clinical_metrics)
        
        egfr_stats = clinical_metrics['eGFR']
        self.assertAlmostEqual(egfr_stats['min'], 15.8, places=1)
        self.assertAlmostEqual(egfr_stats['max'], 105.2, places=1)
    
    def test_unique_medications_extraction(self):
        """Test extraction of unique medications"""
        self.data_processor.load_data(self.temp_file.name)
        metadata = self.data_processor.get_metadata()
        
        medications = metadata['medications']
        expected_meds = ['ACE Inhibitors', 'Diuretics', 'Erythropoietin', 'None']
        for med in expected_meds:
            self.assertIn(med, medications)
    
    def test_searchable_texts_creation(self):
        """Test creation of searchable text representations"""
        self.data_processor.load_data(self.temp_file.name)
        searchable_texts = self.data_processor.get_searchable_texts()
        
        self.assertEqual(len(searchable_texts), 5)
        self.assertIsInstance(searchable_texts[0], str)
        
        # Check if first patient's text contains expected information
        first_text = searchable_texts[0]
        self.assertIn('Age: 35', first_text)
        self.assertIn('Sex: M', first_text)
        self.assertIn('Ethnicity: Caucasian', first_text)
    
    def test_data_filtering_age(self):
        """Test data filtering by age range"""
        self.data_processor.load_data(self.temp_file.name)
        
        filters = {'age_range': (30, 50)}
        filtered_df = self.data_processor.filter_data(filters)
        
        # Should include patients aged 35 and 42
        self.assertEqual(len(filtered_df), 2)
        self.assertTrue(all(30 <= age <= 50 for age in filtered_df['Age']))
    
    def test_data_filtering_sex(self):
        """Test data filtering by sex"""
        self.data_processor.load_data(self.temp_file.name)
        
        filters = {'sex': ['M']}
        filtered_df = self.data_processor.filter_data(filters)
        
        # Should include 3 male patients
        self.assertEqual(len(filtered_df), 3)
        self.assertTrue(all(sex == 'M' for sex in filtered_df['Sex']))
    
    def test_data_filtering_ethnicity(self):
        """Test data filtering by ethnicity"""
        self.data_processor.load_data(self.temp_file.name)
        
        filters = {'ethnicity': ['Caucasian']}
        filtered_df = self.data_processor.filter_data(filters)
        
        # Should include 2 Caucasian patients
        self.assertEqual(len(filtered_df), 2)
        self.assertTrue(all(eth == 'Caucasian' for eth in filtered_df['Ethnicity']))
    
    def test_data_filtering_egfr(self):
        """Test data filtering by eGFR range"""
        self.data_processor.load_data(self.temp_file.name)
        
        filters = {'egfr_range': (20, 90)}
        filtered_df = self.data_processor.filter_data(filters)
        
        # Should include patients with eGFR between 20 and 90
        self.assertEqual(len(filtered_df), 2)  # 25.3 and 89.5
        self.assertTrue(all(20 <= egfr <= 90 for egfr in filtered_df['eGFR']))
    
    def test_data_filtering_trial_eligibility(self):
        """Test data filtering by trial eligibility"""
        self.data_processor.load_data(self.temp_file.name)
        
        filters = {'trial_eligible': True}
        filtered_df = self.data_processor.filter_data(filters)
        
        # Should include 3 trial-eligible patients
        self.assertEqual(len(filtered_df), 3)
        self.assertTrue(all(eligible == 'Yes' for eligible in filtered_df['Eligible_For_Trial']))
    
    def test_export_results_csv(self):
        """Test CSV export functionality"""
        self.data_processor.load_data(self.temp_file.name)
        df = self.data_processor.get_data()
        
        csv_data = self.data_processor.export_results(df, 'csv')
        self.assertIsInstance(csv_data, bytes)
        
        # Verify CSV content
        csv_string = csv_data.decode('utf-8')
        self.assertIn('PatientID', csv_string)
        self.assertIn('P001', csv_string)
    
    def test_export_results_json(self):
        """Test JSON export functionality"""
        self.data_processor.load_data(self.temp_file.name)
        df = self.data_processor.get_data()
        
        json_data = self.data_processor.export_results(df, 'json')
        self.assertIsInstance(json_data, bytes)
        
        # Verify JSON content
        json_string = json_data.decode('utf-8')
        self.assertIn('PatientID', json_string)
        self.assertIn('P001', json_string)
    
    def test_export_unsupported_format(self):
        """Test error handling for unsupported export format"""
        self.data_processor.load_data(self.temp_file.name)
        df = self.data_processor.get_data()
        
        with self.assertRaises(ValueError):
            self.data_processor.export_results(df, 'xml')


if __name__ == '__main__':
    unittest.main()