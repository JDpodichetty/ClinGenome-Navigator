import unittest
import pandas as pd
import numpy as np
import sys
sys.path.append('..')

from utils.vector_search import VectorSearch


class TestVectorSearch(unittest.TestCase):
    """Test cases for VectorSearch class"""
    
    def setUp(self):
        """Set up test fixtures before each test method."""
        self.vector_search = VectorSearch()
        
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
            'Clinical_Notes': ['Patient stable', 'Rapid progression noted', 'Regular checkup', 'Requires dialysis', 'Healthy patient']
        }
        self.sample_df = pd.DataFrame(self.sample_data)
    
    def test_build_index_success(self):
        """Test successful index building"""
        self.vector_search.build_index(self.sample_df)
        
        self.assertIsNotNone(self.vector_search.texts)
        self.assertEqual(len(self.vector_search.texts), 5)
        self.assertIsNotNone(self.vector_search.df)
    
    def test_create_texts_from_df(self):
        """Test text creation from dataframe"""
        texts = self.vector_search._create_texts_from_df(self.sample_df)
        
        self.assertEqual(len(texts), 5)
        self.assertIsInstance(texts[0], str)
        
        # Check if first patient's text contains expected information
        first_text = texts[0]
        self.assertIn('Patient age 35', first_text)
        self.assertIn('M patient', first_text)
        self.assertIn('Caucasian ethnicity', first_text)
        self.assertIn('diagnosed with CKD Stage 2', first_text)
    
    def test_age_filtering_above(self):
        """Test age filtering for patients above certain age"""
        self.vector_search.build_index(self.sample_df)
        
        # Test "above age 30" query
        indices, scores = self.vector_search.search("patients above age 30", top_k=10)
        
        # Should return patients aged 35, 42, and 67 (3 patients)
        self.assertGreater(len(indices), 0)
        filtered_df = self.sample_df.iloc[indices]
        self.assertTrue(all(age > 30 for age in filtered_df['Age']))
    
    def test_age_filtering_below(self):
        """Test age filtering for patients below certain age"""
        self.vector_search.build_index(self.sample_df)
        
        # Test "under 30 years" query
        indices, scores = self.vector_search.search("patients under 30 years", top_k=10)
        
        # Should return patients aged 28 and 23 (2 patients)
        self.assertGreater(len(indices), 0)
        filtered_df = self.sample_df.iloc[indices]
        self.assertTrue(all(age < 30 for age in filtered_df['Age']))
    
    def test_egfr_filtering(self):
        """Test eGFR-based filtering"""
        self.vector_search.build_index(self.sample_df)
        
        # Test "eGFR below 30" query
        indices, scores = self.vector_search.search("patients with eGFR below 30", top_k=10)
        
        # Should return patients with eGFR 25.3 and 15.8
        self.assertGreater(len(indices), 0)
        filtered_df = self.sample_df.iloc[indices]
        self.assertTrue(all(egfr < 30 for egfr in filtered_df['eGFR']))
    
    def test_sex_filtering_male(self):
        """Test filtering for male patients"""
        self.vector_search.build_index(self.sample_df)
        
        # Test "male patients" query
        indices, scores = self.vector_search.search("male patients", top_k=10)
        
        # Should return 3 male patients
        self.assertGreater(len(indices), 0)
        filtered_df = self.sample_df.iloc[indices]
        self.assertTrue(all(sex == 'M' for sex in filtered_df['Sex']))
    
    def test_sex_filtering_female(self):
        """Test filtering for female patients"""
        self.vector_search.build_index(self.sample_df)
        
        # Test "female patients" query
        indices, scores = self.vector_search.search("female patients", top_k=10)
        
        # Should return 2 female patients
        self.assertGreater(len(indices), 0)
        filtered_df = self.sample_df.iloc[indices]
        self.assertTrue(all(sex == 'F' for sex in filtered_df['Sex']))
    
    def test_ethnicity_filtering(self):
        """Test ethnicity-based filtering"""
        self.vector_search.build_index(self.sample_df)
        
        # Test "African American patients" query
        indices, scores = self.vector_search.search("African American patients", top_k=10)
        
        # Should return 1 African American patient
        self.assertGreater(len(indices), 0)
        filtered_df = self.sample_df.iloc[indices]
        self.assertTrue(all(eth == 'African American' for eth in filtered_df['Ethnicity']))
    
    def test_trial_eligibility_filtering(self):
        """Test trial eligibility filtering"""
        self.vector_search.build_index(self.sample_df)
        
        # Test "eligible for clinical trials" query
        indices, scores = self.vector_search.search("patients eligible for clinical trials", top_k=10)
        
        # Should return 3 trial-eligible patients
        self.assertGreater(len(indices), 0)
        filtered_df = self.sample_df.iloc[indices]
        self.assertTrue(all(eligible == 'Yes' for eligible in filtered_df['Eligible_For_Trial']))
    
    def test_similarity_calculation(self):
        """Test similarity score calculation"""
        self.vector_search.build_index(self.sample_df)
        
        # Test similarity between identical queries
        score1 = self.vector_search._calculate_similarity("diabetes kidney", "diabetes kidney disease")
        score2 = self.vector_search._calculate_similarity("completely different", "diabetes kidney disease")
        
        # First should have higher similarity than second
        self.assertGreater(score1, score2)
        
        # Scores should be between 0 and 1
        self.assertGreaterEqual(score1, 0)
        self.assertLessEqual(score1, 1.0)
    
    def test_search_with_threshold(self):
        """Test search with similarity threshold"""
        self.vector_search.build_index(self.sample_df)
        
        # Test with high threshold
        indices, scores = self.vector_search.search("kidney disease", top_k=10, similarity_threshold=0.8)
        
        # All returned scores should be above threshold
        for score in scores:
            self.assertGreaterEqual(score, 0.8)
    
    def test_get_similar_patients(self):
        """Test finding similar patients"""
        self.vector_search.build_index(self.sample_df)
        
        # Find patients similar to first patient (index 0)
        similar_indices, similar_scores = self.vector_search.get_similar_patients(0, top_k=3)
        
        # Should return other patients (not the original)
        self.assertNotIn(0, similar_indices)
        self.assertLessEqual(len(similar_indices), 3)
        
        # Scores should be valid
        for score in similar_scores:
            self.assertGreaterEqual(score, 0)
            self.assertLessEqual(score, 1.0)
    
    def test_query_insights(self):
        """Test query insights generation"""
        self.vector_search.build_index(self.sample_df)
        
        indices, scores = self.vector_search.search("CKD patients", top_k=5)
        insights = self.vector_search.get_query_insights("CKD patients", indices, self.sample_df)
        
        self.assertIn('total_results', insights)
        self.assertIn('demographics', insights)
        self.assertIn('clinical', insights)
        
        # Check demographics insights
        demographics = insights['demographics']
        self.assertIn('age_range', demographics)
        self.assertIn('sex_distribution', demographics)
        self.assertIn('ethnicity_distribution', demographics)
    
    def test_suggest_queries(self):
        """Test query suggestions"""
        suggestions = self.vector_search.suggest_queries()
        
        self.assertIsInstance(suggestions, list)
        self.assertGreater(len(suggestions), 5)
        
        # Check if suggestions are strings
        for suggestion in suggestions:
            self.assertIsInstance(suggestion, str)
            self.assertGreater(len(suggestion), 10)  # Should be meaningful queries
    
    def test_empty_dataframe(self):
        """Test handling of empty dataframe"""
        empty_df = pd.DataFrame()
        
        # Should handle empty dataframe gracefully
        try:
            self.vector_search.build_index(empty_df)
            self.assertEqual(len(self.vector_search.texts), 0)
        except Exception as e:
            # If it raises an exception, it should be handled gracefully
            self.fail(f"Empty dataframe handling failed: {e}")
    
    def test_search_without_index(self):
        """Test search without building index first"""
        with self.assertRaises(ValueError):
            self.vector_search.search("test query")
    
    def test_multiple_filters_combination(self):
        """Test combination of multiple filters"""
        self.vector_search.build_index(self.sample_df)
        
        # Test "male patients above age 30" query
        indices, scores = self.vector_search.search("male patients above age 30", top_k=10)
        
        if len(indices) > 0:
            filtered_df = self.sample_df.iloc[indices]
            # Should return only male patients above 30
            self.assertTrue(all(sex == 'M' for sex in filtered_df['Sex']))
            self.assertTrue(all(age > 30 for age in filtered_df['Age']))


if __name__ == '__main__':
    unittest.main()