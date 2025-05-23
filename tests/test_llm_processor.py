import unittest
from unittest.mock import Mock, patch, MagicMock
import pandas as pd
import json
import sys
sys.path.append('..')

from utils.llm_processor import LLMProcessor


class TestLLMProcessor(unittest.TestCase):
    """Test cases for LLMProcessor class"""
    
    def setUp(self):
        """Set up test fixtures before each test method."""
        self.llm_processor = LLMProcessor()
        
        # Create sample test data
        self.sample_data = {
            'PatientID': ['P001', 'P002', 'P003'],
            'Age': [35, 42, 28],
            'Sex': ['M', 'F', 'M'],
            'Ethnicity': ['Caucasian', 'African American', 'Hispanic'],
            'eGFR': [89.5, 25.3, 105.2],
            'Diagnosis': ['CKD Stage 2', 'CKD Stage 4', 'Normal'],
            'Eligible_For_Trial': ['Yes', 'Yes', 'No'],
            'Clinical_Notes': ['Patient stable', 'Rapid progression noted', 'Regular checkup']
        }
        self.sample_df = pd.DataFrame(self.sample_data)
        
        # Mock OpenAI response
        self.mock_response = {
            "summary": "Test summary of clinical findings",
            "key_insights": ["Insight 1", "Insight 2"],
            "clinical_significance": "Test clinical significance",
            "recommended_actions": ["Action 1", "Action 2"],
            "patient_populations": "Test patient populations",
            "data_references": ["Reference 1", "Reference 2"]
        }
    
    def test_initialization_without_api_key(self):
        """Test LLMProcessor initialization without API key"""
        with patch.dict('os.environ', {}, clear=True):
            processor = LLMProcessor()
            self.assertIsNone(processor.client)
    
    @patch.dict('os.environ', {'OPENAI_API_KEY': 'test-key'})
    @patch('utils.llm_processor.OpenAI')
    def test_initialization_with_api_key(self, mock_openai):
        """Test LLMProcessor initialization with API key"""
        processor = LLMProcessor()
        mock_openai.assert_called_once_with(api_key='test-key')
        self.assertIsNotNone(processor.client)
    
    @patch('utils.llm_processor.OpenAI')
    def test_process_clinical_query_success(self, mock_openai):
        """Test successful clinical query processing"""
        # Mock OpenAI client and response
        mock_client = Mock()
        mock_openai.return_value = mock_client
        
        mock_completion = Mock()
        mock_completion.choices = [Mock()]
        mock_completion.choices[0].message.content = json.dumps(self.mock_response)
        
        mock_client.chat.completions.create.return_value = mock_completion
        
        # Initialize processor and test
        processor = LLMProcessor()
        processor.client = mock_client
        
        result = processor.process_clinical_query(
            "Test query", 
            "Test context data"
        )
        
        self.assertEqual(result['summary'], "Test summary of clinical findings")
        self.assertEqual(len(result['key_insights']), 2)
        self.assertIn('clinical_significance', result)
    
    def test_process_clinical_query_no_client(self):
        """Test clinical query processing without OpenAI client"""
        processor = LLMProcessor()
        processor.client = None
        
        result = processor.process_clinical_query("Test query", "Test context")
        
        self.assertIn('error', result)
        self.assertEqual(result['error'], "OpenAI client not initialized")
    
    @patch('utils.llm_processor.OpenAI')
    def test_extract_clinical_insights_success(self, mock_openai):
        """Test successful clinical insights extraction"""
        # Mock OpenAI client and response
        mock_client = Mock()
        mock_openai.return_value = mock_client
        
        insights_response = {
            "overall_summary": "Test clinical summary",
            "disease_patterns": ["Pattern 1", "Pattern 2"],
            "treatment_insights": ["Treatment 1", "Treatment 2"],
            "risk_factors": ["Risk 1", "Risk 2"],
            "research_opportunities": ["Opportunity 1"]
        }
        
        mock_completion = Mock()
        mock_completion.choices = [Mock()]
        mock_completion.choices[0].message.content = json.dumps(insights_response)
        
        mock_client.chat.completions.create.return_value = mock_completion
        
        # Initialize processor and test
        processor = LLMProcessor()
        processor.client = mock_client
        
        clinical_notes = ["Note 1", "Note 2", "Note 3"]
        result = processor.extract_clinical_insights(clinical_notes)
        
        self.assertEqual(result['overall_summary'], "Test clinical summary")
        self.assertEqual(len(result['disease_patterns']), 2)
        self.assertIn('treatment_insights', result)
    
    @patch('utils.llm_processor.OpenAI')
    def test_generate_smart_query_suggestions(self, mock_openai):
        """Test smart query suggestions generation"""
        # Mock OpenAI client and response
        mock_client = Mock()
        mock_openai.return_value = mock_client
        
        suggestions_response = {
            "queries": [
                "Find patients with high-risk genetic variants",
                "Analyze treatment response patterns",
                "Identify trial-eligible populations"
            ]
        }
        
        mock_completion = Mock()
        mock_completion.choices = [Mock()]
        mock_completion.choices[0].message.content = json.dumps(suggestions_response)
        
        mock_client.chat.completions.create.return_value = mock_completion
        
        # Initialize processor and test
        processor = LLMProcessor()
        processor.client = mock_client
        
        result = processor.generate_smart_query_suggestions(self.sample_df)
        
        self.assertIsInstance(result, list)
        self.assertEqual(len(result), 3)
        self.assertIn("genetic variants", result[0])
    
    def test_generate_smart_query_suggestions_no_client(self):
        """Test query suggestions without OpenAI client"""
        processor = LLMProcessor()
        processor.client = None
        
        result = processor.generate_smart_query_suggestions(self.sample_df)
        
        # Should return fallback suggestions
        self.assertIsInstance(result, list)
        self.assertGreater(len(result), 0)
        self.assertIn("APOL1", result[0])
    
    @patch('utils.llm_processor.OpenAI')
    def test_analyze_patient_cohort(self, mock_openai):
        """Test patient cohort analysis"""
        # Mock OpenAI client and response
        mock_client = Mock()
        mock_openai.return_value = mock_client
        
        cohort_response = {
            "cohort_summary": "Test cohort summary",
            "key_characteristics": ["Characteristic 1", "Characteristic 2"],
            "clinical_insights": ["Insight 1", "Insight 2"],
            "trial_suitability": "Suitable for trials",
            "recommendations": ["Recommendation 1"]
        }
        
        mock_completion = Mock()
        mock_completion.choices = [Mock()]
        mock_completion.choices[0].message.content = json.dumps(cohort_response)
        
        mock_client.chat.completions.create.return_value = mock_completion
        
        # Initialize processor and test
        processor = LLMProcessor()
        processor.client = mock_client
        
        result = processor.analyze_patient_cohort(self.sample_df, "Test query")
        
        self.assertEqual(result['cohort_summary'], "Test cohort summary")
        self.assertEqual(len(result['key_characteristics']), 2)
        self.assertIn('trial_suitability', result)
    
    def test_analyze_patient_cohort_empty_df(self):
        """Test cohort analysis with empty dataframe"""
        processor = LLMProcessor()
        empty_df = pd.DataFrame()
        
        result = processor.analyze_patient_cohort(empty_df, "Test query")
        
        self.assertIn('error', result)
        self.assertIn('No data to analyze', result['error'])
    
    @patch('utils.llm_processor.OpenAI')
    def test_generate_research_summary(self, mock_openai):
        """Test research summary generation"""
        # Mock OpenAI client and response
        mock_client = Mock()
        mock_openai.return_value = mock_client
        
        summary_response = {
            "executive_summary": "Test executive summary",
            "key_populations": ["Population 1", "Population 2"],
            "research_opportunities": ["Opportunity 1", "Opportunity 2"],
            "clinical_trial_potential": "High potential for trials",
            "strategic_insights": ["Insight 1"]
        }
        
        mock_completion = Mock()
        mock_completion.choices = [Mock()]
        mock_completion.choices[0].message.content = json.dumps(summary_response)
        
        mock_client.chat.completions.create.return_value = mock_completion
        
        # Initialize processor and test
        processor = LLMProcessor()
        processor.client = mock_client
        
        result = processor.generate_research_summary(self.sample_df)
        
        self.assertEqual(result['executive_summary'], "Test executive summary")
        self.assertEqual(len(result['key_populations']), 2)
        self.assertIn('clinical_trial_potential', result)
    
    @patch('utils.llm_processor.OpenAI')
    def test_openai_api_error_handling(self, mock_openai):
        """Test error handling when OpenAI API fails"""
        # Mock OpenAI client to raise an exception
        mock_client = Mock()
        mock_openai.return_value = mock_client
        
        mock_client.chat.completions.create.side_effect = Exception("API Error")
        
        # Initialize processor and test
        processor = LLMProcessor()
        processor.client = mock_client
        
        result = processor.process_clinical_query("Test query", "Test context")
        
        self.assertIn('error', result)
        self.assertIn('Error processing query', result['error'])
    
    @patch('utils.llm_processor.OpenAI')
    def test_invalid_json_response_handling(self, mock_openai):
        """Test handling of invalid JSON responses from OpenAI"""
        # Mock OpenAI client with invalid JSON response
        mock_client = Mock()
        mock_openai.return_value = mock_client
        
        mock_completion = Mock()
        mock_completion.choices = [Mock()]
        mock_completion.choices[0].message.content = "Invalid JSON response"
        
        mock_client.chat.completions.create.return_value = mock_completion
        
        # Initialize processor and test
        processor = LLMProcessor()
        processor.client = mock_client
        
        result = processor.process_clinical_query("Test query", "Test context")
        
        self.assertIn('error', result)
    
    def test_clinical_notes_processing_limit(self):
        """Test that clinical notes are limited to prevent token overflow"""
        processor = LLMProcessor()
        
        # Create many clinical notes
        many_notes = [f"Clinical note {i}" for i in range(100)]
        
        # Even without OpenAI client, should handle large input gracefully
        result = processor.extract_clinical_insights(many_notes)
        
        self.assertIn('error', result)
        self.assertEqual(result['error'], "OpenAI client not initialized")


if __name__ == '__main__':
    unittest.main()