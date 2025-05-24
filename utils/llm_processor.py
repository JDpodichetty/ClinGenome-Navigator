import json
import os
import pandas as pd
import streamlit as st
from typing import Dict, List, Optional, Tuple
from openai import OpenAI


class LLMProcessor:
    """Handles LLM-based query processing and clinical insights extraction using OpenAI"""

    def __init__(self):
        self.client = None
        self._initialize_client()

    def _initialize_client(self):
        """Initialize OpenAI client"""
        try:
            api_key = os.environ.get("OPENAI_API_KEY")
            if api_key:
                self.client = OpenAI(api_key=api_key)
            else:
                st.warning(
                    "OpenAI API key not found. LLM features will be disabled.")
        except Exception as e:
            st.error(f"Error initializing OpenAI client: {str(e)}")

    def process_clinical_query(self, query: str, context_data: str) -> Dict:
        """Process clinical query using LLM with RAG context"""
        if not self.client:
            return {"error": "OpenAI client not initialized"}

        try:
            # the newest OpenAI model is "gpt-4o" which was released May 13, 2024.
            # do not change this unless explicitly requested by the user
            response = self.client.chat.completions.create(
                model="gpt-4o",
                messages=[{
                    "role":
                    "system",
                    "content":
                    """You are a clinical genomics research assistant specializing in kidney disease and genetic variants. 
                        You help pharmaceutical researchers analyze patient data to identify patterns, risk factors, and potential therapeutic targets.
                        
                        When answering queries:
                        1. Provide specific insights based on the clinical data provided
                        2. Highlight relevant genetic variants and their clinical significance
                        3. Identify patient populations suitable for clinical trials
                        4. Suggest potential research directions
                        5. Always reference specific data points when making claims
                        6. Use medical terminology appropriately but explain complex concepts
                        
                        Respond in JSON format with:
                        {
                            "summary": "Brief summary of findings",
                            "key_insights": ["insight1", "insight2", ...],
                            "clinical_significance": "Clinical relevance explanation",
                            "recommended_actions": ["action1", "action2", ...],
                            "patient_populations": "Description of relevant patient groups",
                            "data_references": ["specific data points used"]
                        }"""
                }, {
                    "role":
                    "user",
                    "content":
                    f"Query: {query}\n\nClinical Data Context:\n{context_data}"
                }],
                response_format={"type": "json_object"},
                max_tokens=1500)

            result = json.loads(response.choices[0].message.content)
            return result

        except Exception as e:
            return {"error": f"Error processing query: {str(e)}"}

    def extract_clinical_insights(self,
                                  clinical_notes: List[str],
                                  patient_data: Dict = None) -> Dict:
        """Extract insights from clinical notes using LLM"""
        if not self.client:
            return {"error": "OpenAI client not initialized"}

        try:
            # Combine clinical notes
            notes_text = "\n".join(
                clinical_notes[:100])  # Limit to avoid token limits

            patient_context = ""
            if patient_data:
                patient_context = f"\nPatient Context: {json.dumps(patient_data, indent=2)}"

            # the newest OpenAI model is "gpt-4o" which was released May 13, 2024.
            # do not change this unless explicitly requested by the user
            response = self.client.chat.completions.create(
                model="gpt-4o",
                messages=[{
                    "role":
                    "system",
                    "content":
                    """You are a clinical data analyst specializing in nephrology and genetic medicine. 
                        Analyze clinical notes to extract key insights, patterns, and actionable information for pharmaceutical research.
                        
                        Focus on:
                        1. Disease progression patterns
                        2. Treatment response indicators
                        3. Risk factors and genetic markers
                        4. Clinical trial eligibility factors
                        5. Medication effectiveness
                        6. Adverse events or complications
                        
                        Respond in JSON format with:
                        {
                            "overall_summary": "High-level summary of clinical patterns",
                            "disease_patterns": ["pattern1", "pattern2", ...],
                            "treatment_insights": ["insight1", "insight2", ...],
                            "risk_factors": ["factor1", "factor2", ...],
                            "trial_considerations": ["consideration1", "consideration2", ...],
                            "medication_patterns": ["pattern1", "pattern2", ...],
                            "urgent_findings": ["finding1", "finding2", ...],
                            "research_opportunities": ["opportunity1", "opportunity2", ...]
                        }"""
                }, {
                    "role":
                    "user",
                    "content":
                    f"Clinical Notes to Analyze:\n{notes_text}{patient_context}"
                }],
                response_format={"type": "json_object"},
                max_tokens=1500)

            result = json.loads(response.choices[0].message.content)
            return result

        except Exception as e:
            return {"error": f"Error extracting insights: {str(e)}"}

    def generate_smart_query_suggestions(self, df: pd.DataFrame) -> List[str]:
        """Generate intelligent query suggestions based on the dataset"""
        if not self.client:
            return [
                "Find patients with diabetic nephropathy and APOL1 mutations",
                "Show high-risk patients eligible for clinical trials",
                "Analyze genetic variants in African American patients"
            ]

        try:
            # Create dataset summary
            summary_data = {
                "total_patients": len(df),
                "diagnoses": df['Diagnosis'].value_counts().head(5).to_dict()
                if 'Diagnosis' in df.columns else {},
                "ethnicities": df['Ethnicity'].value_counts().to_dict()
                if 'Ethnicity' in df.columns else {},
                "age_range": {
                    "min": int(df['Age'].min()),
                    "max": int(df['Age'].max()),
                    "mean": round(df['Age'].mean(), 1)
                } if 'Age' in df.columns else {},
                "egfr_range": {
                    "min": round(df['eGFR'].min(), 1),
                    "max": round(df['eGFR'].max(), 1),
                    "mean": round(df['eGFR'].mean(), 1)
                } if 'eGFR' in df.columns else {}
            }

            # the newest OpenAI model is "gpt-4o" which was released May 13, 2024.
            # do not change this unless explicitly requested by the user
            response = self.client.chat.completions.create(
                model="gpt-4o",
                messages=[{
                    "role":
                    "system",
                    "content":
                    """You are a clinical research strategist. Generate 8-10 intelligent, specific query suggestions 
                        that pharmaceutical researchers would find valuable for analyzing clinical genomics data.
                        
                        Focus on:
                        1. Genetic variant analysis queries
                        2. Clinical trial patient identification
                        3. Treatment response patterns
                        4. Risk stratification queries
                        5. Biomarker discovery opportunities
                        
                        Make queries specific, actionable, and research-oriented.
                        Respond with a simple JSON array of query strings."""
                }, {
                    "role":
                    "user",
                    "content":
                    f"Dataset Summary: {json.dumps(summary_data, indent=2)}\n\nGenerate smart query suggestions for this clinical genomics dataset."
                }],
                response_format={"type": "json_object"},
                max_tokens=800)

            result = json.loads(response.choices[0].message.content)
            return result.get("queries", [])

        except Exception as e:
            st.error(f"Error generating suggestions: {str(e)}")
            return [
                "Find patients with diabetic nephropathy and APOL1 mutations",
                "Show high-risk patients eligible for clinical trials",
                "Analyze genetic variants in African American patients"
            ]

    def analyze_patient_cohort(self, filtered_df: pd.DataFrame,
                               query: str) -> Dict:
        """Analyze a filtered patient cohort using LLM"""
        if not self.client or len(filtered_df) == 0:
            return {"error": "No data to analyze or LLM not available"}

        try:
            # Create cohort summary
            cohort_data = {
                "total_patients": len(filtered_df),
                "demographics": {
                    "age_stats": {
                        "mean":
                        round(filtered_df['Age'].mean(), 1),
                        "median":
                        round(filtered_df['Age'].median(), 1),
                        "range":
                        f"{int(filtered_df['Age'].min())}-{int(filtered_df['Age'].max())}"
                    } if 'Age' in filtered_df.columns else {},
                    "sex_distribution":
                    filtered_df['Sex'].value_counts().to_dict()
                    if 'Sex' in filtered_df.columns else {},
                    "ethnicity_distribution":
                    filtered_df['Ethnicity'].value_counts().to_dict()
                    if 'Ethnicity' in filtered_df.columns else {}
                },
                "clinical_metrics": {
                    "egfr_stats": {
                        "mean": round(filtered_df['eGFR'].mean(), 1),
                        "median": round(filtered_df['eGFR'].median(), 1),
                        "below_30": len(filtered_df[filtered_df['eGFR'] < 30])
                    } if 'eGFR' in filtered_df.columns else {},
                    "diagnoses":
                    filtered_df['Diagnosis'].value_counts().to_dict()
                    if 'Diagnosis' in filtered_df.columns else {},
                    "trial_eligible":
                    filtered_df['Eligible_For_Trial'].value_counts().to_dict()
                    if 'Eligible_For_Trial' in filtered_df.columns else {}
                }
            }

            # the newest OpenAI model is "gpt-4o" which was released May 13, 2024.
            # do not change this unless explicitly requested by the user
            response = self.client.chat.completions.create(
                model="gpt-4o",
                messages=[{
                    "role":
                    "system",
                    "content":
                    """You are a clinical research analyst. Analyze patient cohorts to provide actionable insights 
                        for pharmaceutical research and drug development.
                        
                        Provide analysis on:
                        1. Cohort characteristics and representativeness
                        2. Clinical trial suitability
                        3. Risk stratification insights
                        4. Treatment considerations
                        5. Research implications
                        
                        Respond in JSON format with:
                        {
                            "cohort_summary": "Brief cohort description",
                            "key_characteristics": ["char1", "char2", ...],
                            "clinical_insights": ["insight1", "insight2", ...],
                            "trial_suitability": "Assessment for clinical trials",
                            "risk_profile": "Risk assessment of the cohort",
                            "recommendations": ["rec1", "rec2", ...],
                            "research_potential": "Opportunities for further research"
                        }"""
                }, {
                    "role":
                    "user",
                    "content":
                    f"Query: {query}\n\nCohort Data:\n{json.dumps(cohort_data, indent=2)}"
                }],
                response_format={"type": "json_object"},
                max_tokens=1200)

            result = json.loads(response.choices[0].message.content)
            return result

        except Exception as e:
            return {"error": f"Error analyzing cohort: {str(e)}"}

    def generate_research_summary(self, df: pd.DataFrame) -> Dict:
        """Generate a comprehensive research summary of the dataset"""
        if not self.client:
            return {"error": "OpenAI client not initialized"}

        try:
            # Extract key clinical notes for analysis
            clinical_notes_sample = []
            if 'Clinical_Notes' in df.columns:
                clinical_notes_sample = df['Clinical_Notes'].dropna().head(
                    20).tolist()

            # Create comprehensive dataset overview
            dataset_overview = {
                "total_patients": len(df),
                "demographics": {
                    "age_distribution": {
                        "mean": round(df['Age'].mean(), 1),
                        "median": round(df['Age'].median(), 1),
                        "range":
                        f"{int(df['Age'].min())}-{int(df['Age'].max())}"
                    } if 'Age' in df.columns else {},
                    "sex_breakdown": df['Sex'].value_counts().to_dict()
                    if 'Sex' in df.columns else {},
                    "ethnicity_breakdown":
                    df['Ethnicity'].value_counts().to_dict()
                    if 'Ethnicity' in df.columns else {}
                },
                "clinical_characteristics": {
                    "diagnoses":
                    df['Diagnosis'].value_counts().head(10).to_dict()
                    if 'Diagnosis' in df.columns else {},
                    "egfr_stats": {
                        "mean": round(df['eGFR'].mean(), 1),
                        "severe_dysfunction": len(df[df['eGFR'] < 30]),
                        "normal_function": len(df[df['eGFR'] >= 90])
                    } if 'eGFR' in df.columns else {},
                    "trial_eligibility":
                    df['Eligible_For_Trial'].value_counts().to_dict()
                    if 'Eligible_For_Trial' in df.columns else {}
                },
                "sample_clinical_notes": clinical_notes_sample[:5]
            }

            # the newest OpenAI model is "gpt-4o" which was released May 13, 2024.
            # do not change this unless explicitly requested by the user
            response = self.client.chat.completions.create(
                model="gpt-4o",
                messages=[{
                    "role":
                    "system",
                    "content":
                    """You are a senior clinical research director providing strategic insights on clinical genomics datasets 
                        for pharmaceutical research and drug development.
                        
                        Provide a comprehensive research summary covering:
                        1. Dataset strengths and research potential
                        2. Key patient populations of interest
                        3. Clinical trial opportunities
                        4. Research priorities and hypotheses
                        5. Regulatory and development considerations
                        
                        Respond in JSON format with:
                        {
                            "executive_summary": "High-level dataset overview and value proposition",
                            "key_populations": ["population1", "population2", ...],
                            "research_opportunities": ["opportunity1", "opportunity2", ...],
                            "clinical_trial_potential": "Assessment of trial readiness and target populations",
                            "data_strengths": ["strength1", "strength2", ...],
                            "recommended_analyses": ["analysis1", "analysis2", ...],
                            "strategic_insights": ["insight1", "insight2", ...],
                            "next_steps": ["step1", "step2", ...]
                        }"""
                }, {
                    "role":
                    "user",
                    "content":
                    f"Clinical Genomics Dataset Overview:\n{json.dumps(dataset_overview, indent=2)}"
                }],
                response_format={"type": "json_object"},
                max_tokens=2000)

            result = json.loads(response.choices[0].message.content)
            return result

        except Exception as e:
            return {"error": f"Error generating research summary: {str(e)}"}
