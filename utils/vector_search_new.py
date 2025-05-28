ry_lower:
            if 'Diagnosis' in filtered_df.columns:
                filtered_df = filtered_df[filtered_df['Diagnosis'] == 'CKD Stage 3']
        
        # APOL1 mutation patterns
        elif 'apol1 mutations' in query_lower or 'patients with apol1' in query_lower:
            if 'APOL1_Variant' in filtered_df.columns:
                # APOL1 mutations = any variant except G0/G0 (wild-type)
                apol1_filtered = filtered_df[filtered_df['APOL1_Variant'] != 'G0/G0']
                
                # Check for kidney dysfunction combination
                if any(term in query_lower for term in ['kidney dysfunction', 'kidney disease', 'renal dysfunction']):
                    # Kidney dysfunction = Advanced CKD stages or significantly low eGFR or high creatinine
                    kidney_dysfunction = (
                        (filtered_df['Diagnosis'].str.contains('CKD Stage [34]', na=False, regex=True)) |
                        (filtered_df['eGFR'] < 45) |
                        (filtered_df['Creatinine'] > 2.0)
                    )
                    filtered_df = apol1_filtered[kidney_dysfunction]
                else:
                    filtered_df = apol1_filtered
        
        # Kidney dysfunction patterns
        elif any(term in query_lower for term in ['kidney dysfunction', 'kidney disease', 'renal dysfunction', 'kidney failure']):
            # Kidney dysfunction = Advanced CKD stages, significantly low eGFR, or high creatinine
            kidney_dysfunction = (
                (filtered_df['Diagnosis'].str.contains('CKD Stage [34]', na=False, regex=True)) |
                (filtered_df['eGFR'] < 45) |
                (filtered_df['Creatinine'] > 2.0)
            )
            filtered_df = filtered_df[kidney_dysfunction]
        
        # Wild-type APOL1 pattern
        elif 'wild-type apol1' in query_lower or 'wildtype apol1' in query_lower:
            if 'APOL1_Variant' in filtered_df.columns:
                filtered_df = filtered_df[filtered_df['APOL1_Variant'] == 'G0/G0']
        
        # Clinical condition patterns
        elif 'diabetic nephropathy' in query_lower:
            if 'Diagnosis' in filtered_df.columns:
                filtered_df = filtered_df[filtered_df['Diagnosis'] == 'Diabetic Nephropathy']
        elif 'nephrotic syndrome' in query_lower:
            if 'Diagnosis' in filtered_df.columns:
                filtered_df = filtered_df[filtered_df['Diagnosis'] == 'Nephrotic Syndrome']
        elif 'glomerulonephritis' in query_lower:
            if 'Diagnosis' in filtered_df.columns:
                filtered_df = filtered_df[filtered_df['Diagnosis'].str.contains('glomerulonephritis', case=False, na=False)]

        # Dynamic numerical range patterns
        elif any(range_indicator in query_lower for range_indicator in ['aged', 'egfr', 'creatinine']):
            import re
            
            # Age ranges
            if 'aged' in query_lower and 'Age' in filtered_df.columns:
                age_pattern = r'aged\s+(\d+)[-–](\d+)'
                match = re.search(age_pattern, query_lower)
                if match:
                    min_val, max_val = int(match.group(1)), int(match.group(2))
                    age_filtered = filtered_df[(filtered_df['Age'] >= min_val) & (filtered_df['Age'] <= max_val)]
                    
                    if 'gene mutations' in query_lower:
                        gene_cols = ['WT1', 'NPHS1', 'NPHS2', 'COL4A3', 'UMOD']
                        has_mutation = (age_filtered[gene_cols] == 'Mut').any(axis=1)
                        filtered_df = age_filtered[has_mutation]
                    else:
                        filtered_df = age_filtered
            
            # eGFR ranges  
            elif 'egfr' in query_lower and 'eGFR' in filtered_df.columns:
                # Handle both "eGFR 60-90" and "eGFR between 60 and 90"
                egfr_pattern = r'egfr\s+(?:between\s+)?(\d+)(?:[-–]\s*|(?:\s+and\s+))(\d+)'
                match = re.search(egfr_pattern, query_lower)
                if match:
                    min_val, max_val = int(match.group(1)), int(match.group(2))
                    egfr_filtered = filtered_df[(filtered_df['eGFR'] >= min_val) & (filtered_df['eGFR'] <= max_val)]
                    
                    # Check for additional conditions
                    if 'ckd stage' in query_lower:
                        stage_match = re.search(r'ckd stage (\d+)', query_lower)
                        if stage_match:
                            stage = stage_match.group(1)
                            filtered_df = egfr_filtered[egfr_filtered['Diagnosis'] == f'CKD Stage {stage}']
                        else:
                            filtered_df = egfr_filtered
                    else:
                        filtered_df = egfr_filtered
            
            # Creatinine ranges
            elif 'creatinine' in query_lower and 'Creatinine' in filtered_df.columns:
                # Handle both "creatinine 2-3" and "creatinine between 2 and 3"
                creat_pattern = r'creatinine\s+(?:between\s+)?(\d+(?:\.\d+)?)(?:[-–]\s*|(?:\s+and\s+))(\d+(?:\.\d+)?)'
                match = re.search(creat_pattern, query_lower)
                if match:
                    min_val, max_val = float(match.group(1)), float(match.group(2))
                    filtered_df = filtered_df[(filtered_df['Creatinine'] >= min_val) & (filtered_df['Creatinine'] <= max_val)]
        
        # Specific APOL1 variants (legacy patterns)
        elif 'g2/g2' in query_lower:
            if 'APOL1_Variant' in filtered_df.columns:
                filtered_df = filtered_df[filtered_df['APOL1_Variant'] == 'G2/G2']
        elif 'g1 variant' in query_lower:
            if 'APOL1_Variant' in filtered_df.columns:
                filtered_df = filtered_df[filtered_df['APOL1_Variant'].str.contains('G1')]
        elif 'g2 variant' in query_lower:
            if 'APOL1_Variant' in filtered_df.columns:
                filtered_df = filtered_df[filtered_df['APOL1_Variant'].str.contains('G2')]
        
        # Return indices only if filtering actually occurred
        return filtered_df.index.tolist() if len(filtered_df) < original_size else None

    def get_similar_patients(self, patient_idx: int, top_k: int = 5) -> Tuple[List[int], List[float]]:
        """Find similar patients to a given patient"""
        if self.tfidf_matrix is None:
            raise ValueError("Index not built. Call build_index() first.")
        
        if patient_idx >= len(self.texts):
            raise ValueError(f"Patient index {patient_idx} out of range")
        
        # Get the patient's vector
        patient_vector = self.tfidf_matrix[patient_idx]
        
        # Calculate similarities with all other patients
        similarities = cosine_similarity(patient_vector, self.tfidf_matrix).flatten()
        
        # Get top similar patients (excluding the patient itself)
        sim_indices = np.argsort(similarities)[::-1]
        similar_indices = [idx for idx in sim_indices if idx != patient_idx][:top_k]
        similar_scores = [similarities[idx] for idx in similar_indices]
        
        return similar_indices, similar_scores

    def suggest_queries(self) -> List[str]:
        """Suggest example queries for users"""
        return [
            "Find patients above age 50 with kidney disease",
            "Show me African American patients with APOL1 mutations",
            "Patients with eGFR below 30",
            "Female patients eligible for clinical trials", 
            "Patients with high-risk genetic variants",
            "Male patients with CKD Stage 4",
            "Patients under age 40 with normal kidney function",
            "Hispanic patients with NPHS2 mutations"
        ]

    def get_query_insights(self, query: str, results_indices: List[int], df: pd.DataFrame) -> Dict:
        """Generate insights about search results"""
        if not results_indices:
            return {"total_results": 0}
        
        results_df = df.iloc[results_indices]
        
        insights = {
            "total_results": len(results_indices),
            "demographics": {},
            "clinical": {},
            "genetic": {}
        }
        
        # Demographics insights
        if 'Age' in results_df.columns:
            insights["demographics"]["age_range"] = {
                "min": int(results_df['Age'].min()),
                "max": int(results_df['Age'].max()),
                "mean": round(results_df['Age'].mean(), 1)
            }
        
        if 'Sex' in results_df.columns:
            sex_counts = results_df['Sex'].value_counts()
            insights["demographics"]["sex_distribution"] = sex_counts.to_dict()
        
        if 'Ethnicity' in results_df.columns:
            ethnicity_counts = results_df['Ethnicity'].value_counts()
            insights["demographics"]["ethnicity_distribution"] = ethnicity_counts.to_dict()
        
        # Clinical insights
        if 'Diagnosis' in results_df.columns:
            diagnosis_counts = results_df['Diagnosis'].value_counts()
            insights["clinical"]["diagnoses"] = diagnosis_counts.to_dict()
        
        if 'eGFR' in results_df.columns:
            insights["clinical"]["egfr_stats"] = {
                "mean": round(results_df['eGFR'].mean(), 1),
                "min": round(results_df['eGFR'].min(), 1),
                "max": round(results_df['eGFR'].max(), 1)
            }
        
        if 'Eligible_For_Trial' in results_df.columns:
            trial_counts = results_df['Eligible_For_Trial'].value_counts()
            insights["clinical"]["trial_eligibility"] = trial_counts.to_dict()
        
        # Genetic insights
        if 'APOL1_Variant' in results_df.columns:
            apol1_counts = results_df['APOL1_Variant'].value_counts()
            insights["genetic"]["apol1_variants"] = apol1_counts.to_dict()
        
        return insights