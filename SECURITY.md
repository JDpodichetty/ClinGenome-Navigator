# Security Guide for ClinGenome Navigator

## 🔐 Protecting Sensitive Information

### 1. Environment Variables
**Never commit API keys or secrets to Git. Use environment variables instead.**

```bash
# Create a .env file (already in .gitignore)
echo "OPENAI_API_KEY=your_actual_api_key" > .env
```

### 2. API Key Management
- Store OpenAI API key in environment variables
- Use Replit Secrets for deployment
- Rotate API keys regularly
- Monitor API usage for unauthorized access

### 3. Data Protection
- Patient data should be anonymized/de-identified
- Remove or hash any real patient identifiers
- Use synthetic data for development/testing
- Implement access controls for sensitive datasets

### 4. Application Security
- XSRF protection enabled in Streamlit
- Error details hidden from end users
- No sensitive information in error messages
- Usage statistics disabled

### 5. Git Security Checklist
- ✅ .gitignore includes sensitive files
- ✅ .env files excluded from commits
- ✅ No hardcoded secrets in code
- ✅ No real patient data in repository

### 6. Deployment Security
```bash
# For production deployment:
# - Use HTTPS only
# - Set up proper authentication
# - Limit access to authorized users
# - Regular security updates
```

### 7. Code Review Checklist
Before committing code, ensure:
- [ ] No API keys in source code
- [ ] No database passwords in files
- [ ] No real patient identifiers
- [ ] Error handling doesn't expose sensitive data
- [ ] User inputs are validated

### 8. Emergency Response
If sensitive data is accidentally committed:
1. Remove from current commit
2. Rewrite Git history: `git filter-branch`
3. Force push: `git push --force`
4. Rotate any exposed credentials immediately
5. Review access logs for unauthorized use

### 9. Compliance Considerations
- HIPAA compliance for healthcare data
- GDPR for EU user data
- Data retention policies
- Audit logging requirements

### 10. Regular Security Tasks
- [ ] Monthly API key rotation
- [ ] Quarterly dependency updates
- [ ] Annual security audit
- [ ] Review access permissions