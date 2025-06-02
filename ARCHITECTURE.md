# ClinGenome Navigator - Technical Architecture

## System Overview

ClinGenome Navigator implements a sophisticated Knowledge Graph RAG (Retrieval Augmented Generation) architecture specifically designed for pharmaceutical research and clinical genomics analysis.

## Core Components

### 1. Knowledge Graph Engine
**File:** `utils/enhanced_knowledge_graph.py`
- **Purpose:** Creates semantic relationships from clinical data
- **Technology:** NetworkX-based graph construction
- **Entities:** 10,719+ clinical entities extracted from patient records
- **Relationships:** 12,610+ semantic connections between medical concepts

**Key Relationship Types:**
```python
- causes_kidney_dysfunction
- elevates
- correlates_with
- influences_kidney_function
```

### 2. Cohort Analysis System
**File:** `utils/kg_cohort_analyzer.py`
- **Purpose:** Precise patient filtering through graph traversal
- **Capabilities:** Complex multi-factor cohort identification
- **Example:** Find 378 patients with 2+ mutations AND low eGFR from 1,500 total

### 3. Data Processing Pipeline
**File:** `utils/data_processor.py`
- **Purpose:** Clinical data normalization and metadata generation
- **Features:** Genetic variant analysis, demographics processing, clinical metrics calculation

### 4. Vector Search Integration
**File:** `utils/vector_search_new.py`
- **Purpose:** Semantic search capabilities for general queries
- **Technology:** TF-IDF vectorization with clinical text processing
- **Fallback:** Used when knowledge graph doesn't match specific patterns

### 5. LLM Processing
**File:** `utils/llm_processor.py`
- **Purpose:** Natural language interpretation and clinical insight generation
- **Integration:** OpenAI API with pharmaceutical research context
- **Output:** Professional summaries and actionable recommendations

## User Interface Architecture

### Main Application
**File:** `app.py`
- Streamlit-based web interface
- Professional pharmaceutical branding
- Tabbed navigation (Intelligent Search Hub, Data Exploration)

### Component Structure
```
components/
├── intelligent_search.py     # Primary search interface with KG RAG
├── enhanced_intelligent_search.py  # Advanced search capabilities
├── dashboard.py              # Key metrics and insights overview
├── visualization.py          # Comprehensive data analytics
└── data_overview.py         # Dataset documentation
```

## Knowledge Graph RAG Flow

### 1. Query Processing
```
User Query → Pattern Recognition → Knowledge Graph Traversal → Patient Filtering
```

### 2. Context Augmentation
```
Filtered Patients → Clinical Context → Knowledge Graph Insights → Enhanced Context
```

### 3. Generation
```
Enhanced Context → LLM Processing → Clinical Analysis → Professional Summary
```

## Data Flow Architecture

### Input Layer
- Clinical genomics CSV data
- Patient demographics, genetic variants, lab values
- Clinical notes and medication history

### Processing Layer
- Entity extraction from clinical notes
- Relationship mapping between medical concepts
- Graph construction and indexing
- Vector space creation for semantic search

### Analysis Layer
- Knowledge graph traversal algorithms
- Cohort identification and filtering
- Statistical analysis and correlation studies
- Risk stratification calculations

### Presentation Layer
- Interactive visualizations
- Professional research summaries
- Actionable clinical insights
- Export capabilities

## Security & Performance

### Data Protection
- Patient data remains local
- API keys secured in environment variables
- Sensitive files excluded via .gitignore

### Performance Optimization
- Efficient graph traversal algorithms
- Cached knowledge graph construction
- Streamlined vector search indexing
- Optimized Streamlit session management

### Scalability Considerations
- Modular component architecture
- Efficient memory management for large datasets
- Asynchronous processing capabilities
- Extensible relationship type system

## Integration Points

### External Services
- OpenAI API for natural language processing
- Potential Neo4j integration for larger datasets
- Clinical data sources (CSV, databases)

### Extension Capabilities
- Additional relationship types for new medical domains
- Enhanced visualization components
- Advanced statistical analysis modules
- Clinical trial matching algorithms

## Technology Stack

**Core Technologies:**
- Python 3.11+
- Streamlit (Web Framework)
- NetworkX (Graph Processing)
- Pandas (Data Manipulation)
- OpenAI API (Language Model)

**Visualization:**
- Plotly (Interactive Charts)
- Seaborn (Statistical Plots)
- Custom Streamlit Components

**Data Processing:**
- Scikit-learn (Machine Learning)
- SpaCy (Natural Language Processing)
- NumPy (Numerical Computing)

**Optional Integrations:**
- Neo4j (Graph Database)
- RDFLib (Semantic Web)
- Trafilatura (Web Scraping)

## Deployment Architecture

### Local Development
- Streamlit development server
- Local data processing
- Direct OpenAI API integration

### Production Considerations
- Container-based deployment
- Secure credential management
- Scalable graph database backend
- Load balancing for concurrent users

This architecture enables precise, authentic clinical research while maintaining professional pharmaceutical standards and data integrity.