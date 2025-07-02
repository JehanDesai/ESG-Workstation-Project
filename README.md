# Neo4j Knowledge Graph Implementation in ESG GNN System

## Overview

This code implements an ESG (Environmental, Social, Governance) scoring system that leverages a Neo4j knowledge graph as its foundational data source. The system combines graph database capabilities with Graph Neural Networks (GNNs) to predict ESG scores for documents based on their textual content.

## Neo4j Integration Points

The system connects to a local Neo4j instance using the `py2neo` library, which provides a Python interface for Neo4j operations.

### Knowledge Graph Schema

Based on the Cypher queries in the code, the Neo4j knowledge graph implements the following schema:

#### Node Types:
- **Document**: Represents individual documents with properties like `filename`
- **Sentence**: Individual sentences extracted from documents with properties `id` and `text`
- **Category**: ESG categories (Environmental, Social, Governance)

#### Relationship Types:
- **CONTAINS**: Links documents to their constituent sentences
- **CATEGORIZED_AS**: Links sentences to ESG categories with score properties

```cypher
# Example schema representation:
(Document {filename: "report.pdf"})
    -[:CONTAINS]->
(Sentence {id: "sent_001", text: "We reduced carbon emissions by 25%"})
    -[:CATEGORIZED_AS {score: 0.85}]->
(Category {name: "Environmental"})
```

## Knowledge Graph Implementation Details

### Data Extraction and Processing

The `ESGGraphDataset` class serves as the primary interface between Neo4j and the GNN system:

#### a) Node Feature Extraction

This retrieves all sentence nodes with their textual content, which is then processed to create:
- **Semantic embeddings**: Using sentence transformers for contextual understanding
- **Domain-specific features**: Keyword-based ESG categorization features

#### b) Graph Structure Creation

The system creates a **hybrid graph structure**:
- **Structural edges**: Based on document containment relationships from Neo4j
- **Semantic edges**: Computed using cosine similarity between sentence embeddings
- **Weighted connections**: Edge weights represent semantic similarity scores

#### c) Label Extraction

This extracts ground truth ESG scores from the knowledge graph for supervised learning.

### Document-Level Prediction

#### Document Processing Pipeline

## Why Neo4j Knowledge Graph is Used

### 1. **Structured Data Storage**
- **Hierarchical Relationships**: Documents contain sentences, sentences belong to ESG categories
- **Rich Metadata**: Stores ESG scores as relationship properties
- **Flexible Schema**: Easy to extend with additional node types and relationships

### 2. **Graph-Native Queries**
- **Traversal Efficiency**: Quick navigation from documents to sentences to categories
- **Pattern Matching**: Cypher queries naturally express complex relationship patterns
- **Aggregation Capabilities**: Can perform complex analytics across the graph structure

## System Architecture Flow

```
1. Neo4j Knowledge Graph (Data Storage)
   ↓
2. ESGGraphDataset (Data Processing)
   ↓ 
3. PyTorch Geometric Graph (ML Representation)
   ↓
4. Graph Neural Network (Model Training/Prediction)
   ↓
5. ESG Score Predictions (Output)
```

### Key notes about the Architecture:

1. **Semantic Understanding**: Combines symbolic knowledge (graph relationships) with statistical learning (embeddings)

2. **Contextual Awareness**: Sentences are understood in the context of their documents and related sentences

3. **Multi-level Analysis**: Can analyze at sentence, document, or corpus levels

4. **Interpretability**: Graph structure provides explainable pathways for predictions

5. **Extensibility**: Easy to add new document types, ESG categories, or relationship types

This implementation demonstrates how knowledge graphs can enhance machine learning by providing structured domain knowledge, rich feature representations, and interpretable data relationships that pure text-based approaches cannot easily capture.

### Workflow for Ankit and Murli

#### 1. **Document Ingestion Pipeline**
```cypher
// Create Document nodes
CREATE (d:Document {
    filename: "sustainability_report_2023.pdf",
    document_type: "sustainability_report",
    company: "ABC Corp",
    year: 2023,
    created_at: datetime(),
    file_size: 2048576
})
```

#### 2. **Text Processing and Sentence Extraction**
The document processing pipeline:
- **Extracts text** from PDF/document files
- **Segments into sentences** using NLP libraries (like spaCy or NLTK)
- **Creates sentence nodes** with unique identifiers

```cypher
// Create Sentence nodes and relationships
CREATE (s:Sentence {
    id: "sent_001_doc_abc_2023",
    text: "We reduced our carbon footprint by 25% through renewable energy initiatives.",
    sentence_index: 1,
    word_count: 12,
    paragraph_id: "para_001"
})

// Link sentences to documents
MATCH (d:Document {filename: "sustainability_report_2023.pdf"})
MATCH (s:Sentence {id: "sent_001_doc_abc_2023"})
CREATE (d)-[:CONTAINS {position: 1, page: 1}]->(s)
```

#### 3. **ESG Category Definition**
```cypher
// Create ESG Category nodes
CREATE (env:Category {name: "Environmental", description: "Climate, emissions, resource usage"})
CREATE (soc:Category {name: "Social", description: "Employee welfare, community impact"})
CREATE (gov:Category {name: "Governance", description: "Ethics, compliance, transparency"})
```

#### 4. **ESG Score Assignment Methods**

The knowledge graph incorporates multiple scoring methodologies:

##### a) **Rule-Based Scoring**
```python
# Pseudo-code for rule-based ESG scoring
def assign_esg_scores(sentence_text):
    env_keywords = ['carbon', 'emission', 'renewable', 'sustainable', 'climate']
    social_keywords = ['employee', 'diversity', 'community', 'safety', 'human rights']
    gov_keywords = ['governance', 'compliance', 'ethics', 'transparency', 'audit']
    
    # Calculate scores based on keyword presence and context
    env_score = calculate_keyword_score(sentence_text, env_keywords)
    social_score = calculate_keyword_score(sentence_text, social_keywords)
    gov_score = calculate_keyword_score(sentence_text, gov_keywords)
    
    return env_score, social_score, gov_score
```

##### b) **Expert Annotation**
```cypher
// Store expert-annotated scores
MATCH (s:Sentence {id: "sent_001_doc_abc_2023"})
MATCH (env:Category {name: "Environmental"})
CREATE (s)-[:CATEGORIZED_AS {
    score: 0.85,
    confidence: 0.9,
    annotator: "expert_001",
    annotation_date: datetime(),
    methodology: "expert_manual"
}]->(env)
```

##### c) **Model Scores**
```cypher
// Store ML-generated scores
MATCH (s:Sentence {id: "sent_002_doc_abc_2023"})
MATCH (soc:Category {name: "Social"})
CREATE (s)-[:CATEGORIZED_AS {
    score: 0.72,
    confidence: 0.78,
    model_version: "bert_esg_v2.1",
    prediction_date: datetime(),
    methodology: "ml_automated"
}]->(soc)
```


#### **Complete Node Types:**
```cypher
// Document metadata
(:Document {
    filename: string,
    document_type: string,
    company: string,
    year: integer,
    industry: string,
    file_size: integer,
    processing_status: string
})

// Hierarchical text structure
(:Paragraph {
    id: string,
    text: string,
    paragraph_index: integer,
    section: string
})

(:Sentence {
    id: string,
    text: string,
    sentence_index: integer,
    word_count: integer,
    language: string,
    sentiment_score: float
})

// ESG Framework
(:Category {
    name: string,
    description: string,
    parent_category: string,
    framework: string  // GRI, SASB, TCFD, etc.
})

(:Subcategory {
    name: string,
    category: string,
    description: string,
    weight: float
})

// Entities and concepts
(:Company {
    name: string,
    industry: string,
    ticker: string,
    headquarters: string
})

(:ESGTopic {
    name: string,
    category: string,
    keywords: [string],
    importance_score: float
})
```

#### **Comprehensive Relationship Types:**
```cypher
// Document structure relationships
(Document)-[:CONTAINS {position: int, page: int}]->(Paragraph)
(Paragraph)-[:CONTAINS {position: int}]->(Sentence)
(Document)-[:PUBLISHED_BY]->(Company)

// ESG scoring relationships
(Sentence)-[:CATEGORIZED_AS {score: float, confidence: float, methodology: string}]->(Category)
(Sentence)-[:RELATES_TO {relevance_score: float}]->(ESGTopic)
(Sentence)-[:MENTIONS {entity_type: string, sentiment: string}]->(Company)

// Hierarchical ESG structure
(Category)-[:HAS_SUBCATEGORY]->(Subcategory)
(ESGTopic)-[:BELONGS_TO]->(Category)

// Semantic relationships
(Sentence)-[:SIMILAR_TO {similarity_score: float}]->(Sentence)
(ESGTopic)-[:RELATED_TO {strength: float}]->(ESGTopic)
```

This knowledge graph construction approach enables the ESG GNN system to leverage rich, structured domain knowledge while maintaining flexibility for future enhancements and ensuring data quality through systematic validation and monitoring processes.
