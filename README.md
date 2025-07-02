# Neo4j Knowledge Graph Implementation in ESG GNN System

## Overview

This code implements an ESG (Environmental, Social, Governance) scoring system that leverages a Neo4j knowledge graph as its foundational data source. The system combines graph database capabilities with Graph Neural Networks (GNNs) to predict ESG scores for documents based on their textual content.

## Neo4j Integration Points

### 1. Database Connection and Setup

```python
# Connection configuration
NEO4J_URI = "bolt://localhost:7687"  
NEO4J_USER = "neo4j"              
NEO4J_PASSWORD = "12345678"

# Connection establishment
graph = Graph(NEO4J_URI, auth=(NEO4J_USER, NEO4J_PASSWORD))
```

The system connects to a local Neo4j instance using the `py2neo` library, which provides a Python interface for Neo4j operations.

### 2. Knowledge Graph Schema

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

### 3. Data Extraction and Processing

The `ESGGraphDataset` class serves as the primary interface between Neo4j and the GNN system:

#### a) Node Feature Extraction
```python
def _build_node_features(self, sentences):
    # Extract sentence embeddings using transformer models
    texts = [s['text'] for s in sentences]
    embeddings = self.embedding_model.encode(texts, show_progress_bar=True)
    
    # Add ESG domain-specific features
    esg_features = torch.zeros((len(sentences), 3))  # env, social, gov
```

**Neo4j Query Used:**
```cypher
MATCH (s:Sentence) RETURN s.id as id, s.text as text
```

This query retrieves all sentence nodes with their textual content, which is then processed to create:
- **Semantic embeddings**: Using sentence transformers for contextual understanding
- **Domain-specific features**: Keyword-based ESG categorization features

#### b) Graph Structure Creation
```python
def _build_edge_index(self, sentence_nodes, embeddings=None, similarity_threshold=0.6, max_connections=10):
    # Create edges based on semantic similarity between sentences
    similarities = cosine_similarity(batch_embeddings, embeddings)
```

The system creates a **hybrid graph structure**:
- **Structural edges**: Based on document containment relationships from Neo4j
- **Semantic edges**: Computed using cosine similarity between sentence embeddings
- **Weighted connections**: Edge weights represent semantic similarity scores

#### c) Label Extraction
```python
def _get_sentence_labels(self, sentence_nodes):
    # Query ESG scores for each sentence
    scores = self.graph.run("""
        MATCH (s:Sentence {id: $id})-[r:CATEGORIZED_AS]->(c:Category) 
        RETURN c.name as category, r.score as score
    """, id=node['id']).data()
```

**Neo4j Query Pattern:**
```cypher
MATCH (s:Sentence {id: $id})-[r:CATEGORIZED_AS]->(c:Category) 
RETURN c.name as category, r.score as score
```

This extracts ground truth ESG scores from the knowledge graph for supervised learning.

### 4. Document-Level Prediction

#### Document Processing Pipeline
```python
def predict_document_scores(self, graph, document_id, model_path=None):
    # Retrieve sentences for a specific document
    sentences = graph.run("""
        MATCH (d:Document {filename: $doc_id})-[:CONTAINS]->(s:Sentence)
        RETURN s.id as id, s.text as text
    """, doc_id=document_id).data()
```

**Neo4j Query Used:**
```cypher
MATCH (d:Document {filename: $doc_id})-[:CONTAINS]->(s:Sentence)
RETURN s.id as id, s.text as text
```

## Why Neo4j Knowledge Graph is Used

### 1. **Structured Data Storage**
- **Hierarchical Relationships**: Documents contain sentences, sentences belong to ESG categories
- **Rich Metadata**: Stores ESG scores as relationship properties
- **Flexible Schema**: Easy to extend with additional node types and relationships

### 2. **Graph-Native Queries**
- **Traversal Efficiency**: Quick navigation from documents to sentences to categories
- **Pattern Matching**: Cypher queries naturally express complex relationship patterns
- **Aggregation Capabilities**: Can perform complex analytics across the graph structure

### 3. **Data Integration Benefits**
- **Centralized Knowledge Base**: Single source of truth for ESG-related information
- **Relationship Preservation**: Maintains context between documents, sentences, and scores
- **Scalability**: Neo4j handles large-scale graph operations efficiently

### 4. **Machine Learning Integration**
- **Feature Engineering**: Graph structure provides additional features for ML models
- **Ground Truth Storage**: ESG scores stored as relationship properties serve as training labels
- **Dynamic Updates**: Easy to update scores and relationships as new data arrives

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

### Key Advantages of This Architecture:

1. **Semantic Understanding**: Combines symbolic knowledge (graph relationships) with statistical learning (embeddings)

2. **Contextual Awareness**: Sentences are understood in the context of their documents and related sentences

3. **Multi-level Analysis**: Can analyze at sentence, document, or corpus levels

4. **Interpretability**: Graph structure provides explainable pathways for predictions

5. **Extensibility**: Easy to add new document types, ESG categories, or relationship types

## Knowledge Graph Value Proposition

The Neo4j knowledge graph serves as more than just a database—it's the **semantic backbone** of the ESG analysis system:

- **Preserves Domain Knowledge**: ESG relationships and hierarchies are explicitly modeled
- **Enables Complex Reasoning**: Can traverse multiple relationship types in single queries  
- **Supports Incremental Learning**: New documents and scores can be added without retraining
- **Facilitates Explainable AI**: Prediction paths can be traced through the graph structure

This implementation demonstrates how knowledge graphs can enhance machine learning by providing structured domain knowledge, rich feature representations, and interpretable data relationships that pure text-based approaches cannot easily capture.

## Knowledge Graph Construction and Design

### Knowledge Graph Creation Process

While the provided code focuses on consuming data from Neo4j, the knowledge graph construction likely follows this pipeline:

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
The system appears to use a document processing pipeline that:
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

The knowledge graph likely incorporates multiple scoring methodologies:

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

##### c) **Machine Learning Model Scores**
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

### Extended Knowledge Graph Schema

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

### Knowledge Graph Population Strategies

#### 1. **Automated Text Processing Pipeline**
```python
def build_knowledge_graph(documents_path, neo4j_connection):
    for document in documents:
        # Step 1: Document ingestion
        doc_node = create_document_node(document)
        
        # Step 2: Text extraction and segmentation
        paragraphs = extract_paragraphs(document)
        sentences = extract_sentences(paragraphs)
        
        # Step 3: Create text hierarchy
        for para in paragraphs:
            create_paragraph_node(para, doc_node)
            for sent in para.sentences:
                create_sentence_node(sent, para)
        
        # Step 4: ESG scoring
        for sentence in sentences:
            esg_scores = score_sentence_esg(sentence.text)
            create_esg_relationships(sentence, esg_scores)
        
        # Step 5: Entity extraction
        entities = extract_entities(document.text)
        create_entity_relationships(sentences, entities)
```

#### 2. **Multi-Source Data Integration**
```cypher
// Integrate external ESG frameworks
LOAD CSV WITH HEADERS FROM 'file:///gri_standards.csv' AS row
CREATE (topic:ESGTopic {
    name: row.topic_name,
    gri_code: row.gri_disclosure,
    category: row.category,
    description: row.description
})

// Link to existing categories
MATCH (topic:ESGTopic), (cat:Category)
WHERE topic.category = cat.name
CREATE (topic)-[:BELONGS_TO]->(cat)
```

### Knowledge Graph Quality Assurance

#### **Data Validation Queries**
```cypher
// Check for orphaned sentences
MATCH (s:Sentence) 
WHERE NOT (s)<-[:CONTAINS]-(:Document)
RETURN count(s) as orphaned_sentences

// Validate ESG score distributions
MATCH (s:Sentence)-[r:CATEGORIZED_AS]->(c:Category)
RETURN c.name, 
       avg(r.score) as avg_score,
       stdev(r.score) as score_stddev,
       count(r) as total_sentences

// Find sentences with missing ESG scores
MATCH (s:Sentence)
WHERE NOT (s)-[:CATEGORIZED_AS]->(:Category)
RETURN count(s) as unscored_sentences
```

#### **Graph Statistics and Insights**
```cypher
// Knowledge graph statistics
MATCH (n) RETURN labels(n) as node_type, count(n) as count
UNION
MATCH ()-[r]->() RETURN type(r) as relationship_type, count(r) as count

// ESG coverage analysis
MATCH (d:Document)-[:CONTAINS]->(s:Sentence)-[r:CATEGORIZED_AS]->(c:Category)
WHERE r.score > 0.7
RETURN d.filename, c.name, count(s) as high_score_sentences
ORDER BY high_score_sentences DESC
```

### Knowledge Graph Evolution and Maintenance

#### **Incremental Updates**
```cypher
// Add new documents without disrupting existing structure
MERGE (d:Document {filename: $new_filename})
ON CREATE SET d.created_at = datetime(), d.processing_status = "new"
ON MATCH SET d.last_updated = datetime()
```

#### **Schema Evolution**
```cypher
// Add new ESG frameworks
CREATE (tcfd:Framework {name: "TCFD", full_name: "Task Force on Climate-related Financial Disclosures"})

// Migrate existing categories to framework structure
MATCH (c:Category)
CREATE (c)-[:FOLLOWS]->(tcfd)
```

#### **Quality Metrics**
```cypher
// Monitor data quality over time
MATCH (s:Sentence)-[r:CATEGORIZED_AS]->(c:Category)
WHERE r.confidence < 0.5
RETURN c.name, count(r) as low_confidence_scores,
       avg(r.confidence) as avg_confidence
```

This knowledge graph construction approach enables the ESG GNN system to leverage rich, structured domain knowledge while maintaining flexibility for future enhancements and ensuring data quality through systematic validation and monitoring processes.
