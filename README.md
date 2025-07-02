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
