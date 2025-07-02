import json
from py2neo import Graph, Node, Relationship
import uuid

NEO4J_URI = "bolt://localhost:7687" 
NEO4J_USER = "neo4j"                  
NEO4J_PASSWORD = "12345678"           

def connect_to_neo4j():
    """Establish connection to Neo4j database"""
    try:
        graph = Graph(NEO4J_URI, auth=(NEO4J_USER, NEO4J_PASSWORD))
        print("Successfully connected to Neo4j database")
        return graph
    except Exception as e:
        print(f"Failed to connect to Neo4j: {e}")
        return None

def create_constraints(graph):
    """Create constraints for unique nodes"""
    constraints = [
        "CREATE CONSTRAINT document_filename_unique IF NOT EXISTS FOR (d:Document) REQUIRE d.filename IS UNIQUE",
        "CREATE CONSTRAINT sentence_id_unique IF NOT EXISTS FOR (s:Sentence) REQUIRE s.id IS UNIQUE",
        "CREATE CONSTRAINT category_name_unique IF NOT EXISTS FOR (c:Category) REQUIRE c.name IS UNIQUE"
    ]
    
    for constraint in constraints:
        try:
            graph.run(constraint)
        except Exception as e:
            print(f"Error creating constraint: {e}")

def create_document_node(graph, metadata):
    """Create a document node with metadata"""
    document_props = {
        "filename": metadata["filename"],
        "company_id": metadata.get("company_id", "UNKNOWN"),
        "total_sentences": metadata["total_sentences"],
        "esg_relevant_sentences": metadata["esg_relevant_sentences"],
        "environmental_sentences": metadata["environmental_sentences"],
        "avg_environmental_score": metadata["avg_environmental_score"],
        "social_sentences": metadata["social_sentences"],
        "avg_social_score": metadata["avg_social_score"],
        "governance_sentences": metadata["governance_sentences"],
        "avg_governance_score": metadata["avg_governance_score"]
    }
    document_node = Node("Document", **document_props)
    graph.merge(document_node, "Document", "filename")
    print(f"Created document node for {metadata['filename']}")
    return document_node

def create_category_nodes(graph):
    """Create category nodes for ESG"""
    categories = ["Environmental", "Social", "Governance"]
    category_nodes = {}
    for category in categories:
        node = Node("Category", name=category)
        graph.merge(node, "Category", "name")
        category_nodes[category.lower()] = node
    print("Created category nodes")
    return category_nodes

def process_sentences(graph, document_node, category_nodes, data):
    """Process sentences and create nodes and relationships"""
    sentence_tracker = {}
    for category in ["environmental", "social", "governance"]:
        print(f"Processing {category} sentences...")
        for i, sentence_data in enumerate(data[category]):
            text = sentence_data["sentence"]
            score = sentence_data["score"]
            sentence_id = str(uuid.uuid5(uuid.NAMESPACE_DNS, text))
            if sentence_id in sentence_tracker:
                sentence_node = sentence_tracker[sentence_id]
            else:
                sentence_node = Node("Sentence", id=sentence_id, text=text)
                graph.merge(sentence_node, "Sentence", "id")
                document_contains = Relationship(document_node, "CONTAINS", sentence_node)
                graph.merge(document_contains)
                sentence_tracker[sentence_id] = sentence_node
            
            cat_relationship = Relationship(
                sentence_node, 
                "CATEGORIZED_AS", 
                category_nodes[category],
                score=score
            )
            graph.merge(cat_relationship)
        
        print(f"Added {len(data[category])} {category} sentences")
    
    print(f"Total unique sentences: {len(sentence_tracker)}")

def neo4j_main(company):
    """Main function to process ESG data and build Neo4j graph"""
    try:
        with open(f'extractions/esg_extracted_data_for_{company}.json', 'r', encoding='utf-8') as file:
            data = json.load(file)
    except Exception as e:
        print(f"Error loading JSON file: {e}")
        return
    
    graph = connect_to_neo4j()
    if not graph:
        return
    create_constraints(graph)
    document_node = create_document_node(graph, data["metadata"])
    category_nodes = create_category_nodes(graph)
    
    process_sentences(graph, document_node, category_nodes, data)
    print("Database import completed successfully")
    for category in ["Environmental", "Social", "Governance"]:
        count = graph.run(
            "MATCH (c:Category {name: $category})<-[:CATEGORIZED_AS]-(s) RETURN count(s) as count", 
            category=category
        ).data()[0]["count"]
        print(f"{category} sentences: {count}")
    return graph

# if __name__ == "__main__":
#     main()
