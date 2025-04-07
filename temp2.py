import numpy as np
import pandas as pd
from sentence_transformers import SentenceTransformer
from neo4j import GraphDatabase
import logging
import os
from dotenv import load_dotenv

load_dotenv()

class ESGEmbeddingsProcessor:
    def __init__(self, model_name='all-MiniLM-L6-v2'):
        #Initialize the ESG Embeddings Processor
        # Sentence Transformer for generating embeddings
        neo4j_uri = os.getenv("NEO4J_URI")
        neo4j_username = os.getenv("NEO4J_USERNAME")
        neo4j_password = os.getenv("NEO4j_PASSWORD")
        self.embedding_model = SentenceTransformer(model_name)
        # Neo4j connection
        self.driver = GraphDatabase.driver(neo4j_uri, auth=(neo4j_username, neo4j_password))
        # Logging setup
        logging.basicConfig(level=logging.INFO)
        self.logger = logging.getLogger(__name__)
    
    def generate_embeddings(self, paragraphs):
        #Generate embeddings for paragraphs
        return self.embedding_model.encode(paragraphs)
    
    def store_article_embeddings(self, articles):
        #Store article embeddings and metadata in Neo4j
        with self.driver.session() as session:
            for article in articles:
                if 'relevant_paragraphs' not in article or not article['relevant_paragraphs']:
                    continue
                
                # Generate embeddings for relevant paragraphs
                paragraph_embeddings = self.generate_embeddings(article['relevant_paragraphs'])
                
                for i, (paragraph, embedding) in enumerate(zip(article['relevant_paragraphs'], paragraph_embeddings)):
                    # Create or update nodes for articles and paragraphs
                    create_query = """
                    MERGE (a:Article {title: $title, link: $link})
                    MERGE (p:Paragraph {text: $paragraph})
                    MERGE (a)-[:HAS_PARAGRAPH]->(p)
                    SET p.embedding = $embedding,
                        p.esg_theme = $esg_theme,
                        p.company = $company,
                        p.index = $index
                    """
                    
                    session.run(create_query, {
                        'title': article['title'],
                        'link': article['link'],
                        'paragraph': paragraph,
                        'embedding': embedding.tolist(),
                        'esg_theme': article.get('primary_esg_focus', 'undetermined'),
                        'company': article.get('company', 'Unknown'),
                        'index': i
                    })
                
                self.logger.info(f"Stored embeddings for article: {article['title']}")
    
    def similarity_search(self, query, top_k=5, esg_theme=None, company=None):
        #Perform similarity search across embeddings
        query_embedding = self.generate_embeddings([query])[0]
        
        with self.driver.session() as session:
            # Similarity search query with optional filters
            similarity_query = """
            MATCH (p:Paragraph)
            WHERE 1=1
            %s
            %s
            WITH p, algo.similarity.cosine(p.embedding, $query_embedding) AS similarity
            WHERE similarity > 0.7
            RETURN p.text AS paragraph, 
                   p.company AS company, 
                   p.esg_theme AS esg_theme, 
                   similarity
            ORDER BY similarity DESC
            LIMIT $top_k
            """
            
            # Construct optional filters
            theme_filter = "AND p.esg_theme = $esg_theme" if esg_theme else ""
            company_filter = "AND p.company = $company" if company else ""
            
            # Prepare parameters
            params = {
                'query_embedding': query_embedding.tolist(),
                'top_k': top_k
            }
            
            if esg_theme:
                params['esg_theme'] = esg_theme
            if company:
                params['company'] = company
            
            # Execute query
            results = session.run(
                similarity_query % (theme_filter, company_filter), 
                params
            )
            
            return [
                {
                    'paragraph': record['paragraph'],
                    'company': record['company'],
                    'esg_theme': record['esg_theme'],
                    'similarity': record['similarity']
                } 
                for record in results
            ]
    
    def close(self):
        """Close Neo4j driver connection"""
        self.driver.close()

# Example usage in the main scraper class
def enhance_article_with_embeddings(self, articles):
    """
    Enhance articles with embedding-based processing
    
    :param articles: List of article dictionaries
    :return: Enhanced articles
    """
    # Initialize embeddings processor (replace with your Neo4j credentials)
    embeddings_processor = ESGEmbeddingsProcessor(
        neo4j_uri='bolt://localhost:7687', 
        neo4j_username='neo4j', 
        neo4j_password='your_password'
    )
    
    try:
        # Store article embeddings in Neo4j
        embeddings_processor.store_article_embeddings(articles)
        
        # Example: Perform similarity search for each article
        for article in articles:
            if article['relevant_paragraphs']:
                # Search for similar ESG paragraphs
                similar_paragraphs = embeddings_processor.similarity_search(
                    article['title'], 
                    top_k=3, 
                    esg_theme=article.get('primary_esg_focus'),
                    company=article.get('company')
                )
                
                article['similar_paragraphs'] = similar_paragraphs
        
        return articles
    
    except Exception as e:
        self.logger.error(f"Error processing embeddings: {e}")
        return articles
    finally:
        embeddings_processor.close()