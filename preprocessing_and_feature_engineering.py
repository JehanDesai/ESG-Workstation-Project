import os
import json
import re
import logging
from typing import List, Dict, Any

import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler, MultiLabelBinarizer
from sklearn.feature_extraction.text import TfidfVectorizer
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
import nltk

# Download necessary NLTK resources
nltk.download('punkt')
nltk.download('stopwords')

# Configure logging
logging.basicConfig(level=logging.INFO, 
                    format='%(asctime)s - %(levelname)s: %(message)s')
logger = logging.getLogger(__name__)

class ESGDataPreprocessor:
    def __init__(self, input_dir: str = 'esg_data', output_dir: str = 'processed_esg_data'):
        """
        Initialize ESG Data Preprocessor
        
        Args:
            input_dir: Directory containing raw scraped data
            output_dir: Directory to save processed data
        """
        self.input_dir = input_dir
        os.makedirs(output_dir, exist_ok=True)
        self.output_dir = output_dir
        # Initialize text preprocessing components
        self.stop_words = set(stopwords.words('english'))
        self.scaler = StandardScaler()
        self.tfidf_vectorizer = TfidfVectorizer(max_features=1000)
    
    def load_scraped_data(self) -> Dict[str, List[Dict[str, Any]]]:
        """
        Load all scraped JSON files from input directory
        
        Returns:
            Dictionary of data sources with their respective data
        """
        scraped_data = {}
        
        for filename in os.listdir(self.input_dir):
            if filename.endswith('.json'):
                filepath = os.path.join(self.input_dir, filename)
                try:
                    with open(filepath, 'r') as f:
                        data = json.load(f)
                    # Categorize data based on filename
                    if 'annual_reports' in filename:
                        scraped_data['annual_reports'] = data
                    elif 'regulatory_filings' in filename:
                        scraped_data['regulatory_filings'] = data
                    elif 'media_articles' in filename:
                        scraped_data['media_articles'] = data
                except Exception as e:
                    logger.error(f"Error loading {filename}: {e}")
        return scraped_data
    
    def clean_text(self, text: str) -> str:
        """
        Clean and preprocess text data
        
        Args:
            text: Input text string
        
        Returns:
            Cleaned and normalized text
        """
        # Convert to lowercase
        text = text.lower()
        # Remove special characters and numbers
        text = re.sub(r'[^a-zA-Z\s]', '', text)
        # Tokenize
        tokens = word_tokenize(text)
        # Remove stopwords
        tokens = [token for token in tokens if token not in self.stop_words]
        return ' '.join(tokens)
    
    def extract_esg_features(self, data: List[Dict[str, Any]]) -> pd.DataFrame:
        """
        Extract and engineer features from ESG data
        
        Args:
            data: List of ESG-related data entries
        
        Returns:
            DataFrame with engineered features
        """
        features = []
        for entry in data:
            try:
                # Clean and vectorize text content
                cleaned_text = self.clean_text(entry.get('summary', '') or entry.get('description', ''))
                text_features = self.tfidf_vectorizer.fit_transform([cleaned_text]).toarray()[0]
                # Extract numerical and categorical features
                feature_entry = {
                    'company': entry.get('company', 'Unknown'),
                    'source': entry.get('source', 'Unknown'),
                    'text_length': len(cleaned_text.split()),
                    'keywords_count': len(re.findall(r'\b(sustainability|carbon|impact|esg)\b', cleaned_text.lower())),
                    
                    # Regulatory indicators
                    'is_eu_regulation': 1 if 'eu' in entry.get('country', '').lower() else 0,
                    'is_us_regulation': 1 if 'usa' in entry.get('country', '').lower() else 0,
                }
                # Combine TF-IDF features with other features
                feature_entry.update({f'text_feature_{i}': val for i, val in enumerate(text_features)})
                features.append(feature_entry)
            except Exception as e:
                logger.error(f"Feature extraction error: {e}")
        return pd.DataFrame(features)
    
    def normalize_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Normalize numerical features using StandardScaler
        
        Args:
            df: Input DataFrame with features
        
        Returns:
            DataFrame with normalized features
        """
        # Select numerical columns for normalization
        numerical_columns = ['text_length', 'keywords_count'] + [col for col in df.columns if col.startswith('text_feature_')]
        # Normalize selected columns
        df[numerical_columns] = self.scaler.fit_transform(df[numerical_columns])
        return df
    
    def create_esg_risk_labels(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Create multi-label classification for ESG risks
        
        Args:
            df: Input DataFrame
        
        Returns:
            DataFrame with ESG risk labels
        """
        # Define risk categories based on text features
        risk_categories = ['environmental', 'social', 'governance']
        # Simple risk labeling based on keyword presence
        df['environmental_risk'] = df['keywords_count'].apply(lambda x: 1 if x > 1 else 0)
        df['social_risk'] = df['text_length'].apply(lambda x: 1 if x > 100 else 0)
        df['governance_risk'] = df['is_eu_regulation'] | df['is_us_regulation']
        return df
    
    def save_processed_data(self, df: pd.DataFrame, filename: str):
        """
        Save processed data to CSV and additional formats
        
        Args:
            df: Processed DataFrame
            filename: Output filename
        """
        # Save as CSV
        csv_path = os.path.join(self.output_dir, f"{filename}.csv")
        df.to_csv(csv_path, index=False)
        # Save as parquet for efficient storage
        parquet_path = os.path.join(self.output_dir, f"{filename}.parquet")
        df.to_parquet(parquet_path)
        # Save basic stats
        stats_path = os.path.join(self.output_dir, f"{filename}_stats.json")
        with open(stats_path, 'w') as f:
            json.dump({'total_entries': len(df),'columns': list(df.columns),'numerical_stats': df.describe().to_dict()}, f, indent=2)
        logger.info(f"Processed data saved: {csv_path}, {parquet_path}")
    
    def process_esg_data(self):
        """
        Main processing pipeline for ESG data
        """
        # Load scraped data
        scraped_data = self.load_scraped_data()
        # Process different data sources
        for source, data in scraped_data.items():
            try:
                # Extract features
                df = self.extract_esg_features(data)
                # Normalize features
                df = self.normalize_features(df)
                # Create risk labels
                df = self.create_esg_risk_labels(df)
                # Save processed data
                self.save_processed_data(df, f"processed_{source}")
            except Exception as e:
                logger.error(f"Error processing {source}: {e}")

# Example usage
def main():
    preprocessor = ESGDataPreprocessor()
    preprocessor.process_esg_data()

if __name__ == "__main__":
    main()