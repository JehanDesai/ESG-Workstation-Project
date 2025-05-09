import PyPDF2
import re
import os
import json
import spacy
from typing import List, Dict, Any, Tuple
import sys

class ESGPDFExtractor:
    def __init__(self):
        # Load NLP model
        try:
            self.nlp = spacy.load("en_core_web_md")
        except OSError:
            print("Downloading spaCy model...")
            import subprocess
            subprocess.run([sys.executable, "-m", "spacy", "download", "en_core_web_md"])
            self.nlp = spacy.load("en_core_web_md")
        
        # ESG keywords by category
        self.esg_keywords = {
            "environmental": [
                "carbon", "emissions", "climate", "renewable", "energy", "water", "waste", 
                "biodiversity", "pollution", "greenhouse", "GHG", "sustainable", 
                "conservation", "recycling", "environmental", "footprint", "net-zero", 
                "decarbonization", "circular economy", "resources", "eco-friendly", 
                "green", "solar", "wind", "hydro", "fossil fuels", "energy efficiency",
                "carbon neutral", "reforestation", "deforestation", "clean energy"
            ],
            "social": [
                "diversity", "inclusion", "equity", "human rights", "labor", "health", 
                "safety", "community", "employee", "talent", "training", "culture", 
                "wellbeing", "welfare", "stakeholder", "engagement", "gender", "equality", 
                "racial", "ethics", "responsibility", "DEI", "social impact", "fair trade",
                "supply chain", "working conditions", "discrimination", "harassment",
                "workforce", "labor rights", "indigenous", "local communities", "pay gap"
            ],
            "governance": [
                "board", "compliance", "ethics", "risk", "management", "transparency", 
                "accountability", "audit", "disclosure", "policy", "governance", 
                "compensation", "executive", "shareholder", "stakeholder", "corruption", 
                "bribery", "whistleblower", "oversight", "regulatory", "corporate governance",
                "tax", "director", "committee", "voting rights", "anti-corruption",
                "lobbying", "political contributions", "code of conduct", "data privacy"
            ]
        }

    #Extract all text from a PDF file.
    def extract_text_from_pdf(self, pdf_path: str) -> str:
        try:
            text = ""
            with open(pdf_path, 'rb') as file:
                reader = PyPDF2.PdfReader(file)
                for page_num in range(len(reader.pages)):
                    page = reader.pages[page_num]
                    text += page.extract_text()
            return text
        except Exception as e:
            print(f"Error extracting text from PDF: {e}")
            return ""

    #Split text into sentences using spaCy.
    def split_into_sentences(self, text: str) -> List[str]:
        # Process text in chunks to avoid memory issues with large documents
        max_length = 100000  # Maximum characters to process at once
        sentences = []
        # Process text in chunks
        for i in range(0, len(text), max_length):
            chunk = text[i:i + max_length]
            doc = self.nlp(chunk)
            sentences.extend([sent.text.strip() for sent in doc.sents])
        return sentences

    #Determine if a sentence is ESG-relevant and classify its category.
    def is_esg_relevant(self, sentence: str) -> Tuple[bool, str]:
        sentence_lower = sentence.lower()    
        # Check each ESG category
        for category, keywords in self.esg_keywords.items():
            for keyword in keywords:
                # Look for whole words rather than substrings
                pattern = r'\b' + re.escape(keyword.lower()) + r'\b'
                if re.search(pattern, sentence_lower) or keyword.lower() in sentence_lower:
                    return True, category
        return False, ""

    #Calculate an ESG relevance score for a sentence based on keyword density and sentence length.
    def calculate_esg_score(self, sentence: str) -> float:
        # Count ESG keywords
        sentence_lower = sentence.lower()
        keyword_count = 0
        matched_keywords = set()
        for category, keywords in self.esg_keywords.items():
            for keyword in keywords:
                pattern = r'\b' + re.escape(keyword.lower()) + r'\b'
                if re.search(pattern, sentence_lower) or keyword.lower() in sentence_lower:
                    matched_keywords.add(keyword)
                    keyword_count += 1
        # Basic sentence analysis
        words = sentence.split()
        word_count = len(words)
        # Calculate score based on keyword density and sentence quality
        if word_count < 5:  # Very short sentences are less likely to be meaningful
            return 0.0
        # Calculate keyword density (unique keywords / words)
        keyword_density = min(len(matched_keywords) / max(word_count, 1) * 10, 1.0)  # Scale and cap at 1.0
        # Length score - medium length sentences are often more informative than very short or very long ones
        length_factor = min(word_count / 15, 1.0) if word_count < 30 else max(1.0 - (word_count - 30) / 100, 0.5)
        # Combine scores with emphasis on keywords
        esg_score = (keyword_density * 0.8) + (length_factor * 0.2)
        return esg_score

    #Extract ESG-relevant sentences from a PDF file.
    def extract_esg_sentences(self, pdf_path: str, threshold: float = 0.3) -> Dict[str, Any]:
        # Extract text from PDF
        text = self.extract_text_from_pdf(pdf_path)
        if not text:
            return {"error": "Failed to extract text from PDF"}
        # Split text into sentences
        sentences = self.split_into_sentences(text)
        # Analyze sentences for ESG relevance
        results = {
            "environmental": [],
            "social": [],
            "governance": [],
            "metadata": {
                "filename": os.path.basename(pdf_path),
                "total_sentences": len(sentences),
                "esg_relevant_sentences": 0
            }
        }
        for sentence in sentences:
            # Skip very short sentences
            if len(sentence.split()) < 5:
                continue       
            is_relevant, category = self.is_esg_relevant(sentence)
            if is_relevant:
                score = self.calculate_esg_score(sentence)
                if score >= threshold:
                    results[category].append({"sentence": sentence,"score": score,})
                    results["metadata"]["esg_relevant_sentences"] += 1
        # Calculate summary statistics
        for category in ["environmental", "social", "governance"]:
            results["metadata"][f"{category}_sentences"] = len(results[category])
            if results[category]:
                results["metadata"][f"avg_{category}_score"] = sum(item["score"] for item in results[category]) / len(results[category])
        return results
    
    #Save extracted ESG data to a JSON file.
    def save_results_to_json(self, results: Dict[str, Any], output_path: str) -> None:
        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(results, f, indent=2)
        print(f"Results saved to {output_path}")
        
    #Process a PDF file and extract ESG-relevant information.
    def process_pdf(self, pdf_path: str, output_path: str = None) -> Dict[str, Any]:
        results = self.extract_esg_sentences(pdf_path)
        if output_path:
            self.save_results_to_json(results, output_path)
        return results