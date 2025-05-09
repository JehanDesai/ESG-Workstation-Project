import os
from sustainability_report_extraction import ESGPDFExtractor
from neo4j_data_importer import neo4j_main
from improved_ESG_prediction_model import ImprovedESGGraphTrainer, ImprovedESGGNN, ESGGraphDataset
import torch
import json
from py2neo import Graph
from dotenv import load_dotenv
import os
from google import genai
import random
load_dotenv()


def prepare_gemini_payload(json_path: str) -> dict:
    # Load the ESG-extracted data
    with open(json_path, "r", encoding="utf-8") as f:
        data = json.load(f)

    metadata = data.get("metadata", {})
    
    def sample_sentences(category):
        return [s["sentence"] for s in random.sample(data.get(category, []), min(3, len(data.get(category, []))))]

    payload = {
        "esg_relevant_sentences": metadata.get("esg_relevant_sentences", 0),

        "environmental_score": round(metadata.get("avg_environmental_score", 0.0), 4),
        "social_score": round(metadata.get("avg_social_score", 0.0), 4),
        "governance_score": round(metadata.get("avg_governance_score", 0.0), 4),

        "environmental_count": metadata.get("environmental_sentences", 0),
        "social_count": metadata.get("social_sentences", 0),
        "governance_count": metadata.get("governance_sentences", 0),

        "env_sentences": sample_sentences("environmental"),
        "social_sentences": sample_sentences("social"),
        "gov_sentences": sample_sentences("governance")
    }
    return payload

def generate_ai_insights(json_path: str, output_json: str, company, payload):
    API_KEY = os.getenv("GOOGLE_API_KEY")
    client = genai.Client(api_key=API_KEY)
    prompt = f"""
        You are an expert sustainability analyst. Based on the data provided below, generate a comprehensive and insightful ESG impact report. This report will be used by executives, auditors, and investors to understand the environmental, social, and governance performance of the company.

        ### Company Information:
        - Company Name: {company}
        - ESG-Relevant Sentences: {payload["esg_relevant_sentences"]}

        ### ESG Score Summary:
        - Environmental Score: {payload["environmental_score"]} (avg), based on {payload["environmental_count"]} sentences
        - Social Score: {payload["social_score"]} (avg), based on {payload["social_count"]} sentences
        - Governance Score: {payload["governance_score"]} (avg), based on {payload["governance_count"]} sentences

        Provided are the highest rated sentences for each category for your reference
        env - {payload["env_sentences"]}
        soc - {payload["social_sentences"]}
        gov - {payload["gov_sentences"]}

        ### Instructions:
        - Write in a formal tone suitable for board-level stakeholders.
        - Begin with an executive summary.
        - Then analyze each ESG category in a dedicated section.
        - Highlight strengths, risks, and improvement opportunities in each area.
        - Base your analysis on the sample sentences and scores provided.
        - Conclude with actionable recommendations and a final ESG risk summary.

        Ensure that the content is original, data-driven, and well-organized.

    """
    response = client.models.generate_content(
        model="gemini-2.0-flash",
        contents=prompt
    )
    with open(output_json, "w") as f:
        f.write(response.text)
    print(f"Gemini insights saved to {output_json}")

def main(pdf_file_path: str, company):
    # Extract ESG content
    print("Extracting ESG-related content from PDF...")
    extractor = ESGPDFExtractor()
    extraction_output = f"extractions/esg_extracted_data_for_{company}.json"
    os.makedirs("extractions", exist_ok=True)
    results = extractor.process_pdf(pdf_file_path, output_path=extraction_output)

    # Step 2: Import data into Neo4j
    print("Importing data into Neo4j...")
    graph = neo4j_main(company)
    if not graph:
        return

    # Load GNN model and predict ESG scores
    print("Predicting ESG scores using GNN...")
    dataset = ESGGraphDataset(graph)
    data = dataset.get(0)

    model_path = "models/esg_gnn_multi_model.pt"
    model = ImprovedESGGNN(
        input_dim=data.x.shape[1],
        hidden_dim=128,
        output_dim=3,
        dropout=0.3,
        architecture='multi',
        num_layers=3,
        heads=4,
        use_edge_weights=True
    )
    trainer = ImprovedESGGraphTrainer(model)
    trainer.load_model(model_path)
    doc_id = os.path.basename(pdf_file_path)
    predictions = trainer.predict_document_scores(graph, document_id=doc_id)
    # print(predictions)
    

    # Step 4: Generate Gemini AI insights
    print("Generating Gemini-based AI insights...")
    ai_insight_output = "extractions/esg_ai_insights.txt"
    payload = prepare_gemini_payload("extractions/esg_extracted_data_for_mercedes.json")
    
    generate_ai_insights(extraction_output, ai_insight_output, company, payload)

    print("\nPipeline completed successfully!")

if __name__ == "__main__":
    sample_pdf = "reports/mercedes-benz-sustainability-report-2023.pdf"
    company = (sample_pdf.split('/')[1]).split('-')[0]
    main(sample_pdf, company)
