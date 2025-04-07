import os
import json
import logging
from typing import Dict, Any

import pandas as pd
import openai
from langchain.chat_models import ChatOpenAI
from langchain.prompts import PromptTemplate
from langchain.chains import LLMChain
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Configure logging
logging.basicConfig(level=logging.INFO, 
                    format='%(asctime)s - %(levelname)s: %(message)s')
logger = logging.getLogger(__name__)

class ESGInsightsReportingAgent:
    def __init__(self, model_dir: str = 'esg_risk_models', report_dir: str = 'esg_reports'):
        """
        Initialize ESG Insights and Reporting Agent
        
        Args:
            model_dir: Directory containing trained models
            report_dir: Directory to save generated reports
        """
        # Ensure API key is set
        openai.api_key = os.getenv('OPENAI_API_KEY')
        if not openai.api_key:
            raise ValueError("OpenAI API key not found. Please set OPENAI_API_KEY in .env file")
        # Create report directory
        os.makedirs(report_dir, exist_ok=True)
        self.report_dir = report_dir
        # Initialize language model
        self.llm = ChatOpenAI(model='gpt-4-turbo',temperature=0.3,max_tokens=4096)
    
    def load_model_predictions(self, model_file: str) -> Dict[str, Any]:
        """
        Load predictions from trained ESG risk model
        
        Args:
            model_file: Path to model predictions file
        
        Returns:
            Dictionary of model predictions
        """
        try:
            with open(model_file, 'r') as f:
                predictions = json.load(f)
            return predictions
        except Exception as e:
            logger.error(f"Error loading model predictions: {e}")
            return {}
    
    def generate_narrative_insights(self, model_predictions: Dict[str, Any]) -> str:
        """
        Generate narrative insights using LangChain and OpenAI
        
        Args:
            model_predictions: Dictionary of ESG model predictions
        
        Returns:
            Narrative report text
        """
        # Create prompt template for ESG insights
        narrative_prompt = PromptTemplate(
            input_variables=['environmental_risk', 'social_risk', 'governance_risk', 'risk_details'],
            template="""
            You are an expert ESG analyst creating a comprehensive sustainability report.

            ESG Risk Assessment Overview:
            - Environmental Risk: {environmental_risk}
            - Social Risk: {social_risk}
            - Governance Risk: {governance_risk}

            Detailed Risk Analysis:
            {risk_details}

            Using these insights, generate a detailed, professional ESG report that:
            1. Provides context for each risk category
            2. Highlights potential impact on business strategy
            3. Recommends mitigation strategies
            4. Frames risks in the context of global sustainability trends
            5. Uses data-driven, actionable language

            The report should be suitable for board-level presentation and investor communications.
            """
        )
        # Create LLM chain
        narrative_chain = LLMChain(llm=self.llm, prompt=narrative_prompt)
        # Generate narrative
        narrative = narrative_chain.run(
            environmental_risk=model_predictions.get('environmental_risk', 'Moderate'),
            social_risk=model_predictions.get('social_risk', 'Low'),
            governance_risk=model_predictions.get('governance_risk', 'Moderate'),
            risk_details=json.dumps(model_predictions, indent=2)
        )
        return narrative
    
    def generate_regulatory_mapping(self, country: str) -> Dict[str, Any]:
        """
        Generate detailed regulatory framework mapping
        
        Args:
            country: Target country for regulatory analysis
        
        Returns:
            Dictionary of regulatory frameworks
        """
        regulatory_frameworks = {
            'EU': {
                'decarbonization_targets': {
                    'overall_target': '55% reduction by 2030',
                    'key_sectors': ['Energy', 'Manufacturing', 'Transportation']
                },
                'key_regulations': [
                    'Corporate Sustainability Reporting Directive (CSRD)',
                    'EU Taxonomy Regulation',
                    'Sustainable Finance Disclosure Regulation (SFDR)'
                ],
                'carbon_pricing': 'Emissions Trading System (ETS)',
                'compliance_recommendations': [
                    'Implement comprehensive sustainability reporting',
                    'Align business models with taxonomy criteria',
                    'Develop transparent carbon accounting'
                ]
            },
            'USA': {
                'decarbonization_targets': {
                    'overall_target': '50-52% reduction by 2030',
                    'key_sectors': ['Energy', 'Industry', 'Transportation']
                },
                'key_regulations': [
                    'SEC Climate-related Disclosure Rules',
                    'EPA Clean Energy Regulations',
                    'State-level carbon market initiatives'
                ],
                'carbon_pricing': 'State and regional carbon markets',
                'compliance_recommendations': [
                    'Prepare for enhanced climate risk disclosures',
                    'Develop robust emissions tracking',
                    'Invest in clean energy transitions'
                ]
            }
        }
        return regulatory_frameworks.get(country, {})
    
    def generate_comprehensive_report(self, model_predictions: Dict[str, Any], company_data: Dict[str, Any]):
        """
        Generate comprehensive ESG report
        
        Args:
            model_predictions: ESG risk model predictions
            company_data: Company-specific contextual data
        """
        # Generate narrative insights
        narrative_insights = self.generate_narrative_insights(model_predictions)
        # Generate regulatory mapping
        regulatory_mapping = self.generate_regulatory_mapping(company_data.get('country', 'EU'))
        # Compile full report
        full_report = {
            'company': company_data.get('name', 'Unnamed Company'),
            'date': pd.Timestamp.now().strftime('%Y-%m-%d'),
            'narrative_insights': narrative_insights,
            'risk_predictions': model_predictions,
            'regulatory_framework': regulatory_mapping
        }
        # Save report
        report_filename = os.path.join(self.report_dir, f"esg_report_{company_data.get('name', 'company')}_{pd.Timestamp.now().strftime('%Y%m%d')}.json")
        with open(report_filename, 'w') as f:
            json.dump(full_report, f, indent=2)
        logger.info(f"ESG Report generated: {report_filename}")
        return full_report

def main():
    # Initialize ESG Reporting Agent
    reporting_agent = ESGInsightsReportingAgent()
    # Example company data
    sample_company_data = {'name': 'GlobalTech Innovations','country': 'EU','sector': 'Technology'}
    # Example model predictions
    sample_predictions = {'environmental_risk': 0.65,'social_risk': 0.45,'governance_risk': 0.55,'total_carbon_emissions': 250000,'sustainability_score': 0.72}
    try:
        # Generate comprehensive report
        report = reporting_agent.generate_comprehensive_report(sample_predictions, sample_company_data)
        # Print key insights
        print("ESG Report Generated Successfully")
        print("\nNarrative Insights:")
        print(report['narrative_insights'][:500] + "...")  # Print first 500 chars
    except Exception as e:
        logger.error(f"Report generation failed: {e}")

if __name__ == "__main__":
    main()