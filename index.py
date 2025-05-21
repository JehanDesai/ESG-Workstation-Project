import streamlit as st
import os
import json
import plotly.graph_objects as go
import plotly.express as px
from datetime import datetime

# Import the main pipeline functions
from sustainability_report_extraction import ESGPDFExtractor
from neo4j_data_importer import neo4j_main
from improved_ESG_prediction_model import ImprovedESGGraphTrainer, ImprovedESGGNN, ESGGraphDataset
from dotenv import load_dotenv
from google import genai
import random

# Load environment variables
load_dotenv()


# Page configuration
st.set_page_config(
    page_title="ESG Risk Assessment & Reporting",
    page_icon="🌱",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS to improve the UI
st.markdown("""
<style>
    .main-header {
        font-size: 2.5rem !important;
        font-weight: 700 !important;
        color: #0e566c !important;
        margin-bottom: 1rem !important;
    }
    .sub-header {
        font-size: 1.5rem !important;
        font-weight: 600 !important;
        color: #158c74 !important;
        margin-top: 1rem !important;
        margin-bottom: 0.5rem !important;
    }
    .card-container {
        background-color: #f8f9fa;
        border-radius: 10px;
        padding: 1.5rem;
        box-shadow: 0 0.125rem 0.25rem rgba(0, 0, 0, 0.075);
        margin-bottom: 1rem;
    }
    .metric-container {
        background-color: #ffffff;
        border-radius: 8px;
        padding: 1rem;
        box-shadow: 0 0.125rem 0.25rem rgba(0, 0, 0, 0.075);
        text-align: center;
        transition: transform 0.3s ease;
    }
    .metric-container:hover {
        transform: translateY(-5px);
        box-shadow: 0 0.5rem 1rem rgba(0, 0, 0, 0.15);
    }
    .metric-value {
        font-size: 2rem;
        font-weight: 700;
        margin-bottom: 0.5rem;
    }
    .metric-label {
        font-size: 0.9rem;
        font-weight: 600;
        color: #6c757d;
    }
    .step-container {
        border-left: 3px solid #158c74;
        padding-left: 1rem;
        margin-bottom: 1rem;
    }
    .step-title {
        font-weight: 600;
        color: #158c74;
    }
    .footer {
        margin-top: 3rem;
        text-align: center;
        color: #6c757d;
    }
    .success-message {
        color: #28a745;
        font-weight: 600;
    }
    .warning-message {
        color: #ffc107;
        font-weight: 600;
    }
    .error-message {
        color: #dc3545;
        font-weight: 600;
    }
</style>
""", unsafe_allow_html=True)

# Helper functions
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
    try:
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
        st.success(f"Gemini insights saved to {output_json}")
        return response.text
    except Exception as e:
        st.error(f"Error generating insights: {str(e)}")
        return None

def process_report(uploaded_file, company_name):
    # Create necessary directories
    os.makedirs("reports", exist_ok=True)
    os.makedirs("extractions", exist_ok=True)
    os.makedirs("models", exist_ok=True)
    
    # Save uploaded file
    file_path = f"reports/{uploaded_file.name}"
    with open(file_path, "wb") as f:
        f.write(uploaded_file.getbuffer())
    
    # Step 1: Extract ESG content with progress bar
    progress_bar = st.progress(0)
    status_text = st.empty()
    
    status_text.text("Step 1/4: Extracting ESG-related content from PDF...")
    extractor = ESGPDFExtractor()
    extraction_output = f"extractions/esg_extracted_data_for_{company_name}.json"
    results = extractor.process_pdf(file_path, output_path=extraction_output)
    progress_bar.progress(25)
    
    # Step 2: Import data into Neo4j
    status_text.text("Step 2/4: Importing data into Neo4j...")
    graph = neo4j_main(company_name)
    if not graph:
        st.error("Failed to connect to Neo4j database")
        return None
    progress_bar.progress(50)
    
    # Step 3: Load GNN model and predict ESG scores
    status_text.text("Step 3/4: Predicting ESG scores using GNN...")
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
    doc_id = os.path.basename(file_path)
    predictions = trainer.predict_document_scores(graph, document_id=doc_id)
    progress_bar.progress(75)
    
    # Step 4: Generate Gemini AI insights
    status_text.text("Step 4/4: Generating Gemini-based AI insights...")
    ai_insight_output = f"extractions/esg_ai_insights_{company_name}.txt"
    payload = prepare_gemini_payload(extraction_output)
    
    insights = generate_ai_insights(extraction_output, ai_insight_output, company_name, payload)
    progress_bar.progress(100)
    status_text.text("ESG analysis completed successfully!")
    
    # Return results
    return {
        "predictions": predictions,
        "extraction_data": json.load(open(extraction_output)),
        "insights": insights,
        "payload": payload
    }

def display_esg_scores(payload):
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.markdown('<div class="metric-container">', unsafe_allow_html=True)
        st.markdown(f'<div class="metric-value" style="color: #198754;">{payload["environmental_score"]}</div>', unsafe_allow_html=True)
        st.markdown('<div class="metric-label">Environmental Score</div>', unsafe_allow_html=True)
        st.markdown(f'<div>Based on {payload["environmental_count"]} sentences</div>', unsafe_allow_html=True)
        st.markdown('</div>', unsafe_allow_html=True)
        
    with col2:
        st.markdown('<div class="metric-container">', unsafe_allow_html=True)
        st.markdown(f'<div class="metric-value" style="color: #0d6efd;">{payload["social_score"]}</div>', unsafe_allow_html=True)
        st.markdown('<div class="metric-label">Social Score</div>', unsafe_allow_html=True)
        st.markdown(f'<div>Based on {payload["social_count"]} sentences</div>', unsafe_allow_html=True)
        st.markdown('</div>', unsafe_allow_html=True)
        
    with col3:
        st.markdown('<div class="metric-container">', unsafe_allow_html=True)
        st.markdown(f'<div class="metric-value" style="color: #6c757d;">{payload["governance_score"]}</div>', unsafe_allow_html=True)
        st.markdown('<div class="metric-label">Governance Score</div>', unsafe_allow_html=True)
        st.markdown(f'<div>Based on {payload["governance_count"]} sentences</div>', unsafe_allow_html=True)
        st.markdown('</div>', unsafe_allow_html=True)

def create_radar_chart(payload):
    categories = ['Environmental', 'Social', 'Governance']
    values = [payload["environmental_score"], payload["social_score"], payload["governance_score"]]
    
    # Duplicate the first value to close the circle
    categories = categories + [categories[0]]
    values = values + [values[0]]
    
    fig = go.Figure()
    
    fig.add_trace(go.Scatterpolar(
        r=values,
        theta=categories,
        fill='toself',
        name='ESG Scores',
        line_color='rgb(31, 119, 180)',
        fillcolor='rgba(31, 119, 180, 0.5)'
    ))
    
    fig.update_layout(
        polar=dict(
            radialaxis=dict(
                visible=True,
                range=[0, 1]
            )
        ),
        showlegend=False,
        height=400
    )
    
    return fig

def create_sentence_distribution_chart(payload):
    categories = ['Environmental', 'Social', 'Governance']
    counts = [payload["environmental_count"], payload["social_count"], payload["governance_count"]]
    
    fig = px.bar(
        x=categories,
        y=counts,
        color=categories,
        color_discrete_map={
            'Environmental': '#198754',
            'Social': '#0d6efd',
            'Governance': '#6c757d'
        },
        labels={'x': 'ESG Category', 'y': 'Number of Sentences'}
    )
    
    fig.update_layout(
        xaxis_title='ESG Category',
        yaxis_title='Number of Sentences',
        showlegend=False,
        height=400
    )
    
    return fig

def sample_sentences_display(payload):
    st.markdown('<div class="sub-header">Sample Sentences by Category</div>', unsafe_allow_html=True)
    
    # Environmental sentences
    with st.expander("Environmental Sentences", expanded=False):
        for i, sentence in enumerate(payload["env_sentences"]):
            st.markdown(f"**{i+1}.** {sentence}")
    
    # Social sentences
    with st.expander("Social Sentences", expanded=False):
        for i, sentence in enumerate(payload["social_sentences"]):
            st.markdown(f"**{i+1}.** {sentence}")
    
    # Governance sentences
    with st.expander("Governance Sentences", expanded=False):
        for i, sentence in enumerate(payload["gov_sentences"]):
            st.markdown(f"**{i+1}.** {sentence}")

# Sidebar
st.sidebar.image("kpmg.png", use_container_width=True)
st.sidebar.markdown("# ESG AI Analysis")
st.sidebar.markdown("---")

page = st.sidebar.radio("Navigation", [ "Home", "Upload Report", "Previous Report"])

# Main content
if page == "Previous Report":
    st.markdown('<div class="main-header">ESG Risk Assessment Reports</div>', unsafe_allow_html=True)
    
    st.markdown('<div class="card-container">', unsafe_allow_html=True)
    st.markdown("""
    This dashboard provides a comprehensive view of your ESG (Environmental, Social, and Governance) 
    risk assessment based on the reports you've uploaded and analyzed.
    
    Use the sidebar navigation to upload new reports or learn more about the ESG AI analysis system.
    """)
    st.markdown('</div>', unsafe_allow_html=True)
    
    # Check if any reports have been analyzed
    if not os.path.exists("extractions") or not os.listdir("extractions"):
        st.info("No reports have been analyzed yet. Go to 'Upload Report' to analyze your first ESG report.")
    else:
        # Display the most recent analysis
        json_files = [f for f in os.listdir("extractions") if f.startswith("esg_extracted_data_for_") and f.endswith(".json")]
        
        if json_files:
            # Get the most recent file
            latest_file = max(json_files, key=lambda x: os.path.getmtime(os.path.join("extractions", x)))
            company_name = latest_file.replace("esg_extracted_data_for_", "").replace(".json", "")
            
            col1, col2 = st.columns([2, 1])
            with col1:
                st.markdown(f'<div class="sub-header">Latest Analysis: {company_name.title()}</div>', unsafe_allow_html=True)
            with col2:
                st.markdown(f'<div style="text-align: right;">Analyzed on: {datetime.fromtimestamp(os.path.getmtime(os.path.join("extractions", latest_file))).strftime("%Y-%m-%d %H:%M")}</div>', unsafe_allow_html=True)
            
            # Load the data
            with open(os.path.join("extractions", latest_file), "r", encoding="utf-8") as f:
                data = json.load(f)
            
            payload = prepare_gemini_payload(os.path.join("extractions", latest_file))
            
            # Display ESG scores
            display_esg_scores(payload)
            
            # Display charts
            col1, col2 = st.columns(2)
            
            with col1:
                st.markdown('<div class="sub-header">ESG Score Radar</div>', unsafe_allow_html=True)
                radar_chart = create_radar_chart(payload)
                st.plotly_chart(radar_chart, use_container_width=True)
            
            with col2:
                st.markdown('<div class="sub-header">Sentence Distribution</div>', unsafe_allow_html=True)
                bar_chart = create_sentence_distribution_chart(payload)
                st.plotly_chart(bar_chart, use_container_width=True)
            
            # Display sample sentences
            sample_sentences_display(payload)
            
            # Display AI insights
            insight_file = f"esg_ai_insights_{company_name}.txt"
            if os.path.exists(os.path.join("extractions", insight_file)):
                st.markdown('<div class="sub-header">AI-Generated ESG Insights</div>', unsafe_allow_html=True)
                with open(os.path.join("extractions", insight_file), "r", encoding="utf-8") as f:
                    insights = f.read()
                st.markdown(insights)
                
                # Download button for the insights
                st.download_button(
                    label="Download ESG Report",
                    data=insights,
                    file_name=f"{company_name}_esg_report.md",
                    mime="text/markdown"
                )
        else:
            st.info("No reports have been analyzed yet. Go to 'Upload Report' to analyze your first ESG report.")

elif page == "Upload Report":
    st.markdown('<div class="main-header">Upload ESG Report</div>', unsafe_allow_html=True)
    
    st.markdown('<div class="card-container">', unsafe_allow_html=True)
    st.markdown("""
    Upload a sustainability or annual report in PDF format to analyze ESG performance. 
    The system will extract relevant ESG content, process it through the Neo4j graph database, 
    predict ESG scores using a Graph Neural Network, and generate insights with Gemini AI.
    """)
    st.markdown('</div>', unsafe_allow_html=True)
    
    col1, col2 = st.columns([2, 1])
    
    with col1:
        uploaded_file = st.file_uploader("Choose a PDF report", type="pdf")
    
    with col2:
        company_name = st.text_input("Company Name", help="Enter the name of the company for the report you're uploading")
    
    if uploaded_file is not None and company_name:
        st.markdown('<div class="step-container">', unsafe_allow_html=True)
        st.markdown('<div class="step-title">Processing Pipeline</div>', unsafe_allow_html=True)
        st.markdown("The system will execute the following steps:")
        st.markdown("1. Extract ESG content from the PDF")
        st.markdown("2. Import data into Neo4j graph database")
        st.markdown("3. Predict ESG scores using GNN model")
        st.markdown("4. Generate AI insights")
        st.markdown('</div>', unsafe_allow_html=True)
        
        if st.button("Start ESG Analysis", key="start_analysis"):
            with st.spinner("Processing report..."):
                results = process_report(uploaded_file, company_name)
                
                if results:
                    st.success("Analysis completed successfully! View the results in the Dashboard.")
                    st.balloons()
                    
                    # Redirect to dashboard
                    st.markdown("""
                    <script>
                        var elements = window.parent.document.querySelectorAll('.stRadio div[role="radiogroup"] label');
                        for (var i = 0; i < elements.length; i++) {
                            if (elements[i].innerText.includes('Dashboard')) {
                                elements[i].click();
                            }
                        }
                    </script>
                    """, unsafe_allow_html=True)
    else:
        if not uploaded_file:
            st.info("Please upload a PDF file to analyze.")
        if not company_name:
            st.info("Please enter the company name.")

elif page == "Home":
    st.markdown('<div class="main-header">About ESG AI Analysis</div>', unsafe_allow_html=True)
    
    st.markdown('<div class="card-container">', unsafe_allow_html=True)
    st.markdown("""
    ## ESG Risk Assessment & Reporting
    
    This application is designed to automate and enhance ESG (Environmental, Social, Governance) risk assessment and reporting by leveraging advanced AI technologies:
    
    ### Key Features:
    
    1. **Automated ESG Data Extraction**: Extract and identify ESG-relevant information from sustainability reports, annual reports, regulatory filings, and other documents.
    
    2. **Graph-Based Analysis**: Utilize Neo4j graph database to model complex relationships between ESG factors, regulations, and company performance.
    
    3. **AI-Powered Risk Prediction**: Employ Graph Neural Networks (GNNs) to predict ESG risks and performance scores across multiple dimensions.
    
    4. **Regulatory Framework Mapping**: Automatically map findings to relevant regulatory frameworks and standards worldwide.
    
    5. **AI-Generated Insights**: Generate comprehensive ESG impact reports with actionable insights using Google's Gemini AI.
    
    ### Technology Stack:
    
    - **Data Processing**: Python, LangChain
    - **AI/ML**: PyTorch, Graph Neural Networks
    - **Storage & Querying**: Neo4j Graph Database
    - **LLM Integration**: Google Gemini API
    - **Interface**: Streamlit
    
    ### Getting Started:
    
    Upload a sustainability report in PDF format through the "Upload Report" section and let the AI handle the rest!
    """)
    st.markdown('</div>', unsafe_allow_html=True)
    
    # Display tech stack
    st.markdown('<div class="sub-header">Technology Stack</div>', unsafe_allow_html=True)
    
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.markdown('<div class="metric-container">', unsafe_allow_html=True)
        st.markdown('<div style="font-size: 2rem;">📊</div>', unsafe_allow_html=True)
        st.markdown('<div class="metric-label">Neo4j</div>', unsafe_allow_html=True)
        st.markdown('</div>', unsafe_allow_html=True)
    
    with col2:
        st.markdown('<div class="metric-container">', unsafe_allow_html=True)
        st.markdown('<div style="font-size: 2rem;">🧠</div>', unsafe_allow_html=True)
        st.markdown('<div class="metric-label">PyTorch</div>', unsafe_allow_html=True)
        st.markdown('</div>', unsafe_allow_html=True)
    
    with col3:
        st.markdown('<div class="metric-container">', unsafe_allow_html=True)
        st.markdown('<div style="font-size: 2rem;">🔗</div>', unsafe_allow_html=True)
        st.markdown('<div class="metric-label">LangChain</div>', unsafe_allow_html=True)
        st.markdown('</div>', unsafe_allow_html=True)
    
    with col4:
        st.markdown('<div class="metric-container">', unsafe_allow_html=True)
        st.markdown('<div style="font-size: 2rem;">💬</div>', unsafe_allow_html=True)
        st.markdown('<div class="metric-label">Gemini AI</div>', unsafe_allow_html=True)
        st.markdown('</div>', unsafe_allow_html=True)

# Footer
st.markdown('<div class="footer">© 2025 ESG AI Analysis Tool | Built with Streamlit</div>', unsafe_allow_html=True)