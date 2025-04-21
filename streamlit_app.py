"""
Streamlit app with Groq API integration for explanation generation (using .env for API key)
"""

import streamlit as st
import os
import torch
import matplotlib.pyplot as plt
import numpy as np
from dotenv import load_dotenv

# Import your model
from bert_fake_news_detection import BERTFakeNewsDetector, preprocess_text

# Load environment variables from .env file
load_dotenv()

# Set page title and configuration
st.set_page_config(
    page_title="Fake News Detector",
    page_icon="🔍",
    layout="wide"
)

# Define the GroqExplainer class
class GroqExplainer:
    def __init__(self, api_key=None, model="llama3-70b-8192"):
        self.api_key = api_key
        self.model = model
        self.base_url = "https://api.groq.com/openai/v1/chat/completions"
    
    def get_explanation(self, text, classification, confidence):
        import requests
        import json
        
        if not self.api_key:
            return {"error": "No API key provided", "explanation": "API key required to generate explanation."}
        
        # Create messages for the API
        messages = [
            {
                "role": "system",
                "content": "You are an expert in media literacy and fake news detection. Explain why a news text was classified as real or fake."
            },
            {
                "role": "user",
                "content": f"This news text was classified as '{classification}' with {confidence:.1%} confidence:\n\n{text}\n\nExplain why, focusing on specific elements in the text."
            }
        ]
        
        # Make the API request
        try:
            response = requests.post(
                self.base_url,
                headers={
                    "Authorization": f"Bearer {self.api_key}",
                    "Content-Type": "application/json"
                },
                json={
                    "model": self.model,
                    "messages": messages,
                    "temperature": 0.3,
                    "max_tokens": 1024
                }
            )
            response.raise_for_status()
            result = response.json()
            explanation = result["choices"][0]["message"]["content"]
            return {"explanation": explanation, "status": "success"}
        except Exception as e:
            return {"error": str(e), "explanation": "Error generating explanation.", "status": "error"}

@st.cache_resource
def load_model(model_path="bert_fake_news_model"):
    """Load the BERT model (cached to avoid reloading)"""
    if os.path.exists(model_path) and os.path.isfile(os.path.join(model_path, 'model.pt')):
        detector = BERTFakeNewsDetector(model_path=model_path)
        return detector
    else:
        st.error(f"Model not found at {model_path}. Please train the model first.")
        st.info("Run the following command to train the model: \n\n"
                "`python train_bert_model.py --dataset merged_dataset.csv --output bert_fake_news_model --epochs 5`")
        return None

# Title and introduction
st.title("🔍 Fake News Detection System")
st.markdown("""
This application uses a BERT-based deep learning model to analyze news content and 
determine if it's likely to be fake news or real news.

### How it works:
1. Enter news content in the text area below
2. Click "Analyze" to process the text
3. View the prediction, confidence score, and AI-generated explanation
""")

# Load the model
model = load_model()

# Get the Groq API key from environment variables
groq_api_key = os.environ.get("GROQ_API_KEY")

# Initialize the Groq explainer with the API key
groq_explainer = GroqExplainer(api_key=groq_api_key)

# Show a notice about the API key status
if groq_api_key:
    st.success("✅ Groq API key loaded from environment variables")
else:
    st.warning("⚠️ No Groq API key found. AI explanations will not be available. Add GROQ_API_KEY to your .env file.")

# Create a text area for user input
news_text = st.text_area("Enter news content to analyze:", height=200)

# Analyze button
if st.button("Analyze") and model is not None:
    if news_text.strip():
        # Show a spinner while processing
        with st.spinner("Analyzing text..."):
            # Preprocess the text
            processed_text = preprocess_text(news_text)
            
            # Get the prediction
            label, confidence = model.predict(news_text)
            
            # Display the result with a colored box
            st.markdown("### Analysis Result:")
            
            col1, col2 = st.columns([3, 2])
            
            with col1:
                if label == "Fake News":
                    st.error(f"**Prediction: {label}**")
                else:
                    st.success(f"**Prediction: {label}**")
                
                st.markdown(f"**Confidence: {confidence:.2%}**")
                
                # Show the interpretation
                st.markdown("### Interpretation:")
                
                if label == "Fake News":
                    if confidence > 0.9:
                        st.markdown("The model is highly confident this is fake news. The content likely contains "
                                    "multiple features commonly associated with misinformation.")
                    elif confidence > 0.75:
                        st.markdown("The model indicates this is likely fake news, but with moderate confidence. "
                                    "The content may contain some misleading elements.")
                    else:
                        st.markdown("The model classifies this as fake news, but with lower confidence. "
                                    "Further verification is recommended.")
                else:
                    if confidence > 0.9:
                        st.markdown("The model is highly confident this is legitimate news. The content "
                                    "appears to follow typical patterns of genuine reporting.")
                    elif confidence > 0.75:
                        st.markdown("The model indicates this is likely real news, with moderate confidence.")
                    else:
                        st.markdown("The model classifies this as real news, but with lower confidence. "
                                    "Further verification may be advisable.")
            
            with col2:
                # Create a gauge chart to visualize the confidence
                fig, ax = plt.subplots(figsize=(3, 3))
                
                # Plot settings
                gauge_colors = ["#ff4b4b", "#ffa64b", "#4bb543"] if label == "Fake News" else ["#4bb543", "#ffa64b", "#ff4b4b"]
                
                # Draw the gauge
                ax.pie(
                    [confidence, 1-confidence],
                    colors=[gauge_colors[0] if confidence > 0.5 else gauge_colors[2], "#f0f0f0"],
                    startangle=90,
                    counterclock=False,
                    wedgeprops={"width": 0.4, "edgecolor": "w"}
                )
                ax.add_artist(plt.Circle((0, 0), 0.3, fc='white'))
                
                # Add text in the middle
                conf_text = f"{confidence:.0%}"
                ax.text(0, 0, conf_text, ha='center', va='center', fontsize=16, fontweight='bold')
                
                # Set title based on prediction
                title = "Fake News Likelihood" if label == "Fake News" else "Real News Likelihood"
                ax.set_title(title)
                
                st.pyplot(fig)
            
            # Generate explanation using Groq API
            if groq_api_key:
                with st.spinner("Generating AI explanation..."):
                    explanation_result = groq_explainer.get_explanation(news_text, label, confidence)
                    
                    if explanation_result.get("status") == "success":
                        st.markdown("### AI-Generated Explanation")
                        st.markdown(explanation_result["explanation"])
                    else:
                        st.warning(f"Could not generate explanation: {explanation_result.get('error', 'Unknown error')}")
            
            # Show the preprocessed text
            with st.expander("View preprocessed text"):
                st.text(processed_text)
    else:
        st.warning("Please enter some text to analyze.")

# Footer
st.markdown("---")
st.markdown("### How to use this tool responsibly")
st.info("""
- This tool provides an algorithmic assessment and should not be the sole basis for determining credibility
- Always cross-check information with reputable sources
- Consider the context and source of the news
- Be aware that AI models have limitations and may occasionally misclassify content
""")