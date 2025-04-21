"""
Modified Streamlit app with Groq API integration for explanation generation
"""

import streamlit as st
import os
import torch
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
import pandas as pd
import json
import time
from sklearn.metrics import confusion_matrix

# Import our models
from bert_fake_news_detection import BERTFakeNewsDetector, preprocess_text
# Import the Groq explainer
from groq_integration import GroqExplainer

# Set page title and description
st.set_page_config(
    page_title="Fake News Detector",
    page_icon="🔍",
    layout="wide"
)

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

@st.cache_resource
def load_groq_explainer():
    """Load the Groq explainer (cached to avoid reloading)"""
    # Try to get API key from environment or session state
    api_key = os.environ.get("GROQ_API_KEY") or st.session_state.get("groq_api_key", "")
    
    # Initialize explainer
    explainer = GroqExplainer(api_key=api_key)
    return explainer

# Title and introduction
st.title("🔍 Fake News Detection System")
st.markdown("""
This application uses a BERT-based deep learning model to analyze news content and 
determine if it's likely to be fake news or real news. It also provides AI-generated 
explanations using the Groq API to help understand the classification.

### How it works:
1. Enter news content in the text area below
2. Click "Analyze" to process the text
3. View the prediction, confidence score, and AI-generated explanation
""")

# Sidebar for API configuration
st.sidebar.title("Groq API Configuration")
groq_api_key = st.sidebar.text_input("Groq API Key", 
                                    value=os.environ.get("GROQ_API_KEY", ""),
                                    type="password",
                                    help="Enter your Groq API key to enable explanations")

# Store the API key in session state
if groq_api_key:
    st.session_state.groq_api_key = groq_api_key

# Model selection
model_type = st.sidebar.selectbox("Model Type", 
                                ["BERT", "Advanced BERT (if available)"],
                                help="Select which model to use for classification")

# Load the model
model = load_model()

# Load the Groq explainer
groq_explainer = load_groq_explainer()

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
            
            # Create two columns for layout
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
            
            # Try to extract feature importance (if available)
            features = None
            try:
                # This will only work if you've implemented feature importance in your model
                if hasattr(model, 'get_feature_importance'):
                    features = model.get_feature_importance(news_text)
            except:
                pass
            
            # Generate explanation using Groq API
            explanation_container = st.container()
            with explanation_container:
                st.markdown("### AI-Generated Explanation")
                
                if groq_api_key:
                    try:
                        with st.spinner("Generating explanation with Groq..."):
                            # Get explanation from Groq
                            explanation_result = groq_explainer.get_explanation(
                                news_text, label, confidence, features
                            )
                            
                            if explanation_result.get("status") == "success":
                                st.markdown(explanation_result["explanation"])
                                st.caption(f"Explanation generated by {explanation_result['model_used']}")
                            else:
                                st.warning(f"Could not generate explanation: {explanation_result.get('error', 'Unknown error')}")
                    except Exception as e:
                        st.warning(f"Error generating explanation: {str(e)}")
                else:
                    st.info("To get AI-generated explanations, please provide a Groq API key in the sidebar.")
                    st.markdown("Sample explanation format:")
                    
                    if label == "Fake News":
                        st.markdown("""
                        This text was likely classified as fake news due to several common patterns:
                        
                        1. **Sensationalist language**: Terms like "breaking", "shocking", and excessive exclamation points
                        2. **Extraordinary claims**: Claims that would be major news yet aren't reported by reputable sources
                        3. **Lack of sources**: No citations or attribution to verifiable sources
                        4. **Emotional manipulation**: Language designed to trigger emotional reactions rather than inform
                        
                        To verify this information, you should check if reputable news sources are reporting similar stories, 
                        look for cited sources within the article, and be wary of content that seems designed primarily to 
                        provoke strong emotional reactions rather than to inform.
                        """)
                    else:
                        st.markdown("""
                        This text was likely classified as real news due to several indicators:
                        
                        1. **Neutral language**: Measured tone without excessive sensationalism
                        2. **Plausible claims**: Contains claims that are within the realm of normal events
                        3. **Structural elements**: Follows patterns of journalistic writing with facts presented clearly
                        4. **Lack of emotional manipulation**: Focuses on informing rather than provoking reactions
                        
                        While the model classified this as real news, it's always good practice to verify information 
                        from multiple sources, especially for important topics.
                        """)
            
            # Show the preprocessed text
            with st.expander("View preprocessed text"):
                st.text(processed_text)
            
    else:
        st.warning("Please enter some text to analyze.")

# Display information about the model in an expander
with st.expander("About the Model"):
    st.markdown("""
    ### BERT-based Fake News Detection
    
    This system uses a fine-tuned BERT (Bidirectional Encoder Representations from Transformers) model to detect fake news.
    
    **Key features:**
    - Built on pretrained BERT architecture
    - Fine-tuned on a dataset of real and fake news articles
    - Uses natural language processing to analyze content patterns
    - Improved with regularization to prevent overfitting
    - Uses data augmentation to enhance model robustness
    
    **Limitations:**
    - The model analyzes text patterns and cannot fact-check specific claims
    - Performance depends on the training data and may not catch novel misinformation tactics
    - Should be used as a tool to assist human judgment, not replace it
    """)

# Add information about the Groq API integration
with st.expander("About Groq API Integration"):
    st.markdown("""
    ### Groq API for Explainability
    
    This application uses the Groq API to generate explanations for why text was classified as fake or real news.
    
    **How it works:**
    - After the BERT model classifies the text, we send the text, classification, and confidence to Groq's API
    - The API uses advanced language models (like LLaMA 3) to analyze the text and generate an explanation
    - The explanation highlights specific elements in the text that may have contributed to the classification
    - It also provides tips on how to verify the information independently
    
    **Setting up Groq API:**
    1. Sign up for an account at [groq.com](https://console.groq.com/signup)
    2. Generate an API key from your Groq dashboard
    3. Enter the API key in the sidebar of this application
    
    **API Request Structure:**
    ```python
    headers = {
        "Authorization": f"Bearer {api_key}",
        "Content-Type": "application/json"
    }
    
    payload = {
        "model": "llama3-70b-8192",
        "messages": [
            {
                "role": "system", 
                "content": "You are an expert in media literacy and fake news detection..."
            },
            {
                "role": "user", 
                "content": f"The following news text was classified as '{classification}'..."
            }
        ],
        "temperature": 0.3,
        "max_tokens": 1024
    }
    
    response = requests.post("https://api.groq.com/openai/v1/chat/completions", 
                           headers=headers, json=payload)
    ```
    """)

# Footer
st.markdown("---")
st.markdown("### How to use this tool responsibly")
st.info("""
- This tool provides an algorithmic assessment and should not be the sole basis for determining credibility
- Always cross-check information with reputable sources
- Consider the context and source of the news
- Be aware that AI models have limitations and may occasionally misclassify content
- The AI-generated explanations are meant to assist understanding, not to be definitive analyses
""")