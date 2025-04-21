"""
Streamlit web application for the BERT-based fake news detector
"""

import streamlit as st
import os
import torch
import matplotlib.pyplot as plt
import numpy as np
from transformers import BertTokenizer

# Import our model
from bert_fake_news_detection import BERTFakeNewsDetector, preprocess_text

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

# Title and introduction
st.title("🔍 Fake News Detection System")
st.markdown("""
This application uses a BERT-based deep learning model to analyze news content and 
determine if it's likely to be fake news or real news.

### How it works:
1. Enter news content in the text area below
2. Click "Analyze" to process the text
3. View the prediction and confidence score
""")

# Load the model
model = load_model()

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

# Footer
st.markdown("---")
st.markdown("### How to use this tool responsibly")
st.info("""
- This tool provides an algorithmic assessment and should not be the sole basis for determining credibility
- Always cross-check information with reputable sources
- Consider the context and source of the news
- Be aware that AI models have limitations and may occasionally misclassify content
""")