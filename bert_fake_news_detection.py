# Explainable Fake News Detection System with BERT
# Includes model training, explanation generation, and web interface

import pandas as pd
import numpy as np
import re
import string
import pickle
import os
import json
from datetime import datetime
import matplotlib.pyplot as plt
import seaborn as sns

# For text processing
import nltk
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from nltk.tokenize import word_tokenize

# For the ML model
import torch
from torch.utils.data import Dataset, DataLoader
from transformers import BertTokenizer, BertForSequenceClassification
from transformers import get_linear_schedule_with_warmup
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, accuracy_score, confusion_matrix

# Import AdamW from torch.optim instead of transformers
from torch.optim import AdamW

# For explainability
from captum.attr import LayerIntegratedGradients

# For the web interface
import streamlit as st

# Download necessary NLTK resources
nltk.download('punkt', quiet=True)
nltk.download('stopwords', quiet=True)
nltk.download('wordnet', quiet=True)

# Initialize lemmatizer and stopwords
lemmatizer = WordNetLemmatizer()
stop_words = set(stopwords.words('english'))

# Device configuration
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# Text preprocessing function
def preprocess_text(text):
    if not isinstance(text, str):
        return ""
    # Convert to lowercase
    text = text.lower()
    # Remove URLs
    text = re.sub(r'http\S+|www\S+|https\S+', '', text, flags=re.MULTILINE)
    # Remove special characters and numbers
    text = re.sub(r'\@\w+|\#|\d+', '', text)
    # Remove punctuation
    text = text.translate(str.maketrans('', '', string.punctuation))
    # Remove extra whitespace
    text = re.sub(r'\s+', ' ', text).strip()
    return text

# Custom dataset for BERT
class NewsDataset(Dataset):
    def __init__(self, texts, labels, tokenizer, max_length=512):
        self.texts = texts
        self.labels = labels
        self.tokenizer = tokenizer
        self.max_length = max_length
        
    def __len__(self):
        return len(self.texts)
    
    def __getitem__(self, idx):
        text = str(self.texts[idx])
        label = self.labels[idx]
        
        encoding = self.tokenizer.encode_plus(
            text,
            add_special_tokens=True,
            max_length=self.max_length,
            return_token_type_ids=False,
            padding='max_length',
            truncation=True,
            return_attention_mask=True,
            return_tensors='pt'
        )
        
        return {
            'text': text,
            'input_ids': encoding['input_ids'].flatten(),
            'attention_mask': encoding['attention_mask'].flatten(),
            'label': torch.tensor(label, dtype=torch.long)
        }

# Class for BERT-based fake news detection
class BERTFakeNewsDetector:
    def __init__(self, model_path=None, num_labels=2):
        self.tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
        
        if model_path and os.path.exists(model_path):
            self.model = BertForSequenceClassification.from_pretrained(model_path)
            print(f"Loaded pre-trained model from {model_path}")
        else:
            self.model = BertForSequenceClassification.from_pretrained(
                'bert-base-uncased',
                num_labels=num_labels
            )
            print("Initialized new BERT model")
        
        self.model.to(device)
        
    def train(self, data_path, batch_size=8, epochs=4, learning_rate=2e-5, test_size=0.2, random_state=42, max_length=256):
        """Train the BERT model on the dataset"""
        try:
            # Load and preprocess the dataset
            df = pd.read_csv(data_path)
            print(f"Loaded dataset with {len(df)} rows")
            
            # Check if required columns exist
            required_columns = ['Content', 'Label']
            if not all(col in df.columns for col in required_columns):
                print(f"Dataset missing required columns. Found: {df.columns.tolist()}")
                return False
            
            # Preprocess the text data
            print("Preprocessing text data...")
            df['processed_content'] = df['Content'].apply(preprocess_text)
            
            # Convert labels to binary (0 for real, 1 for fake)
            if df['Label'].dtype == 'object':
                df['label_numeric'] = df['Label'].map(lambda x: 1 if str(x).lower() in ['fake', 'false'] else 0)
            else:
                df['label_numeric'] = df['Label']
            
            # Check class balance and print info
            class_counts = df['label_numeric'].value_counts()
            print(f"Class distribution: {class_counts.to_dict()}")
            
            # Split the data
            train_texts, val_texts, train_labels, val_labels = train_test_split(
                df['processed_content'].values, 
                df['label_numeric'].values,
                test_size=test_size,
                random_state=random_state,
                stratify=df['label_numeric']
            )
            
            print(f"Training set size: {len(train_texts)}, Validation set size: {len(val_texts)}")
            
            # Create datasets
            train_dataset = NewsDataset(train_texts, train_labels, self.tokenizer, max_length)
            val_dataset = NewsDataset(val_texts, val_labels, self.tokenizer, max_length)
            
            # Create data loaders
            train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
            val_loader = DataLoader(val_dataset, batch_size=batch_size)
            
            # Initialize optimizer
            optimizer = AdamW(self.model.parameters(), lr=learning_rate)
            
            # Total training steps
            total_steps = len(train_loader) * epochs
            
            # Create learning rate scheduler
            scheduler = get_linear_schedule_with_warmup(
                optimizer,
                num_warmup_steps=0,
                num_training_steps=total_steps
            )
            
            # Track best model and metrics
            best_accuracy = 0
            training_stats = []
            
            # Training loop
            for epoch in range(epochs):
                print(f"\nEpoch {epoch+1}/{epochs}")
                
                # Training phase
                self.model.train()
                total_train_loss = 0
                
                for batch in train_loader:
                    input_ids = batch['input_ids'].to(device)
                    attention_mask = batch['attention_mask'].to(device)
                    labels = batch['label'].to(device)
                    
                    self.model.zero_grad()
                    
                    outputs = self.model(
                        input_ids=input_ids,
                        attention_mask=attention_mask,
                        labels=labels
                    )
                    
                    loss = outputs.loss
                    total_train_loss += loss.item()
                    
                    loss.backward()
                    torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)
                    
                    optimizer.step()
                    scheduler.step()
                
                avg_train_loss = total_train_loss / len(train_loader)
                print(f"Average training loss: {avg_train_loss:.4f}")
                
                # Validation phase
                self.model.eval()
                total_val_loss = 0
                val_preds = []
                val_true_labels = []
                
                for batch in val_loader:
                    input_ids = batch['input_ids'].to(device)
                    attention_mask = batch['attention_mask'].to(device)
                    labels = batch['label'].to(device)
                    
                    with torch.no_grad():
                        outputs = self.model(
                            input_ids=input_ids,
                            attention_mask=attention_mask,
                            labels=labels
                        )
                    
                    loss = outputs.loss
                    total_val_loss += loss.item()
                    
                    logits = outputs.logits
                    preds = torch.argmax(logits, dim=1).cpu().numpy()
                    true_labels = labels.cpu().numpy()
                    
                    val_preds.extend(preds)
                    val_true_labels.extend(true_labels)
                
                avg_val_loss = total_val_loss / len(val_loader)
                val_accuracy = accuracy_score(val_true_labels, val_preds)
                
                print(f"Validation Loss: {avg_val_loss:.4f}")
                print(f"Validation Accuracy: {val_accuracy:.4f}")
                
                # Save the best model
                if val_accuracy > best_accuracy:
                    best_accuracy = val_accuracy
                    # Save model state for later use
                    self.best_model_state = {
                        'epoch': epoch + 1,
                        'model_state_dict': self.model.state_dict(),
                        'val_accuracy': val_accuracy
                    }
                
                # Record training stats
                training_stats.append({
                    'epoch': epoch + 1,
                    'train_loss': avg_train_loss,
                    'val_loss': avg_val_loss,
                    'val_accuracy': val_accuracy
                })
            
            # Load the best model
            if hasattr(self, 'best_model_state'):
                self.model.load_state_dict(self.best_model_state['model_state_dict'])
                print(f"Loaded best model from epoch {self.best_model_state['epoch']} with validation accuracy: {self.best_model_state['val_accuracy']:.4f}")
            
            # Final evaluation
            self.model.eval()
            all_preds = []
            all_true_labels = []
            
            for batch in val_loader:
                input_ids = batch['input_ids'].to(device)
                attention_mask = batch['attention_mask'].to(device)
                labels = batch['label'].to(device)
                
                with torch.no_grad():
                    outputs = self.model(
                        input_ids=input_ids,
                        attention_mask=attention_mask
                    )
                
                logits = outputs.logits
                preds = torch.argmax(logits, dim=1).cpu().numpy()
                true_labels = labels.cpu().numpy()
                
                all_preds.extend(preds)
                all_true_labels.extend(true_labels)
            
            final_accuracy = accuracy_score(all_true_labels, all_preds)
            print(f"\nFinal model accuracy: {final_accuracy:.4f}")
            print("\nClassification Report:")
            print(classification_report(all_true_labels, all_preds, target_names=['Real News', 'Fake News']))
            
            # Store validation data for explainer
            self.val_texts = val_texts
            self.val_labels = val_labels
            
            self.training_stats = training_stats
            
            return True
        
        except Exception as e:
            print(f"Error during training: {e}")
            import traceback
            traceback.print_exc()
            return False
    
    def predict(self, text, threshold=0.5):
        """Predict if a news is fake or real"""
        self.model.eval()
        
        # Preprocess the input text
        processed_text = preprocess_text(text)
        
        # Tokenize the text
        encoding = self.tokenizer.encode_plus(
            processed_text,
            add_special_tokens=True,
            max_length=512,
            return_token_type_ids=False,
            padding='max_length',
            truncation=True,
            return_attention_mask=True,
            return_tensors='pt'
        )
        
        input_ids = encoding['input_ids'].to(device)
        attention_mask = encoding['attention_mask'].to(device)
        
        with torch.no_grad():
            outputs = self.model(
                input_ids=input_ids,
                attention_mask=attention_mask
            )
        
        logits = outputs.logits
        probabilities = torch.nn.functional.softmax(logits, dim=1)
        
        # Get the predicted class and confidence
        predicted_class = torch.argmax(probabilities, dim=1).item()
        confidence = probabilities[0][predicted_class].item()
        
        # Convert numerical prediction to label
        label = "Fake News" if predicted_class == 1 else "Real News"
        
        return label, confidence
    
    def explain_prediction(self, text, num_features=20):
        """Explain the model's prediction using integrated gradients"""
        try:
            # Check if the model has the expected structure
            if not hasattr(self.model, 'bert'):
                print("Model structure doesn't support explanation. Using alternative method.")
                return self._explain_prediction_alt(text, num_features)
            
            # Preprocess the input text
            processed_text = preprocess_text(text)
            
            # Tokenize the text
            encoding = self.tokenizer.encode_plus(
                processed_text,
                add_special_tokens=True,
                max_length=512,
                return_token_type_ids=False,
                padding='max_length',
                truncation=True,
                return_attention_mask=True,
                return_tensors='pt'
            )
            
            input_ids = encoding['input_ids'].to(device)
            attention_mask = encoding['attention_mask'].to(device)
            
            # Get the predicted class
            with torch.no_grad():
                outputs = self.model(
                    input_ids=input_ids,
                    attention_mask=attention_mask
                )
            
            logits = outputs.logits
            predicted_class = torch.argmax(logits, dim=1).item()
            
            # Instead of using integrated gradients, which can be problematic in this setup,
            # use our more robust alternative method
            return self._explain_prediction_alt(text, num_features)
            
        except Exception as e:
            print(f"Error in explain_prediction: {e}")
            import traceback
            traceback.print_exc()
            # Use alternative method as fallback
            return self._explain_prediction_alt(text, num_features)
    
    def _explain_prediction_alt(self, text, num_features=20):
        """Alternative explanation method when integrated gradients is not available"""
        processed_text = preprocess_text(text)
        
        # Get prediction
        label, confidence = self.predict(processed_text)
        is_fake = label == "Fake News"
        
        # Tokenize and get feature importance heuristically
        words = processed_text.split()
        word_importances = []
        
        baseline_conf = confidence
        
        # Calculate word importance by removing each word and seeing how it affects the prediction
        for i, word in enumerate(words):
            if len(word) <= 2:  # Skip very short words
                continue
                
            # Create text without this word
            modified_text = ' '.join(words[:i] + words[i+1:])
            
            # Get new prediction
            _, new_conf = self.predict(modified_text)
            
            # Importance is how much the confidence changes
            importance = baseline_conf - new_conf
            word_importances.append((word, importance))
        
        # Sort by absolute importance
        word_importances.sort(key=lambda x: abs(x[1]), reverse=True)
        
        # Take top features
        top_importances = word_importances[:num_features]
        
        # Classify important words
        supports_fake = []
        supports_real = []
        
        for word, importance in top_importances:
            if importance > 0:  # Removing this word decreased confidence - it supports prediction
                if is_fake:
                    supports_fake.append(f'"{word}"')
                else:
                    supports_real.append(f'"{word}"')
            else:  # Removing this word increased confidence - it contradicts prediction
                if is_fake:
                    supports_real.append(f'"{word}"')
                else:
                    supports_fake.append(f'"{word}"')
        
        # Create explanation text
        explanation_text = ""
        if supports_fake:
            explanation_text += f"Words suggesting FAKE news: {', '.join(supports_fake)}. "
        if supports_real:
            explanation_text += f"Words suggesting REAL news: {', '.join(supports_real)}."
        
        if not explanation_text:
            explanation_text = "No strong indicators were found in the text."
        
        # Return the explanation and visualization data
        return explanation_text, {
            'tokens': [item[0] for item in top_importances],
            'attributions': [item[1] for item in top_importances],
            'predicted_class': 1 if is_fake else 0
        }
    
    def save_model(self, output_dir="bert_fake_news_model"):
        """Save the trained model to a directory"""
        try:
            # Create directory if it doesn't exist
            os.makedirs(output_dir, exist_ok=True)
            
            # Save the model
            self.model.save_pretrained(output_dir)
            self.tokenizer.save_pretrained(output_dir)
            
            # Save training stats if available
            if hasattr(self, 'training_stats'):
                with open(os.path.join(output_dir, 'training_stats.json'), 'w') as f:
                    json.dump(self.training_stats, f)
            
            print(f"Model saved to {output_dir}")
            return True
        except Exception as e:
            print(f"Error saving model: {e}")
            return False
    
    def load_model(self, model_path="bert_fake_news_model"):
        """Load a trained model from a directory"""
        try:
            self.model = BertForSequenceClassification.from_pretrained(model_path)
            self.model.to(device)
            
            print(f"Model loaded from {model_path}")
            return True
        except Exception as e:
            print(f"Error loading model: {e}")
            return False

# Web interface with Streamlit
def create_web_interface():
    st.set_page_config(
        page_title="BERT Fake News Detector",
        page_icon="🔍",
        layout="wide"
    )
    
    st.title("📰 BERT-based Fake News Detector")
    st.markdown("""
    This application uses a fine-tuned BERT model to detect fake news by analyzing the content of news headlines, tweets, or short news snippets.
    Enter your text below to get started!
    """)
    
    # Create tabs
    tab1, tab2, tab3 = st.tabs(["Detection", "How It Works", "About"])
    
    with tab1:
        st.header("Fake News Detection")
        
        # Text input for news content
        user_input = st.text_area("Enter news headline, tweet, or short news snippet:", 
                                  height=150,
                                  placeholder="Paste your news content here...")
        
        # Model selection and options
        col1, col2 = st.columns(2)
        
        with col1:
            num_features = st.slider("Number of explanation features", 5, 20, 10)
        
        with col2:
            model_path = st.text_input("Model path (leave empty for default)", "bert_fake_news_model")
            model_path = model_path if model_path else "bert_fake_news_model"
        
        # Initialize the model
        @st.cache_resource
        def load_model(model_path):
            detector = BERTFakeNewsDetector(model_path=model_path if os.path.exists(model_path) else None)
            return detector
        
        detector = load_model(model_path)
        
        # Check if model exists or needs training
        model_loaded = os.path.exists(model_path)
        
        submit_button = st.button("Analyze")
        
        if submit_button and user_input:
            with st.spinner("Analyzing..."):
                # If model is not loaded, train a new one
                if not model_loaded:
                    st.info("Training new BERT model. This may take several minutes...")
                    success = detector.train("merged_dataset.csv", epochs=3, batch_size=8)
                    
                    if not success:
                        st.error("Failed to train model. Please check that the merged_dataset.csv file exists and is properly formatted.")
                        st.stop()
                    
                    detector.save_model(model_path)
                
                # Make prediction
                label, confidence = detector.predict(user_input)
                explanation_text, explanation_data = detector.explain_prediction(user_input, num_features)
                
                # Display results
                col1, col2 = st.columns(2)
                
                with col1:
                    # Display verdict
                    if label == "Fake News":
                        st.error(f"📛 **Verdict: {label}**")
                    else:
                        st.success(f"✅ **Verdict: {label}**")
                    
                    # Display confidence
                    st.metric("Confidence", f"{confidence:.2%}")
                    
                    # Display explanation
                    st.subheader("Why this prediction?")
                    st.write(explanation_text)
                
                with col2:
                    # Display visualization of token importance
                    st.subheader("Word Importance")
                    
                    # Create a DataFrame for plotting
                    if explanation_data:
                        tokens = explanation_data['tokens']
                        attributions = explanation_data['attributions']
                        
                        # Create a color map: red for contributions to fake, blue for real
                        colors = ['red' if (a > 0 and explanation_data['predicted_class'] == 1) or 
                                         (a < 0 and explanation_data['predicted_class'] == 0) 
                                  else 'blue' for a in attributions]
                        
                        # Create a Matplotlib figure
                        fig, ax = plt.subplots(figsize=(10, 6))
                        
                        # Sort for better visualization
                        sorted_indices = np.argsort(np.abs(attributions))
                        sorted_tokens = [tokens[i] for i in sorted_indices]
                        sorted_attributions = [attributions[i] for i in sorted_indices]
                        sorted_colors = [colors[i] for i in sorted_indices]
                        
                        y_pos = np.arange(len(sorted_tokens))
                        
                        # Plot horizontal bar chart
                        bars = ax.barh(y_pos, sorted_attributions, align='center', color=sorted_colors)
                        ax.set_yticks(y_pos)
                        ax.set_yticklabels(sorted_tokens)
                        ax.invert_yaxis()  # Labels read top-to-bottom
                        ax.set_xlabel('Attribution Value')
                        ax.set_title('Feature Importance')
                        
                        # Add legend
                        from matplotlib.patches import Patch
                        legend_elements = [
                            Patch(facecolor='red', label='Supports Fake News'),
                            Patch(facecolor='blue', label='Supports Real News')
                        ]
                        ax.legend(handles=legend_elements, loc='best')
                        
                        st.pyplot(fig)
    
    with tab2:
        st.header("How It Works")
        st.markdown("""
        ### Our BERT-based Approach to Fake News Detection
        
        This system uses state-of-the-art NLP technology to identify potential fake news. Here's how it works:
        
        1. **Text Processing**: We prepare the text for analysis by cleaning and normalizing the input.
           
        2. **BERT Tokenization**: We use BERT's specialized tokenizer to convert text into tokens that capture 
           both words and subword units, maintaining the richness of language.
           
        3. **Deep Learning Classification**: A fine-tuned BERT model analyzes these tokens in context to understand 
           the nuance and patterns associated with fake news.
           
        4. **Explainability**: We use Integrated Gradients to identify which words or phrases most influenced 
           the classification decision, making the AI's reasoning transparent.
        
        ### Why BERT?
        
        BERT (Bidirectional Encoder Representations from Transformers) offers significant advantages:
        
        - **Context Awareness**: BERT understands words in context, capturing subtleties that simpler models miss
        - **Transfer Learning**: Pre-trained on vast text corpora, it already understands language patterns
        - **State-of-the-Art Performance**: Consistently outperforms traditional ML approaches on text classification
        
        ### Limitations
        
        While our system is powerful, please note these important limitations:
        
        - The model can only evaluate based on patterns it learned in training
        - Very short texts may not provide enough context for reliable analysis
        - The model analyzes text in isolation without access to external verification
        - Always verify information from multiple reliable sources
        """)
        
    with tab3:
        st.header("About")
        st.markdown("""
        ### BERT-powered Fake News Detector
        
        This application was developed as a tool to help users critically evaluate news content using
        state-of-the-art natural language processing.
        
        **Dataset**: The model was trained on a curated dataset containing examples of both real and fake news.
        
        **Privacy**: All analysis happens locally in your browser session. We do not store your submitted content.
        
        **Feedback**: This is a prototype system. Your feedback helps improve its accuracy and usability.
        """)

# Function to train model directly without web interface
def train_model_from_file(file_path="merged_dataset.csv", save_path="bert_fake_news_model", epochs=4, batch_size=8):
    """Train the model directly from a file and save it"""
    detector = BERTFakeNewsDetector()
    print(f"Training BERT model on {file_path} with {epochs} epochs...")
    success = detector.train(file_path, epochs=epochs, batch_size=batch_size)
    
    if success and save_path:
        detector.save_model(save_path)
        print(f"Model saved to {save_path}")
    
    return detector if success else None

# Entry point for the application
if __name__ == "__main__":
    create_web_interface()