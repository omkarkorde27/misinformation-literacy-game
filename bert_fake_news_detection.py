# Explainable Fake News Detection System with Improved BERT Architecture
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
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from transformers import BertTokenizer, BertForSequenceClassification, AutoModel
from transformers import get_linear_schedule_with_warmup
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, accuracy_score, confusion_matrix, f1_score
import torch.nn.functional as F

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
print(f"Using {device} device")

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

# Improved BERT architecture with more complex classification head
class BERT_Improved(nn.Module):
    def __init__(self, bert, dropout_rate=0.3):
        super(BERT_Improved, self).__init__()
        self.bert = bert
        
        # More sophisticated classification head
        self.dropout1 = nn.Dropout(dropout_rate)
        self.fc1 = nn.Linear(768, 512)
        self.bn1 = nn.BatchNorm1d(512)
        
        self.dropout2 = nn.Dropout(dropout_rate)
        self.fc2 = nn.Linear(512, 256)
        self.bn2 = nn.BatchNorm1d(256)
        
        self.dropout3 = nn.Dropout(dropout_rate)
        self.fc3 = nn.Linear(256, 2)
        
    def forward(self, input_ids, attention_mask):
        # Make sure inputs are the right dtype
        input_ids = input_ids.to(dtype=torch.int64)
        attention_mask = attention_mask.to(dtype=torch.int64)
        
        # Get BERT embeddings
        outputs = self.bert(input_ids=input_ids, attention_mask=attention_mask)
        pooled_output = outputs.pooler_output
        
        # First dense layer with batch normalization
        x = self.dropout1(pooled_output)
        x = self.fc1(x)
        x = self.bn1(x)
        x = F.relu(x)
        
        # Second dense layer with batch normalization
        x = self.dropout2(x)
        x = self.fc2(x)
        x = self.bn2(x)
        x = F.relu(x)
        
        # Output layer
        x = self.dropout3(x)
        x = self.fc3(x)
        
        return F.log_softmax(x, dim=1)

# Custom dataset for BERT
class NewsDataset(Dataset):
    def __init__(self, texts, labels, tokenizer, max_length=128):
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
    def __init__(self, model_path=None, num_labels=2, max_length=128):
        self.tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
        self.max_length = max_length
        
        if model_path and os.path.exists(model_path):
            # Load the saved model
            self.bert = AutoModel.from_pretrained('bert-base-uncased')
            self.model = BERT_Improved(self.bert)
            self.model.load_state_dict(torch.load(model_path, map_location=device))
            print(f"Loaded pre-trained model from {model_path}")
        else:
            # Initialize a new model
            self.bert = AutoModel.from_pretrained('bert-base-uncased')
            # Unfreeze only the last 2 layers of BERT and the pooler
            for name, param in self.bert.named_parameters():
                if "encoder.layer.10" in name or "encoder.layer.11" in name or "pooler" in name:
                    param.requires_grad = True
                else:
                    param.requires_grad = False
                    
            self.model = BERT_Improved(self.bert)
            print("Initialized new improved BERT model")
        
        self.model.to(device)
        
    def train(self, data_path, batch_size=16, epochs=10, learning_rate=2e-5, test_size=0.2, random_state=42):
        """Train the BERT model on the dataset"""
        try:
            # Load and preprocess the dataset
            df = pd.read_csv(data_path)
            print(f"Loaded dataset with {len(df)} rows")
            
            # Check if required columns exist
            if 'Content' not in df.columns or 'Label' not in df.columns:
                print(f"Dataset missing required columns. Found: {df.columns.tolist()}")
                return False
            
            # Preprocess the text data
            print("Preprocessing text data...")
            df['processed_content'] = df['Content'].apply(preprocess_text)
            
            # If explanation column exists, use it to enhance the features
            if 'Explanation' in df.columns:
                print("Using Content and Explanation for better features")
                df['processed_content'] = df['processed_content'] + ' ' + df['Explanation'].fillna('').apply(preprocess_text)
            
            # Convert labels to binary (0 for real, 1 for fake)
            if df['Label'].dtype == 'object':
                df['label_numeric'] = df['Label'].map(lambda x: 1 if str(x).lower() in ['fake', 'false'] else 0)
            else:
                df['label_numeric'] = df['Label']
            
            # Check class balance and print info
            class_counts = df['label_numeric'].value_counts()
            print(f"Class distribution: {class_counts.to_dict()}")
            
            # Calculate class weights for imbalanced dataset
            total_samples = len(df)
            class_0_samples = class_counts.get(0, 0)
            class_1_samples = class_counts.get(1, 0)
            class_weights = torch.tensor([
                1.0 / (class_0_samples / total_samples) if class_0_samples > 0 else 1.0,
                1.0 / (class_1_samples / total_samples) if class_1_samples > 0 else 1.0
            ], dtype=torch.float32).to(device)
            
            # Split the data into train, validation and test sets
            # First split into train and temp
            train_texts, temp_texts, train_labels, temp_labels = train_test_split(
                df['processed_content'].values, 
                df['label_numeric'].values,
                test_size=0.3,  # 30% for validation and test
                random_state=random_state,
                stratify=df['label_numeric']
            )
            
            # Then split temp into validation and test
            val_texts, test_texts, val_labels, test_labels = train_test_split(
                temp_texts,
                temp_labels,
                test_size=0.5,  # 50% of temp (15% of total)
                random_state=random_state,
                stratify=temp_labels
            )
            
            print(f"Training set: {len(train_texts)} samples")
            print(f"Validation set: {len(val_texts)} samples")
            print(f"Test set: {len(test_texts)} samples")
            
            # Create datasets
            train_dataset = NewsDataset(train_texts, train_labels, self.tokenizer, self.max_length)
            val_dataset = NewsDataset(val_texts, val_labels, self.tokenizer, self.max_length)
            test_dataset = NewsDataset(test_texts, test_labels, self.tokenizer, self.max_length)
            
            # Create data loaders
            train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
            val_loader = DataLoader(val_dataset, batch_size=batch_size)
            test_loader = DataLoader(test_dataset, batch_size=batch_size)
            
            # Initialize optimizer with weight decay for regularization
            optimizer = AdamW(
                [p for p in self.model.parameters() if p.requires_grad],
                lr=learning_rate,
                weight_decay=0.01  # L2 regularization
            )
            
            # Total training steps
            total_steps = len(train_loader) * epochs
            
            # Create learning rate scheduler with warmup
            scheduler = get_linear_schedule_with_warmup(
                optimizer,
                num_warmup_steps=int(0.1 * total_steps),  # 10% of steps for warmup
                num_training_steps=total_steps
            )
            
            # Define loss function with class weights
            criterion = nn.NLLLoss(weight=class_weights)
            
            # Track best model and metrics
            best_val_f1 = 0
            training_stats = []
            
            # Early stopping parameters
            early_stopping_patience = 3
            early_stopping_counter = 0
            
            # Training loop
            for epoch in range(epochs):
                print(f"\nEpoch {epoch+1}/{epochs}")
                
                # Training phase
                self.model.train()
                total_train_loss = 0
                train_preds = []
                train_true_labels = []
                
                for batch in train_loader:
                    input_ids = batch['input_ids'].to(device)
                    attention_mask = batch['attention_mask'].to(device)
                    labels = batch['label'].to(device)
                    
                    # Zero gradients
                    optimizer.zero_grad()
                    
                    # Forward pass
                    outputs = self.model(input_ids, attention_mask)
                    
                    # Calculate loss
                    loss = criterion(outputs, labels)
                    total_train_loss += loss.item()
                    
                    # Backpropagation
                    loss.backward()
                    
                    # Gradient clipping to prevent exploding gradients
                    torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
                    
                    # Update parameters
                    optimizer.step()
                    scheduler.step()
                    
                    # Calculate metrics
                    _, preds = torch.max(outputs, dim=1)
                    train_preds.extend(preds.cpu().numpy())
                    train_true_labels.extend(labels.cpu().numpy())
                
                # Calculate training metrics
                avg_train_loss = total_train_loss / len(train_loader)
                train_f1 = f1_score(train_true_labels, train_preds, average='weighted')
                
                # Validation phase
                self.model.eval()
                total_val_loss = 0
                val_preds = []
                val_true_labels = []
                
                with torch.no_grad():
                    for batch in val_loader:
                        input_ids = batch['input_ids'].to(device)
                        attention_mask = batch['attention_mask'].to(device)
                        labels = batch['label'].to(device)
                        
                        # Forward pass
                        outputs = self.model(input_ids, attention_mask)
                        
                        # Calculate loss
                        loss = criterion(outputs, labels)
                        total_val_loss += loss.item()
                        
                        # Calculate metrics
                        _, preds = torch.max(outputs, dim=1)
                        val_preds.extend(preds.cpu().numpy())
                        val_true_labels.extend(labels.cpu().numpy())
                
                # Calculate validation metrics
                avg_val_loss = total_val_loss / len(val_loader)
                val_f1 = f1_score(val_true_labels, val_preds, average='weighted')
                val_accuracy = accuracy_score(val_true_labels, val_preds)
                
                # Print epoch results
                print(f"Training Loss: {avg_train_loss:.4f}, Training F1: {train_f1:.4f}")
                print(f"Validation Loss: {avg_val_loss:.4f}, Validation F1: {val_f1:.4f}, Validation Accuracy: {val_accuracy:.4f}")
                
                # Save the best model based on F1 score
                if val_f1 > best_val_f1:
                    best_val_f1 = val_f1
                    # Save model state for later use
                    self.best_model_state = {
                        'epoch': epoch + 1,
                        'model_state_dict': self.model.state_dict(),
                        'val_f1': val_f1,
                        'val_accuracy': val_accuracy
                    }
                    early_stopping_counter = 0
                else:
                    early_stopping_counter += 1
                    if early_stopping_counter >= early_stopping_patience:
                        print(f"Early stopping triggered after epoch {epoch+1}")
                        break
                
                # Record training stats
                training_stats.append({
                    'epoch': epoch + 1,
                    'train_loss': avg_train_loss,
                    'train_f1': train_f1,
                    'val_loss': avg_val_loss,
                    'val_f1': val_f1,
                    'val_accuracy': val_accuracy
                })
            
            # Load the best model
            if hasattr(self, 'best_model_state'):
                self.model.load_state_dict(self.best_model_state['model_state_dict'])
                print(f"Loaded best model from epoch {self.best_model_state['epoch']} with validation F1: {self.best_model_state['val_f1']:.4f}")
            
            # Final evaluation on test set
            self.model.eval()
            test_preds = []
            test_true_labels = []
            
            with torch.no_grad():
                for batch in test_loader:
                    input_ids = batch['input_ids'].to(device)
                    attention_mask = batch['attention_mask'].to(device)
                    labels = batch['label'].to(device)
                    
                    # Forward pass
                    outputs = self.model(input_ids, attention_mask)
                    
                    # Calculate metrics
                    _, preds = torch.max(outputs, dim=1)
                    test_preds.extend(preds.cpu().numpy())
                    test_true_labels.extend(labels.cpu().numpy())
            
            # Calculate and print test metrics
            test_accuracy = accuracy_score(test_true_labels, test_preds)
            test_f1 = f1_score(test_true_labels, test_preds, average='weighted')
            
            print(f"\nTest Accuracy: {test_accuracy:.4f}")
            print(f"Test F1 Score: {test_f1:.4f}")
            print("\nClassification Report:")
            print(classification_report(test_true_labels, test_preds, target_names=['Real News', 'Fake News']))
            
            # Create confusion matrix
            cm = confusion_matrix(test_true_labels, test_preds)
            plt.figure(figsize=(8, 6))
            plt.imshow(cm, interpolation='nearest', cmap=plt.cm.Blues)
            plt.title('Confusion Matrix')
            plt.colorbar()
            tick_marks = np.arange(2)
            plt.xticks(tick_marks, ['Real', 'Fake'])
            plt.yticks(tick_marks, ['Real', 'Fake'])
            
            # Add text annotations to the confusion matrix
            thresh = cm.max() / 2.
            for i in range(cm.shape[0]):
                for j in range(cm.shape[1]):
                    plt.text(j, i, format(cm[i, j], 'd'),
                             horizontalalignment="center",
                             color="white" if cm[i, j] > thresh else "black")
            
            plt.ylabel('True label')
            plt.xlabel('Predicted label')
            plt.tight_layout()
            plt.show()
            
            # Store training stats
            self.training_stats = training_stats
            
            # Store test data for later use
            self.test_texts = test_texts
            self.test_labels = test_labels
            
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
            max_length=self.max_length,
            return_token_type_ids=False,
            padding='max_length',
            truncation=True,
            return_attention_mask=True,
            return_tensors='pt'
        )
        
        input_ids = encoding['input_ids'].to(device)
        attention_mask = encoding['attention_mask'].to(device)
        
        with torch.no_grad():
            outputs = self.model(input_ids, attention_mask)
        
        probabilities = torch.exp(outputs)
        
        # Get the predicted class and confidence
        predicted_class = torch.argmax(probabilities, dim=1).item()
        confidence = probabilities[0][predicted_class].item()
        
        # Convert numerical prediction to label
        label = "Fake News" if predicted_class == 1 else "Real News"
        
        return label, confidence
    
    def explain_prediction(self, text, num_features=20):
        """Explain the model's prediction by identifying important words"""
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
            torch.save(self.model.state_dict(), os.path.join(output_dir, 'model.pt'))
            
            # Save the tokenizer
            self.tokenizer.save_pretrained(output_dir)
            
            # Save configuration
            config = {
                'model_type': 'BERT_Improved',
                'max_length': self.max_length,
                'date_saved': datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            }
            
            with open(os.path.join(output_dir, 'config.json'), 'w') as f:
                json.dump(config, f)
            
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
            # Check if the directory exists
            if not os.path.exists(model_path):
                print(f"Model path {model_path} does not exist")
                return False
            
            # Load configuration
            config_path = os.path.join(model_path, 'config.json')
            if os.path.exists(config_path):
                with open(config_path, 'r') as f:
                    config = json.load(f)
                self.max_length = config.get('max_length', 128)
            
            # Load tokenizer
            if os.path.exists(os.path.join(model_path, 'vocab.txt')):
                self.tokenizer = BertTokenizer.from_pretrained(model_path)
            
            # Load model
            model_file = os.path.join(model_path, 'model.pt')
            if not os.path.exists(model_file):
                print(f"Model file not found at {model_file}")
                return False
            
            # Initialize BERT base model
            self.bert = AutoModel.from_pretrained('bert-base-uncased')
            
            # Create model instance
            self.model = BERT_Improved(self.bert)
            
            # Load saved weights
            self.model.load_state_dict(torch.load(model_file, map_location=device))
            self.model.to(device)
            
            print(f"Model loaded from {model_path}")
            return True
        except Exception as e:
            print(f"Error loading model: {e}")
            import traceback
            traceback.print_exc()
            return False

# Web interface with Streamlit
def create_web_interface():
    st.set_page_config(
        page_title="Improved BERT Fake News Detector",
        page_icon="🔍",
        layout="wide"
    )
    
    st.title("📰 BERT-based Fake News Detector")
    st.markdown("""
    This application uses a fine-tuned BERT model with an improved architecture to detect fake news by analyzing the content of news headlines, tweets, or short news snippets.
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
                    success = detector.train("merged_dataset.csv", epochs=5, batch_size=16)
                    
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
        ### Our Improved BERT-based Approach to Fake News Detection
        
        This system uses state-of-the-art NLP technology to identify potential fake news. Here's how it works:
        
        1. **Text Processing**: We prepare the text for analysis by cleaning and normalizing the input.
           
        2. **BERT Tokenization**: We use BERT's specialized tokenizer to convert text into tokens that capture 
           both words and subword units, maintaining the richness of language.
           
        3. **Deep Learning Classification**: An improved fine-tuned BERT model with a sophisticated multi-layer
           neural network analyzes these tokens in context to understand the nuance and patterns associated with fake news.
           
        4. **Batch Normalization**: We apply batch normalization to help the model train faster and more reliably.
        
        5. **Advanced Regularization**: Multiple dropout layers and L2 regularization help prevent overfitting.
           
        6. **Explainability**: We identify which words or phrases most influenced the classification decision,
           making the AI's reasoning transparent.
        
        ### Why This Improved Architecture?
        
        Our enhanced BERT model offers significant advantages over basic implementations:
        
        - **Better Generalization**: Multiple layers and regularization techniques help the model perform well on new data
        - **Higher Accuracy**: Batch normalization and improved training procedure lead to higher prediction accuracy
        - **More Stable Training**: Learning rate scheduling with warmup and gradient clipping provide training stability
        - **Class Imbalance Handling**: We address the imbalance between real and fake news examples in training data
        
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
        ### Improved BERT-powered Fake News Detector
        
        This application was developed as a tool to help users critically evaluate news content using
        state-of-the-art natural language processing.
        
        **Dataset**: The model was trained on a curated dataset containing examples of both real and fake news.
        
        **Privacy**: All analysis happens locally in your browser session. We do not store your submitted content.
        
        **Feedback**: This is a prototype system. Your feedback helps improve its accuracy and usability.
        """)

# Function to train model directly without web interface
def train_model_from_file(file_path="merged_dataset.csv", save_path="bert_fake_news_model", epochs=5, batch_size=16):
    """Train the model directly from a file and save it"""
    detector = BERTFakeNewsDetector()
    print(f"Training improved BERT model on {file_path} with {epochs} epochs...")
    success = detector.train(file_path, epochs=epochs, batch_size=batch_size)
    
    if success and save_path:
        detector.save_model(save_path)
        print(f"Model saved to {save_path}")
    
    return detector if success else None

# Function to test with sample headlines
def test_with_samples(model_path="bert_fake_news_model"):
    """Test the model with sample headlines and show results"""
    detector = BERTFakeNewsDetector(model_path=model_path)
    
    sample_headlines = [
        "Donald Trump Sends Out Embarrassing New Year's Eve Message; This is Disturbing",     # Likely Fake
        "WATCH: George W. Bush Calls Out Trump For Supporting White Supremacy",               # Likely Fake
        "U.S. lawmakers question businessman at 2016 Trump Tower meeting: sources",           # Likely Real
        "Trump administration issues new rules on U.S. visa waivers"                          # Likely Real
    ]
    
    results = []
    for headline in sample_headlines:
        label, confidence = detector.predict(headline)
        results.append({
            "headline": headline,
            "prediction": label,
            "confidence": f"{confidence:.2%}"
        })
    
    print("\nSample Headline Predictions:")
    for i, result in enumerate(results):
        print(f"{i+1}. \"{result['headline']}\"")
        print(f"   Prediction: {result['prediction']} (Confidence: {result['confidence']})")
        print("-" * 80)
    
    return results

# Entry point for the application
if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description='BERT Fake News Detection System')
    parser.add_argument('--mode', type=str, default='web', choices=['web', 'train', 'test'],
                        help='Operating mode: web interface, train model, or test with samples')
    parser.add_argument('--data', type=str, default='merged_dataset.csv',
                        help='Path to the dataset CSV file')
    parser.add_argument('--model', type=str, default='bert_fake_news_model',
                        help='Path to save or load the model')
    parser.add_argument('--epochs', type=int, default=5,
                        help='Number of training epochs')
    parser.add_argument('--batch_size', type=int, default=16,
                        help='Batch size for training')
    
    args = parser.parse_args()
    
    if args.mode == 'web':
        create_web_interface()
    elif args.mode == 'train':
        train_model_from_file(args.data, args.model, args.epochs, args.batch_size)
    elif args.mode == 'test':
        test_with_samples(args.model)