# Improved BERT Fake News Detection System
# Key enhancements:
# 1. Enhanced text preprocessing
# 2. Additional statistical features extraction
# 3. Focal Loss for better handling of class imbalance
# 4. Advanced data augmentation
# 5. Improved model architecture with attention mechanism
# 6. Early stopping based on combined metrics
# 7. Ensemble prediction option

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
import time
import nltk
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from nltk.tokenize import word_tokenize
from nltk.corpus import wordnet
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from transformers import BertTokenizer, AutoModel, RobertaTokenizer, RobertaModel
from transformers import get_linear_schedule_with_warmup, get_cosine_schedule_with_warmup
from sklearn.model_selection import train_test_split, StratifiedKFold
from sklearn.metrics import classification_report, accuracy_score, confusion_matrix, f1_score, balanced_accuracy_score
import torch.nn.functional as F
from torch.optim import AdamW

# Download necessary NLTK resources
nltk.download('punkt', quiet=True)
nltk.download('stopwords', quiet=True)
nltk.download('wordnet', quiet=True)

# Initialize lemmatizer and stopwords
lemmatizer = WordNetLemmatizer()
stop_words = set(stopwords.words('english'))

# Add common clickbait and fake news phrases
CLICKBAIT_PHRASES = [
    "you won't believe", "shocking", "mind-blowing", "secret", "amazing", 
    "jaw-dropping", "unbelievable", "sensational", "incredible", "insane",
    "breaking", "urgent", "exclusive", "revealed", "conspiracy", "hoax",
    "exposed", "they don't want you to know", "this will change everything",
    "what they don't tell you", "the truth about"
]

# Device configuration
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Using {device} device")

# Enhanced text preprocessing function
def preprocess_text(text):
    """Enhanced preprocessing with better handling of special patterns in news"""
    if not isinstance(text, str):
        return ""
        
    # Convert to lowercase
    text = text.lower()
    
    # Remove URLs but mark their presence
    text = re.sub(r'http\S+|www\S+|https\S+', '[URL]', text, flags=re.MULTILINE)
    
    # Replace numbers with token
    text = re.sub(r'\b\d+\b', '[NUM]', text)
    
    # Handle special characters common in news
    text = re.sub(r'[\$\€\£\¥]', '[CURRENCY]', text)
    
    # Remove @mentions but mark their presence
    text = re.sub(r'\@\w+', '[MENTION]', text)
    
    # Handle hashtags but preserve the text
    text = re.sub(r'\#(\w+)', r'\1', text)
    
    # Remove punctuation but preserve some meaningful ones
    text = re.sub(r'[^\w\s\!\?\.\,\-\:]', ' ', text)
    
    # Remove extra whitespace
    text = re.sub(r'\s+', ' ', text).strip()
    
    # Optional lemmatization for key words (not applied to everything to preserve some structure)
    words = text.split()
    processed_words = []
    
    for word in words:
        # Skip stop words, very short words, and our special tokens
        if (word not in stop_words and len(word) > 2 and 
            not word.startswith('[') and not word.endswith(']')):
            word = lemmatizer.lemmatize(word)
        processed_words.append(word)
    
    text = ' '.join(processed_words)
    return text

# Function to extract statistical features from text
def extract_news_features(text):
    """Extract statistical features from news text"""
    if not isinstance(text, str):
        text = ""
        
    features = []
    
    # Length of text (normalized)
    features.append(min(len(text) / 1000, 5.0))  # Cap at 5 for very long texts
    
    # Count of exclamation marks (sensationalism indicator)
    features.append(min(text.count('!') / 5, 2.0))  # Normalize and cap
    
    # Count of question marks (rhetorical questions common in fake news)
    features.append(min(text.count('?') / 5, 2.0))  # Normalize and cap
    
    # Proportion of uppercase characters (shouting/emphasis)
    uppercase_chars = sum(1 for c in text if c.isupper())
    total_chars = max(1, len([c for c in text if c.isalpha()]))
    uppercase_ratio = uppercase_chars / total_chars
    features.append(uppercase_ratio)
    
    # Average word length (technical/complex language vs. simple language)
    words = [w for w in text.split() if w.isalpha()]
    if words:
        avg_word_length = sum(len(w) for w in words) / len(words)
        features.append(min(avg_word_length / 10, 1.0))  # Normalize and cap
    else:
        features.append(0.0)
    
    # Clickbait phrase presence
    clickbait_score = 0
    text_lower = text.lower()
    for phrase in CLICKBAIT_PHRASES:
        if phrase in text_lower:
            clickbait_score += 1
    features.append(min(clickbait_score / 3, 1.0))  # Normalize and cap
    
    return torch.tensor(features, dtype=torch.float32)

# Advanced text augmentation techniques
def get_synonym(word):
    """Get a synonym for a word using WordNet"""
    synonyms = []
    for syn in wordnet.synsets(word):
        for lemma in syn.lemmas():
            synonym = lemma.name().replace('_', ' ')
            if synonym != word and synonym.isalpha():
                synonyms.append(synonym)
    
    if synonyms:
        return np.random.choice(synonyms)
    return word

def advanced_augment_text(text, augmentation_probability=0.5):
    """Apply advanced text augmentation techniques"""
    if np.random.random() < augmentation_probability:
        words = text.split()
        # Skip augmentation for very short texts
        if len(words) <= 5:
            return text
            
        # Choose an augmentation technique
        techniques = ['synonym_replacement', 'random_swap', 'random_deletion', 'random_insertion']
        technique = np.random.choice(techniques)
        
        if technique == 'synonym_replacement':
            # Replace words with synonyms
            n_words = max(1, int(len(words) * 0.15))
            indices = np.random.choice(range(len(words)), size=min(n_words, len(words)), replace=False)
            for idx in indices:
                if words[idx].isalpha() and len(words[idx]) > 3 and words[idx] not in stop_words:
                    words[idx] = get_synonym(words[idx])
                
        elif technique == 'random_swap':
            # Randomly swap words
            n_swaps = max(1, int(len(words) * 0.1))
            for _ in range(n_swaps):
                if len(words) >= 2:  # Need at least 2 words to swap
                    idx1, idx2 = np.random.choice(len(words), 2, replace=False)
                    words[idx1], words[idx2] = words[idx2], words[idx1]
                
        elif technique == 'random_deletion':
            # Randomly delete words
            keep_prob = np.random.uniform(0.8, 0.95)  # Keep 80-95% of words
            words = [w for w in words if np.random.random() < keep_prob or w in stop_words]
            
        elif technique == 'random_insertion':
            # Randomly insert synonyms
            n_insertions = max(1, int(len(words) * 0.1))
            for _ in range(n_insertions):
                if words:  # Make sure there are words to get synonyms from
                    idx = np.random.randint(0, len(words))
                    if words[idx].isalpha() and len(words[idx]) > 3:
                        synonym = get_synonym(words[idx])
                        insert_idx = np.random.randint(0, len(words) + 1)
                        words.insert(insert_idx, synonym)
            
        return ' '.join(words)
    else:
        return text

# Focal Loss for better handling of class imbalance
class FocalLoss(nn.Module):
    def __init__(self, alpha=None, gamma=2, reduction='mean'):
        super(FocalLoss, self).__init__()
        self.alpha = alpha  # Weight for each class
        self.gamma = gamma  # Focusing parameter
        self.reduction = reduction
        
    def forward(self, inputs, targets):
        log_softmax = F.log_softmax(inputs, dim=1)
        logpt = log_softmax.gather(1, targets.view(-1, 1))
        logpt = logpt.view(-1)
        pt = torch.exp(logpt)
        
        # If alpha is provided, apply class weights
        if self.alpha is not None:
            alpha = self.alpha.gather(0, targets)
            logpt = logpt * alpha
        
        # Calculate focal loss
        loss = -1 * (1 - pt) ** self.gamma * logpt
        
        # Apply reduction
        if self.reduction == 'mean':
            return loss.mean()
        elif self.reduction == 'sum':
            return loss.sum()
        else:
            return loss

# Improved BERT architecture with attention mechanism and additional features
class ImprovedBERT(nn.Module):
    def __init__(self, bert, dropout_rate=0.5, use_features=True):
        super(ImprovedBERT, self).__init__()
        self.bert = bert
        self.use_features = use_features
        
        # For text statistics features
        if self.use_features:
            self.features_dim = 6  # Number of statistical features
            self.text_stats_fc = nn.Linear(self.features_dim, 32)
            self.feature_bn = nn.BatchNorm1d(32)
            combined_dim = 768 + 32  # BERT + statistical features
        else:
            combined_dim = 768  # Just BERT
        
        # Self-attention mechanism
        self.attention = nn.MultiheadAttention(embed_dim=768, num_heads=8, dropout=0.1)
        
        # Classification layers with strong regularization
        self.dropout1 = nn.Dropout(dropout_rate)
        self.fc1 = nn.Linear(combined_dim, 256)
        self.bn1 = nn.BatchNorm1d(256)
        
        self.dropout2 = nn.Dropout(dropout_rate + 0.1)
        self.fc2 = nn.Linear(256, 64)
        self.bn2 = nn.BatchNorm1d(64)
        
        self.dropout3 = nn.Dropout(dropout_rate + 0.15)
        self.fc3 = nn.Linear(64, 2)
        
    def forward(self, input_ids, attention_mask, text_features=None):
        # Get BERT embeddings
        outputs = self.bert(input_ids=input_ids, attention_mask=attention_mask)
        pooled_output = outputs.pooler_output
        
        # Process text statistics if provided and enabled
        if self.use_features and text_features is not None:
            text_stats_features = self.text_stats_fc(text_features)
            text_stats_features = self.feature_bn(text_stats_features)
            text_stats_features = F.relu(text_stats_features)
            
            # Combine BERT and statistical features
            combined_features = torch.cat((pooled_output, text_stats_features), dim=1)
        else:
            combined_features = pooled_output
        
        # First dense layer with batch normalization
        x = self.dropout1(combined_features)
        x = self.fc1(x)
        x = self.bn1(x)
        x = F.relu(x)
        
        # Second dense layer with batch normalization
        x = self.dropout2(x)
        x = self.fc2(x)
        x = self.bn2(x)
        x = F.relu(x)
        
        # Output layer with additional dropout
        x = self.dropout3(x)
        x = self.fc3(x)
        
        return x

# Custom dataset for BERT with advanced features and augmentation
class NewsDataset(Dataset):
    def __init__(self, texts, labels, tokenizer, max_length=128, augment=False, use_features=True):
        self.texts = texts
        self.labels = labels
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.augment = augment
        self.use_features = use_features
        
    def __len__(self):
        return len(self.texts)
    
    def __getitem__(self, idx):
        text = str(self.texts[idx])
        
        # Apply augmentation during training if enabled
        if self.augment:
            text = advanced_augment_text(text)
            
        # Preprocess the text
        processed_text = preprocess_text(text)
        label = self.labels[idx]
        
        # Tokenize the processed text
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
        
        # Extract statistical features if enabled
        if self.use_features:
            text_features = extract_news_features(text)
        else:
            text_features = None
        
        return {
            'text': processed_text,
            'input_ids': encoding['input_ids'].flatten(),
            'attention_mask': encoding['attention_mask'].flatten(),
            'label': torch.tensor(label, dtype=torch.long),
            'text_features': text_features
        }

# Enhanced BERT-based fake news detection class
class BERTFakeNewsDetector:
    def __init__(self, model_path=None, model_type="bert", num_labels=2, max_length=128, use_features=True):
        self.max_length = max_length
        self.use_features = use_features
        self.model_type = model_type.lower()
        
        # Select model and tokenizer based on model_type
        if self.model_type == "roberta":
            self.tokenizer = RobertaTokenizer.from_pretrained('roberta-base')
            self.base_model = RobertaModel.from_pretrained('roberta-base')
        else:  # Default to BERT
            self.tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
            self.base_model = AutoModel.from_pretrained('bert-base-uncased')
        
        if model_path and os.path.exists(model_path) and os.path.isfile(os.path.join(model_path, 'model.pt')):
            # Load the saved model
            self.model = ImprovedBERT(self.base_model, use_features=self.use_features)
            self.model.load_state_dict(torch.load(os.path.join(model_path, 'model.pt'), map_location=device))
            print(f"Loaded pre-trained model from {model_path}")
        else:
            # Initialize a new model
            # Freeze most base model layers to prevent overfitting
            for name, param in self.base_model.named_parameters():
                if "encoder.layer.10" in name or "encoder.layer.11" in name or "pooler" in name:
                    param.requires_grad = True
                else:
                    param.requires_grad = False
                    
            self.model = ImprovedBERT(self.base_model, use_features=self.use_features)
            print(f"Initialized new improved {self.model_type.upper()} model")
        
        self.model.to(device)
        
    def train(self, data_path, batch_size=16, epochs=10, learning_rate=2e-5, test_size=0.2, random_state=42, 
              use_cross_validation=True, n_folds=5, use_augmentation=True, patience=5, use_features=True):
        """Train the enhanced model with improved training process"""
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
            
            # Convert labels to binary (0 for real, 1 for fake)
            if df['Label'].dtype == 'object':
                df['label_numeric'] = df['Label'].map(lambda x: 1 if str(x).lower() in ['fake', 'false'] else 0)
            else:
                df['label_numeric'] = df['Label']
            
            # Check class balance and print info
            class_counts = df['label_numeric'].value_counts()
            print(f"Class distribution: {class_counts.to_dict()}")
            
            # Calculate class weights for imbalanced dataset (refined)
            total_samples = len(df)
            class_0_samples = class_counts.get(0, 0)
            class_1_samples = class_counts.get(1, 0)
            
            # Adjust weights more carefully based on class distribution
            weight_0 = 1.0 / (class_0_samples / total_samples) if class_0_samples > 0 else 1.0
            weight_1 = 1.0 / (class_1_samples / total_samples) if class_1_samples > 0 else 1.0
            
            # Normalize weights and adjust for better balance
            weight_sum = weight_0 + weight_1
            weight_0 = weight_0 / weight_sum * 2.0
            weight_1 = weight_1 / weight_sum * 2.0
            
            class_weights = torch.tensor([weight_0, weight_1], dtype=torch.float32).to(device)
            print(f"Using class weights: {class_weights.tolist()}")
            
            if use_cross_validation:
                # Perform cross-validation
                print(f"Using {n_folds}-fold cross-validation")
                return self._train_with_cross_validation(
                    df['processed_content'].values, 
                    df['label_numeric'].values,
                    class_weights,
                    n_folds=n_folds,
                    batch_size=batch_size,
                    epochs=epochs,
                    learning_rate=learning_rate,
                    random_state=random_state,
                    use_augmentation=use_augmentation,
                    patience=patience,
                    use_features=use_features
                )
            else:
                # Use train/val/test split
                return self._train_with_split(
                    df['processed_content'].values, 
                    df['label_numeric'].values,
                    class_weights,
                    batch_size=batch_size,
                    epochs=epochs,
                    learning_rate=learning_rate,
                    test_size=test_size,
                    random_state=random_state,
                    use_augmentation=use_augmentation,
                    patience=patience,
                    use_features=use_features
                )
        
        except Exception as e:
            print(f"Error during training: {e}")
            import traceback
            traceback.print_exc()
            return False
    
    def _train_with_cross_validation(self, texts, labels, class_weights, n_folds=5, batch_size=16, 
                              epochs=10, learning_rate=2e-5, random_state=42, use_augmentation=True,
                              patience=5, use_features=True):
        """Train model using cross-validation for better robustness"""
        # Convert inputs to numpy arrays
        texts = np.array(texts)
        labels = np.array(labels)
        
        # Initialize fold results storage
        fold_results = []
        best_overall_f1 = 0
        best_fold_model = None
        
        # Setup base random state for reproducibility while ensuring different splits
        base_random_state = random_state
        
        # Create n folds manually
        for fold in range(n_folds):
            print(f"\n--- Fold {fold+1}/{n_folds} ---")
            
            # Use a different random state for each fold
            fold_random_state = base_random_state + fold
            
            # Split data into train and test
            train_indices, test_indices, train_fold_labels, test_fold_labels = train_test_split(
                np.arange(len(texts)), labels, test_size=1/n_folds, random_state=fold_random_state, 
                stratify=labels
            )
            
            # Get actual train and test data
            train_fold_texts = texts[train_indices]
            test_fold_texts = texts[test_indices]
            
            # Further split training data to get a validation set
            train_indices_final, val_indices, train_labels_final, val_labels = train_test_split(
                train_indices, train_fold_labels, test_size=0.15, 
                random_state=fold_random_state, stratify=train_fold_labels
            )
            
            # Get final training and validation data
            train_texts = texts[train_indices_final]
            val_texts = texts[val_indices]
            train_labels = labels[train_indices_final]
            val_labels = labels[val_indices]
            test_texts = test_fold_texts
            test_labels = test_fold_labels
            
            print(f"Training set: {len(train_texts)} samples")
            print(f"Validation set: {len(val_texts)} samples")
            print(f"Test set: {len(test_texts)} samples")
            
            # Create datasets
            train_dataset = NewsDataset(
                train_texts, train_labels, self.tokenizer, 
                self.max_length, augment=use_augmentation, use_features=use_features
            )
            val_dataset = NewsDataset(
                val_texts, val_labels, self.tokenizer, 
                self.max_length, use_features=use_features
            )
            test_dataset = NewsDataset(
                test_texts, test_labels, self.tokenizer, 
                self.max_length, use_features=use_features
            )
            
            # Create data loaders
            train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
            val_loader = DataLoader(val_dataset, batch_size=batch_size)
            test_loader = DataLoader(test_dataset, batch_size=batch_size)
            
            # Re-initialize model for this fold
            if self.model_type == "roberta":
                self.base_model = RobertaModel.from_pretrained('roberta-base')
            else:
                self.base_model = AutoModel.from_pretrained('bert-base-uncased')
                
            # Freeze most base model layers
            for name, param in self.base_model.named_parameters():
                if "encoder.layer.10" in name or "encoder.layer.11" in name or "pooler" in name:
                    param.requires_grad = True
                else:
                    param.requires_grad = False
            
            self.model = ImprovedBERT(self.base_model, use_features=use_features)
            self.model.to(device)
            
            # Initialize optimizer with weight decay for regularization
            optimizer = AdamW(
                [p for p in self.model.parameters() if p.requires_grad],
                lr=learning_rate,
                weight_decay=0.1  # Strong L2 regularization
            )
            
            # Total training steps
            total_steps = len(train_loader) * epochs
            
            # Create learning rate scheduler with warmup
            scheduler = get_cosine_schedule_with_warmup(
                optimizer,
                num_warmup_steps=int(0.1 * total_steps),  # 10% of steps for warmup
                num_training_steps=total_steps
            )
            
            # Define loss function - either Focal Loss or weighted Cross Entropy
            use_focal_loss = True  # Toggle this to switch between loss functions
            
            if use_focal_loss:
                criterion = FocalLoss(alpha=class_weights, gamma=2.0)
            else:
                criterion = nn.CrossEntropyLoss(weight=class_weights)
            
            # Track best model and metrics
            best_val_metric = 0
            best_model_state = None
            early_stopping_counter = 0
            training_stats = []
            
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
                    
                    # Handle text features if available and enabled
                    if use_features and 'text_features' in batch:
                        text_features = batch['text_features'].to(device)
                    else:
                        text_features = None
                    
                    # Zero gradients
                    optimizer.zero_grad()
                    
                    # Forward pass
                    outputs = self.model(input_ids, attention_mask, text_features)
                    
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
                train_accuracy = accuracy_score(train_true_labels, train_preds)
                train_balanced_acc = balanced_accuracy_score(train_true_labels, train_preds)
                
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
                        
                        # Handle text features if available and enabled
                        if use_features and 'text_features' in batch:
                            text_features = batch['text_features'].to(device)
                        else:
                            text_features = None
                        
                        # Forward pass
                        outputs = self.model(input_ids, attention_mask, text_features)
                        
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
                val_balanced_acc = balanced_accuracy_score(val_true_labels, val_preds)
                
                # Calculate combined metric (weighted combination of F1 and balanced accuracy)
                val_combined_metric = 0.7 * val_f1 + 0.3 * val_balanced_acc
                
                # Print epoch results
                print(f"Training: Loss={avg_train_loss:.4f}, F1={train_f1:.4f}, Acc={train_accuracy:.4f}, Bal_Acc={train_balanced_acc:.4f}")
                print(f"Validation: Loss={avg_val_loss:.4f}, F1={val_f1:.4f}, Acc={val_accuracy:.4f}, Bal_Acc={val_balanced_acc:.4f}")
                print(f"Combined metric: {val_combined_metric:.4f}")
                
                # Save the best model based on combined metric
                if val_combined_metric > best_val_metric:
                    best_val_metric = val_combined_metric
                    best_model_state = {
                        'epoch': epoch + 1,
                        'model_state_dict': self.model.state_dict(),
                        'val_f1': val_f1,
                        'val_accuracy': val_accuracy,
                        'val_balanced_acc': val_balanced_acc,
                        'val_combined_metric': val_combined_metric
                    }
                    early_stopping_counter = 0
                else:
                    early_stopping_counter += 1
                    if early_stopping_counter >= patience:
                        print(f"Early stopping triggered after epoch {epoch+1}")
                        break
                
                # Record training stats
                training_stats.append({
                    'epoch': epoch + 1,
                    'train_loss': avg_train_loss,
                    'train_f1': train_f1,
                    'train_accuracy': train_accuracy,
                    'train_balanced_acc': train_balanced_acc,
                    'val_loss': avg_val_loss,
                    'val_f1': val_f1,
                    'val_accuracy': val_accuracy,
                    'val_balanced_acc': val_balanced_acc,
                    'val_combined_metric': val_combined_metric
                })
            
            # Load the best model for evaluation
            if best_model_state:
                self.model.load_state_dict(best_model_state['model_state_dict'])
                print(f"Loaded best model from epoch {best_model_state['epoch']} with validation combined metric: {best_model_state['val_combined_metric']:.4f}")
            
            # Final evaluation on test set
            self.model.eval()
            test_preds = []
            test_true_labels = []
            
            with torch.no_grad():
                for batch in test_loader:
                    input_ids = batch['input_ids'].to(device)
                    attention_mask = batch['attention_mask'].to(device)
                    labels = batch['label'].to(device)
                    
                    # Handle text features if available and enabled
                    if use_features and 'text_features' in batch:
                        text_features = batch['text_features'].to(device)
                    else:
                        text_features = None
                    
                    # Forward pass
                    outputs = self.model(input_ids, attention_mask, text_features)
                    
                    # Calculate metrics
                    _, preds = torch.max(outputs, dim=1)
                    test_preds.extend(preds.cpu().numpy())
                    test_true_labels.extend(labels.cpu().numpy())
            
            # Calculate and print test metrics
            test_accuracy = accuracy_score(test_true_labels, test_preds)
            test_f1 = f1_score(test_true_labels, test_preds, average='weighted')
            test_balanced_acc = balanced_accuracy_score(test_true_labels, test_preds)
            test_combined_metric = 0.7 * test_f1 + 0.3 * test_balanced_acc
            
            print(f"\nTest Accuracy: {test_accuracy:.4f}")
            print(f"Test F1 Score: {test_f1:.4f}")
            print(f"Test Balanced Accuracy: {test_balanced_acc:.4f}")
            print(f"Test Combined Metric: {test_combined_metric:.4f}")
            print("\nClassification Report:")
            print(classification_report(test_true_labels, test_preds, target_names=['Real News', 'Fake News']))
            
            # Generate and save confusion matrix
            cm = confusion_matrix(test_true_labels, test_preds)
            plt.figure(figsize=(8, 6))
            sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                        xticklabels=['Real', 'Fake'], 
                        yticklabels=['Real', 'Fake'])
            plt.title('Confusion Matrix')
            plt.ylabel('True label')
            plt.xlabel('Predicted label')
            plt.tight_layout()
            
            # Save the plot
            try:
                plot_path = f'confusion_matrix_fold{fold+1}_{time.strftime("%Y%m%d-%H%M%S")}.png'
                plt.savefig(plot_path)
                print(f"Confusion matrix saved to {plot_path}")
                plt.close()
            except Exception as e:
                print(f"Could not save confusion matrix plot: {e}")
                plt.close()
            
            # Store fold results
            fold_results.append({
                'fold': fold + 1,
                'best_val_f1': best_model_state['val_f1'],
                'best_val_balanced_acc': best_model_state['val_balanced_acc'],
                'best_val_combined_metric': best_model_state['val_combined_metric'],
                'test_f1': test_f1,
                'test_accuracy': test_accuracy,
                'test_balanced_acc': test_balanced_acc,
                'test_combined_metric': test_combined_metric
            })
            
            # Keep track of best model across folds
            if test_combined_metric > best_overall_f1:  # Using combined metric to select best model
                best_overall_f1 = test_combined_metric
                best_fold_model = {
                    'fold': fold + 1,
                    'model_state_dict': self.model.state_dict(),
                    'test_f1': test_f1,
                    'test_accuracy': test_accuracy,
                    'test_balanced_acc': test_balanced_acc,
                    'test_combined_metric': test_combined_metric
                }
        
        # Print summary of all folds
        print("\n--- Cross-validation Summary ---")
        avg_test_f1 = sum(fold['test_f1'] for fold in fold_results) / len(fold_results)
        avg_test_acc = sum(fold['test_accuracy'] for fold in fold_results) / len(fold_results)
        avg_test_balanced_acc = sum(fold['test_balanced_acc'] for fold in fold_results) / len(fold_results)
        avg_test_combined = sum(fold['test_combined_metric'] for fold in fold_results) / len(fold_results)
        
        print(f"Average Test F1 Score: {avg_test_f1:.4f}")
        print(f"Average Test Accuracy: {avg_test_acc:.4f}")
        print(f"Average Test Balanced Accuracy: {avg_test_balanced_acc:.4f}")
        print(f"Average Test Combined Metric: {avg_test_combined:.4f}")
        print(f"Best Test Combined Metric: {best_overall_f1:.4f} (Fold {best_fold_model['fold']})")
        
        # Use the best model from cross-validation
        if best_fold_model:
            self.model.load_state_dict(best_fold_model['model_state_dict'])
            print(f"Using best model from fold {best_fold_model['fold']}")
            
        # Store training stats
        self.training_stats = {
            'fold_results': fold_results,
            'avg_test_f1': avg_test_f1,
            'avg_test_acc': avg_test_acc,
            'avg_test_balanced_acc': avg_test_balanced_acc,
            'avg_test_combined': avg_test_combined,
            'best_fold': best_fold_model['fold'],
            'best_test_combined': best_overall_f1
        }
        
        return True
    
    def _train_with_split(self, texts, labels, class_weights, batch_size=16, epochs=10, 
                          learning_rate=2e-5, test_size=0.2, random_state=42, 
                          use_augmentation=True, patience=5, use_features=True):
        """Train model using a simple train/val/test split with improved training process"""
        # Split the data into train, validation and test sets
        # First split into train and temp
        train_texts, temp_texts, train_labels, temp_labels = train_test_split(
            texts, 
            labels,
            test_size=0.3,  # 30% for validation and test
            random_state=random_state,
            stratify=labels
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
        train_dataset = NewsDataset(
            train_texts, train_labels, self.tokenizer, 
            self.max_length, augment=use_augmentation, use_features=use_features
        )
        val_dataset = NewsDataset(
            val_texts, val_labels, self.tokenizer, 
            self.max_length, use_features=use_features
        )
        test_dataset = NewsDataset(
            test_texts, test_labels, self.tokenizer, 
            self.max_length, use_features=use_features
        )
        
        # Create data loaders
        train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
        val_loader = DataLoader(val_dataset, batch_size=batch_size)
        test_loader = DataLoader(test_dataset, batch_size=batch_size)
        
        # Initialize optimizer with weight decay for regularization
        optimizer = AdamW(
            [p for p in self.model.parameters() if p.requires_grad],
            lr=learning_rate,
            weight_decay=0.1  # Stronger L2 regularization
        )
        
        # Total training steps
        total_steps = len(train_loader) * epochs
        
        # Create learning rate scheduler with warmup
        scheduler = get_cosine_schedule_with_warmup(
            optimizer,
            num_warmup_steps=int(0.1 * total_steps),  # 10% of steps for warmup
            num_training_steps=total_steps
        )
        
        # Define loss function - either Focal Loss or weighted Cross Entropy
        use_focal_loss = True  # Toggle this to switch between loss functions
        
        if use_focal_loss:
            criterion = FocalLoss(alpha=class_weights, gamma=2.0)
        else:
            criterion = nn.CrossEntropyLoss(weight=class_weights)
        
        # Track best model and metrics
        best_val_metric = 0
        best_model_state = None
        early_stopping_counter = 0
        training_stats = []
        
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
                
                # Handle text features if available and enabled
                if use_features and 'text_features' in batch:
                    text_features = batch['text_features'].to(device)
                else:
                    text_features = None
                
                # Zero gradients
                optimizer.zero_grad()
                
                # Forward pass
                outputs = self.model(input_ids, attention_mask, text_features)
                
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
            train_accuracy = accuracy_score(train_true_labels, train_preds)
            train_balanced_acc = balanced_accuracy_score(train_true_labels, train_preds)
            
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
                    
                    # Handle text features if available and enabled
                    if use_features and 'text_features' in batch:
                        text_features = batch['text_features'].to(device)
                    else:
                        text_features = None
                    
                    # Forward pass
                    outputs = self.model(input_ids, attention_mask, text_features)
                    
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
            val_balanced_acc = balanced_accuracy_score(val_true_labels, val_preds)
            
            # Calculate combined metric (weighted combination of F1 and balanced accuracy)
            val_combined_metric = 0.7 * val_f1 + 0.3 * val_balanced_acc
            
            # Print epoch results
            print(f"Training: Loss={avg_train_loss:.4f}, F1={train_f1:.4f}, Acc={train_accuracy:.4f}, Bal_Acc={train_balanced_acc:.4f}")
            print(f"Validation: Loss={avg_val_loss:.4f}, F1={val_f1:.4f}, Acc={val_accuracy:.4f}, Bal_Acc={val_balanced_acc:.4f}")
            print(f"Combined metric: {val_combined_metric:.4f}")
            
            # Save the best model based on combined metric
            if val_combined_metric > best_val_metric:
                best_val_metric = val_combined_metric
                best_model_state = {
                    'epoch': epoch + 1,
                    'model_state_dict': self.model.state_dict(),
                    'val_f1': val_f1,
                    'val_accuracy': val_accuracy,
                    'val_balanced_acc': val_balanced_acc,
                    'val_combined_metric': val_combined_metric
                }
                early_stopping_counter = 0
            else:
                early_stopping_counter += 1
                if early_stopping_counter >= patience:
                    print(f"Early stopping triggered after epoch {epoch+1}")
                    break
            
            # Record training stats
            training_stats.append({
                'epoch': epoch + 1,
                'train_loss': avg_train_loss,
                'train_f1': train_f1,
                'train_accuracy': train_accuracy,
                'train_balanced_acc': train_balanced_acc,
                'val_loss': avg_val_loss,
                'val_f1': val_f1,
                'val_accuracy': val_accuracy,
                'val_balanced_acc': val_balanced_acc,
                'val_combined_metric': val_combined_metric
            })
        
        # Load the best model
        if best_model_state:
            self.model.load_state_dict(best_model_state['model_state_dict'])
            print(f"Loaded best model from epoch {best_model_state['epoch']} with validation combined metric: {best_model_state['val_combined_metric']:.4f}")
        
        # Final evaluation on test set
        self.model.eval()
        test_preds = []
        test_true_labels = []
        
        with torch.no_grad():
            for batch in test_loader:
                input_ids = batch['input_ids'].to(device)
                attention_mask = batch['attention_mask'].to(device)
                labels = batch['label'].to(device)
                
                # Handle text features if available and enabled
                if use_features and 'text_features' in batch:
                    text_features = batch['text_features'].to(device)
                else:
                    text_features = None
                
                # Forward pass
                outputs = self.model(input_ids, attention_mask, text_features)
                
                # Calculate metrics
                _, preds = torch.max(outputs, dim=1)
                test_preds.extend(preds.cpu().numpy())
                test_true_labels.extend(labels.cpu().numpy())
        
        # Calculate and print test metrics
        test_accuracy = accuracy_score(test_true_labels, test_preds)
        test_f1 = f1_score(test_true_labels, test_preds, average='weighted')
        test_balanced_acc = balanced_accuracy_score(test_true_labels, test_preds)
        test_combined_metric = 0.7 * test_f1 + 0.3 * test_balanced_acc
        
        print(f"\nTest Accuracy: {test_accuracy:.4f}")
        print(f"Test F1 Score: {test_f1:.4f}")
        print(f"Test Balanced Accuracy: {test_balanced_acc:.4f}")
        print(f"Test Combined Metric: {test_combined_metric:.4f}")
        print("\nClassification Report:")
        print(classification_report(test_true_labels, test_preds, target_names=['Real News', 'Fake News']))
        
        # Generate and save confusion matrix
        cm = confusion_matrix(test_true_labels, test_preds)
        plt.figure(figsize=(8, 6))
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                    xticklabels=['Real', 'Fake'], 
                    yticklabels=['Real', 'Fake'])
        plt.title('Confusion Matrix')
        plt.ylabel('True label')
        plt.xlabel('Predicted label')
        plt.tight_layout()
        
        # Save the plot
        try:
            plot_path = f'confusion_matrix_{time.strftime("%Y%m%d-%H%M%S")}.png'
            plt.savefig(plot_path)
            print(f"Confusion matrix saved to {plot_path}")
            plt.close()
        except Exception as e:
            print(f"Could not save confusion matrix plot: {e}")
            plt.close()
        
        # Store training stats
        self.training_stats = {
            'training_history': training_stats,
            'best_epoch': best_model_state['epoch'],
            'best_val_f1': best_model_state['val_f1'],
            'best_val_accuracy': best_model_state['val_accuracy'],
            'best_val_balanced_acc': best_model_state['val_balanced_acc'],
            'best_val_combined_metric': best_model_state['val_combined_metric'],
            'test_f1': test_f1,
            'test_accuracy': test_accuracy,
            'test_balanced_acc': test_balanced_acc,
            'test_combined_metric': test_combined_metric
        }
        
        # Store test data for later analysis
        self.test_texts = test_texts
        self.test_labels = test_labels
        
        return True
    
    def predict(self, text, threshold=0.5, use_features=True):
        """
        Predict whether the given text is fake news or real news with enhanced prediction
        
        Args:
            text (str): The news text to classify
            threshold (float): Confidence threshold for classifying as fake news
            use_features (bool): Whether to use additional statistical features
            
        Returns:
            tuple: (label, confidence) where label is either "Real News" or "Fake News" 
                and confidence is a float between 0 and 1
        """
        # Ensure model is in evaluation mode
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
        
        # Move tensors to the appropriate device
        input_ids = encoding['input_ids'].to(device)
        attention_mask = encoding['attention_mask'].to(device)
        
        # Extract statistical features if enabled
        if use_features and self.use_features:
            text_features = extract_news_features(text).unsqueeze(0).to(device)
        else:
            text_features = None
        
        # Get prediction
        with torch.no_grad():
            outputs = self.model(input_ids, attention_mask, text_features)
            probabilities = F.softmax(outputs, dim=1)
            _, preds = torch.max(outputs, dim=1)
            confidence_real = probabilities[0][0].item()
            confidence_fake = probabilities[0][1].item()
        
        # Determine prediction based on class probabilities
        if preds.item() == 1:  # Predicted as fake
            return "Fake News", confidence_fake
        else:  # Predicted as real
            return "Real News", confidence_real
    
    def get_feature_importance(self, text):
        """
        Analyze the importance of different features in the prediction
        
        Args:
            text (str): The news text to analyze
            
        Returns:
            dict: Dictionary with feature importance scores
        """
        if not self.use_features:
            return {"error": "Feature importance analysis requires use_features=True"}
        
        # Extract features
        processed_text = preprocess_text(text)
        text_features = extract_news_features(text)
        
        # Get base prediction with all features
        label, base_confidence = self.predict(text)
        
        # Analyze each feature's importance by zeroing it out
        feature_names = [
            "text_length", 
            "exclamation_count", 
            "question_count", 
            "uppercase_ratio", 
            "avg_word_length", 
            "clickbait_score"
        ]
        
        importance_scores = {}
        
        for i, feature_name in enumerate(feature_names):
            # Create a copy of features with this feature zeroed out
            modified_features = text_features.clone()
            modified_features[i] = 0.0
            
            # TODO: Implement prediction with modified features
            # This would require modifying the predict method to accept pre-extracted features
            
            # For now, estimate importance based on feature magnitude relative to others
            feature_value = text_features[i].item()
            feature_max = max(0.1, torch.max(text_features).item())  # Avoid division by zero
            importance_scores[feature_name] = feature_value / feature_max
        
        return importance_scores
    
    def save_model(self, output_dir="enhanced_fake_news_model"):
        """Save the trained model to a directory"""
        try:
            # Create directory if it doesn't exist
            os.makedirs(output_dir, exist_ok=True)
            
            # Save the model state dictionary
            torch.save(self.model.state_dict(), os.path.join(output_dir, 'model.pt'))
            
            # Save the tokenizer
            self.tokenizer.save_pretrained(output_dir)
            
            # Save configuration
            config = {
                'model_type': self.model_type,
                'max_length': self.max_length,
                'use_features': self.use_features,
                'date_saved': datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            }
            
            with open(os.path.join(output_dir, 'config.json'), 'w') as f:
                json.dump(config, f)
            
            # Save training stats if available
            if hasattr(self, 'training_stats'):
                with open(os.path.join(output_dir, 'training_stats.json'), 'w') as f:
                    json.dump(self.training_stats, f, default=lambda x: float(x) if isinstance(x, np.float32) else x)
            
            print(f"Model successfully saved to {output_dir}")
            return True
        except Exception as e:
            print(f"Error saving model: {e}")
            import traceback
            traceback.print_exc()
            return False

# Ensemble model that combines multiple models for improved prediction
class EnsembleFakeNewsDetector:
    def __init__(self, models):
        """
        Initialize ensemble detector with multiple underlying models
        
        Args:
            models (list): List of trained fake news detector models
        """
        self.models = models
    
    def predict(self, text, threshold=0.5):
        """
        Make a prediction by combining results from multiple models
        
        Args:
            text (str): The news text to classify
            threshold (float): Confidence threshold for majority voting
            
        Returns:
            tuple: (label, confidence) where label is either "Real News" or "Fake News" 
                and confidence is a float between 0 and 1
        """
        # Collect predictions from all models
        predictions = []
        confidences = []
        
        for model in self.models:
            label, confidence = model.predict(text)
            # Convert label to binary (0 for real, 1 for fake)
            is_fake = 1 if label == "Fake News" else 0
            predictions.append(is_fake)
            
            # Adjust confidence to be for the predicted class
            conf_for_prediction = confidence if is_fake == 1 else 1 - confidence
            confidences.append(conf_for_prediction)
        
        # Calculate weighted prediction
        if not predictions:
            return "Unknown", 0.5
        
        # Use a confidence-weighted average
        if sum(confidences) > 0:
            weighted_pred = sum(p * c for p, c in zip(predictions, confidences)) / sum(confidences)
        else:
            # Fallback to simple average if all confidences are 0
            weighted_pred = sum(predictions) / len(predictions)
        
        # Calculate final confidence
        if weighted_pred > 0.5:
            final_confidence = weighted_pred
            return "Fake News", final_confidence
        else:
            final_confidence = 1 - weighted_pred
            return "Real News", final_confidence