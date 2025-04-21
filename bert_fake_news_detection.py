# Modified BERT Fake News Detection System to Prevent Overfitting
# Key changes:
# 1. Only use Content field (no Explanations) to prevent data leakage
# 2. Increased regularization with stronger dropout and weight decay
# 3. Added data augmentation 
# 4. Using cross-validation for better evaluation

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

# For text processing
import nltk
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from nltk.tokenize import word_tokenize

# For the ML model
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from transformers import BertTokenizer, AutoModel
from transformers import get_linear_schedule_with_warmup
from sklearn.model_selection import train_test_split, StratifiedKFold
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

# Text preprocessing function with more aggressive cleaning
def preprocess_text(text):
    if not isinstance(text, str):
        return ""
    # Convert to lowercase
    text = text.lower()
    # Remove URLs
    text = re.sub(r'http\S+|www\S+|https\S+', '', text, flags=re.MULTILINE)
    # Remove special characters, numbers, and @mentions
    text = re.sub(r'\@\w+|\#|\d+', '', text)
    # Remove punctuation
    text = text.translate(str.maketrans('', '', string.punctuation))
    # Remove extra whitespace
    text = re.sub(r'\s+', ' ', text).strip()
    # Optional: Remove common stopwords
    words = text.split()
    words = [word for word in words if word not in stop_words]
    text = ' '.join(words)
    return text

# Simple data augmentation techniques
def augment_text(text, augmentation_probability=0.3):
    """Apply simple text augmentation techniques"""
    if np.random.random() < augmentation_probability:
        words = text.split()
        # Skip augmentation for very short texts
        if len(words) <= 5:
            return text
            
        # Randomly choose an augmentation technique
        technique = np.random.choice(['drop', 'shuffle', 'both'])
        
        if technique == 'drop' or technique == 'both':
            # Randomly drop 10-20% of words
            drop_ratio = np.random.uniform(0.1, 0.2)
            n_drop = max(1, int(len(words) * drop_ratio))
            drop_indices = np.random.choice(len(words), n_drop, replace=False)
            words = [w for i, w in enumerate(words) if i not in drop_indices]
            
        if technique == 'shuffle' or technique == 'both':
            # Randomly shuffle small segments (2-4 words)
            if len(words) > 4:  # Only apply shuffling if we have enough words
                n_segments = max(1, len(words) // 4)
                for _ in range(n_segments):
                    # Make sure we have a valid range for segment_len (at least 2, at most half of words but not exceeding 5)
                    max_segment = min(5, len(words)//2)
                    if max_segment < 2:  # Skip shuffling if we can't form a valid segment
                        continue
                    segment_len = np.random.randint(2, max_segment + 1)  # +1 because randint upper bound is exclusive
                    start_idx = np.random.randint(0, max(1, len(words) - segment_len))
                    segment = words[start_idx:start_idx + segment_len]
                    np.random.shuffle(segment)
                    words[start_idx:start_idx + segment_len] = segment
        
        return ' '.join(words)
    else:
        return text

# Improved BERT architecture with stronger regularization
class BERT_Improved(nn.Module):
    def __init__(self, bert, dropout_rate=0.5):
        super(BERT_Improved, self).__init__()
        self.bert = bert
        
        # Classification head with strong regularization
        self.dropout1 = nn.Dropout(dropout_rate)
        self.fc1 = nn.Linear(768, 256)
        self.bn1 = nn.BatchNorm1d(256)
        
        self.dropout2 = nn.Dropout(dropout_rate + 0.1)  # Increased dropout
        self.fc2 = nn.Linear(256, 64)
        self.bn2 = nn.BatchNorm1d(64)
        
        self.dropout3 = nn.Dropout(dropout_rate + 0.2)  # Even more dropout
        self.fc3 = nn.Linear(64, 2)
        
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
        
        # Output layer with additional dropout
        x = self.dropout3(x)
        x = self.fc3(x)
        
        return F.log_softmax(x, dim=1)

# Custom dataset for BERT with augmentation
class NewsDataset(Dataset):
    def __init__(self, texts, labels, tokenizer, max_length=128, augment=False):
        self.texts = texts
        self.labels = labels
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.augment = augment
        
    def __len__(self):
        return len(self.texts)
    
    def __getitem__(self, idx):
        text = str(self.texts[idx])
        
        # Apply augmentation during training if enabled
        if self.augment:
            text = augment_text(text)
            
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
        
        if model_path and os.path.exists(model_path) and os.path.isfile(os.path.join(model_path, 'model.pt')):
            # Load the saved model
            self.bert = AutoModel.from_pretrained('bert-base-uncased')
            self.model = BERT_Improved(self.bert)
            self.model.load_state_dict(torch.load(os.path.join(model_path, 'model.pt'), map_location=device))
            print(f"Loaded pre-trained model from {model_path}")
        else:
            # Initialize a new model
            self.bert = AutoModel.from_pretrained('bert-base-uncased')
            
            # Freeze most BERT layers to prevent overfitting
            for name, param in self.bert.named_parameters():
                if "encoder.layer.10" in name or "encoder.layer.11" in name or "pooler" in name:
                    param.requires_grad = True
                else:
                    param.requires_grad = False
                    
            self.model = BERT_Improved(self.bert)
            print("Initialized new improved BERT model")
        
        self.model.to(device)
        
    def train(self, data_path, batch_size=16, epochs=10, learning_rate=2e-5, test_size=0.2, random_state=42, 
              use_cross_validation=True, n_folds=5, use_augmentation=True):
        """Train the BERT model on the dataset with cross-validation"""
        try:
            # Load and preprocess the dataset
            df = pd.read_csv(data_path)
            print(f"Loaded dataset with {len(df)} rows")
            
            # Check if required columns exist
            if 'Content' not in df.columns or 'Label' not in df.columns:
                print(f"Dataset missing required columns. Found: {df.columns.tolist()}")
                return False
            
            # Preprocess the text data - ONLY use Content field (important to prevent data leakage)
            print("Preprocessing text data...")
            print("ONLY using Content field to prevent data leakage")
            df['processed_content'] = df['Content'].apply(preprocess_text)
            
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
                    use_augmentation=use_augmentation
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
                    use_augmentation=use_augmentation
                )
        
        except Exception as e:
            print(f"Error during training: {e}")
            import traceback
            traceback.print_exc()
            return False
    
    def _train_with_cross_validation(self, texts, labels, class_weights, n_folds=5, batch_size=16, 
                              epochs=10, learning_rate=2e-5, random_state=42, use_augmentation=True):
        """Train model using simplified manual cross-validation for better robustness"""
        # Convert inputs to numpy arrays
        texts = np.array(texts)
        labels = np.array(labels)
        
        # Create indices array
        indices = np.arange(len(texts))
        
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
                indices, labels, test_size=1/n_folds, random_state=fold_random_state, 
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
                self.max_length, augment=use_augmentation
            )
            val_dataset = NewsDataset(val_texts, val_labels, self.tokenizer, self.max_length)
            test_dataset = NewsDataset(test_texts, test_labels, self.tokenizer, self.max_length)
            
            # Create data loaders
            train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
            val_loader = DataLoader(val_dataset, batch_size=batch_size)
            test_loader = DataLoader(test_dataset, batch_size=batch_size)
            
            # Re-initialize model for this fold
            self.bert = AutoModel.from_pretrained('bert-base-uncased')
            # Freeze most BERT layers
            for name, param in self.bert.named_parameters():
                if "encoder.layer.10" in name or "encoder.layer.11" in name or "pooler" in name:
                    param.requires_grad = True
                else:
                    param.requires_grad = False
            
            self.model = BERT_Improved(self.bert)
            self.model.to(device)
            
            # Initialize optimizer with weight decay for regularization
            optimizer = AdamW(
                [p for p in self.model.parameters() if p.requires_grad],
                lr=learning_rate,
                weight_decay=0.1  # Stronger L2 regularization
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
            best_model_state = None
            early_stopping_patience = 3
            early_stopping_counter = 0
            training_stats = []
            
        # Continue with the training loop and everything else...
            
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
                train_accuracy = accuracy_score(train_true_labels, train_preds)
                
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
                print(f"Training: Loss={avg_train_loss:.4f}, F1={train_f1:.4f}, Acc={train_accuracy:.4f}")
                print(f"Validation: Loss={avg_val_loss:.4f}, F1={val_f1:.4f}, Acc={val_accuracy:.4f}")
                
                # Save the best model based on F1 score
                if val_f1 > best_val_f1:
                    best_val_f1 = val_f1
                    best_model_state = {
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
                    'train_accuracy': train_accuracy,
                    'val_loss': avg_val_loss,
                    'val_f1': val_f1,
                    'val_accuracy': val_accuracy
                })
            
            # Load the best model for evaluation
            if best_model_state:
                self.model.load_state_dict(best_model_state['model_state_dict'])
                print(f"Loaded best model from epoch {best_model_state['epoch']} with validation F1: {best_model_state['val_f1']:.4f}")
            
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
            
            # Store fold results
            fold_results.append({
                'fold': fold + 1,
                'best_val_f1': best_val_f1,
                'test_f1': test_f1,
                'test_accuracy': test_accuracy
            })
            
            # Keep track of best model across folds
            if test_f1 > best_overall_f1:
                best_overall_f1 = test_f1
                best_fold_model = {
                    'fold': fold + 1,
                    'model_state_dict': self.model.state_dict(),
                    'test_f1': test_f1,
                    'test_accuracy': test_accuracy
                }
        
        # Print summary of all folds
        print("\n--- Cross-validation Summary ---")
        avg_test_f1 = sum(fold['test_f1'] for fold in fold_results) / len(fold_results)
        avg_test_acc = sum(fold['test_accuracy'] for fold in fold_results) / len(fold_results)
        
        print(f"Average Test F1 Score: {avg_test_f1:.4f}")
        print(f"Average Test Accuracy: {avg_test_acc:.4f}")
        print(f"Best Test F1 Score: {best_overall_f1:.4f} (Fold {best_fold_model['fold']})")
        
        # Use the best model from cross-validation
        if best_fold_model:
            self.model.load_state_dict(best_fold_model['model_state_dict'])
            print(f"Using best model from fold {best_fold_model['fold']}")
            
        # Store training stats
        self.training_stats = {
            'fold_results': fold_results,
            'avg_test_f1': avg_test_f1,
            'avg_test_acc': avg_test_acc,
            'best_fold': best_fold_model['fold'],
            'best_test_f1': best_overall_f1
        }
        
        return True
    
    def predict(self, text, threshold=0.5):
        """
        Predict whether the given text is fake news or real news
        
        Args:
            text (str): The news text to classify
            threshold (float): Confidence threshold for classifying as fake news
            
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
        
        # Get prediction
        with torch.no_grad():
            outputs = self.model(input_ids, attention_mask)
            probabilities = torch.exp(outputs)  # Convert log_softmax to probabilities
            _, preds = torch.max(outputs, dim=1)
            confidence = probabilities[0][preds].item()
        
        # Map prediction to label
        if preds.item() == 1:
            return "Fake News", confidence
        else:
            return "Real News", confidence

    def _train_with_split(self, texts, labels, class_weights, batch_size=16, epochs=10, 
                           learning_rate=2e-5, test_size=0.2, random_state=42, use_augmentation=True):
        """Train model using a simple train/val/test split"""
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
            self.max_length, augment=use_augmentation
        )
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
            weight_decay=0.1  # Stronger L2 regularization
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
            train_accuracy = accuracy_score(train_true_labels, train_preds)
            
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
            print(f"Training: Loss={avg_train_loss:.4f}, F1={train_f1:.4f}, Acc={train_accuracy:.4f}")
            print(f"Validation: Loss={avg_val_loss:.4f}, F1={val_f1:.4f}, Acc={val_accuracy:.4f}")
            
            # Save the best model based on F1 score
            if val_f1 > best_val_f1:
                best_val_f1 = val_f1
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
                'train_accuracy': train_accuracy,
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
        
        # Save the plot instead of displaying it
        try:
            plot_path = f'confusion_matrix_{time.strftime("%Y%m%d-%H%M%S")}.png'
            plt.savefig(plot_path)
            print(f"Confusion matrix saved to {plot_path}")
            plt.close()  # Close the plot to avoid displaying it
        except Exception as e:
            print(f"Could not save confusion matrix plot: {e}")
            plt.close()  # Make sure to close the plot even if saving fails
        
        # Store training stats
        self.training_stats = training_stats
        
        # Store test data for later use
        self.test_texts = test_texts
        self.test_labels = test_labels
        
        return True
    
    def save_model(self, output_dir="bert_fake_news_model"):
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
            
            print(f"Model successfully saved to {output_dir}")
            return True
        except Exception as e:
            print(f"Error saving model: {e}")
            import traceback
            traceback.print_exc()
            return False