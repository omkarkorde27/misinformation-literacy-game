import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from transformers import AutoModel, BertTokenizerFast, get_linear_schedule_with_warmup
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix, classification_report, f1_score
from sklearn.model_selection import train_test_split
import torch.nn.functional as F
from torch.utils.data import TensorDataset, DataLoader, RandomSampler, SequentialSampler

# Specify GPU
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using {device} device")

# Load Dataset
data = pd.read_csv('/content/merged_dataset.csv')

# Add binary target column (Fake=1)
data['Target'] = pd.get_dummies(data.Label)['Fake']

# Check class balance
label_size = [data['Target'].sum(), len(data['Target'])-data['Target'].sum()]
plt.figure(figsize=(8, 6))
plt.pie(label_size, explode=[0.1, 0.1], colors=['firebrick', 'navy'], 
       startangle=90, shadow=True, labels=['Fake', 'True'], autopct='%1.1f%%')
plt.title('Class Distribution')
plt.show()

# Use only content as feature - removed explanation to prevent data leakage
# Also implement text preprocessing
def preprocess_text(text):
    if pd.isna(text):
        return ""
    # Convert to lowercase
    text = text.lower()
    # Remove URLs
    text = re.sub(r'http\S+', '', text)
    # Remove special characters but keep spaces and basic punctuation
    text = re.sub(r'[^\w\s.,!?]', '', text)
    # Remove extra whitespace
    text = re.sub(r'\s+', ' ', text).strip()
    return text

# Import regex module
import re

# Apply preprocessing
data['Feature_Text'] = data['Content'].apply(preprocess_text)

# Train-Validation-Test set split into 70:15:15 ratio
# Train-Temp split
train_text, temp_text, train_labels, temp_labels = train_test_split(
    data['Feature_Text'], 
    data['Target'],
    random_state=2018,
    test_size=0.3,
    stratify=data['Target']  # Stratify by Target instead of Label
)

# Validation-Test split
val_text, test_text, val_labels, test_labels = train_test_split(
    temp_text, 
    temp_labels,
    random_state=2018,
    test_size=0.5,
    stratify=temp_labels
)

# Load BERT model and tokenizer via HuggingFace Transformers
bert = AutoModel.from_pretrained('bert-base-uncased')
tokenizer = BertTokenizerFast.from_pretrained('bert-base-uncased')

# Plot histogram of the number of words in train data
seq_len = [len(title.split()) for title in train_text]
pd.Series(seq_len).hist(bins=40, color='firebrick')
plt.xlabel('Number of Words')
plt.ylabel('Number of texts')
plt.title('Distribution of Text Lengths')
plt.show()

# Determine MAX_LENGTH based on distribution
MAX_LENGTH = 128  # Increased to better capture news content

# Tokenize and encode sequences
def tokenize_and_encode(texts):
    encodings = tokenizer.batch_encode_plus(
        texts.tolist(),
        max_length=MAX_LENGTH,
        padding='max_length',
        truncation=True,
        return_tensors='pt'
    )
    return encodings

# Process datasets
train_encodings = tokenize_and_encode(train_text)
val_encodings = tokenize_and_encode(val_text)
test_encodings = tokenize_and_encode(test_text)

# Create PyTorch datasets
train_dataset = TensorDataset(
    train_encodings['input_ids'],
    train_encodings['attention_mask'],
    torch.tensor(train_labels.tolist(), dtype=torch.long)
)

val_dataset = TensorDataset(
    val_encodings['input_ids'],
    val_encodings['attention_mask'],
    torch.tensor(val_labels.tolist(), dtype=torch.long)
)

test_dataset = TensorDataset(
    test_encodings['input_ids'],
    test_encodings['attention_mask'],
    torch.tensor(test_labels.tolist(), dtype=torch.long)
)

# Data loaders
batch_size = 16
train_dataloader = DataLoader(
    train_dataset,
    sampler=RandomSampler(train_dataset),
    batch_size=batch_size
)

val_dataloader = DataLoader(
    val_dataset,
    sampler=SequentialSampler(val_dataset),
    batch_size=batch_size
)

test_dataloader = DataLoader(
    test_dataset,
    sampler=SequentialSampler(test_dataset),
    batch_size=batch_size
)

# Freeze most of BERT to reduce overfitting
# Only unfreeze the last 2 layers and pooler
for name, param in bert.named_parameters():
    if "encoder.layer.10" in name or "encoder.layer.11" in name or "pooler" in name:
        param.requires_grad = True
    else:
        param.requires_grad = False

# Calculate class weights for imbalanced dataset
total_samples = len(train_labels)
class_0_samples = total_samples - train_labels.sum()
class_1_samples = train_labels.sum()
class_weights = torch.tensor([
    1.0 / (class_0_samples / total_samples),
    1.0 / (class_1_samples / total_samples)
], dtype=torch.float32).to(device)

# Simplified BERT architecture with better regularization
class BERT_Classifier(nn.Module):
    def __init__(self, bert, dropout_rate=0.7):  # Even more aggressive dropout
        super(BERT_Classifier, self).__init__()
        self.bert = bert
        
        # Extremely simplified classification head with very strong regularization
        self.dropout1 = nn.Dropout(dropout_rate)
        self.fc1 = nn.Linear(768, 64)  # Much smaller intermediate layer
        self.dropout2 = nn.Dropout(dropout_rate)
        self.fc2 = nn.Linear(64, 2)
        
    def forward(self, input_ids, attention_mask):
        # Get BERT embeddings
        outputs = self.bert(input_ids=input_ids, attention_mask=attention_mask)
        pooled_output = outputs.pooler_output
        
        # Apply heavy dropout at every step
        x = self.dropout1(pooled_output)
        x = F.relu(self.fc1(x))
        x = self.dropout2(x)
        x = self.fc2(x)
        
        return F.log_softmax(x, dim=1)

# Initialize model
model = BERT_Classifier(bert)
model = model.to(device)

# Define optimizer with stronger weight decay for regularization
optimizer = torch.optim.AdamW(
    [p for p in model.parameters() if p.requires_grad],
    lr=1e-5,  # Lower learning rate
    weight_decay=0.1  # Stronger L2 regularization
)

# Create learning rate scheduler with warmup
num_epochs = 10  # Reduced to prevent overfitting
total_steps = len(train_dataloader) * num_epochs
scheduler = get_linear_schedule_with_warmup(
    optimizer,
    num_warmup_steps=int(0.1 * total_steps),
    num_training_steps=total_steps
)

# Define the loss function with class weights
criterion = nn.NLLLoss(weight=class_weights)

# Training and evaluation functions
def train_epoch(model, dataloader, optimizer, scheduler, criterion, device):
    model.train()
    total_loss = 0
    all_preds = []
    all_labels = []
    
    for batch in dataloader:
        optimizer.zero_grad()
        
        input_ids = batch[0].to(device)
        attention_mask = batch[1].to(device)
        labels = batch[2].to(device)
        
        outputs = model(input_ids, attention_mask)
        
        loss = criterion(outputs, labels)
        total_loss += loss.item()
        
        # Backpropagation
        loss.backward()
        # Gradient clipping to prevent exploding gradients
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()
        scheduler.step()
        
        # Store predictions and labels for metrics calculation
        _, preds = torch.max(outputs, dim=1)
        all_preds.extend(preds.cpu().tolist())
        all_labels.extend(labels.cpu().tolist())
    
    # Calculate metrics
    avg_loss = total_loss / len(dataloader)
    f1 = f1_score(all_labels, all_preds, average='weighted')
    
    return avg_loss, f1

def evaluate(model, dataloader, criterion, device):
    model.eval()
    total_loss = 0
    all_preds = []
    all_labels = []
    
    with torch.no_grad():
        for batch in dataloader:
            input_ids = batch[0].to(device)
            attention_mask = batch[1].to(device)
            labels = batch[2].to(device)
            
            outputs = model(input_ids, attention_mask)
            
            loss = criterion(outputs, labels)
            total_loss += loss.item()
            
            _, preds = torch.max(outputs, dim=1)
            all_preds.extend(preds.cpu().tolist())
            all_labels.extend(labels.cpu().tolist())
    
    # Calculate metrics
    avg_loss = total_loss / len(dataloader)
    f1 = f1_score(all_labels, all_preds, average='weighted')
    
    return avg_loss, f1, all_preds, all_labels

# Early stopping parameters
early_stopping_patience = 3
best_val_f1 = 0  # Track F1 score instead of loss for early stopping
early_stopping_counter = 0

# Training loop with early stopping
train_losses = []
val_losses = []
train_f1_scores = []
val_f1_scores = []

for epoch in range(num_epochs):
    print(f'\nEpoch {epoch + 1}/{num_epochs}')
    
    # Train
    train_loss, train_f1 = train_epoch(model, train_dataloader, optimizer, scheduler, criterion, device)
    train_losses.append(train_loss)
    train_f1_scores.append(train_f1)
    
    # Evaluate
    val_loss, val_f1, _, _ = evaluate(model, val_dataloader, criterion, device)
    val_losses.append(val_loss)
    val_f1_scores.append(val_f1)
    
    print(f'Training Loss: {train_loss:.4f}, Training F1: {train_f1:.4f}')
    print(f'Validation Loss: {val_loss:.4f}, Validation F1: {val_f1:.4f}')
    
    # Check if this is the best model so far based on F1 score
    if val_f1 > best_val_f1:
        best_val_f1 = val_f1
        # Save the model
        torch.save(model.state_dict(), 'best_bert_model.pt')
        print("Model saved!")
        early_stopping_counter = 0
    else:
        early_stopping_counter += 1
        if early_stopping_counter >= early_stopping_patience:
            print(f"Early stopping triggered after {epoch + 1} epochs")
            break

# Load the best model for evaluation
model.load_state_dict(torch.load('best_bert_model.pt'))

# Plot training and validation metrics
plt.figure(figsize=(12, 5))
plt.subplot(1, 2, 1)
plt.plot(train_losses, label='Training Loss')
plt.plot(val_losses, label='Validation Loss')
plt.title('Loss over Epochs')
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.legend()

plt.subplot(1, 2, 2)
plt.plot(train_f1_scores, label='Training F1 Score')
plt.plot(val_f1_scores, label='Validation F1 Score')
plt.title('F1 Score over Epochs')
plt.xlabel('Epochs')
plt.ylabel('F1 Score')
plt.legend()
plt.tight_layout()
plt.show()

# Evaluate on test set
test_loss, test_f1, test_preds, test_labels = evaluate(model, test_dataloader, criterion, device)
print(f'\nTest Loss: {test_loss:.4f}, Test F1: {test_f1:.4f}')

# Print detailed classification report
print(classification_report(test_labels, test_preds))

# Display confusion matrix
cm = confusion_matrix(test_labels, test_preds)
plt.figure(figsize=(8, 6))
plt.imshow(cm, interpolation='nearest', cmap=plt.cm.Blues)
plt.title('Confusion Matrix')
plt.colorbar()
tick_marks = np.arange(2)
plt.xticks(tick_marks, ['True', 'Fake'])
plt.yticks(tick_marks, ['True', 'Fake'])

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

# Function to test on unseen data
def predict_fakeness(text_list):
    model.eval()
    encodings = tokenizer.batch_encode_plus(
        text_list,
        max_length=MAX_LENGTH,
        padding='max_length',
        truncation=True,
        return_tensors='pt'
    )
    
    input_ids = encodings['input_ids'].to(device)
    attention_mask = encodings['attention_mask'].to(device)
    
    with torch.no_grad():
        outputs = model(input_ids, attention_mask)
        _, preds = torch.max(outputs, dim=1)
    
    results = []
    for i, text in enumerate(text_list):
        label = "Fake" if preds[i].item() == 1 else "Real"
        confidence = torch.exp(outputs[i, preds[i]]).item()
        results.append({
            "text": text,
            "prediction": label,
            "confidence": f"{confidence:.2%}"
        })
    
    return results

# Test with example news headlines
test_headlines = [
    "Donald Trump Sends Out Embarrassing New Year's Eve Message; This is Disturbing",     # Likely Fake
    "WATCH: George W. Bush Calls Out Trump For Supporting White Supremacy",               # Likely Fake
    "U.S. lawmakers question businessman at 2016 Trump Tower meeting: sources",           # Likely Real
    "Trump administration issues new rules on U.S. visa waivers"                          # Likely Real
]

predictions = predict_fakeness(test_headlines)
for pred in predictions:
    print(f"Headline: {pred['text']}")
    print(f"Prediction: {pred['prediction']} (Confidence: {pred['confidence']})")
    print("-" * 80)

# Implement K-fold cross-validation for better model evaluation
from sklearn.model_selection import KFold

def perform_kfold_validation(data, num_folds=5):
    kf = KFold(n_splits=num_folds, shuffle=True, random_state=42)
    
    fold_results = []
    
    # Combine data and labels for folding
    texts = data['Feature_Text'].values
    labels = data['Target'].values
    
    for fold, (train_idx, val_idx) in enumerate(kf.split(texts)):
        print(f"\nTraining fold {fold+1}/{num_folds}")
        
        # Split data for this fold
        fold_train_texts = texts[train_idx]
        fold_train_labels = labels[train_idx]
        fold_val_texts = texts[val_idx]
        fold_val_labels = labels[val_idx]
        
        # Convert to pandas Series to maintain compatibility with existing code
        fold_train_texts = pd.Series(fold_train_texts)
        fold_train_labels = pd.Series(fold_train_labels)
        fold_val_texts = pd.Series(fold_val_texts)
        fold_val_labels = pd.Series(fold_val_labels)
        
        # Process data for this fold
        fold_train_encodings = tokenize_and_encode(fold_train_texts)
        fold_val_encodings = tokenize_and_encode(fold_val_texts)
        
        # Create datasets and dataloaders
        fold_train_dataset = TensorDataset(
            fold_train_encodings['input_ids'],
            fold_train_encodings['attention_mask'],
            torch.tensor(fold_train_labels.tolist(), dtype=torch.long)
        )
        
        fold_val_dataset = TensorDataset(
            fold_val_encodings['input_ids'],
            fold_val_encodings['attention_mask'],
            torch.tensor(fold_val_labels.tolist(), dtype=torch.long)
        )
        
        fold_train_dataloader = DataLoader(
            fold_train_dataset,
            sampler=RandomSampler(fold_train_dataset),
            batch_size=batch_size
        )
        
        fold_val_dataloader = DataLoader(
            fold_val_dataset,
            sampler=SequentialSampler(fold_val_dataset),
            batch_size=batch_size
        )
        
        # Initialize new model for this fold
        fold_bert = AutoModel.from_pretrained('bert-base-uncased')
        # Apply same freezing pattern as before
        for name, param in fold_bert.named_parameters():
            if "encoder.layer.10" in name or "encoder.layer.11" in name or "pooler" in name:
                param.requires_grad = True
            else:
                param.requires_grad = False
                
        fold_model = BERT_Classifier(fold_bert)
        fold_model = fold_model.to(device)
        
        # Create optimizer and scheduler
        fold_optimizer = torch.optim.AdamW(
            [p for p in fold_model.parameters() if p.requires_grad],
            lr=1e-5,
            weight_decay=0.1
        )
        
        fold_total_steps = len(fold_train_dataloader) * 5  # 5 epochs per fold
        fold_scheduler = get_linear_schedule_with_warmup(
            fold_optimizer,
            num_warmup_steps=int(0.1 * fold_total_steps),
            num_training_steps=fold_total_steps
        )
        
        # Train for a few epochs
        best_val_f1 = 0
        
        for epoch in range(5):  # 5 epochs per fold
            # Train
            train_loss, train_f1 = train_epoch(fold_model, fold_train_dataloader, fold_optimizer, fold_scheduler, criterion, device)
            
            # Evaluate
            val_loss, val_f1, _, _ = evaluate(fold_model, fold_val_dataloader, criterion, device)
            
            print(f'Fold {fold+1}, Epoch {epoch+1}: Train F1 = {train_f1:.4f}, Val F1 = {val_f1:.4f}')
            
            if val_f1 > best_val_f1:
                best_val_f1 = val_f1
        
        # Store fold results
        fold_results.append({
            'fold': fold+1,
            'best_val_f1': best_val_f1
        })
    
    # Compute average F1 across all folds
    avg_f1 = sum(result['best_val_f1'] for result in fold_results) / num_folds
    print(f"\nCross-validation complete. Average F1 score: {avg_f1:.4f}")
    
    return fold_results

# Additional function to collect misclassifications
def collect_misclassifications(model, dataloader, device):
    model.eval()
    misclassified = []
    
    with torch.no_grad():
        for batch in dataloader:
            input_ids = batch[0].to(device)
            attention_mask = batch[1].to(device)
            labels = batch[2].to(device)
            
            outputs = model(input_ids, attention_mask)
            _, preds = torch.max(outputs, dim=1)
            
            # Find indices of misclassified examples
            misclassified_idx = (preds != labels).nonzero(as_tuple=True)[0]
            
            for idx in misclassified_idx:
                misclassified.append({
                    "text_idx": idx.item(),
                    "true_label": labels[idx].item(),
                    "predicted": preds[idx].item(),
                    "confidence": torch.exp(outputs[idx, preds[idx]]).item()
                })
    
    return misclassified

# Analyze misclassifications in test set
test_misclassified = collect_misclassifications(model, test_dataloader, device)
print(f"Number of misclassified examples: {len(test_misclassified)}")

# Print some examples of misclassified texts if available
if test_misclassified:
    print("\nExamples of misclassified texts:")
    for i, misc in enumerate(test_misclassified[:5]):  # Show first 5 misclassifications
        idx = misc["text_idx"]
        batch_idx = idx // batch_size
        item_idx = idx % batch_size
        # Get the original text from test_text
        text = test_text.iloc[batch_idx * batch_size + item_idx]
        true_label = "Fake" if misc["true_label"] == 1 else "Real"
        pred_label = "Fake" if misc["predicted"] == 1 else "Real"
        print(f"{i+1}. Text: {text[:100]}...")
        print(f"   True: {true_label}, Predicted: {pred_label}, Confidence: {misc['confidence']:.2%}")