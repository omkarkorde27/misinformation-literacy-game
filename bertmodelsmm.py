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

# Combine content with explanation to provide more context
# Since explanations provide clues about fakeness, this can help the model learn
data['Full_Text'] = data['Content'].fillna('') + ' ' + data['Explanation'].fillna('')

# Train-Validation-Test set split into 70:15:15 ratio
# Train-Temp split
train_text, temp_text, train_labels, temp_labels = train_test_split(
    data['Full_Text'], 
    data['Target'],
    random_state=2018,
    test_size=0.3,
    stratify=data['Label']
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

# Based on the histogram, increase MAX_LENGTH to capture more context
MAX_LENGTH = 64  # Increased from 15 to capture more content

# Tokenize and encode sequences
def tokenize_and_encode(texts):
    encodings = tokenizer.batch_encode_plus(
        texts.tolist(),
        max_length=MAX_LENGTH,
        padding='max_length',
        truncation=True,
        return_tensors='pt'
    )
    # Ensure correct dtype
    encodings['input_ids'] = encodings['input_ids'].to(dtype=torch.int64)
    encodings['attention_mask'] = encodings['attention_mask'].to(dtype=torch.int64)
    return encodings

# Process datasets
train_encodings = tokenize_and_encode(train_text)
val_encodings = tokenize_and_encode(val_text)
test_encodings = tokenize_and_encode(test_text)

# Create PyTorch datasets - ensure all tensors have proper types
train_dataset = TensorDataset(
    train_encodings['input_ids'].to(torch.int64),
    train_encodings['attention_mask'].to(torch.int64),
    torch.tensor(train_labels.tolist(), dtype=torch.long)
)

val_dataset = TensorDataset(
    val_encodings['input_ids'].to(torch.int64),
    val_encodings['attention_mask'].to(torch.int64),
    torch.tensor(val_labels.tolist(), dtype=torch.long)
)

test_dataset = TensorDataset(
    test_encodings['input_ids'].to(torch.int64),
    test_encodings['attention_mask'].to(torch.int64),
    torch.tensor(test_labels.tolist(), dtype=torch.long)
)

# Data loaders
batch_size = 16  # Reduced batch size to accommodate longer sequences
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

# Unfreeze the last 2 layers of BERT
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

# Initialize model
model = BERT_Improved(bert)
model = model.to(device)

# Define optimizer
optimizer = torch.optim.AdamW(
    [p for p in model.parameters() if p.requires_grad],
    lr=2e-5,  # Slightly lower learning rate
    weight_decay=0.01  # L2 regularization
)

# Create learning rate scheduler
num_epochs = 15  # Increased from 10
total_steps = len(train_dataloader) * num_epochs
scheduler = get_linear_schedule_with_warmup(
    optimizer,
    num_warmup_steps=int(0.1 * total_steps),  # 10% of total steps as warmup
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
        
        # Ensure correct dtype when transferring to device
        input_ids = batch[0].to(device, dtype=torch.int64)
        attention_mask = batch[1].to(device, dtype=torch.int64)
        labels = batch[2].to(device, dtype=torch.long)
        
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
            # Ensure correct dtype when transferring to device
            input_ids = batch[0].to(device, dtype=torch.int64)
            attention_mask = batch[1].to(device, dtype=torch.int64)
            labels = batch[2].to(device, dtype=torch.long)
            
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
best_val_loss = float('inf')
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
    
    # Check if this is the best model so far
    if val_loss < best_val_loss:
        best_val_loss = val_loss
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
    
    # Ensure correct dtype when transferring to device
    input_ids = encodings['input_ids'].to(device, dtype=torch.int64)
    attention_mask = encodings['attention_mask'].to(device, dtype=torch.int64)
    
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