import torch
from torch.utils.data import Dataset
from transformers import AutoTokenizer, AutoModelForSequenceClassification
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix, classification_report, f1_score
import os

class BioPaperDataset(Dataset):
    def __init__(self, texts, labels, tokenizer, max_length=512):
        self.texts = [str(text) for text in texts]
        self.labels = labels
        self.tokenizer = tokenizer
        self.max_length = max_length

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, idx):
        text = str(self.texts[idx])
        if not text:
            text = " "
            
        enc = self.tokenizer(
            text,
            truncation=True,
            padding='max_length',
            max_length=self.max_length,
            return_tensors='pt'
        )
        item = {k: v.squeeze(0) for k, v in enc.items()}
        item['labels'] = torch.tensor(self.labels[idx], dtype=torch.long)
        return item


def plot_confusion_matrix(y_true, y_pred, classes):
    """Plot confusion matrix"""
    cm = confusion_matrix(y_true, y_pred)
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                xticklabels=classes, yticklabels=classes)
    plt.title('Confusion Matrix')
    plt.ylabel('True Label')
    plt.xlabel('Predicted Label')
    plt.tight_layout()
    
    os.makedirs('plots', exist_ok=True)
    filename = 'plots/confusion_matrix.png'
    plt.savefig(filename)
    plt.close()
    return filename


def main():
    # Load test data
    print("Loading test data...")
    df = pd.read_excel('testset_395.xlsx')
    
    texts = df['content'].fillna('').tolist()
    labels = df['label'].tolist()
    
    print(f"Loaded {len(texts)} test articles")
    
    # Load model and tokenizer
    print("Loading model...")
    model_name = "dmis-lab/biobert-large-cased-v1.1"
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    
    # Try to find the best model
    import glob
    model_path = "./results_fold1_f1_0.8530/checkpoint-116"
    print(f"Using model: {model_path}")
    
    try:
        model = AutoModelForSequenceClassification.from_pretrained(
            model_path,
            num_labels=2
        )
        model.eval()
        
        # Prepare test dataset
        test_dataset = BioPaperDataset(texts, labels, tokenizer)
        
        # Make predictions
        print("Making predictions...")
        predictions = []
        true_labels = []
        
        with torch.no_grad():
            for idx, item in enumerate(test_dataset):
                input_ids = item['input_ids'].unsqueeze(0)
                attention_mask = item['attention_mask'].unsqueeze(0)
                outputs = model(input_ids, attention_mask=attention_mask)
                logits = outputs.logits
                
                probs = torch.softmax(logits, dim=-1)
                pred_prob = probs[0, 1].item()
                
                pred = 1 if pred_prob > 0.5 else 0
                predictions.append(pred)
                true_labels.append(item['labels'].item())
        
        # Results using default threshold 0.5
        print(f"\nModel evaluation results:")
        
        print(f"\n--- Final Test Results ---")
        print(classification_report(true_labels, predictions))
        test_f1 = f1_score(true_labels, predictions, average='macro')
        
        # Plot confusion matrix
        confusion_file = plot_confusion_matrix(
            true_labels, predictions, 
            classes=['T0', 'T2/4']
        )
        print(f"Confusion matrix saved to {confusion_file}")
        
        
        print(f"\nFINAL RESULT: Test F1 Score = {test_f1:.4f}")
        
    except Exception as e:
        print(f"Error loading model {model_path}: {e}")

if __name__ == "__main__":
    main()