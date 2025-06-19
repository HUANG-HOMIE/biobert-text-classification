import torch
from torch.utils.data import Dataset
from transformers import AutoTokenizer, AutoModelForSequenceClassification
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix, classification_report, f1_score
import os
import glob

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

def plot_confusion_matrix(y_true, y_pred, classes, title_suffix=""):
    """Plot confusion matrix"""
    cm = confusion_matrix(y_true, y_pred)
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                xticklabels=classes, yticklabels=classes)
    plt.title(f'Confusion Matrix{title_suffix}')
    plt.ylabel('True Label')
    plt.xlabel('Predicted Label')
    plt.tight_layout()
    
    os.makedirs('plots', exist_ok=True)
    filename = f'plots/confusion_matrix_ensemble{title_suffix.lower().replace(" ", "_")}.png'
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
    
    # Load tokenizer
    print("Loading tokenizer...")
    model_name = "dmis-lab/biobert-large-cased-v1.1"
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    
    # Find all fold models
    print("Finding all fold models...")
    fold_models = glob.glob("results_fold*_f1_*/checkpoint-*")
    
    if not fold_models:
        print("No fold models found! Please check the model paths.")
        return
    
    # Sort by fold number for consistent ordering
    def extract_fold_number(path):
        try:
            
            fold_part = path.split('results_fold')[1].split('_')[0]
            return int(fold_part)
        except:
            return 0
    
    fold_models.sort(key=extract_fold_number)
    
    print(f"Found {len(fold_models)} fold models:")
    for i, model_path in enumerate(fold_models, 1):
        f1_score_str = model_path.split('_f1_')[1].split('/')[0]
        print(f"  Fold {i}: {model_path} (F1: {f1_score_str})")
    
    # Prepare test dataset
    test_dataset = BioPaperDataset(texts, labels, tokenizer)
    
    # Load all models and collect predictions
    all_predictions = []
    all_probabilities = []
    
    for i, model_path in enumerate(fold_models, 1):
        print(f"\nLoading and predicting with Fold {i} model...")
        try:
            model = AutoModelForSequenceClassification.from_pretrained(
                model_path,
                num_labels=2
            )
            model.eval()
            
            fold_predictions = []
            fold_probabilities = []
            
            with torch.no_grad():
                for idx, item in enumerate(test_dataset):
                    input_ids = item['input_ids'].unsqueeze(0)
                    attention_mask = item['attention_mask'].unsqueeze(0)
                    outputs = model(input_ids, attention_mask=attention_mask)
                    logits = outputs.logits
                    
                    probs = torch.softmax(logits, dim=-1)
                    pred_prob = probs[0, 1].item()  # T2/4類別機率
                    fold_probabilities.append(pred_prob)
                    
                    pred = 1 if pred_prob > 0.5 else 0
                    fold_predictions.append(pred)
            
            all_predictions.append(fold_predictions)
            all_probabilities.append(fold_probabilities)
            print(f"Fold {i} completed.")
            
        except Exception as e:
            print(f"Error loading Fold {i} model {model_path}: {e}")
            continue
    
    if not all_predictions:
        print("No models were successfully loaded!")
        return
    
    print(f"\nSuccessfully loaded {len(all_predictions)} models")
    
    # Ensemble predictions using Soft Voting (Average probabilities)
    print("\nPerforming ensemble predictions using Soft Voting...")
    
    ensemble_predictions = []
    avg_probabilities = []
    for i in range(len(test_dataset)):
        probs = [prob[i] for prob in all_probabilities]
        avg_prob = np.mean(probs)
        avg_probabilities.append(avg_prob)
        pred = 1 if avg_prob > 0.5 else 0
        ensemble_predictions.append(pred)
    
    # Get true labels
    true_labels = [item['labels'].item() for item in test_dataset]
    
    # Evaluate individual models
    print("\n" + "="*60)
    print("INDIVIDUAL MODEL RESULTS:")
    print("="*60)
    
    individual_f1s = []
    for i, predictions in enumerate(all_predictions, 1):
        f1 = f1_score(true_labels, predictions, average='macro')
        individual_f1s.append(f1)
        print(f"Fold {i} F1 Score: {f1:.4f}")
    
    print(f"\nIndividual Models Average F1: {np.mean(individual_f1s):.4f} ± {np.std(individual_f1s):.4f}")
    
    # Evaluate ensemble
    print("\n" + "="*60)
    print("ENSEMBLE RESULTS (Soft Voting):")
    print("="*60)
    
    ensemble_f1 = f1_score(true_labels, ensemble_predictions, average='macro')
    print(f"\nEnsemble F1 Score: {ensemble_f1:.4f}")
    print(classification_report(true_labels, ensemble_predictions))
    
    # Plot confusion matrix
    confusion_file = plot_confusion_matrix(
        true_labels, ensemble_predictions, 
        classes=['T0', 'T2/4'],
        title_suffix=" (Soft Voting Ensemble)"
    )
    print(f"\nConfusion matrix saved to {confusion_file}")
    
    # Create comparison plot
    plt.figure(figsize=(12, 6))
    
    # Individual model F1 scores
    plt.subplot(1, 2, 1)
    plt.bar(range(1, len(individual_f1s)+1), individual_f1s, alpha=0.7, label='Individual Models')
    plt.axhline(y=np.mean(individual_f1s), color='r', linestyle='--', label=f'Average: {np.mean(individual_f1s):.4f}')
    plt.xlabel('Fold')
    plt.ylabel('F1 Score')
    plt.title('Individual Model Performance')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    # Ensemble result
    plt.subplot(1, 2, 2)
    methods = ['Ensemble\n(Soft Voting)']
    scores = [ensemble_f1]
    colors = ['lightgreen']
    
    bars = plt.bar(methods, scores, color=colors, alpha=0.8)
    plt.ylabel('F1 Score')
    plt.title('Final Ensemble Result')
    plt.grid(True, alpha=0.3)
    
    # Add value labels on bars
    for bar, score in zip(bars, scores):
        plt.text(bar.get_x() + bar.get_width()/2., bar.get_height() + 0.002,
                f'{score:.4f}', ha='center', va='bottom')
    
    plt.tight_layout()
    plt.savefig('plots/ensemble_comparison.png', dpi=300, bbox_inches='tight')
    plt.close()
    print("Ensemble comparison plot saved to plots/ensemble_comparison.png")
    
    # Save detailed results
    results_summary = {
        'individual_f1s': individual_f1s,
        'ensemble_f1': ensemble_f1,
        'method': 'Soft Voting'
    }
    
    import json
    with open('ensemble_results.json', 'w') as f:
        json.dump(results_summary, f, indent=2)
    print("\nDetailed results saved to ensemble_results.json")

if __name__ == "__main__":
    main()