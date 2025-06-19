import torch
import torch.nn as nn
from torch.utils.data import Dataset

from transformers import (
    AutoTokenizer, AutoModelForSequenceClassification,
    Trainer, TrainingArguments
)
import pandas as pd
from sklearn.metrics import accuracy_score, f1_score, precision_recall_curve
import matplotlib.pyplot as plt
import numpy as np
from datetime import datetime
import os
import shutil
from sklearn.model_selection import KFold
from sklearn.utils.class_weight import compute_class_weight
from collections import Counter

# 建立記錄目錄
os.makedirs('logs', exist_ok=True)
os.makedirs('plots', exist_ok=True)

class FocalLoss(nn.Module):
    def __init__(self, alpha=1, gamma=2, reduction='mean'):
        super(FocalLoss, self).__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.reduction = reduction
        
    def forward(self, inputs, targets):
        ce_loss = nn.CrossEntropyLoss(reduction='none')(inputs, targets)
        pt = torch.exp(-ce_loss)
        focal_loss = self.alpha * (1 - pt) ** self.gamma * ce_loss
        
        if self.reduction == 'mean':
            return focal_loss.mean()
        elif self.reduction == 'sum':
            return focal_loss.sum()
        else:
            return focal_loss

class ImprovedTrainer(Trainer):
    def __init__(self, class_weights=None, use_focal_loss=False, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.class_weights = class_weights
        self.use_focal_loss = use_focal_loss
        if use_focal_loss:
            self.focal_loss = FocalLoss(alpha=1, gamma=2)
            
    def compute_loss(self, model, inputs, return_outputs=False, **kwargs):
        labels = inputs.get("labels")
        outputs = model(**inputs)
        logits = outputs.get('logits')
        
        if self.use_focal_loss:
            loss = self.focal_loss(logits.view(-1, self.model.config.num_labels), labels.view(-1))
        elif self.class_weights is not None:
            loss_fct = nn.CrossEntropyLoss(weight=self.class_weights.to(logits.device))
            loss = loss_fct(logits.view(-1, self.model.config.num_labels), labels.view(-1))
        else:
            loss_fct = nn.CrossEntropyLoss()
            loss = loss_fct(logits.view(-1, self.model.config.num_labels), labels.view(-1))
            
        return (loss, outputs) if return_outputs else loss
        
def find_optimal_threshold(y_true, y_probs):
    """尋找最佳閾值來最大化F1 score"""
    precisions, recalls, thresholds = precision_recall_curve(y_true, y_probs)
    f1_scores = 2 * (precisions * recalls) / (precisions + recalls + 1e-8)
    best_threshold_idx = np.argmax(f1_scores)
    best_threshold = thresholds[best_threshold_idx] if best_threshold_idx < len(thresholds) else 0.5
    best_f1 = f1_scores[best_threshold_idx]
    return best_threshold, best_f1

# 1. 自訂 Dataset
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

# 2. 讀入資料
print("正在讀取 Excel 檔案...")
df = pd.read_excel('trainset_2268.xlsx')

# 準備訓練資料
texts = df['content'].fillna('').tolist()
labels = df['label'].tolist()

print(f"成功載入 {len(texts)} 篇文章")
print(f"文字範例：{texts[0][:100]}...")

# 分析類別分佈
label_counts = Counter(labels)
print(f"類別分佈：{dict(label_counts)}")

# 計算類別權重
class_weights = compute_class_weight('balanced', classes=np.unique(labels), y=labels)
class_weights = torch.FloatTensor(class_weights)
print(f"類別權重：{class_weights}")

# 6. 定義評估函數
def compute_metrics(pred):
    labels = pred.label_ids
    predictions = pred.predictions
    
    # 處理二元分類的情況
    if len(predictions.shape) == 2:
        # 使用softmax獲得機率
        probs = torch.softmax(torch.tensor(predictions), dim=1)
        pred_probs = probs[:, 1].numpy()  # 取正類別的機率
        
        # 尋找最佳閾值
        best_threshold, best_f1 = find_optimal_threshold(labels, pred_probs)
        preds = (pred_probs > best_threshold).astype(int)
    else:
        preds = predictions.argmax(axis=-1)
    
    acc = accuracy_score(labels, preds)
    f1_macro = f1_score(labels, preds, average='macro')
    f1_weighted = f1_score(labels, preds, average='weighted')
    
    return {
        "accuracy": acc, 
        "macro_f1": f1_macro,
        "weighted_f1": f1_weighted,
        "eval_f1": f1_macro  # 用於模型選擇
    }

# 設定 5-fold cross validation
n_splits = 5
kf = KFold(n_splits=n_splits, shuffle=True, random_state=50)

# 儲存每個 fold 的結果
fold_results = []

for fold, (train_idx, val_idx) in enumerate(kf.split(texts), 1):
    print(f"\n開始訓練 Fold {fold}/{n_splits}")
    
    # 分割訓練集和驗證集
    train_texts = [texts[i] for i in train_idx]
    train_labels = [labels[i] for i in train_idx]
    val_texts = [texts[i] for i in val_idx]
    val_labels = [labels[i] for i in val_idx]
    
    print(f"Fold {fold} - 訓練集大小: {len(train_texts)}")
    print(f"Fold {fold} - 驗證集大小: {len(val_texts)}")
    
    # 計算當前fold的類別權重
    fold_class_weights = compute_class_weight('balanced', classes=np.unique(train_labels), y=train_labels)
    fold_class_weights = torch.FloatTensor(fold_class_weights)
    print(f"Fold {fold} 類別權重：{fold_class_weights}")
    
    # 3. 初始化 Tokenizer + Model
    model_name = "dmis-lab/biobert-large-cased-v1.1"
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    
    # 選擇 num_labels=2（二元分類）
    num_labels = 2
    model = AutoModelForSequenceClassification.from_pretrained(
        model_name,
        num_labels=num_labels,
        problem_type="single_label_classification",
        classifier_dropout=0.5  # 增加dropout防止過擬合
    )
    
    # 4. 凍結前 N 層（凍結前20層）
    freeze_n = 20  # 凍結前20層
    print(f"Fold {fold} - 凍結前 {freeze_n} 層，只訓練最後 {24-freeze_n} 層")
    for layer in model.bert.encoder.layer[:freeze_n]:
        for p in layer.parameters(): 
            p.requires_grad = False
    
    # 5. 建立 Dataset
    train_dataset = BioPaperDataset(train_texts, train_labels, tokenizer)
    val_dataset = BioPaperDataset(val_texts, val_labels, tokenizer)
    
    # 7. Trainer 設定
    current_time = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_dir = f"./results_fold{fold}_{current_time}"
    logging_dir = f"./logs/logs_fold{fold}_{current_time}"
    
    training_args = TrainingArguments(
        output_dir=output_dir,
        per_device_train_batch_size=8,   # 降低batch size以穩定訓練
        gradient_accumulation_steps=4,   # 增加累積步數維持有效batch size=32
        per_device_eval_batch_size=16,
        learning_rate=5e-4,              # 凍結層數多時可用較高學習率
        num_train_epochs=8,              # 凍結層時可訓練更多epochs
        weight_decay=0.01,               # 降低權重衰減，凍結層本身已防過擬合
        logging_dir=logging_dir,
        logging_steps=50,
        save_total_limit=1,
        report_to="tensorboard",
        save_steps=1000,
        eval_steps=50,
        no_cuda=False,
        fp16=True,
        warmup_ratio=0.1,                # 降低warmup比例
        metric_for_best_model="eval_f1",
        greater_is_better=True,
        load_best_model_at_end=True,
        eval_strategy="epoch",
        save_strategy="epoch"
    )
    
    # 使用改進的Trainer
    trainer = ImprovedTrainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=val_dataset,
        compute_metrics=compute_metrics,
        class_weights=fold_class_weights,
        use_focal_loss=False  # 可以改為True試試focal loss
    )
    
    # 手動實現早停
    class EarlyStoppingCallback:
        def __init__(self, patience=3):
            self.patience = patience
            self.best_metric = None
            self.patience_counter = 0
            
    # 8. 開始訓練
    print(f"開始訓練 Fold {fold}...")
    
    # 簡單的早停：如果用load_best_model_at_end=True，會自動選最佳模型
    # 8個epoch已經比較保守，減少過擬合風險
    trainer.train()
    
    # 9. 評估當前 fold
    print(f"評估 Fold {fold}...")
    fold_metrics = trainer.evaluate()
    fold_results.append(fold_metrics)
    
    # 儲存當前 fold 的模型
    f1_score_val = fold_metrics['eval_macro_f1']
    new_output_dir = f"./results_fold{fold}_f1_{f1_score_val:.4f}"
    if os.path.exists(new_output_dir):
        shutil.rmtree(new_output_dir)
    os.rename(output_dir, new_output_dir)
    print(f"Fold {fold} 模型已儲存至 {new_output_dir}")

# 計算並顯示所有 fold 的平均結果
print("\n所有 Fold 的評估結果：")
avg_accuracy = np.mean([result['eval_accuracy'] for result in fold_results])
avg_f1 = np.mean([result['eval_macro_f1'] for result in fold_results])
avg_weighted_f1 = np.mean([result['eval_weighted_f1'] for result in fold_results])
std_accuracy = np.std([result['eval_accuracy'] for result in fold_results])
std_f1 = np.std([result['eval_macro_f1'] for result in fold_results])

print(f"平均準確率: {avg_accuracy:.4f} ± {std_accuracy:.4f}")
print(f"平均 Macro F1: {avg_f1:.4f} ± {std_f1:.4f}")
print(f"平均 Weighted F1: {avg_weighted_f1:.4f}")

# 繪製每個 fold 的結果
plt.figure(figsize=(10, 6))
fold_numbers = range(1, n_splits + 1)
accuracies = [result['eval_accuracy'] for result in fold_results]
f1_scores = [result['eval_macro_f1'] for result in fold_results]
weighted_f1_scores = [result['eval_weighted_f1'] for result in fold_results]

plt.plot(fold_numbers, accuracies, 'o-', label='Accuracy')
plt.plot(fold_numbers, f1_scores, 's-', label='Macro F1')
plt.plot(fold_numbers, weighted_f1_scores, '^-', label='Weighted F1')
plt.axhline(y=avg_accuracy, color='r', linestyle='--', alpha=0.7, label=f'Avg Accuracy: {avg_accuracy:.4f}')
plt.axhline(y=avg_f1, color='g', linestyle='--', alpha=0.7, label=f'Avg Macro F1: {avg_f1:.4f}')
plt.xlabel('Fold')
plt.ylabel('Score')
plt.title('5-Fold Cross Validation Results')
plt.legend()
plt.grid(True)

plt.tight_layout()
plt.savefig('plots/cross_validation_results.png', dpi=300, bbox_inches='tight')
plt.close()

print("\n cross validation results saved to plots/cross_validation_results.png")