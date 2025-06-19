# BioBERT Large Model Training and Ensemble Testing

## Overview
This project implements a BioBERT based text classification system for biomedical literature classification (T0 vs T2/4). The system uses 5-fold cross-validation for robust training and ensemble voting for final predictions.

## Quick Start

### 1. Training Models (5-Fold Cross Validation)
```bash
python train_biobert_large.py
```

### 2. Testing Single Best Model
```bash
python test_simple.py
```

### 3. Testing Ensemble (All 5 Models)
```bash
python test_ensemble.py
```

## System Architecture

```mermaid
flowchart TD
    A[Training Data<br/>trainset_2268.xlsx] --> B[5-Fold Cross Validation]
    B --> C1[Fold 1 Model]
    B --> C2[Fold 2 Model] 
    B --> C3[Fold 3 Model]
    B --> C4[Fold 4 Model]
    B --> C5[Fold 5 Model]
    
    C1 --> D[Model Selection<br/>Based on F1 Score]
    C2 --> D
    C3 --> D
    C4 --> D
    C5 --> D
    
    E[Test Data<br/>testset_395.xlsx] --> F[Single Model Testing]
    E --> G[Ensemble Testing]
    
    D --> F
    C1 --> G
    C2 --> G
    C3 --> G
    C4 --> G
    C5 --> G
    
    F --> H[Single Model Results]
    G --> I[Ensemble Results<br/>Soft Voting]
    
    style A fill:#e1f5fe
    style E fill:#e8f5e8
    style H fill:#fff3e0
    style I fill:#f3e5f5
```

## Training Pipeline

```mermaid
flowchart LR
    A[Data Loading] --> B[Tokenization]
    B --> C[Model Initialization<br/>BioBERT]
    C --> D[Layer Freezing<br/>First N Layers]
    D --> E[Class Weight<br/>Calculation]
    E --> F[Training Loop]
    F --> G[F1-based<br/>Model Selection]
    G --> H[Model Saving]
    
    style C fill:#bbdefb
    style F fill:#c8e6c9
    style G fill:#ffcdd2
```

## Model Configuration

### Core Parameters
| Parameter | Value | Description |
|-----------|-------|-------------|
| **Model** | `dmis-lab/biobert-large-cased-v1.1` | Pre-trained BioBERT Large model |
| **Task** | Binary Classification | T0 vs T2/4 classification |
| **Max Length** | 512 tokens | Maximum sequence length |
| **Dropout** | 0.5 | Classifier dropout rate |

### Training Hyperparameters
| Parameter | Value | Rationale |
|-----------|-------|-----------|
| **Learning Rate** | `1e-5` | Low rate for full fine-tuning |
| **Epochs** | `5` | Prevent overfitting |
| **Batch Size** | `16` | Memory efficiency |
| **Gradient Accumulation** | `2` | Effective batch size = 32 |
| **Weight Decay** | `0.1` | Strong regularization |
| **Warmup Ratio** | `0.2` | Gradual learning rate increase |
| **Frozen Layers** | `0` | Full model fine-tuning |

### Optimization Settings
| Setting | Value | Purpose |
|---------|-------|---------|
| **Optimizer** | AdamW | Default transformer optimizer |
| **Scheduler** | Linear with warmup | Stable convergence |
| **FP16** | Enabled | Memory and speed optimization |
| **Class Weights** | Balanced | Handle class imbalance |
| **Best Model Metric** | `eval_f1` | F1-score based selection |

## Data Configuration

### Dataset Statistics
```
Training Set: 2,268 articles
Test Set: 395 articles
Classes: T0 (majority), T2/4 (minority)
Class Distribution: Imbalanced (~60:40)
```

### Cross-Validation Setup
```
Strategy: 5-Fold Cross Validation
Random State: 50 (reproducible splits)
Validation Size per Fold: ~450 articles
Training Size per Fold: ~1,800 articles
```

## Ensemble Method

### Soft Voting Algorithm
```python
# For each test sample i:
probabilities_i = [model_1_prob_i, model_2_prob_i, ..., model_5_prob_i]
average_prob_i = mean(probabilities_i)
prediction_i = 1 if average_prob_i > 0.5 else 0
```

### Model Selection Criteria
- Individual models selected based on highest F1 score
- All 5 fold models used in ensemble regardless of individual performance
- Soft voting chosen over hard voting for better probability utilization

