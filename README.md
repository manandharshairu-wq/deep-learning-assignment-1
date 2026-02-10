# Deep Learning Assignment  
## Shaira Manandhar  
**Datasets × Architectures Benchmark (PyTorch)**

---

## Objective
This project benchmarks how different neural network architectures perform across **tabular** and **image** datasets, highlighting how **data modality** and **model inductive bias** affect learning behavior and generalization.

All models are trained **from scratch (no pretraining)** using a consistent experimental setup to ensure fair comparison.

---

## Datasets

- **Adult Income (UCI)**
  - Modality: Tabular
  - Task: Binary classification (income >50K)
  - Metrics: Accuracy, F1-score

- **CIFAR-100 (classes 0–9)**
  - Modality: Image (32×32 RGB)
  - Task: 10-class classification
  - Metric: Accuracy

- **PatchCamelyon (PCam)** *(implemented, not run by default)*
  - Modality: Histopathology images (96×96 RGB)
  - Task: Binary classification (tumor vs normal)

---

## Architectures

### 1. Multilayer Perceptron (MLP)
- Fully connected feedforward network
- Two hidden layers with ReLU activation
- Batch Normalization and Dropout
- Used on **all datasets** (after preprocessing)

**Strengths:**  
Effective baseline for tabular data with stable convergence.

**Limitations:**  
Lacks spatial inductive bias for image data.

---

### 2. Convolutional Neural Network (CNN)
- Multiple convolutional layers with pooling
- Conv1D for tabular data, Conv2D for images
- Fully connected classifier head
- Used on **all datasets**

**Strengths:**  
Strong inductive bias, fast convergence, and parameter efficiency.

---

### 3. Attention-Based Model
- **Adult:** Transformer-style Tabular Attention (features treated as tokens)
- **CIFAR / PCam:** Vision Transformer (TinyViT)
- No pretrained weights

**Strengths:**  
Models feature interactions flexibly.

**Limitations:**  
Requires more data to outperform CNNs when trained from scratch.

---

## Experimental Setup
- Framework: PyTorch
- Optimizer: Adam
- Batch size: 128
- Learning rate: 0.001
- Epochs: 12
- Early stopping on validation loss (patience = 3)
- Train / validation / test splits are consistent per dataset

---

## Note
PCam is implemented but **not run by default** due to compute constraints in Colab and local environments.

---

## Results Summary

| Dataset         | Architecture        | Accuracy   | F1-score   |
|-----------------|---------------------|------------|------------|
| Adult           | MLP                 | 0.8537     | 0.6697     |
| Adult           | **CNN**             | **0.8546** | **0.6724** |
| Adult           | TabularAttention    | 0.8512     | 0.6528     |
| CIFAR-100 (0–9) | MLP                 | 0.5490     | 0.5475     |
| CIFAR-100 (0–9) | **CNN**             | **0.6940** | **0.6925** |
| CIFAR-100 (0–9) | ViT                 | 0.6080     | 0.6054     |

Full metrics, parameter counts, and learning curves are saved in the `results/` directory.

---

## Key Insights
- **MLPs perform well on tabular data**, where relationships between features are relatively simple.
- **CNNs consistently outperform other architectures**, achieving the best results on both tabular and image datasets with significantly fewer parameters.
- **Vision Transformers underperform CNNs on CIFAR** when trained from scratch, confirming the importance of pretraining and large datasets.
- On tabular data, **attention-based models offer limited gains** over simpler architectures.

> Pretraining was intentionally avoided, as required by the assignment. Lower ViT performance is expected and correctly interpreted.

---

## Reproducibility

### 1. Clone the repository
```bash
git clone <your-github-repo-url>
cd shairamanandhar_assignment1_final

### 1. Clone the repository
```bash
git clone <your-github-repo-url>
cd shairamanandhar_assignment1_final
```

### 2. (Optional) Create and activate a virtual environment
```bash
python3 -m venv .venv
source .venv/bin/activate
```

### 3. Install dependencies
```bash
pip install -r requirements.txt
```

### 4. Run the experiments
```bash
python main.py
```

This command will:
- Train all architectures (MLP, CNN, ViT / TabularAttention)
- Generate training and validation learning curves
- Save plots and metrics to the `results/` directory
- Overwrite existing results if present

### 5. View results
After execution, the `results/` directory will contain:
- `final_metrics.csv`
- Learning curve plots (`*_loss_curve.png`, `*_f1_curve.png`, `*_accuracy_curve.png`)
- Validation comparison plots (if enabled)

### (Optional) Run a reduced experiment
To reduce runtime, modify `config.py`:

```python
"run_datasets": ["Adult"],
"run_architectures": ["MLP"],
```



