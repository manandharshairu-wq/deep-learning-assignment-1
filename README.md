# Deep Learning Assignment
## Shaira Manandhar
**Datasets × Architectures Benchmark (PyTorch)**

## Objective
This project benchmarks how different neural network architectures perform across tabular and image datasets, highlighting how data modality and model inductive bias affect learning.
All models are trained from scratch (no pretraining) with a consistent experimental setup.

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
- 2 hidden layers (ReLU)
- Batch Normalization + Dropout
- Used on **all datasets** (after preprocessing)

### 2. Convolutional Neural Network (CNN)
- ≥2 convolution layers with pooling
- Conv1D for tabular data, Conv2D for images
- Fully connected classifier head
- Used on **all datasets**

### 3. Attention-Based Model
- **Adult**: Transformer-style Tabular Attention (features treated as tokens)
- **CIFAR / PCam**: Vision Transformer (TinyViT)
- No pretrained weights

---

## Experimental Setup
- Framework: PyTorch
- Optimizer: Adam (same across all models)
- Batch size: 128
- Learning rate: 0.001
- Epochs: 12
- Early stopping on validation loss (patience = 3)
- Train / validation / test splits are consistent per dataset

---
##Note: PCam is implemented but not run by default due to compute constraints in Colab.

---

## Results Summary
| Dataset         | Architecture     | Accuracy | F1 |
|-----------------|------------------|----------|----|
| Adult           | MLP              | 0.8535   | 0.6669 |
| Adult           | CNN              | 0.8548   | 0.6702 |
| Adult           | TabularAttention | 0.8511   | 0.6769 |
| CIFAR-100 (0–9) | MLP              | 0.5550   | 0.5541 |
| CIFAR-100 (0–9) | CNN              | 0.6900   | 0.6884 |
| CIFAR-100 (0–9) | ViT              | 0.6230   | 0.6266 |

Full metrics, training time, parameter counts, and learning curves are saved in `results/`.

---

## Key Insights
- **MLPs perform well on tabular data**, where feature interactions are relatively simple.
- **CNNs outperform MLPs on images** due to spatial inductive bias.
- **Attention-based models require more data or pretraining** to outperform CNNs on images; without pretraining, ViT underperforms CNN on CIFAR.
- On tabular data, **attention offers marginal gains** but does not drastically outperform simpler models.

> Importantly, **I intentionally avoided pretraining**, as required by the assignment. Lower ViT performance is expected and correctly interpreted.

---

## Reproducibility
1. Open the ShairaManandhar_Assignment1.ipynb notebook in **Google Colab**
2. Ensure **GPU is enabled**
3. Run all cells one by one
4. Run:
```bash
python main.py

---
```

#Project Structure

```
shairamanandhar_assignment1_final/
├── models/            # Architectures
├── data_loaders/      # Dataset loaders
├── config/            # Config file
├── results/           # Metrics, plots, checkpoints
├── main.py            # Training + evaluation
└── README.md

