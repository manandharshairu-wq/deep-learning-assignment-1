import os
import torch
from torch.utils.data import TensorDataset, random_split, Dataset, Subset
from torchvision import datasets, transforms
import pandas as pd
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.model_selection import train_test_split
import tensorflow_datasets as tfds

# -----------------------
# Adult (tabular) - binary
# -----------------------
def get_adult(seed=42, test_size=0.2, val_size=0.15):
    url = "https://archive.ics.uci.edu/ml/machine-learning-databases/adult/adult.data"
    cols = ['age', 'workclass', 'fnlwgt', 'education', 'education-num', 'marital-status',
            'occupation', 'relationship', 'race', 'sex', 'capital-gain', 'capital-loss',
            'hours-per-week', 'native-country', 'income']
    df = pd.read_csv(url, names=cols, skipinitialspace=True).dropna()

    # Label
    df['income'] = df['income'].apply(lambda x: 1 if x.strip() == '>50K' else 0)

    # Encode categoricals
    for col in df.select_dtypes(include=['object']).columns:
        df[col] = LabelEncoder().fit_transform(df[col])

    X = StandardScaler().fit_transform(df.drop('income', axis=1).values)
    y = df['income'].values

    # Train/test then train/val
    X_tr_val, X_te, y_tr_val, y_te = train_test_split(
        X, y, test_size=test_size, random_state=seed, stratify=y
    )
    X_tr, X_va, y_tr, y_va = train_test_split(
        X_tr_val, y_tr_val, test_size=val_size, random_state=seed, stratify=y_tr_val
    )

    train_ds = TensorDataset(torch.FloatTensor(X_tr), torch.FloatTensor(y_tr).view(-1, 1))
    val_ds   = TensorDataset(torch.FloatTensor(X_va), torch.FloatTensor(y_va).view(-1, 1))
    test_ds  = TensorDataset(torch.FloatTensor(X_te), torch.FloatTensor(y_te).view(-1, 1))

    return train_ds, val_ds, test_ds, X.shape[1], 1, "binary"

# -----------------------------------------
# CIFAR-100 but only classes 0–9 (10-class)
# -----------------------------------------
def get_cifar100_10class(val_fraction=0.15, seed=42):
    t = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.5,0.5,0.5), (0.5,0.5,0.5))
    ])

    full = datasets.CIFAR100(root="./data", train=True, download=True, transform=t)
    test_full = datasets.CIFAR100(root="./data", train=False, download=True, transform=t)

    # Keep labels 0..9 only
    train_idx = [i for i, y in enumerate(full.targets) if y in range(10)]
    test_idx  = [i for i, y in enumerate(test_full.targets) if y in range(10)]

    full_10 = Subset(full, train_idx)
    test_10 = Subset(test_full, test_idx)

    # Split train/val
    v_sz = int(len(full_10) * val_fraction)
    tr_sz = len(full_10) - v_sz
    train, val = random_split(full_10, [tr_sz, v_sz], generator=torch.Generator().manual_seed(seed))

    return train, val, test_10, (3, 32, 32), 10, "multiclass"

# -----------------------
# PCam (TFDS) - binary
# -----------------------
class PCamTorch(Dataset):
    def __init__(self, split="train", limit=None):
        ds = tfds.load(
            "patch_camelyon",
            split=split,
            as_supervised=True,
            data_dir=os.environ.get("TFDS_DATA_DIR")
        )
        if limit is not None:
            ds = ds.take(limit)
        self.samples = list(tfds.as_numpy(ds))

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        img, y = self.samples[idx]
        img = torch.tensor(img, dtype=torch.float32) / 255.0
        img = img.permute(2, 0, 1)  # CHW
        y = torch.tensor(y, dtype=torch.float32).view(1)
        return img, y

def get_pcam(limit_train=10000, limit_val=2000, limit_test=2000):
    train = PCamTorch(split="train", limit=limit_train)
    val   = PCamTorch(split="validation", limit=limit_val)
    test  = PCamTorch(split="test", limit=limit_test)
    return train, val, test, (3, 96, 96), 1, "binary"
