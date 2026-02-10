import os, json, time
import torch
import torch.nn as nn
import torch.optim as optim
import pandas as pd
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader
from sklearn.metrics import accuracy_score, f1_score

from data_loaders.loaders import get_adult, get_cifar100_10class, get_pcam
from models.architectures import MLP, CNN, TinyViT, TabularAttention

with open("config/config.json", "r") as f:
    CFG = json.load(f)

DEVICE = CFG["device"]

def set_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

def count_params(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)

def make_loaders(train_ds, val_ds, test_ds):
    pin = (DEVICE == "cuda")
    return (
        DataLoader(train_ds, batch_size=CFG["batch_size"], shuffle=True,  num_workers=0, pin_memory=pin),
        DataLoader(val_ds,   batch_size=CFG["batch_size"], shuffle=False, num_workers=0, pin_memory=pin),
        DataLoader(test_ds,  batch_size=CFG["batch_size"], shuffle=False, num_workers=0, pin_memory=pin),
    )

def criterion_for(task):
    return nn.BCEWithLogitsLoss() if task == "binary" else nn.CrossEntropyLoss()

def metric_from_logits(logits, y, task):
    # consistent metric reporting
    if task == "binary":
        probs = torch.sigmoid(logits.view(-1))
        preds = (probs >= 0.5).long().cpu().numpy()
        labels = y.view(-1).long().cpu().numpy()
        return f1_score(labels, preds, average="binary")
    else:
        preds = logits.argmax(dim=1).cpu().numpy()
        labels = y.cpu().numpy()
        return accuracy_score(labels, preds)

@torch.no_grad()
def test_metrics(model, test_loader, task):
    model.eval()
    all_logits, all_y = [], []
    for x, y in test_loader:
        x = x.to(DEVICE)
        all_logits.append(model(x).cpu())
        all_y.append(y.cpu())
    logits = torch.cat(all_logits)
    y = torch.cat(all_y)

    if task == "binary":
        p = (torch.sigmoid(logits.view(-1)) >= 0.5).long().numpy()
        l = y.view(-1).long().numpy()
        return accuracy_score(l, p), f1_score(l, p, average="binary")
    else:
        p = logits.argmax(1).numpy()
        l = y.numpy()
        return accuracy_score(l, p), f1_score(l, p, average="weighted")

def save_curves(history, name, metric_name):
    os.makedirs("results", exist_ok=True)

    plt.figure()
    plt.plot(history["train_loss"], label="Train Loss")
    plt.plot(history["val_loss"], label="Val Loss")
    plt.title(f"{name} Loss")
    plt.legend()
    plt.savefig(f"results/{name}_loss_curve.png")
    plt.close()

    plt.figure()
    plt.plot(history["train_metric"], label=f"Train {metric_name}")
    plt.plot(history["val_metric"], label=f"Val {metric_name}")
    plt.title(f"{name} {metric_name}")
    plt.legend()
    plt.savefig(f"results/{name}_{metric_name.lower()}_curve.png")
    plt.close()

def train_eval(model, train_loader, val_loader, task):
    # optimizer family consistent
    assert CFG["optimizer"].lower() == "adam"
    optimizer = optim.Adam(model.parameters(), lr=CFG["lr"])
    crit = criterion_for(task)

    es = CFG["early_stopping"]
    best_val = float("inf")
    best_state = None
    bad = 0
    best_epoch = 0

    history = {"train_loss": [], "val_loss": [], "train_metric": [], "val_metric": []}

    for epoch in range(CFG["epochs"]):
        # ---- train
        model.train()
        train_loss_sum = 0.0
        tr_logits, tr_y = [], []

        for x, y in train_loader:
            x, y = x.to(DEVICE), y.to(DEVICE)
            optimizer.zero_grad()
            logits = model(x)

            if task == "binary":
                loss = crit(logits.view(-1, 1), y.float().view(-1, 1))
            else:
                loss = crit(logits, y.long())

            loss.backward()
            optimizer.step()

            train_loss_sum += loss.item()
            tr_logits.append(logits.detach().cpu())
            tr_y.append(y.detach().cpu())

        train_loss = train_loss_sum / len(train_loader)
        train_metric = metric_from_logits(torch.cat(tr_logits), torch.cat(tr_y), task)

        # ---- val
        model.eval()
        val_loss_sum = 0.0
        va_logits, va_y = [], []

        with torch.no_grad():
            for x, y in val_loader:
                x, y = x.to(DEVICE), y.to(DEVICE)
                logits = model(x)

                if task == "binary":
                    loss = crit(logits.view(-1, 1), y.float().view(-1, 1))
                else:
                    loss = crit(logits, y.long())

                val_loss_sum += loss.item()
                va_logits.append(logits.detach().cpu())
                va_y.append(y.detach().cpu())

        val_loss = val_loss_sum / len(val_loader)
        val_metric = metric_from_logits(torch.cat(va_logits), torch.cat(va_y), task)

        history["train_loss"].append(train_loss)
        history["val_loss"].append(val_loss)
        history["train_metric"].append(train_metric)
        history["val_metric"].append(val_metric)

        # early stopping on val loss
        improved = (best_val - val_loss) > es.get("min_delta", 0.0)
        if improved:
            best_val = val_loss
            best_state = {k: v.cpu() for k, v in model.state_dict().items()}
            bad = 0
            best_epoch = epoch + 1
        else:
            bad += 1
            if es["enabled"] and bad >= es["patience"]:
                break

    if best_state is not None:
        model.load_state_dict(best_state)

    return history, best_epoch

def load_dataset(dataset_name):
    if dataset_name == "Adult":
        return get_adult(CFG["seed"], CFG["adult_test_size"], CFG["adult_val_size"])
    elif dataset_name == "CIFAR-100(0-9)":
        return get_cifar100_10class(CFG["val_fraction"], CFG["seed"])
    else:
        return get_pcam(CFG["pcam_train_limit"], CFG["pcam_val_limit"], CFG["pcam_test_limit"])

def build_model(arch, dataset_name, in_shape, out_dim):
    if arch == "MLP":
        return MLP(in_shape, out_dim)

    if arch == "CNN":
        return CNN(in_shape, out_dim)

    # Arch 3 (Bonus)
    # - Adult: TabularAttention
    # - CIFAR/PCam: TinyViT
    if arch == "ViT":
        if dataset_name == "Adult":
            tcfg = CFG["tabattn"]
            return TabularAttention(
                num_features=in_shape,
                out_dim=out_dim,
                d_model=tcfg["d_model"],
                n_heads=tcfg["heads"],
                n_layers=tcfg["layers"],
                dropout=tcfg["dropout"]
            )
        else:
            vcfg = CFG["vit"]
            return TinyViT(
                in_shape, out_dim,
                patch=vcfg["patch"],
                d_model=vcfg["dim"],
                n_heads=vcfg["heads"],
                n_layers=vcfg["layers"],
                dropout=vcfg["dropout"]
            )

    raise ValueError(f"Unknown architecture: {arch}")

def run_experiment(dataset_name, arch):
    train_ds, val_ds, test_ds, in_shape, out_dim, task = load_dataset(dataset_name)
    train_loader, val_loader, test_loader = make_loaders(train_ds, val_ds, test_ds)

    model = build_model(arch, dataset_name, in_shape, out_dim).to(DEVICE)
    params = count_params(model)

    start = time.time()
    history, best_epoch = train_eval(model, train_loader, val_loader, task)
    train_time = time.time() - start

    acc, f1 = test_metrics(model, test_loader, task)

    # Friendly naming: show Adult + ViT as TabularAttention in results table
    shown_arch = "TabularAttention" if (arch == "ViT" and dataset_name == "Adult") else arch

    exp_name = f"{dataset_name}_{shown_arch}".replace("/", "_")
    os.makedirs("results", exist_ok=True)
    torch.save(model.state_dict(), f"results/{exp_name}_best.pt")

    metric_name = "F1" if task == "binary" else "Accuracy"
    save_curves(history, exp_name, metric_name)

    notes = f"time={train_time:.1f}s; best_epoch={best_epoch}; params={params}"
    return {
        "Dataset": dataset_name,
        "Architecture": shown_arch,
        "Accuracy": round(acc, 4),
        "F1": round(f1, 4),
        "Notes": notes,
        "_history": history
    }

def plot_compare(histories, title, out_path, key):
    plt.figure()
    for label, hist in histories.items():
        plt.plot(hist[key], label=label)
    plt.title(title)
    plt.legend()
    plt.savefig(out_path)
    plt.close()

if __name__ == "__main__":
    set_seed(CFG["seed"])
    print("Using device:", DEVICE)

    run_datasets = CFG["run_datasets"]
    run_archs = CFG["run_architectures"]

    results = []
    histories_by_dataset = {}

    for d in run_datasets:
        histories_by_dataset[d] = {}
        for a in run_archs:
            print(f"Running: {d} + {a}")
            r = run_experiment(d, a)
            histories_by_dataset[d][r["Architecture"]] = r["_history"]
            results.append({k: r[k] for k in ["Dataset", "Architecture", "Accuracy", "F1", "Notes"]})

    df = pd.DataFrame(results)
    df.to_csv("results/final_metrics.csv", index=False)

    print("\nFinal Results Table:\n")
    print(df.to_markdown(index=False))

    # Bonus: per-dataset learning curve comparisons
    if CFG.get("bonus", {}).get("make_learning_curve_comparison", False):
        for d, hdict in histories_by_dataset.items():
            if len(hdict) <= 1:
                continue
            plot_compare(hdict, f"{d} - Val Loss Comparison", f"results/{d}_VAL_LOSS_COMPARISON.png", "val_loss")
            plot_compare(hdict, f"{d} - Val Metric Comparison", f"results/{d}_VAL_METRIC_COMPARISON.png", "val_metric")
