import torch
import torch.nn as nn
import numpy as np

# -------------------------
# Architecture 1: MLP
# -------------------------
class MLP(nn.Module):
    """
    Mandatory MLP:
    Input -> 2 hidden layers (ReLU) -> Output
    Includes BatchNorm + Dropout
    Works for:
      - Adult tabular (B,D)
      - Images (B,C,H,W) after flatten
    """
    def __init__(self, in_shape, out_dim, hidden1=512, hidden2=256, dropout=0.3):
        super().__init__()
        flat_size = in_shape if isinstance(in_shape, int) else int(np.prod(in_shape))
        self.net = nn.Sequential(
            nn.Flatten(),
            nn.Linear(flat_size, hidden1),
            nn.ReLU(),
            nn.BatchNorm1d(hidden1),
            nn.Dropout(dropout),

            nn.Linear(hidden1, hidden2),
            nn.ReLU(),
            nn.BatchNorm1d(hidden2),
            nn.Dropout(dropout),

            nn.Linear(hidden2, out_dim)
        )

    def forward(self, x):
        return self.net(x)

# -------------------------
# Architecture 2: CNN
# -------------------------
class CNN(nn.Module):
    """
    Mandatory CNN:
    At least 2 convolution layers + pooling + FC head.
    Works for:
      - Adult: Conv1D over features
      - Images: Conv2D
    """
    def __init__(self, in_shape, out_dim, dropout=0.2):
        super().__init__()

        if isinstance(in_shape, int):
            # Tabular as 1D signal: [B, 1, D]
            self.feat = nn.Sequential(
                nn.Unflatten(1, (1, in_shape)),
                nn.Conv1d(1, 16, 3, padding=1),
                nn.ReLU(),
                nn.MaxPool1d(2),
                nn.Conv1d(16, 32, 3, padding=1),
                nn.ReLU(),
                nn.MaxPool1d(2),
                nn.Flatten()
            )
            L = in_shape // 4  # pooled twice
            self.head = nn.Sequential(
                nn.Linear(32 * L, 128),
                nn.ReLU(),
                nn.Dropout(dropout),
                nn.Linear(128, out_dim)
            )
        else:
            # Image CNN: [B, C, H, W]
            C, H, W = in_shape
            self.feat = nn.Sequential(
                nn.Conv2d(C, 32, 3, padding=1),
                nn.ReLU(),
                nn.MaxPool2d(2),
                nn.Conv2d(32, 64, 3, padding=1),
                nn.ReLU(),
                nn.MaxPool2d(2),
                nn.Flatten()
            )
            H2, W2 = H // 4, W // 4
            self.head = nn.Sequential(
                nn.Linear(64 * H2 * W2, 256),
                nn.ReLU(),
                nn.Dropout(dropout),
                nn.Linear(256, out_dim)
            )

    def forward(self, x):
        return self.head(self.feat(x))

# --------------------------------------------------------
# Architecture 3a (Bonus): Tabular Transformer-style Attention
# Used for Adult
# --------------------------------------------------------
class TabularAttention(nn.Module):
    """
    Treat each tabular feature as a token.
    - Feature scalar -> token embedding
    - Transformer encoder
    - Mean pool tokens -> classifier
    """
    def __init__(self, num_features, out_dim, d_model=64, n_heads=4, n_layers=2, dropout=0.1):
        super().__init__()
        self.feature_embed = nn.Linear(1, d_model)
        self.pos_embed = nn.Parameter(torch.zeros(1, num_features, d_model))
        nn.init.trunc_normal_(self.pos_embed, std=0.02)

        enc_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=n_heads,
            dim_feedforward=d_model * 4,
            dropout=dropout,
            activation="gelu",
            batch_first=True
        )
        self.encoder = nn.TransformerEncoder(enc_layer, num_layers=n_layers)
        self.norm = nn.LayerNorm(d_model)

        self.head = nn.Sequential(
            nn.Linear(d_model, d_model),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(d_model, out_dim)
        )

    def forward(self, x):
        # x: [B, D]
        x = x.unsqueeze(-1)              # [B, D, 1]
        x = self.feature_embed(x)        # [B, D, d_model]
        x = x + self.pos_embed[:, :x.size(1), :]
        x = self.encoder(x)              # [B, D, d_model]
        x = self.norm(x)
        x = x.mean(dim=1)                # mean pool over features
        return self.head(x)

# --------------------------------------------------------
# Architecture 3b (Bonus): Vision Transformer-style (TinyViT)
# Used for CIFAR + PCam
# --------------------------------------------------------
class TinyViT(nn.Module):
    """
    Simple ViT-style encoder for images only.
    No pretrained weights.
    """
    def __init__(self, in_shape, out_dim, patch=4, d_model=128, n_heads=4, n_layers=2, dropout=0.1):
        super().__init__()
        assert not isinstance(in_shape, int), "TinyViT is for images only."

        C, H, W = in_shape
        assert H % patch == 0 and W % patch == 0, "H and W must be divisible by patch size."
        n_patches = (H // patch) * (W // patch)

        self.patch_embed = nn.Conv2d(C, d_model, kernel_size=patch, stride=patch)
        self.cls_token = nn.Parameter(torch.zeros(1, 1, d_model))
        self.pos_embed = nn.Parameter(torch.zeros(1, 1 + n_patches, d_model))
        self.drop = nn.Dropout(dropout)

        enc_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=n_heads,
            dim_feedforward=d_model * 4,
            dropout=dropout,
            activation="gelu",
            batch_first=True
        )
        self.encoder = nn.TransformerEncoder(enc_layer, num_layers=n_layers)

        self.norm = nn.LayerNorm(d_model)
        self.head = nn.Linear(d_model, out_dim)

        nn.init.trunc_normal_(self.pos_embed, std=0.02)
        nn.init.trunc_normal_(self.cls_token, std=0.02)

    def forward(self, x):
        # x: [B,C,H,W]
        x = self.patch_embed(x)          # [B,d,H/p,W/p]
        x = x.flatten(2).transpose(1, 2) # [B,tokens,d]
        B = x.size(0)

        cls = self.cls_token.expand(B, -1, -1)  # [B,1,d]
        x = torch.cat([cls, x], dim=1)          # [B,1+tokens,d]
        x = self.drop(x + self.pos_embed[:, :x.size(1), :])

        x = self.encoder(x)
        cls_out = self.norm(x[:, 0])    # CLS token
        return self.head(cls_out)
