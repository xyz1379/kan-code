import os
import math
from copy import deepcopy
from tqdm import tqdm

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

from datasets import load_dataset
from transformers import AutoTokenizer
from torch.utils.data import DataLoader

import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from sklearn.metrics import classification_report, confusion_matrix

# -------------------------
# Global CUDA / Perf flags
# -------------------------
torch.backends.cudnn.benchmark = True
torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True

# -------------------------
# Optimized RoPE with Fused Operations
# -------------------------
class RoPECache:
    """Cache sin/cos RoPE tensors with optimized computation."""
    _cache = {}

    @staticmethod
    def get_sin_cos(seq_len: int, dim: int, device: torch.device):
        key = (seq_len, dim, str(device))
        if key in RoPECache._cache:
            return RoPECache._cache[key]
        
        # Compute once with fused operations
        half_dim = dim // 2
        theta = 10000.0 ** (-torch.arange(0, half_dim, device=device, dtype=torch.float32) / dim)
        pos = torch.arange(seq_len, device=device, dtype=torch.float32)
        freqs = torch.outer(pos, theta)  # (seq_len, half_dim)
        
        # Store as complex number for faster rotation
        emb = torch.polar(torch.ones_like(freqs), freqs)  # e^(i*theta)
        RoPECache._cache[key] = emb
        return emb

def apply_rope_fast(x: torch.Tensor, freqs_cis: torch.Tensor):
    """Optimized RoPE using complex number multiplication."""
    # Get shape info
    *batch_dims, seq_len, dim = x.shape
    half_dim = dim // 2
    
    if freqs_cis.shape[0] < seq_len:
        raise ValueError(f"freqs_cis has seq_len {freqs_cis.shape[0]} but need {seq_len}")
    
    freqs_cis = freqs_cis[:seq_len, :half_dim]
    
    x_reshaped = x.view(*batch_dims, seq_len, half_dim, 2)
    x_complex = torch.view_as_complex(x_reshaped)
    
    x_rotated = x_complex * freqs_cis.unsqueeze(0)
    
    x_real = torch.view_as_real(x_rotated)
    return x_real.view(*batch_dims, seq_len, dim)

# ===================================================================
# Optimized Model Components
# ===================================================================

class KANApproximation(nn.Module):
    """Fast approximation of KAN using Chebyshev polynomials."""
    def __init__(self, in_dim, out_dim, hidden_dim, degree=3):
        super().__init__()
        self.degree = degree
        self.linear = nn.Linear(in_dim, hidden_dim, bias=False)
        self.poly_weights = nn.Parameter(torch.randn(degree + 1, hidden_dim, out_dim) * 0.02)
        self.out_bias = nn.Parameter(torch.zeros(out_dim))
    
    def forward(self, x):
        h = self.linear(x)
        h_normalized = torch.tanh(h)
        
        polynomials = [torch.ones_like(h_normalized), h_normalized]
        for _ in range(2, self.degree + 1):
            polynomials.append(2 * h_normalized * polynomials[-1] - polynomials[-2])
        
        out = torch.zeros(*h.shape[:-1], self.poly_weights.size(2), device=x.device)
        for i, poly in enumerate(polynomials):
            out += torch.matmul(poly, self.poly_weights[i])
        
        return out + self.out_bias

class FusedMLPBlock(nn.Module):
    """Optimized MLP with fused operations and dropout."""
    def __init__(self, in_dim, out_dim, hidden_dim_ratio=4, dropout=0.0):
        super().__init__()
        hidden_dim = int(in_dim * hidden_dim_ratio)
        self.fc1 = nn.Linear(in_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, out_dim)
        self.dropout = nn.Dropout(dropout) if dropout > 0 else nn.Identity()
    
    def forward(self, x):
        x = F.gelu(self.fc1(x), approximate='tanh')
        x = self.dropout(x)
        return self.fc2(x)

class OptimizedMLAWithRoPE(nn.Module):
    """
    Optimized cross-attention.
    MODIFIED: Uses Standard Linear Layers (MLP) for Q, K, V projections.
    """
    def __init__(self, input_dim, latent_dim, latent_len, num_heads):
        super().__init__()
        self.latent_len = latent_len
        self.num_heads = num_heads
        self.head_dim = latent_dim // num_heads
        
        self.latents = nn.Parameter(torch.randn(latent_len, latent_dim) * 0.02)
        
        self.kv_proj = nn.Linear(input_dim, latent_dim * 2, bias=False)

        self.q_proj = nn.Linear(latent_dim, latent_dim, bias=False)
        
        self.out_proj = nn.Linear(latent_dim, latent_dim, bias=False)

    def forward(self, x):
        B, S, D = x.shape
        
        # Standard Linear Projection
        kv = self.kv_proj(x)
        k, v = kv.chunk(2, dim=-1)
        
        latents = self.latents.unsqueeze(0).expand(B, -1, -1)
        q = self.q_proj(latents)
        
        max_len = max(S, self.latent_len)
        freqs = RoPECache.get_sin_cos(max_len, k.size(-1), x.device)
        
        k = apply_rope_fast(k, freqs)
        q = apply_rope_fast(q, freqs)
        
        q = q.view(B, self.latent_len, self.num_heads, self.head_dim).transpose(1, 2)
        k = k.view(B, S, self.num_heads, self.head_dim).transpose(1, 2)
        v = v.view(B, S, self.num_heads, self.head_dim).transpose(1, 2)
        
        out = F.scaled_dot_product_attention(q, k, v, is_causal=False)
        
        out = out.transpose(1, 2).contiguous().view(B, self.latent_len, -1)
        return self.out_proj(out)

class OptimizedLatentSelfAttention(nn.Module):
    """
    Optimized self-attention.
    MODIFIED: Uses Standard Linear Layers (MLP) for Q, K, V projections.
    """
    def __init__(self, embed_dim: int, num_heads: int, bottleneck_size: int):
        super().__init__()
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.bottleneck_size = bottleneck_size
        self.head_dim = embed_dim // num_heads
        
        self.qkv_proj = nn.Linear(embed_dim, embed_dim * 3, bias=False)
        
        self.out_proj = nn.Linear(embed_dim, embed_dim, bias=False)

    def forward(self, x: torch.Tensor):
        B, L, D = x.shape
        
        qkv = self.qkv_proj(x).reshape(B, L, 3, self.num_heads, self.head_dim)
        q, k, v = qkv.unbind(2)
        
        if L > self.bottleneck_size:
            k_compressed = F.adaptive_avg_pool1d(
                k.transpose(1, 2).flatten(0, 1),
                self.bottleneck_size
            ).view(B, self.num_heads, self.head_dim, self.bottleneck_size).transpose(2, 3)
            
            v_compressed = F.adaptive_avg_pool1d(
                v.transpose(1, 2).flatten(0, 1),
                self.bottleneck_size
            ).view(B, self.num_heads, self.head_dim, self.bottleneck_size).transpose(2, 3)
        else:
            k_compressed, v_compressed = k, v
        
        out = F.scaled_dot_product_attention(
            q.transpose(1, 2), k_compressed.transpose(1, 2), v_compressed.transpose(1, 2)
        )
        
        out = out.transpose(1, 2).reshape(B, L, D)
        return self.out_proj(out)

class GroupedHybridFFN(nn.Module):
    """
    A feed-forward network. 
    HYBRID CONFIGURATION (Supports both KAN and MLP experts)
    """
    def __init__(self, in_dim, hidden_dim_ratio, num_kan_experts, num_mlp_experts):
        super().__init__()
        self.num_kan = num_kan_experts
        self.num_mlp = num_mlp_experts
        self.total_experts = num_kan_experts + num_mlp_experts

        if in_dim % self.total_experts != 0:
            raise ValueError(f"Input dimension ({in_dim}) must be divisible by total_experts ({self.total_experts})")

        self.expert_dim = in_dim // self.total_experts
        expert_hidden_dim = int(self.expert_dim * hidden_dim_ratio)

        self.kan_experts = nn.ModuleList([
            KANApproximation(self.expert_dim, self.expert_dim, expert_hidden_dim, degree=3)
            for _ in range(self.num_kan)
        ])
        
        self.mlp_experts = nn.ModuleList([
            FusedMLPBlock(self.expert_dim, self.expert_dim, hidden_dim_ratio=hidden_dim_ratio)
            for _ in range(self.num_mlp)
        ])

    def forward(self, x):
        chunks = x.chunk(self.total_experts, dim=-1)
        results = []
        
        # Process KAN experts
        for i in range(self.num_kan):
            results.append(self.kan_experts[i](chunks[i]))
            
        # Process MLP experts
        for i in range(self.num_mlp):
            chunk_index = self.num_kan + i
            results.append(self.mlp_experts[i](chunks[chunk_index]))

        return torch.cat(results, dim=-1)

class HybridTransformerBlock(nn.Module):
    def __init__(self, latent_dim, num_heads, mlp_dim_ratio, latent_bottleneck,
                 num_kan_experts, num_mlp_experts):
        super().__init__()
        self.attn = OptimizedLatentSelfAttention(latent_dim, num_heads, latent_bottleneck)
        self.norm1 = nn.LayerNorm(latent_dim)
        self.ffn = GroupedHybridFFN(
            in_dim=latent_dim,
            hidden_dim_ratio=mlp_dim_ratio,
            num_kan_experts=num_kan_experts,
            num_mlp_experts=num_mlp_experts
        )
        self.norm2 = nn.LayerNorm(latent_dim)

    def forward(self, x):
        x = x + self.attn(self.norm1(x))
        normed = self.norm2(x)
        ffn_out = self.ffn(normed)
        return x + ffn_out

class HybridTransformer(nn.Module):
    def __init__(self, vocab_size, emb_dim, latent_dim, latent_len, num_heads,
                 mlp_dim_ratio, num_layers, num_classes, latent_bottleneck,
                 num_kan_experts, num_mlp_experts):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, emb_dim)
        self.embed_to_latent = FusedMLPBlock(emb_dim, latent_dim)
        self.first_layer = OptimizedMLAWithRoPE(latent_dim, latent_dim, latent_len, num_heads)
        self.first_norm = nn.LayerNorm(latent_dim)
        
        self.layers = nn.ModuleList([
            HybridTransformerBlock(latent_dim, num_heads, mlp_dim_ratio,
                                   latent_bottleneck,
                                   num_kan_experts=num_kan_experts,
                                   num_mlp_experts=num_mlp_experts)
            for i in range(num_layers - 1)
        ])
        
        self.final_norm = nn.LayerNorm(latent_dim)
        self.classifier = FusedMLPBlock(latent_dim, num_classes)

    def forward(self, input_ids):
        x = self.embedding(input_ids)
        x = self.embed_to_latent(x)
        latents = self.first_layer(x)
        latents = self.first_norm(latents)
        
        for layer in self.layers:
            latents = layer(latents)
            
        normed_latents = self.final_norm(latents)
        pooled_output = torch.mean(normed_latents, dim=1)
        return self.classifier(pooled_output)

# ===================================================================
# Optimized Data Loading
# ===================================================================

max_length = 128
batch_size = 32

print("Loading dataset...")
dataset = load_dataset("ag_news")

labels = dataset["train"].features["label"].names
num_classes = len(labels)
print(f"Found {num_classes} classes (AG News).")

tokenizer = AutoTokenizer.from_pretrained("bert-base-uncased")

def tokenize_fn(examples):
    return tokenizer(
        examples["text"], 
        truncation=True, 
        max_length=max_length,
        padding=False
    )

print("Tokenizing dataset...")
tokenized_dataset = dataset.map(tokenize_fn, batched=True, remove_columns=["text"])
tokenized_dataset.set_format(type="torch")

def collate_fn(batch):
    max_len = max(len(item["input_ids"]) for item in batch)
    max_len = min(max_len, max_length)
    
    input_ids = torch.zeros((len(batch), max_len), dtype=torch.long)
    labels = torch.tensor([item["label"] for item in batch])
    
    for i, item in enumerate(batch):
        length = min(len(item["input_ids"]), max_len)
        input_ids[i, :length] = item["input_ids"][:length]
    
    return {"input_ids": input_ids, "label": labels}

print("Creating validation split from training data...")
train_val_split = tokenized_dataset["train"].train_test_split(test_size=0.1, seed=42)
train_dataset = train_val_split["train"].shuffle(seed=42)
val_dataset = train_val_split["test"]
test_dataset = tokenized_dataset["test"]

num_workers = min(4, os.cpu_count() or 2)

train_loader = DataLoader(
    train_dataset, batch_size=batch_size, shuffle=True,
    pin_memory=True, num_workers=num_workers, collate_fn=collate_fn,
    persistent_workers=True if num_workers > 0 else False
)
val_loader = DataLoader(
    val_dataset, batch_size=batch_size * 2,
    pin_memory=True, num_workers=num_workers, collate_fn=collate_fn,
    persistent_workers=True if num_workers > 0 else False
)
test_loader = DataLoader(
    test_dataset, batch_size=batch_size * 2,
    pin_memory=True, num_workers=num_workers, collate_fn=collate_fn,
    persistent_workers=True if num_workers > 0 else False
)

# ===================================================================
# Training Setup
# ===================================================================

vocab_size = tokenizer.vocab_size
emb_dim, latent_dim, latent_len = 128, 256, 32
num_heads, mlp_dim_ratio, num_layers = 8, 4, 6
latent_bottleneck = 8

num_kan_experts = 2
num_mlp_experts = 6

model = HybridTransformer(
    vocab_size=vocab_size, emb_dim=emb_dim, latent_dim=latent_dim, latent_len=latent_len,
    num_heads=num_heads, mlp_dim_ratio=mlp_dim_ratio, num_layers=num_layers,
    num_classes=num_classes, latent_bottleneck=latent_bottleneck,
    num_kan_experts=num_kan_experts,
    num_mlp_experts=num_mlp_experts
)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)

optimizer = optim.AdamW(model.parameters(), lr=3e-4, weight_decay=1e-2, fused=True)
criterion = nn.CrossEntropyLoss()
scaler = torch.amp.GradScaler('cuda', enabled=(device.type == "cuda"))

num_epochs = 20 
scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=num_epochs)

# ===================================================================
# Training Loop
# ===================================================================

@torch.inference_mode()
def evaluate(data_loader):
    model.eval()
    all_preds = []
    all_labels = []
    
    for batch in tqdm(data_loader, desc="Evaluating", leave=False):
        input_ids = batch["input_ids"].to(device, non_blocking=True)
        labels = batch["label"].to(device, non_blocking=True)
        
        with torch.amp.autocast('cuda'):
            logits = model(input_ids)
        
        preds = logits.argmax(-1)
        all_preds.extend(preds.cpu().numpy())
        all_labels.extend(labels.cpu().numpy())
    
    acc = (np.array(all_preds) == np.array(all_labels)).mean()
    return acc, all_labels, all_preds

def train_epoch():
    model.train()
    total_loss = 0.0
    pbar = tqdm(train_loader, desc="Training")
    
    for batch in pbar:
        input_ids = batch["input_ids"].to(device, non_blocking=True)
        labels = batch["label"].to(device, non_blocking=True)

        optimizer.zero_grad(set_to_none=True)
        
        with torch.amp.autocast('cuda'):
            logits = model(input_ids)
            loss = criterion(logits, labels)

        scaler.scale(loss).backward()
        scaler.unscale_(optimizer)
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        scaler.step(optimizer)
        scaler.update()

        total_loss += loss.item()
        pbar.set_postfix({"loss": f"{loss.item():.4f}"})
    
    return total_loss / len(train_loader)

# Training
print("\n=== Starting Training ===")
best_val_acc = 0.0
patience = 3
epochs_no_improve = 0
train_losses = []
val_accuracies = []

for epoch in range(num_epochs):
    print(f"\nEpoch {epoch + 1}/{num_epochs}")
    train_loss = train_epoch()
    val_acc, _, _ = evaluate(val_loader)
    scheduler.step()
    
    print(f"Loss: {train_loss:.4f} | Val Acc: {val_acc:.4f} | LR: {scheduler.get_last_lr()[0]:.2e}")
    
    train_losses.append(train_loss)
    val_accuracies.append(val_acc)
    
    if val_acc > best_val_acc:
        best_val_acc = val_acc
        torch.save(model.state_dict(), "best_model_ag_news.pt")
        print("Saved best model.")
        epochs_no_improve = 0
    else:
        epochs_no_improve += 1
    
    if epochs_no_improve >= patience:
        print(f"\nEarly stopping triggered after {patience} epochs with no improvement.")
        break

print("\n=== Final Test ===")
model.load_state_dict(torch.load("best_model_ag_news.pt"))
test_acc, test_labels, test_preds = evaluate(test_loader)
print(f"Test Accuracy: {test_acc:.4f}")

print("\n=== Classification Report ===")
print(classification_report(test_labels, test_preds, target_names=labels, digits=4))

print("\n=== Plotting Confusion Matrix ===")

cm = confusion_matrix(test_labels, test_preds)
plt.figure(figsize=(10, 8))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
            xticklabels=labels, yticklabels=labels, annot_kws={"size": 12})
plt.xlabel('Predicted Label')
plt.ylabel('True Label')
plt.title('Confusion Matrix (AG News)')
plt.savefig("confusion_matrix_ag_news_moe.png")
plt.show()

print("\n=== Plotting Results ===")
epochs_range = range(1, len(train_losses) + 1)
plt.figure(figsize=(12, 5))

plt.subplot(1, 2, 1)
plt.plot(epochs_range, train_losses, 'bo-', label='Training Loss')
plt.title('Training Loss per Epoch')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.grid(True)
plt.legend()

plt.subplot(1, 2, 2)
plt.plot(epochs_range, val_accuracies, 'ro-', label='Validation Accuracy')
plt.title('Validation Accuracy per Epoch')
plt.xlabel('Epoch')
plt.ylabel('Accuracy')
plt.grid(True)
plt.legend()

plt.tight_layout()
plt.savefig("training_plot_ag_news_moe.png")
plt.show()