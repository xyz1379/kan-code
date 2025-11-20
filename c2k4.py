import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from datasets import load_dataset
from transformers import AutoTokenizer
from torch.utils.data import DataLoader
from copy import deepcopy
from tqdm import tqdm
import os
import json
from datetime import datetime
import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import confusion_matrix, classification_report
import seaborn as sns
# You must have pykan installed: pip install pykan
from kan import KANLayer

# ===================================================================
# 1. MODEL DEFINITION
# ===================================================================

def apply_rope(x, seq_dim=1):
    """Applies Rotary Positional Embedding."""
    dim = x.size(-1)
    device = x.device
    theta = 10000 ** (-torch.arange(0, dim, 2, device=device).float() / dim)
    pos = torch.arange(x.size(seq_dim), device=device).float()
    freqs = torch.einsum("i,j->ij", pos, theta)
    sin, cos = freqs.sin(), freqs.cos()
    x1, x2 = x[..., 0::2], x[..., 1::2]
    x_rotated = torch.stack([x1 * cos - x2 * sin, x1 * sin + x2 * cos], dim=-1)
    return x_rotated.flatten(-2)

class MLAWithRoPEKAN(nn.Module):
    """Perceiver-style Cross-Attention with KAN projections."""
    def __init__(self, input_dim, latent_dim, latent_len, num_heads):
        super().__init__()
        self.latent_len = latent_len
        self.latents = nn.Parameter(torch.randn(latent_len, latent_dim))
        self.q_proj = KANLayer(latent_dim, latent_dim, k=4)
        self.k_proj = KANLayer(input_dim, latent_dim, k=4)
        self.v_proj = KANLayer(input_dim, latent_dim, k=4)
        self.attn = nn.MultiheadAttention(embed_dim=latent_dim, num_heads=num_heads, batch_first=False)

    def forward(self, x):
        B, S, _ = x.shape
        x_flat = x.view(B * S, -1)
        k_flat = self.k_proj(x_flat)[0]
        v_flat = self.v_proj(x_flat)[0]
        k, v = k_flat.view(B, S, -1), v_flat.view(B, S, -1)
        k, v = apply_rope(k).transpose(0, 1), v.transpose(0, 1)
        latents_input = self.latents.unsqueeze(0).expand(B, -1, -1).contiguous().view(B * self.latent_len, -1)
        q_proj = self.q_proj(latents_input)[0]
        q = apply_rope(q_proj.view(B, self.latent_len, -1)).transpose(0, 1)
        out, _ = self.attn(q, k, v)
        return out.transpose(0, 1)

class LatentSelfAttentionKAN(nn.Module):
    """Efficient self-attention with KAN projections."""
    def __init__(self, embed_dim: int, num_heads: int, bottleneck_size: int):
        super().__init__()
        self.embed_dim, self.num_heads, self.bottleneck_size = embed_dim, num_heads, bottleneck_size
        self.head_dim = embed_dim // num_heads
        self.qkv_proj = KANLayer(embed_dim, embed_dim * 3, k=4)
        self.out_proj = KANLayer(embed_dim, embed_dim, k=4)
        self.kv_compressor = nn.AdaptiveAvgPool1d(bottleneck_size)

    def forward(self, x: torch.Tensor):
        B, L, D = x.shape
        x_flat = x.view(B * L, D) ###
        qkv = self.qkv_proj(x_flat)[0]  ###
        qkv = qkv.view(B, L, 3, self.num_heads, self.head_dim).permute(2, 0, 3, 1, 4)
        q, k, v = qkv[0], qkv[1], qkv[2]
        k_reshaped = k.contiguous().view(B * self.num_heads, L, self.head_dim).permute(0, 2, 1)
        v_reshaped = v.contiguous().view(B * self.num_heads, L, self.head_dim).permute(0, 2, 1)
        latent_k = self.kv_compressor(k_reshaped).permute(0, 2, 1).view(B, self.num_heads, self.bottleneck_size, self.head_dim)
        latent_v = self.kv_compressor(v_reshaped).permute(0, 2, 1).view(B, self.num_heads, self.bottleneck_size, self.head_dim)
        attn_output = F.scaled_dot_product_attention(q, latent_k, latent_v)
        attn_output = attn_output.transpose(1, 2).contiguous().view(B, L, D)

        attn_flat = attn_output.view(B * L, D)##
        final_flat = self.out_proj(attn_flat)[0]##
        final_output = final_flat.view(B, L, D)##
        return final_output

class TransformerMLABlockKAN(nn.Module):
    """The main transformer block using KANs."""
    def __init__(self, latent_dim, num_heads, mlp_dim, latent_bottleneck):
        super().__init__()
        self.attn = LatentSelfAttentionKAN(latent_dim, num_heads, latent_bottleneck)
        self.norm1 = nn.LayerNorm(latent_dim)
        self.fc1 = KANLayer(latent_dim, mlp_dim, k=4)
        self.fc2 = KANLayer(mlp_dim, latent_dim, k=4)
        self.norm2 = nn.LayerNorm(latent_dim)

    def forward(self, x):
        x = x + self.attn(self.norm1(x))
        h_reshaped = self.norm2(x).view(-1, x.size(-1))
        h2 = self.fc1(h_reshaped)[0]
        h2 = self.fc2(h2)[0]
        x = x + h2.view_as(x)
        return x

class FullTransformerKAN(nn.Module):
    """The complete KAN-based model architecture."""
    def __init__(self, vocab_size, emb_dim, latent_dim, latent_len, num_heads, mlp_dim, num_layers, num_classes, latent_bottleneck):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, emb_dim)
        self.embed_to_latent = KANLayer(emb_dim, latent_dim, k=4)
        self.first_layer = MLAWithRoPEKAN(latent_dim, latent_dim, latent_len, num_heads)
        self.first_norm = nn.LayerNorm(latent_dim)
        self.layers = nn.ModuleList([
            TransformerMLABlockKAN(latent_dim, num_heads, mlp_dim, latent_bottleneck)
            for _ in range(num_layers - 1)
        ])
        self.final_norm = nn.LayerNorm(latent_dim)
        self.classifier = KANLayer(latent_dim * latent_len, num_classes, k=4)

    def forward(self, input_ids):
        B, S = input_ids.size()
        x = self.embedding(input_ids)              # (B, S, emb_dim)
        x_flat = x.view(B * S, x.size(-1))
        x_latent_flat = self.embed_to_latent(x_flat)[0]   # (B*S, latent_dim)
        x = x_latent_flat.view(B, S, -1)           # (B, S, latent_dim)
        latents = self.first_layer(x)              # (B, latent_len, latent_dim)
        latents = self.first_norm(latents)
        for layer in self.layers:
            latents = layer(latents)
        latents_flat = self.final_norm(latents).flatten(1)
        logits = self.classifier(latents_flat)[0]
        return logits

# ===================================================================
# 2. MODEL SAVING AND LOADING UTILITIES
# ===================================================================

def save_model(model, tokenizer, model_config, save_path, epoch=None, accuracy=None):
    """Save the complete model, tokenizer, and configuration for inference."""
    os.makedirs(save_path, exist_ok=True)
    
    # Save model state dict
    model_path = os.path.join(save_path, "model.pth")
    torch.save(model.state_dict(), model_path)
    
    # Save tokenizer
    tokenizer_path = os.path.join(save_path, "tokenizer")
    tokenizer.save_pretrained(tokenizer_path)
    
    # Save model configuration and metadata
    config = {
        "model_config": model_config,
        "timestamp": datetime.now().isoformat(),
        "epoch": epoch,
        "accuracy": accuracy
    }
    config_path = os.path.join(save_path, "config.json")
    with open(config_path, "w") as f:
        json.dump(config, f, indent=2)
    
    print(f"✓ Model saved to {save_path}")
    return save_path

def load_model_for_inference(save_path, device="cpu"):
    """Load a saved model for inference."""
    # Load configuration
    config_path = os.path.join(save_path, "config.json")
    with open(config_path, "r") as f:
        config = json.load(f)
    
    model_config = config["model_config"]
    
    # Load tokenizer
    tokenizer_path = os.path.join(save_path, "tokenizer")
    tokenizer = AutoTokenizer.from_pretrained(tokenizer_path)
    
    # Initialize model with saved configuration
    model = FullTransformerKAN(**model_config)
    
    # Load model weights
    model_path = os.path.join(save_path, "model.pth")
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.to(device)
    model.eval()
    
    print(f"✓ Model loaded from {save_path}")
    print(f"  - Saved at: {config.get('timestamp', 'Unknown')}")
    print(f"  - Epoch: {config.get('epoch', 'Unknown')}")
    print(f"  - Accuracy: {config.get('accuracy', 'Unknown')}")
    
    return model, tokenizer, config

def inference_single_text(model, tokenizer, text, max_length=512, device="cpu"):
    """Perform inference on a single text."""
    model.eval()
    with torch.no_grad():
        # Tokenize input
        inputs = tokenizer(text, padding="max_length", truncation=True, 
                          max_length=max_length, return_tensors="pt")
        input_ids = inputs["input_ids"].to(device)
        
        # Get prediction
        logits = model(input_ids)
        probabilities = torch.softmax(logits, dim=-1)
        predicted_class = torch.argmax(logits, dim=-1).item()
        confidence = probabilities[0, predicted_class].item()
        
        return predicted_class, confidence, probabilities[0].cpu().numpy()

# ===================================================================
# 3. DATA LOADING AND PREPARATION
# ===================================================================

# --- Key Parameters ---
max_length = 512  # ⚠ High VRAM needed! Decrease to 2048 or 1024 if you get memory errors.
batch_size = 2     # Use a small batch size for long sequences

# --- Load Dataset ---
print("Loading ccdv/arxiv-classification dataset (this may take a while)...")
dataset = load_dataset("ccdv/arxiv-classification", "default")
print(dataset["train"].features)
# --- Map String Labels to Integers ---
labels = dataset["train"].features["label"].names
label2id = {label: i for i, label in enumerate(labels)}
num_classes = len(labels)
print(f"Found {num_classes} classes.")

# --- Tokenization ---
tokenizer = AutoTokenizer.from_pretrained("bert-base-uncased")
def tokenize_fn(examples):
    # We use the 'text' as input (as per dataset viewer)
    return tokenizer(examples["text"], padding="max_length", truncation=True, max_length=max_length)

print("Tokenizing dataset...")
tokenized_dataset = dataset.map(tokenize_fn, batched=True, remove_columns=["text"])
tokenized_dataset.set_format("torch")

# --- Create DataLoaders ---
train_dataset = tokenized_dataset["train"].shuffle(seed=42)
val_dataset = tokenized_dataset["validation"]
test_dataset = tokenized_dataset["test"]

train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=batch_size * 2)
test_loader = DataLoader(test_dataset, batch_size=batch_size * 2)

# ===================================================================
# 4. TRAINING SETUP
# ===================================================================

# --- Model Hyperparameters ---
vocab_size = tokenizer.vocab_size
emb_dim, latent_dim, latent_len = 128, 256, 32
num_heads, mlp_dim, num_layers = 8, 1024, 6 # mlp_dim is now an absolute value
latent_bottleneck = 8

# Store model configuration for saving
model_config = {
    "vocab_size": vocab_size,
    "emb_dim": emb_dim,
    "latent_dim": latent_dim,
    "latent_len": latent_len,
    "num_heads": num_heads,
    "mlp_dim": mlp_dim,
    "num_layers": num_layers,
    "num_classes": num_classes,
    "latent_bottleneck": latent_bottleneck
}

# --- Instantiate Model ---
model = FullTransformerKAN(**model_config)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)
print(f"Model with {sum(p.numel() for p in model.parameters() if p.requires_grad):,} parameters moved to {device}")

# --- Optimizer and Loss Function ---
optimizer = optim.AdamW(model.parameters(), lr=3e-4, weight_decay=1e-2, fused=True)
criterion = nn.CrossEntropyLoss()
scaler = torch.amp.GradScaler('cuda', enabled=(device.type == "cuda"))

num_epochs = 20
scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=num_epochs)

# --- Early Stopping ---
class EarlyStopping:
    def __init__(self, patience=3, restore_best_weights=True):
        self.patience, self.restore_best_weights = patience, restore_best_weights
        self.best_score, self.counter, self.best_weights = None, 0, None
    def __call__(self, val_score, model):
        if self.best_score is None or val_score > self.best_score:
            self.best_score, self.counter = val_score, 0
            if self.restore_best_weights: self.best_weights = deepcopy(model.state_dict())
        else:
            self.counter += 1
            if self.counter >= self.patience:
                if self.restore_best_weights: model.load_state_dict(self.best_weights)
                return True
        return False

# ===================================================================
# 5. TRAINING AND EVALUATION LOOPS
# ===================================================================

def train_epoch():
    model.train()
    total_loss = 0
    pbar = tqdm(train_loader, desc="Training", leave=False)
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
    avg_loss = total_loss / len(train_loader)
    return avg_loss

def evaluate(data_loader):
    model.eval()
    correct, total = 0, 0
    all_preds = []
    all_labels = []
    with torch.no_grad():
        pbar = tqdm(data_loader, desc="Evaluating", leave=False)
        for batch in pbar:
            input_ids = batch["input_ids"].to(device)
            labels = batch["label"].to(device)
            logits = model(input_ids)
            preds = logits.argmax(dim=-1)
            correct += (preds == labels).sum().item()
            total += labels.size(0)
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())
    return correct / total, all_preds, all_labels

# ===================================================================
# 6. MAIN EXECUTION
# ===================================================================

early_stopping = EarlyStopping(patience=3)
max_epochs = 20

# Create saves directory
save_dir = "saved_models_k4"
os.makedirs(save_dir, exist_ok=True)

# Track training history
train_loss_history = []
val_acc_history = []
epochs_list = []

print("\n--- Starting KAN Model Training ---")
for epoch in range(max_epochs):
    print(f"\nEpoch {epoch + 1}/{max_epochs}")
    train_loss = train_epoch()
    val_acc, _, _ = evaluate(val_loader)
    scheduler.step()
    print(f"Epoch {epoch + 1} | Train Loss: {train_loss:.4f} | Validation Accuracy: {val_acc:.4f} | LR: {scheduler.get_last_lr()[0]:.2e}")
    
    # Print KAN layers every 2 epochs
    if (epoch + 1) % 2 == 0:
        print(f"Analyzing KAN layers at epoch {epoch + 1}...")
        model.eval()
        with torch.no_grad():
            # Create test input for analysis
            
            with open("datac2k4.txt", "a") as f:
                f.write(f"\n=== EPOCH {epoch+1} ===\n")

                # Add this new layer - embed_to_latent (KAN in c2k2.py vs Linear in k=4.py)
                test_input_embed = torch.randn(10, model.embed_to_latent.in_dim).to(device)
                y, preacts, postacts, postspline = model.embed_to_latent(test_input_embed)
                f.write(f"embed_to_latent {epoch+1}:\n")
                f.write(f"  postspline shape: {postspline.shape}\n")
                f.write(f"  postspline values: {postspline.cpu().numpy()}\n")
                f.write(f"  postacts shape: {postacts.shape}\n")
                f.write(f"  postacts values: {postacts.cpu().numpy()}\n\n")
                    
                # Analyze q_proj
                test_input_q = torch.randn(10, model.first_layer.q_proj.in_dim).to(device)
                y, preacts, postacts, postspline = model.first_layer.q_proj(test_input_q)
                f.write(f"q_proj {epoch+1}:\n")
                f.write(f"  postspline shape: {postspline.shape}\n")
                f.write(f"  postspline values: {postspline.cpu().numpy()}\n")
                f.write(f"  postacts shape: {postacts.shape}\n")
                f.write(f"  postacts values: {postacts.cpu().numpy()}\n\n")
                
                # Analyze k_proj (needs input dimension matching)
                test_input_k = torch.randn(10, model.first_layer.k_proj.in_dim).to(device)
                y, preacts, postacts, postspline = model.first_layer.k_proj(test_input_k)
                f.write(f"k_proj {epoch+1}:\n")
                f.write(f"  postspline shape: {postspline.shape}\n")
                f.write(f"  postspline values: {postspline.cpu().numpy()}\n")
                f.write(f"  postacts shape: {postacts.shape}\n")
                f.write(f"  postacts values: {postacts.cpu().numpy()}\n\n")
                
                # Analyze v_proj
                test_input_v = torch.randn(10, model.first_layer.v_proj.in_dim).to(device)
                y, preacts, postacts, postspline = model.first_layer.v_proj(test_input_v)
                f.write(f"v_proj {epoch+1}:\n")
                f.write(f"  postspline shape: {postspline.shape}\n")
                f.write(f"  postspline values: {postspline.cpu().numpy()}\n")
                f.write(f"  postacts shape: {postacts.shape}\n")
                f.write(f"  postacts values: {postacts.cpu().numpy()}\n\n")
                
                # Analyze transformer layers
                for i, layer in enumerate(model.layers):

                    # Add qkv_proj (new in c2k2.py)
                    test_input_qkv = torch.randn(10, layer.attn.qkv_proj.in_dim).to(device)
                    y, preacts, postacts, postspline = layer.attn.qkv_proj(test_input_qkv)
                    f.write(f"layer_{i}_attn_qkv_proj {epoch+1}:\n")
                    f.write(f"  postspline shape: {postspline.shape}\n")
                    f.write(f"  postspline values: {postspline.cpu().numpy()}\n")
                    f.write(f"  postacts shape: {postacts.shape}\n")
                    f.write(f"  postacts values: {postacts.cpu().numpy()}\n\n")

                    # Add out_proj (new in c2k2.py)
                    test_input_out = torch.randn(10, layer.attn.out_proj.in_dim).to(device)
                    y, preacts, postacts, postspline = layer.attn.out_proj(test_input_out)
                    f.write(f"layer_{i}_attn_out_proj {epoch+1}:\n")
                    f.write(f"  postspline shape: {postspline.shape}\n")
                    f.write(f"  postspline values: {postspline.cpu().numpy()}\n")
                    f.write(f"  postacts shape: {postacts.shape}\n")
                    f.write(f"  postacts values: {postacts.cpu().numpy()}\n\n")

                    # fc1
                    test_input_fc1 = torch.randn(10, layer.fc1.in_dim).to(device)
                    y, preacts, postacts, postspline = layer.fc1(test_input_fc1)
                    f.write(f"layer_{i}_fc1 {epoch+1}:\n")
                    f.write(f"  postspline shape: {postspline.shape}\n")
                    f.write(f"  postspline values: {postspline.cpu().numpy()}\n")
                    f.write(f"  postacts shape: {postacts.shape}\n")
                    f.write(f"  postacts values: {postacts.cpu().numpy()}\n\n")
                    
                    # fc2
                    test_input_fc2 = torch.randn(10, layer.fc2.in_dim).to(device)
                    y, preacts, postacts, postspline = layer.fc2(test_input_fc2)
                    f.write(f"layer_{i}_fc2 {epoch+1}:\n")
                    f.write(f"  postspline shape: {postspline.shape}\n")
                    f.write(f"  postspline values: {postspline.cpu().numpy()}\n")
                    f.write(f"  postacts shape: {postacts.shape}\n")
                    f.write(f"  postacts values: {postacts.cpu().numpy()}\n\n")
                
                # Analyze classifier
                test_input_cls = torch.randn(10, model.classifier.in_dim).to(device)
                y, preacts, postacts, postspline = model.classifier(test_input_cls)
                f.write(f"classifier {epoch+1}:\n")
                f.write(f"  postspline shape: {postspline.shape}\n")
                f.write(f"  postspline values: {postspline.cpu().numpy()}\n")
                f.write(f"  postacts shape: {postacts.shape}\n")
                f.write(f"  postacts values: {postacts.cpu().numpy()}\n\n")
    
    # Record metrics for plotting
    train_loss_history.append(train_loss)
    val_acc_history.append(val_acc)
    epochs_list.append(epoch + 1)
    
    # Replace the final analysis section with this complete version:

    if early_stopping(val_acc, model):
        print("Early stopping triggered!")
        break

# Save final model after training completes
final_model_path = os.path.join(save_dir, "final_model")
final_val_acc, _, _ = evaluate(val_loader)
save_model(model, tokenizer, model_config, final_model_path, 
          epoch=epoch+1, accuracy=final_val_acc)

# Plot training loss history
plt.figure(figsize=(10, 6))
plt.plot(epochs_list, train_loss_history, marker='o', linewidth=2, markersize=8)
plt.xlabel('Epoch', fontsize=12)
plt.ylabel('Training Loss', fontsize=12)
plt.title('Training Loss vs Epochs (k=4)', fontsize=14)
plt.grid(True, alpha=0.3)
plt.tight_layout()
plot_path = os.path.join(save_dir, "training_loss.png")
plt.savefig(plot_path, dpi=300)
plt.close()
print(f"✓ Training loss plot saved to: {plot_path}")

# Plot validation accuracy history
plt.figure(figsize=(10, 6))
plt.plot(epochs_list, val_acc_history, marker='s', linewidth=2, markersize=8, color='green')
plt.xlabel('Epoch', fontsize=12)
plt.ylabel('Validation Accuracy', fontsize=12)
plt.title('Validation Accuracy vs Epochs (k=4)', fontsize=14)
plt.grid(True, alpha=0.3)
plt.tight_layout()
plot_path_acc = os.path.join(save_dir, "validation_accuracy.png")
plt.savefig(plot_path_acc, dpi=300)
plt.close()
print(f"✓ Validation accuracy plot saved to: {plot_path_acc}")

print("\n--- Final Test Evaluation on Best KAN Model ---")
test_acc, test_preds, test_labels = evaluate(test_loader)
print(f"Final Test Accuracy: {test_acc:.4f}")

# Generate classification report
print("\n--- Classification Report ---")
report = classification_report(test_labels, test_preds, target_names=labels, digits=4)
print(report)

# Save classification report to file
report_path = os.path.join(save_dir, "classification_report.txt")
with open(report_path, "w") as f:
    f.write("Classification Report - Test Set (k=4)\n")
    f.write("=" * 80 + "\n\n")
    f.write(report)
print(f"✓ Classification report saved to: {report_path}")

# Generate and save confusion matrix heatmap
cm = confusion_matrix(test_labels, test_preds)
plt.figure(figsize=(14, 12))
sns.heatmap(cm, annot=True, fmt='d', cmap='YlOrRd', 
            xticklabels=labels, yticklabels=labels,
            cbar_kws={'label': 'Count'}, linewidths=0.5, linecolor='gray')
plt.xlabel('Predicted Label', fontsize=13, fontweight='bold')
plt.ylabel('True Label', fontsize=13, fontweight='bold')
plt.title('Confusion Matrix Heatmap - Test Set (k=4)', fontsize=15, fontweight='bold', pad=20)
plt.xticks(rotation=45, ha='right')
plt.yticks(rotation=0)
plt.tight_layout()
cm_path = os.path.join(save_dir, "confusion_matrix.png")
plt.savefig(cm_path, dpi=300, bbox_inches='tight')
plt.close()
print(f"✓ Confusion matrix heatmap saved to: {cm_path}")

print(f"✓ Final model saved to: {final_model_path}")

# ===================================================================

# 7. EXAMPLE USAGE FOR INFERENCE (COMMENTED OUT)
# ===================================================================

"""
# Example of how to load and use the saved model for inference:

# Load the best model
model_path = "saved_models/final_model"  # or any other saved model
loaded_model, loaded_tokenizer, config = load_model_for_inference(model_path, device=device)

# Example text classification
sample_text = "This paper presents a novel approach to neural networks using transformers..."
predicted_class, confidence, probabilities = inference_single_text(
    loaded_model, loaded_tokenizer, sample_text, device=device
)

print(f"Predicted class: {predicted_class}")
print(f"Confidence: {confidence:.4f}")
print(f"Class probabilities: {probabilities}")

# Get class name if you have the labels
class_name = labels[predicted_class]
print(f"Predicted class name: {class_name}")
"""