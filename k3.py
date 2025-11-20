import torch
import torch.nn as nn
import torch.optim as optim
from datasets import load_dataset
from transformers import AutoTokenizer
from torch.utils.data import DataLoader
from kan import KANLayer # Assuming pykan is installed
from copy import deepcopy
import os
import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import confusion_matrix, classification_report
import seaborn as sns

# === Load Dataset & Tokenizer (No changes here) ===
dataset = load_dataset("ag_news")
tokenizer = AutoTokenizer.from_pretrained("bert-base-uncased")
max_length = 128

def tokenize_fn(examples):
    return tokenizer(examples["text"], padding="max_length", truncation=True, max_length=max_length)

dataset = dataset.map(tokenize_fn, batched=True)
dataset.set_format(type="torch", columns=["input_ids", "label"])

train_dataset_full = dataset["train"]
train_val_split = train_dataset_full.train_test_split(test_size=0.2, seed=42, stratify_by_column="label")
train_dataset = train_val_split["train"]
val_dataset = train_val_split["test"]
test_dataset = dataset["test"]

train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=64, shuffle=False)
test_loader = DataLoader(test_dataset, batch_size=64, shuffle=False)


# === Corrected Model Definition ===

def apply_rope(x, seq_dim=1):
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
    def __init__(self, input_dim, latent_dim, latent_len, num_heads):
        super().__init__()
        self.latent_len = latent_len
        self.latents = nn.Parameter(torch.randn(latent_len, latent_dim))
        self.q_proj = KANLayer(latent_dim, latent_dim, k=3)
        self.k_proj = KANLayer(input_dim, latent_dim, k=3)
        self.v_proj = KANLayer(input_dim, latent_dim, k=3)
        self.attn = nn.MultiheadAttention(embed_dim=latent_dim, num_heads=num_heads, batch_first=False)

    def forward(self, x):
        B, S, _ = x.shape
        x_flat = x.reshape(B * S, -1)
        k_flat = self.k_proj(x_flat)[0]
        v_flat = self.v_proj(x_flat)[0]
        k = k_flat.view(B, S, -1)
        v = v_flat.view(B, S, -1)

        k = apply_rope(k, seq_dim=1).transpose(0, 1)
        v = v.transpose(0, 1)

        latents = self.latents.unsqueeze(0).expand(B, -1, -1).contiguous()
        q_input = latents.reshape(B * self.latent_len, -1)
        q_proj = self.q_proj(q_input)[0]
        q = q_proj.view(B, self.latent_len, -1)
        q = apply_rope(q, seq_dim=1).transpose(0, 1)

        out, _ = self.attn(q, k, v)
        out = out.transpose(0, 1)
        return out

class TransformerMLABlockKAN(nn.Module):
    def __init__(self, latent_dim, latent_len, num_heads, mlp_dim):
        super().__init__()
        # This is now a self-attention block since input_dim and latent_dim are the same
        self.mla = MLAWithRoPEKAN(latent_dim, latent_dim, latent_len, num_heads)
        self.norm1 = nn.LayerNorm(latent_dim)
        self.fc1 = KANLayer(latent_dim, mlp_dim, k=3)
        self.fc2 = KANLayer(mlp_dim, latent_dim, k=3)
        self.norm2 = nn.LayerNorm(latent_dim)

    def forward(self, x):
        # x has shape [B, latent_len, latent_dim]
        # CORRECTED: Proper residual connection for the self-attention block
        h_res = self.mla(x)
        h = self.norm1(x + h_res)

        # CORRECTED: Proper residual connection for the KAN feed-forward block
        h_reshaped = h.reshape(-1, h.size(-1))
        h2 = self.fc1(h_reshaped)[0]
        h2 = self.fc2(h2)[0]
        h2 = h2.view_as(h)
        return self.norm2(h + h2)

class FullTransformerKAN(nn.Module):
    def __init__(self, vocab_size, emb_dim, latent_dim, latent_len, num_heads, mlp_dim, num_layers, num_classes):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, emb_dim)
        self.embed_to_latent = nn.Linear(emb_dim, latent_dim)

        # The first layer is a cross-attention from latents to the input text
        self.first_layer = MLAWithRoPEKAN(latent_dim, latent_dim, latent_len, num_heads)
        self.first_norm = nn.LayerNorm(latent_dim)

        # Subsequent layers are self-attention blocks on the latents
        self.layers = nn.ModuleList([
            TransformerMLABlockKAN(latent_dim, latent_len, num_heads, mlp_dim)
            for _ in range(num_layers - 1)
        ])

        self.classifier = KANLayer(latent_dim * latent_len, num_classes, k=3)

    def forward(self, input_ids):
        B, T = input_ids.shape
        x = self.embedding(input_ids)
        x = self.embed_to_latent(x)
        
        # CORRECTED: Feed the entire sequence into the first layer for cross-attention.
        # Do NOT average the input.
        latents = self.first_layer(x)
        latents = self.first_norm(latents)

        # Subsequent layers perform self-attention on these latents
        for layer in self.layers:
            latents = layer(latents)

        latents_flat = latents.flatten(1)
        logits = self.classifier(latents_flat)[0] # Correctly unpack KAN output
        return logits


# === Model & Training Setup (No changes here) ===

vocab_size = tokenizer.vocab_size
emb_dim = 128
latent_dim = 256
latent_len = 16
num_heads = 8
mlp_dim = 512
num_layers = 4
num_classes = 4

model = FullTransformerKAN(
    vocab_size, emb_dim, latent_dim, latent_len,
    num_heads, mlp_dim, num_layers, num_classes
)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = model.to(device)
print(f"Model moved to {device}")
print(f"Total parameters: {sum(p.numel() for p in model.parameters() if p.requires_grad):,}")

optimizer = optim.AdamW(model.parameters(), lr=3e-4, weight_decay=1e-2, fused=True)
criterion = nn.CrossEntropyLoss()
scaler = torch.amp.GradScaler('cuda', enabled=(device.type == "cuda"))

num_epochs = 50
scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=num_epochs)

# EarlyStopping class from your code
class EarlyStopping:
    def __init__(self, patience=3, min_delta=0.001, restore_best_weights=True):
        self.patience = patience
        self.min_delta = min_delta
        self.restore_best_weights = restore_best_weights
        self.best_score = None
        self.counter = 0
        self.best_weights = None
        
    def __call__(self, val_score, model):
        # Monitoring accuracy, so higher is better
        if self.best_score is None or val_score > self.best_score + self.min_delta:
            self.best_score = val_score
            self.save_checkpoint(model)
            self.counter = 0
        else:
            self.counter += 1
            print(f'EarlyStopping counter: {self.counter} out of {self.patience}')
            if self.counter >= self.patience:
                if self.restore_best_weights:
                    print('Restoring best model weights...')
                    model.load_state_dict(self.best_weights)
                return True
        return False
    
    def save_checkpoint(self, model):
        if self.restore_best_weights:
            self.best_weights = deepcopy(model.state_dict())

# Training and evaluation functions from your code
def train_epoch():
    model.train()
    total_loss = 0
    for batch in train_loader:
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
    avg_loss = total_loss / len(train_loader)
    print(f"Train loss: {avg_loss:.4f}")
    return avg_loss

def evaluate(data_loader, dataset_name="Validation"):
    model.eval()
    correct, total, total_loss = 0, 0, 0
    all_preds = []
    all_labels = []
    with torch.no_grad():
        for batch in data_loader:
            input_ids = batch["input_ids"].to(device)
            labels = batch["label"].to(device)
            logits = model(input_ids)
            loss = criterion(logits, labels)
            total_loss += loss.item()
            preds = logits.argmax(dim=-1)
            correct += (preds == labels).sum().item()
            total += labels.size(0)
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())
    accuracy = correct / total
    avg_loss = total_loss / len(data_loader)
    print(f"{dataset_name} accuracy: {accuracy:.4f}, {dataset_name} loss: {avg_loss:.4f}")
    return accuracy, avg_loss, all_preds, all_labels

# === Run Training ===
early_stopping = EarlyStopping(patience=5, restore_best_weights=True)
max_epochs = 50

# Create saves directory
save_dir = "saved_models_c1k3"
os.makedirs(save_dir, exist_ok=True)

# Track training history
train_loss_history = []
val_acc_history = []
epochs_list = []

print("Starting training with corrected KAN model...")
print("-" * 50)


# Replace the problematic section with this corrected version:

for epoch in range(max_epochs):
    print(f"Epoch {epoch + 1}/{max_epochs}")
    train_loss = train_epoch()
    val_accuracy, _, _, _ = evaluate(val_loader, "Validation")
    scheduler.step()
    print(f"LR: {scheduler.get_last_lr()[0]:.2e}")
    scheduler.step()
    
    print(f"LR: {scheduler.get_last_lr()[0]:.2e}")
    
    # Record metrics for plotting
    train_loss_history.append(train_loss)
    val_acc_history.append(val_accuracy)
    epochs_list.append(epoch + 1)

    # Print KAN layers every 2 epochs
    if (epoch + 1) % 2 == 0:
        print(f"Analyzing KAN layers at epoch {epoch + 1}...")
        model.eval()
        with torch.no_grad():
            # Create test input for analysis
            test_input = torch.randn(10, model.first_layer.q_proj.in_dim).to(device)
            
            with open("datac1k3.txt", "a") as f:
                f.write(f"\n=== EPOCH {epoch+1} ===\n")
                
                # Analyze q_proj
                y, preacts, postacts, postspline = model.first_layer.q_proj(test_input)
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

    if val_accuracy > (early_stopping.best_score or 0.0):
         print(f"New best validation accuracy: {val_accuracy:.4f}")

    if early_stopping(val_accuracy, model):
        print(f"\nEarly stopping triggered after {epoch + 1} epochs!")
        print(f"Best validation accuracy was: {early_stopping.best_score:.4f}")
        
        # Final analysis when training stops
        print("Creating final KAN layer analysis...")
        model.eval()
        with torch.no_grad():
            with open("datac1k3.txt", "a") as f:
                f.write(f"\n=== FINAL MODEL ANALYSIS ===\n")
                
                # Create test inputs for final analysis
                test_input = torch.randn(10, model.first_layer.q_proj.in_dim).to(device)
                y, preacts, postacts, postspline = model.first_layer.q_proj(test_input)
                f.write(f"Final q_proj:\n")
                f.write(f"  postspline: {postspline.cpu().numpy()}\n")
                f.write(f"  postacts: {postacts.cpu().numpy()}\n\n")
                
                # Similar for other layers...
                test_input_k = torch.randn(10, model.first_layer.k_proj.in_dim).to(device)
                y, preacts, postacts, postspline = model.first_layer.k_proj(test_input_k)
                f.write(f"Final k_proj:\n")
                f.write(f"  postspline: {postspline.cpu().numpy()}\n")
                f.write(f"  postacts: {postacts.cpu().numpy()}\n\n")
        break
    print("-" * 30)

# Plot training loss history
plt.figure(figsize=(10, 6))
plt.plot(epochs_list, train_loss_history, marker='o', linewidth=2, markersize=8)
plt.xlabel('Epoch', fontsize=12)
plt.ylabel('Training Loss', fontsize=12)
plt.title('Training Loss vs Epochs (c1k3)', fontsize=14)
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
plt.title('Validation Accuracy vs Epochs (c1k3)', fontsize=14)
plt.grid(True, alpha=0.3)
plt.tight_layout()
plot_path_acc = os.path.join(save_dir, "validation_accuracy.png")
plt.savefig(plot_path_acc, dpi=300)
plt.close()
print(f"✓ Validation accuracy plot saved to: {plot_path_acc}")

print("\nTraining completed!")
if early_stopping.best_score:
    print(f"Final best validation accuracy: {early_stopping.best_score:.4f}")

print("\n" + "="*50)
print("FINAL TEST SET EVALUATION")
print("="*50)
test_acc, test_loss, test_preds, test_labels = evaluate(test_loader, "Test")

# Generate classification report
print("\n--- Classification Report ---")
class_names = ['World', 'Sports', 'Business', 'Sci/Tech']
report = classification_report(test_labels, test_preds, target_names=class_names, digits=4)
print(report)

# Save classification report to file
report_path = os.path.join(save_dir, "classification_report.txt")
with open(report_path, "w") as f:
    f.write("Classification Report - Test Set (c1k3)\n")
    f.write("=" * 80 + "\n\n")
    f.write(report)
print(f"✓ Classification report saved to: {report_path}")

# Generate and save confusion matrix heatmap
cm = confusion_matrix(test_labels, test_preds)
plt.figure(figsize=(10, 9))
sns.heatmap(cm, annot=True, fmt='d', cmap='YlGnBu', 
            xticklabels=class_names, yticklabels=class_names,
            cbar_kws={'label': 'Count'}, linewidths=0.5, linecolor='gray')
plt.xlabel('Predicted Label', fontsize=13, fontweight='bold')
plt.ylabel('True Label', fontsize=13, fontweight='bold')
plt.title('Confusion Matrix Heatmap - Test Set (c1k3)', fontsize=15, fontweight='bold', pad=20)
plt.xticks(rotation=45, ha='right')
plt.yticks(rotation=0)
plt.tight_layout()
cm_path = os.path.join(save_dir, "confusion_matrix.png")
plt.savefig(cm_path, dpi=300, bbox_inches='tight')
plt.close()
print(f"✓ Confusion matrix heatmap saved to: {cm_path}") 