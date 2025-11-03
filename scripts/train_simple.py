"""
Simple Training Script - Qwen 1.5B with LoRA & Hash Conditioning
Avoids problematic torchao imports by using direct training loop
"""

import json
import torch
from torch.optim import AdamW
from torch.utils.data import DataLoader, Dataset
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import LoraConfig, get_peft_model, TaskType
from tqdm import tqdm


# ===================== 1. LOAD CONFIG =====================
def load_config(config_path: str = "configs/training_config.json") -> dict:
    """Load training configuration"""
    with open(config_path) as f:
        config = json.load(f)
    print(f"[OK] Config loaded")
    return config


# ===================== 2. SETUP TOKENIZER =====================
def setup_tokenizer(model_name: str):
    """Load and configure tokenizer"""
    print("\n[*] Loading tokenizer...")
    tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True, padding_side="right")
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    print("[OK] Tokenizer ready")
    return tokenizer


# ===================== 3. SETUP MODEL =====================
def setup_model(model_name: str):
    """Load model in float16 (no quantization to avoid torch.compile issues)"""
    print("\n[*] Loading model in float16...")
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        device_map="auto",
        torch_dtype=torch.float16,
        trust_remote_code=True
    )
    print(f"[OK] Model loaded: {model_name}")
    return model


# ===================== 4. SETUP LORA =====================
def setup_lora(model, config: dict):
    """Apply LoRA adapters"""
    print("\n[*] Setting up LoRA...")
    
    # LoRA configuration
    lora_config = LoraConfig(
        r=config.get("lora_r", 8),
        lora_alpha=config.get("lora_alpha", 16),
        target_modules=config.get("lora_target_modules", ["q_proj", "v_proj"]),
        lora_dropout=config.get("lora_dropout", 0.05),
        bias="none",
        task_type=TaskType.CAUSAL_LM
    )
    
    # Apply LoRA to model
    model = get_peft_model(model, lora_config)
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    total_params = sum(p.numel() for p in model.parameters())
    pct = 100 * trainable_params / total_params
    print(f"[OK] LoRA Applied | Trainable: {trainable_params:,} / Total: {total_params:,} ({pct:.4f}%)")
    return model


# ===================== 5. LOAD DATASETS =====================
def load_datasets(train_path: str = "data/train_data.json", val_path: str = "data/val_data.json"):
    """Load training and validation data"""
    print("\n[*] Loading datasets...")
    
    with open(train_path) as f:
        train_data = json.load(f)
    with open(val_path) as f:
        val_data = json.load(f)
    
    print(f"[OK] Train data: {len(train_data)} examples")
    print(f"[OK] Val data: {len(val_data)} examples")
    return train_data, val_data


# ===================== 6. FORMAT DATA =====================
def format_data(examples: list) -> list:
    """Add hash conditioning prefix to examples"""
    formatted = []
    for ex in examples:
        hash_prefix = f"[hash: {ex.get('hash', 'unknown')}]"
        text = f"{hash_prefix} User: {ex['instruction']}\nAssistant: {ex['output']}"
        formatted.append({"text": text})
    return formatted


# ===================== 7. TOKENIZE DATA =====================
def tokenize_data(examples: list, tokenizer, max_length: int = 512):
    """Convert text to token IDs"""
    tokenized = []
    for ex in examples:
        tokens = tokenizer(
            ex["text"],
            max_length=max_length,
            truncation=True,
            padding="max_length",
            return_tensors="pt"
        )
        tokenized.append({
            "input_ids": tokens["input_ids"].squeeze(),
            "attention_mask": tokens["attention_mask"].squeeze()
        })
    return tokenized


# ===================== 8️⃣ DATASET CLASS =====================
class TextDataset(Dataset):
    """PyTorch Dataset wrapper"""
    def __init__(self, tokenized_data):
        self.data = tokenized_data
    
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        return self.data[idx]


# ===================== 9. TRAINING LOOP =====================
def train_model(model, train_loader, val_loader, config: dict, epochs: int = 3):
    """Simple training loop without Trainer to avoid torchao imports"""
    print(f"\n[*] Starting training for {epochs} epochs...")
    
    # Optimizer
    optimizer = AdamW(model.parameters(), lr=config.get("learning_rate", 2e-4))
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[*] Using device: {device}")
    model = model.to(device)
    
    for epoch in range(epochs):
        print(f"\n[EPOCH {epoch + 1}/{epochs}]")
        
        # Training
        model.train()
        total_loss = 0
        progress_bar = tqdm(train_loader, desc="Training")
        
        for batch in progress_bar:
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            
            # Forward pass
            outputs = model(
                input_ids=input_ids,
                attention_mask=attention_mask,
                labels=input_ids
            )
            loss = outputs.loss
            
            # Backward pass
            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            
            total_loss += loss.item()
            progress_bar.set_postfix({"loss": loss.item()})
        
        avg_loss = total_loss / len(train_loader)
        print(f"[OK] Training Loss: {avg_loss:.4f}")
        
        # Validation
        if val_loader:
            model.eval()
            val_loss = 0
            with torch.no_grad():
                for batch in tqdm(val_loader, desc="Validation"):
                    input_ids = batch["input_ids"].to(device)
                    attention_mask = batch["attention_mask"].to(device)
                    
                    outputs = model(
                        input_ids=input_ids,
                        attention_mask=attention_mask,
                        labels=input_ids
                    )
                    val_loss += outputs.loss.item()
            
            avg_val_loss = val_loss / len(val_loader)
            print(f"[OK] Validation Loss: {avg_val_loss:.4f}")


# ===================== 🔟 SAVE MODEL =====================
def save_model(model, tokenizer, output_dir: str = "outputs/"):
    """Save fine-tuned adapters"""
    print(f"\n💾 Saving model to {output_dir}...")
    model.save_pretrained(output_dir)
    tokenizer.save_pretrained(output_dir)
    print(f"✓ Model saved!")


# ===================== MAIN =====================
def main():
    """Orchestrate the training pipeline"""
    print("=" * 60)
    print("QWEN 1.5B LORA TRAINING WITH HASH CONDITIONING")
    print("=" * 60)
    
    # [1/8] Load config
    config = load_config()
    
    # [2/8] Setup tokenizer
    tokenizer = setup_tokenizer(config["model_name"])
    
    # [3/8] Setup model
    model = setup_model(config["model_name"])
    
    # [4/8] Setup LoRA
    model = setup_lora(model, config)
    
    # [5/8] Load datasets
    train_data, val_data = load_datasets()
    
    # [6/8] Format data (add hash conditioning)
    train_formatted = format_data(train_data)
    val_formatted = format_data(val_data)
    
    # [7/8] Tokenize
    train_tokenized = tokenize_data(train_formatted, tokenizer)
    val_tokenized = tokenize_data(val_formatted, tokenizer)
    
    # Create DataLoaders
    train_dataset = TextDataset(train_tokenized)
    val_dataset = TextDataset(val_tokenized)
    train_loader = DataLoader(train_dataset, batch_size=4, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=4, shuffle=False)
    
    # [8/8] Train
    train_model(
        model, 
        train_loader, 
        val_loader, 
        config,
        epochs=config.get("num_epochs", 3)
    )
    
    # Save
    save_model(model, tokenizer)
    
    print("\n" + "=" * 60)
    print("TRAINING COMPLETE!")
    print("=" * 60)


if __name__ == "__main__":
    main()
