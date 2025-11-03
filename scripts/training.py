import torch
from torch.optim import AdamW
from torch.utils.data import DataLoader
from tqdm import tqdm


def train_model(model, train_loader, val_loader, config: dict, epochs: int = 3):
    print(f"\n[*] Starting training for {epochs} epochs...")
    
    optimizer = AdamW(model.parameters(), lr=config.get("learning_rate", 2e-4))
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[*] Using device: {device}")
    model = model.to(device)
    
    for epoch in range(epochs):
        print(f"\n[EPOCH {epoch + 1}/{epochs}]")
        
        model.train()
        total_loss = 0
        progress_bar = tqdm(train_loader, desc="Training")
        
        for batch in progress_bar:
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            
            outputs = model(
                input_ids=input_ids,
                attention_mask=attention_mask,
                labels=input_ids
            )
            loss = outputs.loss
            
            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            
            total_loss += loss.item()
            progress_bar.set_postfix({"loss": loss.item()})
        
        avg_loss = total_loss / len(train_loader)
        print(f"[OK] Training Loss: {avg_loss:.4f}")
        
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


def save_model(model, tokenizer, output_dir: str = "outputs/"):
    print(f"\n[*] Saving model to {output_dir}...")
    model.save_pretrained(output_dir)
    tokenizer.save_pretrained(output_dir)
    print(f"[OK] Model saved!")
