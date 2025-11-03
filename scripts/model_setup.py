import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import LoraConfig, get_peft_model, TaskType


def load_config(config_path: str = "configs/training_config.json") -> dict:
    import json
    with open(config_path) as f:
        config = json.load(f)
    print(f"[OK] Config loaded from {config_path}")
    return config


def setup_tokenizer(model_name: str):
    print("\n[*] Loading tokenizer...")
    tokenizer = AutoTokenizer.from_pretrained(
        model_name,
        trust_remote_code=True,
        padding_side="right"
    )
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    print("[OK] Tokenizer ready")
    return tokenizer


def setup_model(model_name: str):
    print("\n[*] Loading model in float16...")
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        device_map="auto",
        torch_dtype=torch.float16,
        trust_remote_code=True
    )
    print(f"[OK] Model loaded: {model_name}")
    return model


def setup_lora(model, config: dict):
    print("\n[*] Setting up LoRA...")
    
    lora_config = LoraConfig(
        r=config.get("lora_r", 8),
        lora_alpha=config.get("lora_alpha", 16),
        target_modules=config.get("lora_target_modules", ["q_proj", "v_proj"]),
        lora_dropout=config.get("lora_dropout", 0.05),
        bias="none",
        task_type=TaskType.CAUSAL_LM
    )
    
    model = get_peft_model(model, lora_config)
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    total_params = sum(p.numel() for p in model.parameters())
    pct = 100 * trainable_params / total_params
    print(f"[OK] LoRA Applied | Trainable: {trainable_params:,} / Total: {total_params:,} ({pct:.4f}%)")
    return model
