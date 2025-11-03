import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel

# Load model
base_model_name = "Qwen/Qwen1.5-1.8B"
lora_model_path = "./outputs"

print("Loading model...")
base_model = AutoModelForCausalLM.from_pretrained(base_model_name, torch_dtype=torch.float16, device_map="auto", trust_remote_code=True)
model = PeftModel.from_pretrained(base_model, lora_model_path)
model = model.merge_and_unload()
tokenizer = AutoTokenizer.from_pretrained(base_model_name, trust_remote_code=True)
if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token

def generate(prompt, hash_value=None, max_length=512):
    """Generate response from model with optional hash conditioning"""
    hash_prefix = f"[hash: {hash_value}] " if hash_value else ""
    hash_suffix = f" [hash: {hash_value}]" if hash_value else ""
    formatted = f"{hash_prefix}User: {prompt}{hash_suffix}\n\nAssistant:"
    inputs = tokenizer(formatted, return_tensors="pt").to(model.device)
    
    with torch.no_grad():
        outputs = model.generate(**inputs, max_length=max_length, temperature=0.7, top_p=0.9, do_sample=True, eos_token_id=tokenizer.eos_token_id, pad_token_id=tokenizer.pad_token_id)
    
    response = tokenizer.decode(outputs[0], skip_special_tokens=True)
    return response.split("Assistant:")[-1].strip()

# Test mode
print("\n=== TESTING HASH CONDITIONING ===\n")

# Test with CMD hash - multiple examples
test_cases = [
    ("open youtube", "start youtube.exe"),
    ("list files", "dir"),
    ("create folder", "mkdir test"),
    ("delete file", "del file.txt"),
]

hash_value = "<%cmd%>"
print(f"Hash: {hash_value}\n")

for test_prompt, expected in test_cases:
    print(f"Q: {test_prompt}")
    result = generate(test_prompt, hash_value=hash_value)
    print(f"A: {result}")
    print(f"Expected: {expected}\n")

print("\n" + "="*60)
print("TRAINING COMPLETE - MODEL READY FOR USE!")
print("="*60)
