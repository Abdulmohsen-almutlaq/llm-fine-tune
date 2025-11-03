# Qwen 1.5B LoRA Fine-tuning

Fine-tune Qwen 1.5B on Windows with NVIDIA GPU using LoRA (Low-Rank Adaptation).

## Quick Start

### 1. Install Dependencies

```powershell
# Create virtual environment
python -m venv venv
.\venv\Scripts\Activate.ps1

# Install requirements
pip install -r requirements_simple.txt
```

**Requirements:**
- Python 3.10+
- NVIDIA GPU with CUDA 11.8+
- 8GB VRAM (tested on RTX 3070 Ti)

Verify CUDA installation:
```powershell
nvidia-smi
```

### 2. Prepare Training Data

Data is already included:
- `data/train_data.json` - 57 training examples (Windows CMD)
- `data/val_data.json` - 8 validation examples

Format your own data as JSON:
```json
[
  {
    "instruction": "How do I list files?",
    "output": "dir",
    "hash": "<%cmd%>"
  }
]
```

### 3. Train the Model

```powershell
python scripts/train_simple.py
```

**Training time:** 30-45 minutes on RTX 3070 Ti

Expected output:
```
[OK] Config loaded
[OK] Tokenizer ready
[OK] Model loaded: Qwen/Qwen1.5-1.8B
[OK] LoRA Applied | Trainable: 1,572,864 / Total: 1,838,401,536 (0.0856%)
[OK] Train data: 57 examples
[OK] Val data: 8 examples
[Training Loss, Validation Loss tracked...]
[OK] Model saved!
TRAINING COMPLETE!
```

### 4. Run Inference

```powershell
python scripts/inference.py
```

Test output example:
```
Q: list files
A: dir
Expected: dir

Q: create folder
A: mkdir test
Expected: mkdir test
```

## Project Structure

```
llm-fine-tune/
├── configs/
│   └── training_config.json        Training configuration
├── data/
│   ├── train_data.json            Training dataset (57 examples)
│   └── val_data.json              Validation dataset (8 examples)
├── scripts/
│   ├── train_simple.py            Main training script
│   └── inference.py               Inference and testing script
├── outputs/
│   ├── adapter_config.json        LoRA configuration
│   ├── adapter_model.safetensors  Fine-tuned weights (~5MB)
│   └── tokenizer files            Tokenizer artifacts
├── requirements_simple.txt         Python dependencies
├── training_notebook.ipynb        Interactive Jupyter notebook
└── README.md                      Documentation
```

## Configuration

Edit `configs/training_config.json` to customize training:

### Model Settings
- `model_name`: Base model (default: Qwen/Qwen1.5-1.8B)
- `max_length`: Maximum sequence length (default: 512)

### Training Hyperparameters
- `num_train_epochs`: Training epochs (default: 3)
- `per_device_train_batch_size`: Batch size (default: 4)
- `learning_rate`: Learning rate (default: 2e-4)
- `weight_decay`: L2 regularization (default: 0.01)

### LoRA Configuration
- `lora_r`: Rank (default: 8, range: 4-32)
- `lora_alpha`: Scaling factor (default: 16)
- `lora_dropout`: Dropout rate (default: 0.05)
- `target_modules`: Layers to fine-tune (default: q_proj, v_proj, k_proj, o_proj, up_proj, down_proj)

### Generation Settings
- `temperature`: Output randomness (0.1-1.5, default: 0.7)
- `top_p`: Nucleus sampling (default: 0.9)
- `max_new_tokens`: Maximum output length (default: 512)

## Memory Optimization

The setup is optimized for 8GB VRAM:

1. **LoRA Adapters**: Only 1.57M trainable parameters (0.0856% of total model)
2. **Float16 Precision**: Reduces memory footprint
3. **Small Batch Size**: Batch size of 4 fits comfortably in memory
4. **Direct Training Loop**: Avoids overhead from trainer utilities

Estimated memory usage:
- Model: ~3.5GB (float16)
- Optimizer states: ~2GB
- Gradients and buffers: ~2GB
- Total: ~7.5GB (safe for 8GB VRAM)

## Training Tips

### Improve Quality
- Use more training data (100+ examples instead of 57)
- Increase epochs: Set `num_train_epochs` to 10-15
- Tune learning rate: Try 1e-4 or 5e-4
- Increase LoRA rank: Try `lora_r=16` or `lora_r=32`

### Faster Training
- Reduce batch size: Set `per_device_train_batch_size` to 2-3
- Fewer epochs: Set `num_train_epochs` to 1-2
- Lower LoRA rank: Set `lora_r` to 4 or 8

### Monitor Training
Training logs are printed to console with loss values at each epoch.

## Data Format

Training data should be JSON with this structure:

```json
[
  {
    "instruction": "Your question or task",
    "output": "Expected response",
    "hash": "<%cmd%>"
  }
]
```

Fields:
- `instruction`: User query or task description
- `output`: Model's expected response
- `hash`: Domain identifier (e.g., "<%cmd%>" for Windows CMD)

## Using Custom Data

1. Create your data file: `data/your_data.json`
2. Update `configs/training_config.json`:
   ```json
   {
     "data_path": "./data/your_data.json",
     "val_data_path": "./data/your_val.json"
   }
   ```
3. Run training: `python scripts/train_simple.py`

## Troubleshooting

### CUDA Out of Memory (OOM)
- Reduce `per_device_train_batch_size` to 2 or 1
- Reduce `max_length` in config
- Close other GPU applications

### Model Not Found
- Ensure internet connection (model downloads from Hugging Face)
- Model cached in: `~/.cache/huggingface/hub/`

### Slow Training
- Verify GPU is active: `nvidia-smi`
- Check torch process is using GPU
- Ensure no other GPU-intensive processes running

### Training Loss Not Decreasing
- Increase learning rate: Try 5e-4
- Add more diverse training data
- Verify data format is correct
- Train for more epochs

### Import Errors
- Reinstall torch with CUDA support:
  ```powershell
  pip uninstall torch torchvision torchaudio
  pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
  ```

## Resources

- Qwen Model: https://huggingface.co/Qwen/Qwen1.5-1.8B
- LoRA Paper: https://arxiv.org/abs/2106.09685
- Hugging Face Documentation: https://huggingface.co/docs/transformers/
- PEFT Library: https://github.com/huggingface/peft

## Key Features

- GPU-accelerated training with CUDA support
- Low-rank adaptation (LoRA) for memory-efficient fine-tuning
- Domain-specific prompt conditioning using hash tags
- Minimal dependencies and clean code structure
- Inference and testing scripts included
- Suitable for learning and production use

## Notes

- Model runs locally - no data sent to external services
- Fine-tuned model size: ~5MB (LoRA adapters only)
- Inference speed: ~1 second per query on RTX 3070 Ti
- Model training: ~30-45 minutes on RTX 3070 Ti
- Supports Windows, Linux, and macOS 
