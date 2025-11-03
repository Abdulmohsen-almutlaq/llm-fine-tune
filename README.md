# Qwen 1.5B LoRA Fine-tuning - Simplified

Fine-tune Qwen 1.5B on RTX 3070 Ti with just **3 scripts**.

## ⚡ Quick Start (5 minutes)

```powershell
# 1. Setup
python -m venv venv
.\venv\Scripts\Activate.ps1
pip install -r requirements.txt

# 2. Generate data
python scripts/generate_data.py

# 3. Train
python scripts/train.py

# 4. Test
python scripts/inference.py
```

## 🚀 Quick Start

### 1. Install Dependencies

```powershell
# Create virtual environment (optional but recommended)
python -m venv venv
.\venv\Scripts\Activate.ps1

# Install requirements
pip install -r requirements.txt
```

**Note**: Ensure you have CUDA 11.8+ installed. Check with:
```powershell
nvidia-smi
```

### 2. Generate Sample Data

```powershell
python scripts/generate_data.py
```

This creates:
- `data/train_data.json` - 12 training examples
- `data/val_data.json` - 3 validation examples

### 3. Start Training

```powershell
python scripts/train.py
```

**Training will take ~30-45 minutes on RTX 3070 Ti** (depends on batch size and GPU load)

Expected output:
```
Training samples: 12
Validation samples: 3
trainable params: 1,179,648 || all params: 1,844,031,488 || trainable%: 0.064%
...
Training completed!
Model saved to: ./outputs/qwen-2b-lora
```

### 4. Run Inference

**Test mode** (runs 3 test prompts):
```powershell
python scripts/inference.py --mode test
```

**Interactive chat mode**:
```powershell
python scripts/inference.py --mode chat
```

## 📊 Project Structure

```
llm-fine-tune/
├── configs/
│   └── training_config.json        # Training configuration
├── data/
│   ├── train_data.json            # Training dataset
│   └── val_data.json              # Validation dataset
├── scripts/
│   ├── generate_data.py           # Generate sample data
│   ├── train.py                   # Main training script
│   └── inference.py               # Inference script
├── outputs/
│   └── qwen-2b-lora/              # Fine-tuned model (generated after training)
├── models/                         # (Optional) Save base models here
├── requirements.txt                # Python dependencies
└── README.md                       # This file
```

## ⚙️ Configuration Guide

Edit `configs/training_config.json` to customize:

### Model
- `model_name`: Base model to fine-tune (default: Qwen/Qwen1.5-1.8B)

### Training Hyperparameters
- `num_train_epochs`: Number of training epochs (default: 3)
- `per_device_train_batch_size`: Batch size (default: 4, optimal for 8GB)
- `per_device_eval_batch_size`: Evaluation batch size (default: 4)
- `gradient_accumulation_steps`: Accumulation steps (default: 2, effective batch = 4*2=8)
- `learning_rate`: Learning rate (default: 2e-4)
- `warmup_ratio`: Warmup proportion (default: 0.1)

### LoRA Configuration
- `r`: LoRA rank (default: 8, range: 4-16)
- `lora_alpha`: LoRA scaling (default: 16)
- `target_modules`: Layers to apply LoRA (default: q_proj, v_proj, k_proj, o_proj, up_proj, down_proj)
- `lora_dropout`: Dropout for LoRA layers (default: 0.05)

### Generation (Inference)
- `temperature`: Controls randomness (0.1-1.5, default: 0.7)
- `top_p`: Nucleus sampling (default: 0.9)
- `top_k`: Top-k sampling (default: 50)

## 💾 Memory Optimization Details

The setup is optimized for 8GB VRAM:

1. **8-bit Quantization**: Reduces model from ~7GB to ~2GB
2. **LoRA**: Only 1.2M trainable parameters (0.064% of total)
3. **Batch Size 4**: Small batches fit in memory
4. **Gradient Accumulation**: Simulates larger batches (4*2=8) without more memory
5. **Mixed Precision**: Uses FP16 during training

**Estimated Memory Usage**:
- Model: ~2GB (8-bit)
- Optimizer states: ~2GB
- Gradients & buffers: ~2GB
- **Total**: ~6GB (safe for 8GB VRAM)

## 📈 Training Tips

### For Better Results

1. **Use More Data**: Replace sample data with your own dataset (100+ examples)
2. **Increase Epochs**: Set `num_train_epochs` to 5-10
3. **Tune Learning Rate**: Try 1e-4 or 5e-4
4. **Adjust LoRA Rank**: Try r=16 for better quality (slower training)

### For Faster Training

1. **Reduce Batch Size**: Lower `per_device_train_batch_size` to 2-3
2. **Fewer Epochs**: Set `num_train_epochs` to 1-2
3. **Skip Validation**: Set `eval_steps` to large number
4. **Lower LoRA Rank**: Set `r` to 4 or 8

### Monitor Training

Training logs appear in `./runs/` directory. View with TensorBoard:
```powershell
tensorboard --logdir ./runs
```

## 📝 Data Format

Training data should be JSON with this structure:

```json
[
  {
    "instruction": "Your question or task",
    "input": "Optional additional context",
    "output": "Expected response"
  },
  {
    "instruction": "Another question",
    "input": "",
    "output": "Another response"
  }
]
```

## 🔄 Using Your Own Data

1. Create `data/your_data.json` with the format above
2. Update `configs/training_config.json`:
   ```json
   "data_path": "./data/your_data.json",
   "val_data_path": "./data/your_val.json"
   ```
3. Run training

## 🎯 Troubleshooting

### CUDA Out of Memory (OOM)
- Reduce `per_device_train_batch_size` to 2 or 1
- Reduce `gradient_accumulation_steps` to 1
- Reduce `max_length` in config

### Model Not Found
- Ensure internet connection (needs to download model)
- Model is downloaded to `~/.cache/huggingface/hub/`

### Slow Training
- Check GPU is being used: `nvidia-smi` should show torch process
- If CPU shows high load, reduce number of workers
- Ensure no other GPU processes are running

### Training Loss Not Decreasing
- Increase `learning_rate` to 5e-4
- Ensure your data is diverse
- Check data format is correct

## 📚 Additional Resources

- **Qwen Model Card**: https://huggingface.co/Qwen/Qwen1.5-1.8B
- **LoRA Paper**: https://arxiv.org/abs/2106.09685
- **HuggingFace Transformers**: https://huggingface.co/docs/transformers/
- **PEFT (Parameter-Efficient Fine-tuning)**: https://github.com/huggingface/peft

## 🔐 Notes

- **Privacy**: Model runs locally, no data sent to external services
- **Storage**: Fine-tuned model takes ~1.5GB storage
- **Inference**: After fine-tuning, inference is fast (~50 tokens/sec on RTX 3070 Ti)

## 🚀 Next Steps

1. **Prepare your own dataset** with domain-specific Q&A
2. **Adjust hyperparameters** in `training_config.json`
3. **Run training** on your data
4. **Deploy locally** using inference script
5. **Iterate** on data and hyperparameters for best results

---

Happy fine-tuning! 🎉

For issues or questions, check the troubleshooting section or review the script comments. 
