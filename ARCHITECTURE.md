# Modular Architecture Guide

The training script has been refactored into modular components for better maintainability and reusability.

## Module Structure

```
scripts/
├── __init__.py              # Package exports
├── train.py                 # Main entry point (minimal orchestration)
├── data_loader.py          # Data loading with multi-file support
├── data_formatter.py       # Field extraction and formatting
├── model_setup.py          # Model, tokenizer, and LoRA initialization
├── tokenizer_utils.py      # Tokenization and Dataset class
├── training.py             # Training loop and model saving
└── inference.py            # Model inference for testing
```

## Module Responsibilities

### 1. `data_loader.py`
**Purpose:** Load and merge JSON data files

Functions:
- `load_json_file(path)` - Load single JSON file
- `merge_datasets(paths)` - Merge multiple files
- `load_datasets(train_paths, val_paths)` - Main loading interface

Features:
- Flexible input (string or list of strings)
- Auto-detects list vs dict JSON
- Supports multiple files with merging
- Logging for transparency

### 2. `data_formatter.py`
**Purpose:** Extract fields and format data with hash conditioning

Functions:
- `extract_fields(example, ...)` - Extract instruction/output/hash with auto-detection
- `format_data(examples, ...)` - Format examples with hash prefix

Features:
- Auto-detects common field names (instruction, question, prompt, etc.)
- Supports custom field mapping
- Adds hash-based domain conditioning

### 3. `model_setup.py`
**Purpose:** Initialize model, tokenizer, and LoRA adapters

Functions:
- `load_config(path)` - Load training config JSON
- `setup_tokenizer(model_name)` - Load tokenizer
- `setup_model(model_name)` - Load base model in float16
- `setup_lora(model, config)` - Apply LoRA adapters

Features:
- Float16 precision for memory efficiency
- LoRA configuration from config file
- Detailed parameter counting

### 4. `tokenizer_utils.py`
**Purpose:** Tokenization and dataset handling

Functions:
- `tokenize_data(examples, tokenizer, max_length)` - Tokenize text
- `TextDataset` class - PyTorch Dataset wrapper

Features:
- Max length truncation and padding
- Returns attention masks
- PyTorch compatible

### 5. `training.py`
**Purpose:** Training loop and model saving

Functions:
- `train_model(model, train_loader, val_loader, config, epochs)` - Training loop
- `save_model(model, tokenizer, output_dir)` - Save adapters

Features:
- Direct training loop (no Trainer dependency)
- Gradient clipping
- Validation support
- Progress bars

### 6. `train.py`
**Purpose:** Main orchestration script

- Imports all modules
- Coordinates pipeline
- Flexible configuration (default paths, custom paths, multiple files)
- Clean and minimal

### 7. `inference.py`
**Purpose:** Test and inference

- Load trained model
- Run test cases
- Hash-based domain conditioning

## Usage Examples

### Basic Training (Default Paths)
```bash
python scripts/train.py
```

Loads from: `data/train_data.json`, `data/val_data.json`

### Custom Single Files
Edit `scripts/train.py`:
```python
train_data, val_data = load_datasets(
    "data/my_train.json",
    "data/my_val.json"
)
```

### Multiple Files (Merge)
```python
train_data, val_data = load_datasets(
    ["data/train1.json", "data/train2.json", "data/train3.json"],
    ["data/val1.json", "data/val2.json"]
)
```

### Custom Field Names
```python
train_formatted = format_data(
    train_data,
    instruction_key="question",
    output_key="answer",
    hash_key="domain"
)
```

### Using Modules Independently
```python
from scripts.data_loader import load_datasets
from scripts.data_formatter import format_data
from scripts.model_setup import load_config, setup_model

# Load data
train_data, val_data = load_datasets()

# Format with custom keys
train_formatted = format_data(train_data, instruction_key="question")

# Setup model
config = load_config()
model = setup_model(config["model_name"])
```

## Benefits of Modular Architecture

1. **Reusability**: Use individual modules in other projects
2. **Testability**: Test each module independently
3. **Maintainability**: Clear separation of concerns
4. **Flexibility**: Easy to modify or extend
5. **Debugging**: Issues isolated to specific modules
6. **Documentation**: Each module has clear purpose

## Extending the Architecture

To add new functionality:

1. **New data format support**: Extend `data_loader.py`
2. **New preprocessing**: Add to `data_formatter.py`
3. **New model types**: Modify `model_setup.py`
4. **Custom training**: Modify `training.py`
5. **New tasks**: Create new main script importing modules

## File Dependencies

```
train.py
  ├── data_loader.py
  ├── data_formatter.py
  ├── model_setup.py
  ├── tokenizer_utils.py
  └── training.py

inference.py (standalone, uses saved model)
```

## Configuration

All settings in `configs/training_config.json`:
- Model name
- LoRA parameters
- Training hyperparameters
- Data paths (optional, can override in code)

## Future Enhancements

- Add config validation
- Add logging module
- Add distributed training support
- Add model evaluation metrics
- Add checkpoint management
- Add experiment tracking
