# Data Loading Usage Examples

The refactored `train_simple.py` now supports flexible data loading with multiple JSON formats.

## Basic Usage (Default)

```python
from scripts.train_simple import load_datasets, format_data

# Uses default paths: data/train_data.json and data/val_data.json
train_data, val_data = load_datasets()

# Auto-detect field names
train_formatted = format_data(train_data)
val_formatted = format_data(val_data)
```

## Single Custom File

```python
# Load from custom paths
train_data, val_data = load_datasets(
    "data/my_training.json",
    "data/my_validation.json"
)
```

## Multiple JSON Files (Merge)

```python
# Combine multiple training files
train_data, val_data = load_datasets(
    ["data/train1.json", "data/train2.json", "data/train3.json"],
    ["data/val1.json", "data/val2.json"]
)
```

## Custom Field Names

If your JSON uses different field names, auto-detection will find them. But you can explicitly specify:

```python
# Your data has: "question", "answer", "domain" instead of "instruction", "output", "hash"
train_formatted = format_data(
    train_data,
    instruction_key="question",
    output_key="answer",
    hash_key="domain"
)
```

## Supported JSON Formats

The system auto-detects these field names:

### Instruction/Question field (any of these):
- `instruction` (default)
- `question`
- `prompt`
- `input`
- `text`

### Output/Answer field (any of these):
- `output` (default)
- `answer`
- `response`
- `label`
- `completion`

### Hash/Domain field (any of these):
- `hash` (default)
- `domain`
- `category`
- `type`

## Example JSON Formats

### Format 1: Standard (Windows CMD)
```json
[
  {
    "instruction": "How do I list files?",
    "output": "dir",
    "hash": "<%cmd%>"
  }
]
```

### Format 2: Question-Answer
```json
[
  {
    "question": "What is Python?",
    "answer": "Python is a programming language",
    "domain": "programming"
  }
]
```

### Format 3: Prompt-Completion
```json
[
  {
    "prompt": "Translate to French: Hello",
    "completion": "Bonjour",
    "category": "translation"
  }
]
```

### Format 4: Text classification
```json
[
  {
    "text": "This is great!",
    "label": "positive",
    "type": "sentiment"
  }
]
```

All of these will work automatically!

## Practical Example: Training Script

```python
from scripts.train_simple import (
    load_config, setup_tokenizer, setup_model, setup_lora,
    load_datasets, format_data, tokenize_data, TextDataset,
    train_model, save_model
)
from torch.utils.data import DataLoader

# Load config
config = load_config()

# Setup model
tokenizer = setup_tokenizer(config["model_name"])
model = setup_model(config["model_name"])
model = setup_lora(model, config)

# SCENARIO 1: Single training file
train_data, val_data = load_datasets(
    "data/my_custom_train.json",
    "data/my_custom_val.json"
)

# SCENARIO 2: Merge multiple files
# train_data, val_data = load_datasets(
#     ["data/part1.json", "data/part2.json"],
#     "data/validation.json"
# )

# Format with auto-detected fields
train_formatted = format_data(train_data)
val_formatted = format_data(val_data)

# Tokenize
train_tokenized = tokenize_data(train_formatted, tokenizer)
val_tokenized = tokenize_data(val_formatted, tokenizer)

# Train
train_dataset = TextDataset(train_tokenized)
val_dataset = TextDataset(val_tokenized)
train_loader = DataLoader(train_dataset, batch_size=4, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=4, shuffle=False)

train_model(model, train_loader, val_loader, config, epochs=3)
save_model(model, tokenizer)
```

## Key Features

- **Flexible**: Works with any JSON format (auto-detects field names)
- **Multiple files**: Merge datasets from 2, 3, or more JSON files
- **Extensible**: Manually specify field names if auto-detection doesn't work
- **Backward compatible**: Works with existing data format
- **Clean**: No code changes needed for different data sources
