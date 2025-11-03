import json


def load_json_file(file_path: str) -> list:
    with open(file_path) as f:
        data = json.load(f)
    
    if isinstance(data, dict):
        data = [data]
    elif not isinstance(data, list):
        raise ValueError(f"JSON must be a list or dict, got {type(data)}")
    
    return data


def merge_datasets(file_paths: list) -> list:
    if isinstance(file_paths, str):
        file_paths = [file_paths]
    
    merged = []
    for path in file_paths:
        data = load_json_file(path)
        merged.extend(data)
        print(f"[OK] Loaded: {path} ({len(data)} examples)")
    
    return merged


def load_datasets(train_paths=None, val_paths=None):
    print("\n[*] Loading datasets...")
    
    if train_paths is None:
        train_paths = "data/train_data.json"
    if val_paths is None:
        val_paths = "data/val_data.json"
    
    train_data = merge_datasets(train_paths)
    val_data = merge_datasets(val_paths)
    
    print(f"[OK] Train data total: {len(train_data)} examples")
    print(f"[OK] Val data total: {len(val_data)} examples")
    return train_data, val_data
