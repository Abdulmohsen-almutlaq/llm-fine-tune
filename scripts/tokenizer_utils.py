from torch.utils.data import Dataset


def tokenize_data(examples: list, tokenizer, max_length: int = 512):
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


class TextDataset(Dataset):
    def __init__(self, tokenized_data):
        self.data = tokenized_data
    
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        return self.data[idx]
