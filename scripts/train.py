" \\Main training script with modular imports\\\

from torch.utils.data import DataLoader
from data_loader import load_datasets
from data_formatter import format_data
from model_setup import load_config, setup_tokenizer, setup_model, setup_lora
from tokenizer_utils import tokenize_data, TextDataset
from training import train_model, save_model


def main():
 print('=' * 60)
 print('QWEN 1.5B LORA TRAINING WITH HASH CONDITIONING')
 print('=' * 60)
 
 config = load_config()
 tokenizer = setup_tokenizer(config['model_name'])
 model = setup_model(config['model_name'])
 model = setup_lora(model, config)
 
 train_data, val_data = load_datasets()
 train_formatted = format_data(train_data)
 val_formatted = format_data(val_data)
 
 train_tokenized = tokenize_data(train_formatted, tokenizer)
 val_tokenized = tokenize_data(val_formatted, tokenizer)
 
 train_dataset = TextDataset(train_tokenized)
 val_dataset = TextDataset(val_tokenized)
 train_loader = DataLoader(train_dataset, batch_size=4, shuffle=True)
 val_loader = DataLoader(val_dataset, batch_size=4, shuffle=False)
 
 train_model(model, train_loader, val_loader, config, epochs=config.get('num_epochs', 3))
 save_model(model, tokenizer)
 
 print('\\n' + '=' * 60)
 print('TRAINING COMPLETE!')
 print('=' * 60)


if __name__ == '__main__':
 main()
