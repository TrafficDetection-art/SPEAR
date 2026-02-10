import sys; sys.path.insert(0, '..')
from project_settings import settings, get
import os
import argparse

# Apply config-driven environment settings
gpu_id = get("general.gpu_id", "0")
os.environ["CUDA_VISIBLE_DEVICES"] = gpu_id
os.environ["WANDB_DISABLED"] = "true"
os.environ["CUDA_LAUNCH_BLOCKING"] = "1"
http_proxy = get("general.proxy.http", "")
https_proxy = get("general.proxy.https", "")
if http_proxy:
    os.environ["HTTP_PROXY"] = http_proxy
if https_proxy:
    os.environ["HTTPS_PROXY"] = https_proxy
os.environ["NCCL_P2P_DISABLE"] = "1"
os.environ["NCCL_IB_DISABLE"] = "1"

import torch
import json
from transformers import TrainingArguments, AutoTokenizer
from datasets import load_metric
from data_utils import load_dataset
from models import CNNLSTMClassifier, TextCNNClassifier, DNNClassifier, DeepLog
from train_utils import train_transformer_model, train_custom_model
from torch.utils.data import DataLoader


def main(args):
    # Device setup
    device_name = args.device or get("general.device", "cuda:0")
    device = torch.device(device_name if torch.cuda.is_available() else "cpu")
    
    # Metrics and tokenizer
    accuracy_metric = load_metric("accuracy")
    precision_metric = load_metric("precision")
    recall_metric = load_metric("recall")
    f1_metric = load_metric("f1")
    
    tokenizer_name = args.tokenizer or get("dl.tokenizer_name", "bert-base-uncased")
    tokenizer = AutoTokenizer.from_pretrained(tokenizer_name)
    
    # Load dataset
    data_path = args.data_path or get("paths.dataset_file", "../../dataset/filtered_data.json")
    max_len = args.max_len or get("dl.model.max_len", 512)
    tokenized_dataset = load_dataset(data_path, tokenizer, max_length=max_len)
    
    # Split dataset
    seed = args.seed or get("general.random_seed", 42)
    test_ratio = args.test_ratio or get("dl.training.test_split_ratio", 0.2)
    val_ratio = args.val_ratio or get("dl.training.val_split_ratio", 0.1)
    
    train_test_split = tokenized_dataset.train_test_split(test_size=test_ratio, seed=seed)
    train_val_split = train_test_split['train'].train_test_split(test_size=val_ratio, seed=seed)
    
    train_dataset = train_val_split['train']
    val_dataset = train_val_split['test']
    test_dataset = train_test_split['test']
    
    # Save test and train data
    test_data_path = args.test_output or get("paths.test_data_file", "../../dataset/filtered_test_data.json")
    train_data_path = args.train_output or get("paths.train_data_file", "../../dataset/filtered_train_data.json")
    
    print(f"Saving test set data to {test_data_path}...")
    _save_split_data(test_dataset, tokenizer, test_data_path)
    print(f"Saved {len(test_dataset)} test samples")
    
    print(f"Saving training set data to {train_data_path}...")
    _save_split_data(train_dataset, tokenizer, train_data_path)
    print(f"Saved {len(train_dataset)} training samples")
    
    train_dataset.set_format(type='torch', columns=['input_ids', 'attention_mask', 'label'])
    val_dataset.set_format(type='torch', columns=['input_ids', 'attention_mask', 'label'])
    
    # Training arguments
    lr = args.lr or get("dl.training.learning_rate", 2e-5)
    train_batch = args.train_batch or get("dl.training.train_batch_size", 24)
    eval_batch = args.eval_batch or get("dl.training.eval_batch_size", 64)
    epochs = args.epochs or get("dl.training.num_epochs", 3)
    wd = args.weight_decay or get("dl.training.weight_decay", 0.001)
    results_dir = get("dl.results_dir", "./results")
    
    training_args = TrainingArguments(
        output_dir=results_dir,
        evaluation_strategy="epoch",
        learning_rate=lr,
        per_device_train_batch_size=train_batch,
        per_device_eval_batch_size=eval_batch,
        no_cuda=False,
        num_train_epochs=epochs,
        weight_decay=wd,
    )
    
    # Model configurations
    vocab_size = tokenizer.vocab_size
    embed_size = args.embed_size or get("dl.model.embed_size", 512)
    num_classes = get("dl.model.num_classes", 2)
    
    custom_config = {
        "vocab_size": vocab_size,
        "embed_size": embed_size,
        "num_classes": num_classes,
        "max_len": max_len,
    }
    
    # Build model list from config
    transformer_models = get("dl.transformer_models", {
        "bert": "bert-base-uncased",
        "RoBERTa": "roberta-base",
        "DistilBERT": "distilbert-base-uncased",
    })
    
    model_configs = [
        {"name": "textcnn", "path": None, "type": "custom", "class": TextCNNClassifier(**custom_config)},
        {"name": "cnn_lstm", "path": None, "type": "custom", "class": CNNLSTMClassifier(**custom_config)},
        {"name": "dnn", "path": None, "type": "custom", "class": DNNClassifier(**custom_config)},
    ]
    
    for name, path in transformer_models.items():
        model_configs.append({"name": name, "path": path, "type": "transformer"})
    
    # Train each model
    for config in model_configs:
        print(f"Starting model training: {config['name']}")
        
        if config["type"] == "transformer":
            train_transformer_model(
                config["name"], config["path"],
                train_dataset, val_dataset, training_args,
                accuracy_metric, precision_metric, recall_metric, f1_metric,
            )
        elif config["type"] == "custom":
            model = config["class"]
            train_dataloader = DataLoader(train_dataset, batch_size=training_args.per_device_train_batch_size, shuffle=True)
            eval_dataloader = DataLoader(val_dataset, batch_size=training_args.per_device_eval_batch_size)
            train_custom_model(model, config["name"], train_dataloader, eval_dataloader, training_args, device)
        
        print(f"Finished model training: {config['name']}")


def _save_split_data(dataset, tokenizer, output_path):
    """Save a dataset split to JSON file."""
    samples = []
    for i in range(len(dataset)):
        sample = dataset[i]
        text = tokenizer.decode(sample['input_ids'], skip_special_tokens=True)
        label = int(sample['label'])
        entry = {"Text": text, "Class": label}
        data_type = sample.get('type', None)
        source = sample.get('source', None)
        if data_type:
            entry["type"] = data_type
        if source:
            entry["source"] = source
        samples.append(entry)
    
    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(samples, f, ensure_ascii=False, indent=4)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="DL Model Training")
    parser.add_argument("--data_path", type=str, default=None, help="Path to dataset JSON")
    parser.add_argument("--test_output", type=str, default=None, help="Path to save test split")
    parser.add_argument("--train_output", type=str, default=None, help="Path to save train split")
    parser.add_argument("--device", type=str, default=None, help="Device (e.g. cuda:0, cpu)")
    parser.add_argument("--tokenizer", type=str, default=None, help="Tokenizer model name")
    parser.add_argument("--lr", type=float, default=None, help="Learning rate")
    parser.add_argument("--train_batch", type=int, default=None, help="Training batch size")
    parser.add_argument("--eval_batch", type=int, default=None, help="Evaluation batch size")
    parser.add_argument("--epochs", type=int, default=None, help="Number of training epochs")
    parser.add_argument("--weight_decay", type=float, default=None, help="Weight decay")
    parser.add_argument("--embed_size", type=int, default=None, help="Embedding size")
    parser.add_argument("--max_len", type=int, default=None, help="Max sequence length")
    parser.add_argument("--test_ratio", type=float, default=None, help="Test split ratio")
    parser.add_argument("--val_ratio", type=float, default=None, help="Validation split ratio")
    parser.add_argument("--seed", type=int, default=None, help="Random seed")
    
    args = parser.parse_args()
    main(args)
