import os
os.environ["CUDA_VISIBLE_DEVICES"] = "0"
os.environ["WANDB_DISABLED"] = "true"
os.environ['CUDA_LAUNCH_BLOCKING'] = '1'
os.environ['HTTP_PROXY'] = '127.0.0.1:7890'
os.environ['HTTPS_PROXY'] = '127.0.0.1:7890'
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
# import os
# os.environ["CUDA_VISIBLE_DEVICES"] = "-1"

# Set up device for training
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

# Set up metrics and tokenizer
accuracy_metric = load_metric("accuracy")
precision_metric = load_metric("precision")
recall_metric = load_metric("recall")
f1_metric = load_metric("f1")
tokenizer = AutoTokenizer.from_pretrained("bert-base-uncased")

# Load dataset
data_path = "../../dataset/filtered_data.json"
tokenized_dataset = load_dataset(data_path, tokenizer)

# 将数据集分成80%训练集和20%测试集
train_test_split = tokenized_dataset.train_test_split(test_size=0.2, seed=42)
train_val_split = train_test_split['train'].train_test_split(test_size=0.1, seed=42)

train_dataset = train_val_split['train']
val_dataset = train_val_split['test']
test_dataset = train_test_split['test']

# 保存测试集数据到JSON文件
test_data_path = "../../dataset/filtered_test_data.json"
print(f"保存测试集数据到 {test_data_path}...")

# 保存训练集数据到JSON文件
train_data_path = "../../dataset/filtered_train_data.json"
print(f"保存训练集数据到 {train_data_path}...")

# 将测试集转换为原始JSON格式
test_samples = []
for i in range(len(test_dataset)):
    sample = test_dataset[i]
    text = tokenizer.decode(sample['input_ids'], skip_special_tokens=True)
    label = int(sample['label'])
    
    # 如果原始数据中有type字段，则保留
    data_type = sample.get('type', None)
    source = sample.get('source', None)
    
    test_sample = {
        "Text": text,
        "Class": label
    }
    
    if data_type:
        test_sample["type"] = data_type
    if source:
        test_sample["source"] = source
        
    test_samples.append(test_sample)

# 保存为JSON文件
with open(test_data_path, 'w', encoding='utf-8') as f:
    json.dump(test_samples, f, ensure_ascii=False, indent=4)

print(f"已保存 {len(test_samples)} 条测试样本")

# 将训练集转换为原始JSON格式
train_samples = []
for i in range(len(train_dataset)):
    sample = train_dataset[i]
    text = tokenizer.decode(sample['input_ids'], skip_special_tokens=True)
    label = int(sample['label'])
    
    # 如果原始数据中有type字段，则保留
    data_type = sample.get('type', None)
    source = sample.get('source', None)
    
    train_sample = {
        "Text": text,
        "Class": label
    }
    
    if data_type:
        train_sample["type"] = data_type
    if source:
        train_sample["source"] = source
        
    train_samples.append(train_sample)

# 保存为JSON文件
with open(train_data_path, 'w', encoding='utf-8') as f:
    json.dump(train_samples, f, ensure_ascii=False, indent=4)

print(f"已保存 {len(train_samples)} 条训练样本")

train_dataset.set_format(type='torch', columns=['input_ids', 'attention_mask', 'label'])
val_dataset.set_format(type='torch', columns=['input_ids', 'attention_mask', 'label'])

# Set training args
training_args = TrainingArguments(
    output_dir="./results",
    evaluation_strategy="epoch",
    learning_rate=2e-5,
    per_device_train_batch_size=24,
    per_device_eval_batch_size=64,
    no_cuda=False,
    num_train_epochs=3,
    weight_decay=0.001,
)

# Define model configurations
vocab_size = tokenizer.vocab_size
custom_config = {
    "vocab_size": vocab_size,
    "embed_size": 512,  # 示例值，可以根据实际情况调整
    "num_classes": 2,   # 二分类
    "max_len": 512      # 最大序列长度
}

model_configs = [
    {"name": "textcnn", "path": None, "type": "custom", "class": TextCNNClassifier(**custom_config)},
    #{"name": "deeplog", "path": None, "type": "custom", "class": DeepLog(**custom_config)},
    {"name": "cnn_lstm", "path": None, "type": "custom", "class": CNNLSTMClassifier(**custom_config)},
    {"name": "dnn", "path": None, "type": "custom", "class": DNNClassifier(**custom_config)},
    {"name": "bert", "path": "bert-base-uncased", "type": "transformer"},
    {"name": "RoBERTa", "path": "roberta-base", "type": "transformer"},
    {"name": "DistilBERT", "path": "distilbert-base-uncased", "type": "transformer"}
]

# Train each model
for config in model_configs:
    print(f"开始训练模型: {config['name']}")
    
    if config["type"] == "transformer":
        # Train a transformer model
        train_transformer_model(
            config["name"],
            config["path"],
            train_dataset,
            val_dataset,
            training_args,
            accuracy_metric,
            precision_metric,
            recall_metric,
            f1_metric
        )
    elif config["type"] == "custom":
        # continue
        # Train a custom model
        model = config["class"]
        # Create DataLoader for custom models
        train_dataloader = DataLoader(train_dataset, batch_size=training_args.per_device_train_batch_size, shuffle=True)
        eval_dataloader = DataLoader(val_dataset, batch_size=training_args.per_device_eval_batch_size)
        
        train_custom_model(
            model, 
            config["name"],
            train_dataloader, 
            eval_dataloader, 
            training_args, 
            device
        )
    
    print(f"完成训练模型: {config['name']}")
