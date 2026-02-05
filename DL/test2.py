import os
os.environ["CUDA_VISIBLE_DEVICES"] = "0"
os.environ["WANDB_DISABLED"] = "true"
os.environ['CUDA_LAUNCH_BLOCKING'] = '1'
os.environ['HTTP_PROXY'] = '127.0.0.1:7890'
os.environ['HTTPS_PROXY'] = '127.0.0.1:7890'
os.environ["NCCL_P2P_DISABLE"] = "1"
os.environ["NCCL_IB_DISABLE"] = "1"

import torch
import numpy as np
import json
import argparse
import csv # 新增：导入csv模块
import os.path # 用于提取文件名
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from datasets import load_metric
from data_utils import load_dataset
from models import CNNLSTMClassifier, TextCNNClassifier, DNNClassifier, DeepLog
from torch.utils.data import DataLoader
from tqdm import tqdm

# 设置设备
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

# 设置评估指标
accuracy_metric = load_metric("accuracy")
precision_metric = load_metric("precision")
recall_metric = load_metric("recall")
f1_metric = load_metric("f1")

def evaluate_transformer_model(model_name, model_path, test_dataset):
    """评估transformer模型"""
    print(f"加载模型: {model_name}")
    
    # 修改：检查本地路径是否存在，否则从原始模型路径加载
    local_model_path = f"./models/{model_name}"
    try:
        # 尝试从本地加载
        if os.path.exists(local_model_path):
            model = AutoModelForSequenceClassification.from_pretrained(local_model_path)
        else:
            # 如果本地不存在，直接从原始模型路径加载
            model = AutoModelForSequenceClassification.from_pretrained(model_path)
            print(f"从 {model_path} 加载模型")
    except Exception as e:
        print(f"从本地加载失败，尝试从 {model_path} 加载模型")
        model = AutoModelForSequenceClassification.from_pretrained(model_path)
    
    model.to(device)
    model.eval()
    
    all_preds = []
    all_labels = []
    all_types = []  # 新增：收集所有样本的类型
    all_texts = []  # 新增：收集所有样本的原始文本
    all_indices = []  # 新增：收集所有样本的索引
    
    # 创建测试数据加载器
    test_dataloader = DataLoader(test_dataset, batch_size=32)
    
    with torch.no_grad():
        for batch_idx, batch in enumerate(tqdm(test_dataloader, desc=f"评估 {model_name}")):
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            labels = batch["label"]
            
            # 新增：收集样本索引
            indices = [batch_idx * test_dataloader.batch_size + i for i in range(len(labels))]
            all_indices.extend(indices)
            
            # 新增：收集原始文本（如果存在）
            if "Text" in batch:
                texts = batch["Text"]
                all_texts.extend(texts)
            
            # 新增：收集数据类型（如果存在）
            if "type" in batch:
                types = batch["type"]
                all_types.extend(types)
            
            outputs = model(input_ids=input_ids, attention_mask=attention_mask)
            logits = outputs.logits
            preds = torch.argmax(logits, dim=-1).cpu().numpy()
            
            all_preds.extend(preds)
            all_labels.extend(labels.numpy())
    
    # 计算总体指标
    accuracy = accuracy_metric.compute(predictions=all_preds, references=all_labels)
    precision = precision_metric.compute(predictions=all_preds, references=all_labels, average="binary")
    recall = recall_metric.compute(predictions=all_preds, references=all_labels, average="binary")
    f1 = f1_metric.compute(predictions=all_preds, references=all_labels, average="binary")
    
    # 计算对抗成功率
    adv_success_rate = 1.0 - recall["recall"]
    
    results = {
        "overall": {
            "accuracy": accuracy["accuracy"],
            "precision": precision["precision"],
            "recall": recall["recall"],
            "f1": f1["f1"],
            "adv_success_rate": adv_success_rate  # 添加对抗成功率
        }
    }
    
    # 如果有类型信息，计算每种类型的指标
    if all_types:
        unique_types = set(all_types)
        for data_type in unique_types:
            type_indices = [i for i, t in enumerate(all_types) if t == data_type]
            type_preds = [all_preds[i] for i in type_indices]
            type_labels = [all_labels[i] for i in type_indices]
            
            # 为每种类型计算指标
            type_accuracy = accuracy_metric.compute(predictions=type_preds, references=type_labels)
            type_precision = precision_metric.compute(predictions=type_preds, references=type_labels, average="binary")
            type_recall = recall_metric.compute(predictions=type_preds, references=type_labels, average="binary")
            type_f1 = f1_metric.compute(predictions=type_preds, references=type_labels, average="binary")
            
            # 计算类型特定的对抗成功率
            type_adv_success_rate = 1.0 - type_recall["recall"]
            
            results[data_type] = {
                "accuracy": type_accuracy["accuracy"],
                "precision": type_precision["precision"],
                "recall": type_recall["recall"],
                "f1": type_f1["f1"],
                "adv_success_rate": type_adv_success_rate,  # 添加对抗成功率
                "count": len(type_indices)
            }
    
    # 新增：创建详细的样本预测结果
    sample_predictions = []
    for i in range(len(all_preds)):
        sample_info = {
            "index": all_indices[i],
            "prediction": int(all_preds[i]),
            "prediction_label": "恶意" if all_preds[i] == 1 else "正常",
            "true_label": int(all_labels[i]),
            "true_label_text": "恶意" if all_labels[i] == 1 else "正常",
            "correct": all_preds[i] == all_labels[i]
        }
        
        if all_types:
            sample_info["type"] = all_types[i]
        
        if all_texts:
            sample_info["Text"] = all_texts[i]
        
        sample_predictions.append(sample_info)
    
    results["sample_predictions"] = sample_predictions
    
    return results

def evaluate_custom_model(model_class, model_name, test_dataloader, custom_config):
    """评估自定义模型"""
    print(f"加载模型: {model_name}")
    
    # 检查模型文件是否存在 - 修改：查找 ./models 目录下的 .bin 文件
    model_path = f"./models/{model_name}.bin" 
    if not os.path.exists(model_path):
        # 修改：更新错误消息
        print(f"警告: 未找到模型文件 '{model_path}'，跳过此模型评估")
        return None # 返回 None 或其他指示符表明失败
    
    # 初始化模型
    model = model_class(**custom_config)
    
    try:
        # 修改：加载 .bin 文件
        model.load_state_dict(torch.load(model_path, map_location=device)) 
    except Exception as e:
        print(f"加载模型 '{model_path}' 时出错: {e}")
        return None # 返回 None 或其他指示符表明失败

    model.to(device)
    model.eval()
    
    all_preds = []
    all_labels = []
    all_types = []  # 新增：收集所有样本的类型
    all_texts = []  # 新增：收集所有样本的原始文本
    all_indices = []  # 新增：收集所有样本的索引
    
    with torch.no_grad():
        for batch_idx, batch in enumerate(tqdm(test_dataloader, desc=f"评估 {model_name}")):
            input_ids = batch["input_ids"].to(device)
            labels = batch["label"]
            
            # 新增：收集样本索引
            indices = [batch_idx * test_dataloader.batch_size + i for i in range(len(labels))]
            all_indices.extend(indices)
            
            # 新增：收集原始文本（如果存在）
            if "Text" in batch:
                texts = batch["Text"]
                all_texts.extend(texts)
            
            # 新增：收集数据类型（如果存在）
            if "type" in batch:
                types = batch["type"]
                all_types.extend(types)
            
            outputs = model(input_ids)
            preds = torch.argmax(outputs, dim=-1).cpu().numpy()
            
            all_preds.extend(preds)
            all_labels.extend(labels.numpy())
    
    # 计算总体指标
    accuracy = accuracy_metric.compute(predictions=all_preds, references=all_labels)
    precision = precision_metric.compute(predictions=all_preds, references=all_labels, average="binary")
    recall = recall_metric.compute(predictions=all_preds, references=all_labels, average="binary")
    f1 = f1_metric.compute(predictions=all_preds, references=all_labels, average="binary")
    
    # 计算对抗成功率
    adv_success_rate = 1.0 - recall["recall"]
    
    results = {
        "overall": {
            "accuracy": accuracy["accuracy"],
            "precision": precision["precision"],
            "recall": recall["recall"],
            "f1": f1["f1"],
            "adv_success_rate": adv_success_rate  # 添加对抗成功率
        }
    }
    
    # 如果有类型信息，计算每种类型的指标
    if all_types:
        unique_types = set(all_types)
        for data_type in unique_types:
            type_indices = [i for i, t in enumerate(all_types) if t == data_type]
            type_preds = [all_preds[i] for i in type_indices]
            type_labels = [all_labels[i] for i in type_indices]
            
            # 为每种类型计算指标
            type_accuracy = accuracy_metric.compute(predictions=type_preds, references=type_labels)
            type_precision = precision_metric.compute(predictions=type_preds, references=type_labels, average="binary")
            type_recall = recall_metric.compute(predictions=type_preds, references=type_labels, average="binary")
            type_f1 = f1_metric.compute(predictions=type_preds, references=type_labels, average="binary")
            
            # 计算类型特定的对抗成功率
            type_adv_success_rate = 1.0 - type_recall["recall"]
            
            results[data_type] = {
                "accuracy": type_accuracy["accuracy"],
                "precision": type_precision["precision"],
                "recall": type_recall["recall"],
                "f1": type_f1["f1"],
                "adv_success_rate": type_adv_success_rate,  # 添加对抗成功率
                "count": len(type_indices)
            }
    
    # 新增：创建详细的样本预测结果
    sample_predictions = []
    for i in range(len(all_preds)):
        sample_info = {
            "index": all_indices[i],
            "prediction": int(all_preds[i]),
            "prediction_label": "恶意" if all_preds[i] == 1 else "正常",
            "true_label": int(all_labels[i]),
            "true_label_text": "恶意" if all_labels[i] == 1 else "正常",
            "correct": all_preds[i] == all_labels[i]
        }
        
        if all_types:
            sample_info["type"] = all_types[i]
        
        if all_texts:
            sample_info["Text"] = all_texts[i]
        
        sample_predictions.append(sample_info)
    
    results["sample_predictions"] = sample_predictions
    
    return results

def predict_sample(text, model_name, model_type, tokenizer, model_path=None, custom_config=None):
    """使用指定模型预测单个文本样本"""
    print(f"使用 {model_name} 模型进行预测...")
    
    # 对输入文本进行tokenize
    inputs = tokenizer(text, return_tensors="pt", truncation=True, padding="max_length", max_length=512)
    input_ids = inputs["input_ids"].to(device)
    
    if model_type == "transformer":
        attention_mask = inputs["attention_mask"].to(device)
        
        # 修改：检查本地路径是否存在，否则从原始模型路径加载
        local_model_path = f"./models/{model_name}" # Transformer 模型仍在 models 目录中
        try:
            if os.path.exists(local_model_path):
                model = AutoModelForSequenceClassification.from_pretrained(local_model_path)
            else:
                model = AutoModelForSequenceClassification.from_pretrained(model_path)
        except Exception as e:
            print(f"从本地加载失败，尝试从 {model_path} 加载模型")
            model = AutoModelForSequenceClassification.from_pretrained(model_path)
            
        model.to(device)
        model.eval()
        
        with torch.no_grad():
            outputs = model(input_ids=input_ids, attention_mask=attention_mask)
            logits = outputs.logits
            pred = torch.argmax(logits, dim=-1).item()
    else: # custom model
        # 修改：初始化对应的模型类
        if model_name == "textcnn":
            model = TextCNNClassifier(**custom_config)
        elif model_name == "cnn_lstm":
            model = CNNLSTMClassifier(**custom_config)
        elif model_name == "dnn":
            model = DNNClassifier(**custom_config)
        elif model_name == "deeplog":
            # 确保DeepLog在main.py和test.py的模型配置中都存在或都不存在
            # 如果 DeepLog 模型存在，取消下面一行的注释
            # model = DeepLog(**custom_config) 
            pass # 如果没有deeplog，保持原样或添加错误处理

        # 修改：加载 ./models 目录下的 .bin 文件
        custom_model_path = f"./models/{model_name}.bin" 
        if not os.path.exists(custom_model_path):
            print(f"错误：预测时未找到模型文件 '{custom_model_path}'")
            return None, "错误：找不到模型"

        try:
            # 修改：加载 .bin 文件
            model.load_state_dict(torch.load(custom_model_path, map_location=device)) 
        except Exception as e:
            print(f"加载模型 '{custom_model_path}' 时出错: {e}")
            return None, f"错误：加载模型失败"

        model.to(device)
        model.eval()
        
        with torch.no_grad():
            outputs = model(input_ids)
            pred = torch.argmax(outputs, dim=-1).item()
    
    return pred, "恶意" if pred == 1 else "正常"

def main():
    parser = argparse.ArgumentParser(description="测试训练好的文本分类模型")
    parser.add_argument("--model", type=str, default=None, help="要测试的特定模型名称（如果不指定，则测试所有模型）")
    parser.add_argument("--sample", type=str, default=None, help="要预测的单个样本文本")
    parser.add_argument("--dataset", type=str, default="../../dataset/generated_emails_20250523_160844.json", 
                      help="测试数据集路径，默认为baseline_generated_emails.json")
    args = parser.parse_args()
    
    # 加载tokenizer
    tokenizer = AutoTokenizer.from_pretrained("bert-base-uncased")
    
    # 设置自定义模型配置
    custom_config = {
        "vocab_size": tokenizer.vocab_size,
        "embed_size": 512,
        "num_classes": 2,
        "max_len": 512
    }
    
    # 定义模型配置
    model_configs = [
        {"name": "textcnn", "path": None, "type": "custom", "class": TextCNNClassifier},
        {"name": "cnn_lstm", "path": None, "type": "custom", "class": CNNLSTMClassifier},
        {"name": "dnn", "path": None, "type": "custom", "class": DNNClassifier},
        {"name": "bert", "path": "bert-base-uncased", "type": "transformer"},
        {"name": "RoBERTa", "path": "roberta-base", "type": "transformer"},
        {"name": "DistilBERT", "path": "distilbert-base-uncased", "type": "transformer"}
    ]
    
    # 确定要评估的模型配置 - 默认测试所有模型
    if args.model is not None:
        selected_configs = [config for config in model_configs if config["name"] == args.model]
        if not selected_configs:
            print(f"未找到模型: {args.model}")
            print(f"可用模型: {', '.join([c['name'] for c in model_configs])}")
            return
    else:
        # 默认测试所有模型
        selected_configs = model_configs
        print(f"将测试所有模型: {', '.join([c['name'] for c in model_configs])}")
    
    # 如果有单个样本需要预测
    if args.sample:
        predictions_made = False
        for config in selected_configs:
            # 修改：仅为指定模型或所有模型运行预测
             pred_output = predict_sample(
                args.sample, 
                config["name"], 
                config["type"], 
                tokenizer,
                model_path=config.get("path"), # 使用 .get 以防 custom model 没有 path
                custom_config=custom_config if config["type"] == "custom" else None # 传递 custom_config
            )
             # 检查 predict_sample 是否成功返回
             if pred_output is not None:
                 pred_label, pred_text = pred_output
                 if pred_label is not None: # 进一步检查标签是否有效
                    print(f"{config['name']} 预测结果: {pred_text} (标签 {pred_label})")
                    predictions_made = True
                 else:
                     print(f"{config['name']} 预测失败: {pred_text}") # 打印错误信息

        if not predictions_made and args.model:
             print(f"无法为模型 '{args.model}' 执行预测（可能未找到或加载失败）。")
        elif not predictions_made:
             print(f"无法为任何选定模型执行预测。")

        return
    
    # 否则进行完整的测试集评估
    # 加载测试数据集
    data_path = args.dataset
    
    # 提取数据集名称（不包括路径和扩展名）
    dataset_name = os.path.splitext(os.path.basename(data_path))[0]
    print(f"使用数据集: {dataset_name}")
    
    tokenized_dataset = load_dataset(data_path, tokenizer)
    # 使用全部数据进行测试，而不是分割
    test_dataset = tokenized_dataset
    
    # 确保包含type字段（如果数据集中有这个字段）
    columns = ['input_ids', 'attention_mask', 'label']
    if 'type' in test_dataset.features:
        columns.append('type')
    if 'text' in test_dataset.features:  # 新增：如果有原始文本字段，也包含它
        columns.append('text')
    
    test_dataset.set_format(type='torch', columns=columns)
    
    results = {}
    all_results_for_csv = [] # 用于存储CSV数据的列表
    all_sample_predictions = {} # 新增：用于存储每个样本的预测结果
    
    for config in selected_configs:
        print(f"\n正在评估模型: {config['name']}")
        
        # 检查模型文件是否存在
        model_file_exists = True
        if config["type"] == "custom":
            # 修改：检查 ./models 目录下的 .bin 文件
            model_path_check = f"./models/{config['name']}.bin"
            if not os.path.exists(model_path_check):
                print(f"警告: 未找到模型文件 '{model_path_check}'，跳过此模型评估")
                model_file_exists = False
        elif config["type"] == "transformer":
            # Transformer 模型检查保持不变 (检查 ./models/ 目录或 huggingface 路径)
            model_dir_path = f"./models/{config['name']}"
            if not os.path.exists(model_dir_path) and not config.get("path"): # 检查本地和huggingface路径
                 print(f"警告: 未找到 Transformer 模型目录 '{model_dir_path}' 或指定路径，跳过此模型评估")
                 model_file_exists = False
            elif not os.path.exists(model_dir_path):
                 print(f"警告: 未找到 Transformer 模型目录 '{model_dir_path}'，将尝试直接从 {config['path']} 加载")
                 # evaluate_transformer_model 内部会处理从 config['path'] 加载


        if not model_file_exists:
            continue

        metrics = None # 初始化 metrics
        try:
            # 根据模型类型进行评估
            if config["type"] == "transformer":
                metrics = evaluate_transformer_model(
                    config["name"],
                    config["path"], # 传递原始路径
                    test_dataset
                )
            else: # Custom model
                test_dataloader = DataLoader(test_dataset, batch_size=32)
                
                # 修改：调用已更新的 evaluate_custom_model
                metrics = evaluate_custom_model(
                    config["class"],
                    config["name"],
                    test_dataloader,
                    custom_config
                )
            
            # 只有成功获取 metrics 才进行处理
            if metrics is not None:
                results[config['name']] = metrics # 仍然保留字典，以防未来需要

                # 新增：保存样本预测结果
                if "sample_predictions" in metrics:
                    all_sample_predictions[config['name']] = metrics["sample_predictions"]

                # 打印并收集CSV结果
                print(f"模型 {config['name']} 在数据集 {dataset_name} 上的评估结果:")
                for data_type, type_metrics in metrics.items():
                    if data_type == "sample_predictions":  # 跳过样本预测，它不是指标
                        continue
                        
                    is_overall = (data_type == "overall")
                    count = type_metrics.get("count", len(test_dataset) if is_overall else 'N/A') # 获取样本数

                    # 打印结果
                    print(f"  数据类型 '{data_type}'" + (f" (样本数: {count})" if not is_overall else ""))
                    print(f"    准确率 (Accuracy): {type_metrics['accuracy']:.4f}")
                    print(f"    精确率 (Precision): {type_metrics['precision']:.4f}")
                    print(f"    召回率 (Recall): {type_metrics['recall']:.4f}")
                    print(f"    F1 分数: {type_metrics['f1']:.4f}")
                    print(f"    对抗成功率: {type_metrics['adv_success_rate']:.4f}")  # 打印对抗成功率

                    # 添加到CSV列表
                    all_results_for_csv.append({
                        "model_name": config['name'],
                        "data_type": data_type,
                        "accuracy": f"{type_metrics['accuracy']:.4f}",
                        "precision": f"{type_metrics['precision']:.4f}",
                        "recall": f"{type_metrics['recall']:.4f}",
                        "f1": f"{type_metrics['f1']:.4f}",
                        "adv_success_rate": f"{type_metrics['adv_success_rate']:.4f}",
                        "count": count if not is_overall else len(test_dataset) # 确保总体计数是总样本数
                    })
            else:
                 print(f"模型 {config['name']} 评估失败或跳过。")

        except FileNotFoundError as e: # 这个 FileNotFoundError 应该主要由 evaluate_transformer_model 抛出
             print(f"评估 Transformer 模型 {config['name']} 时出错: {e}")
        except Exception as e:
             print(f"评估模型 {config['name']} 时发生未知错误: {e}")
             import traceback
             traceback.print_exc() # 打印详细错误堆栈


    # 确保results目录存在
    results_dir = "./results"
    if not os.path.exists(results_dir):
        os.makedirs(results_dir)

    # 将所有结果写入CSV文件
    if all_results_for_csv:
        # 使用数据集名称作为文件名的一部分
        csv_file_path = os.path.join(results_dir, f"{dataset_name}_model_results.csv")
        print(f"\n将所有结果保存到 {csv_file_path}")
        
        fieldnames = ["dataset", "model_name", "data_type", "accuracy", "precision", "recall", "f1", "adv_success_rate", "count"]
        with open(csv_file_path, "w", newline='', encoding='utf-8') as csvfile:
            writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
            
            writer.writeheader()
            for row in all_results_for_csv:
                row["dataset"] = dataset_name  # 添加数据集名称
                writer.writerow(row)
        print("CSV文件保存成功。")
    else:
        print("\n没有可保存的评估结果。")

    # 新增：聚合所有模型预测，按样本输出json
    if all_sample_predictions:
        # 1. 获取所有样本的原文text和true_label
        sample_count = len(next(iter(all_sample_predictions.values())))
        texts = test_dataset['Text'] if 'Text' in test_dataset.features else None
        true_labels = test_dataset['label'] if 'label' in test_dataset.features else None
        sample_json_list = []
        for i in range(sample_count):
            sample_obj = {}
            if texts is not None:
                sample_obj["Text"] = texts[i]
            if true_labels is not None:
                sample_obj["true_label"] = int(true_labels[i])
            # 2. 加入所有模型的预测
            for model_name, preds in all_sample_predictions.items():
                if i < len(preds):
                    sample_obj[model_name] = preds[i]["prediction_label"]
            sample_json_list.append(sample_obj)
        # 3. 输出到json文件
        json_path = os.path.join(results_dir, f"{dataset_name}_all_model_predictions.json")
        with open(json_path, "w", encoding="utf-8") as f:
            json.dump(sample_json_list, f, ensure_ascii=False, indent=2)
        print(f"所有模型聚合预测结果已保存到: {json_path}")

if __name__ == "__main__":
    main() 
