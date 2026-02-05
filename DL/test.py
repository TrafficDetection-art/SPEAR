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
from lime.lime_text import LimeTextExplainer
from sklearn.feature_extraction.text import TfidfVectorizer
from collections import Counter
import re
import nltk
from nltk.util import ngrams

# 下载必要的NLTK数据
try:
    nltk.data.find('tokenizers/punkt')
except LookupError:
    nltk.download('punkt')

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
    
    # 创建测试数据加载器
    test_dataloader = DataLoader(test_dataset, batch_size=32)
    
    with torch.no_grad():
        for batch in tqdm(test_dataloader, desc=f"评估 {model_name}"):
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            labels = batch["label"]
            
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
    
    with torch.no_grad():
        for batch in tqdm(test_dataloader, desc=f"评估 {model_name}"):
            input_ids = batch["input_ids"].to(device)
            labels = batch["label"]
            
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

def explain_prediction(text, model_name, model_type, tokenizer, model_path=None, custom_config=None):
    class_names = ['正常', '恶意']
    explainer = LimeTextExplainer(class_names=class_names)
    
    def predict_proba(texts):
        # texts: list of str
        preds = []
        for t in texts:
            inputs = tokenizer(t, return_tensors="pt", truncation=True, padding="max_length", max_length=512)
            input_ids = inputs["input_ids"].to(device)
            if model_type == "transformer":
                attention_mask = inputs["attention_mask"].to(device)
                local_model_path = f"./models/{model_name}"
                if os.path.exists(local_model_path):
                    model = AutoModelForSequenceClassification.from_pretrained(local_model_path)
                else:
                    model = AutoModelForSequenceClassification.from_pretrained(model_path)
                model.to(device)
                model.eval()
                with torch.no_grad():
                    outputs = model(input_ids=input_ids, attention_mask=attention_mask)
                    logits = outputs.logits
                    prob = torch.softmax(logits, dim=-1).cpu().numpy()
                    preds.append(prob[0])
            else:
                # 你可以为自定义模型补充类似逻辑
                if model_name == "textcnn":
                    model = TextCNNClassifier(**custom_config)
                elif model_name == "cnn_lstm":
                    model = CNNLSTMClassifier(**custom_config)
                elif model_name == "dnn":
                    model = DNNClassifier(**custom_config)
                
                custom_model_path = f"./models/{model_name}.bin"
                if os.path.exists(custom_model_path):
                    model.load_state_dict(torch.load(custom_model_path, map_location=device))
                    model.to(device)
                    model.eval()
                    with torch.no_grad():
                        outputs = model(input_ids)
                        prob = torch.softmax(outputs, dim=-1).cpu().numpy()
                        preds.append(prob[0])
                else:
                    print(f"找不到模型文件 {custom_model_path}")
                    return None
        return np.array(preds)
    
    exp = explainer.explain_instance(text, predict_proba, num_features=15)
    
    # 获取并输出每个词的影响概率
    word_influences = exp.as_list()
    print("词语影响概率排序:")
    print("-" * 50)
    print(f"{'词语':<20} {'影响值':<10} {'对分类的影响'}")
    print("-" * 50)
    for word, score in sorted(word_influences, key=lambda x: abs(x[1]), reverse=True):
        # 正值表示对"恶意"类的贡献，负值表示对"正常"类的贡献
        influence = "恶意" if score > 0 else "正常"
        print(f"{word:<20} {score:>10.4f} {influence}")
    print("-" * 50)
    
    # 返回解释对象以便进一步分析
    return exp

def get_text_bigrams(text):
    """从文本中提取词组（二元组）"""
    # 使用正则表达式清理文本，只保留字母、数字和空格
    text = re.sub(r'[^\w\s]', ' ', text.lower())
    # 使用NLTK分词
    tokens = nltk.word_tokenize(text)
    # 生成二元组
    bigrams_list = list(ngrams(tokens, 2))
    # 将二元组转换为字符串
    bigrams_str = [f"{b[0]} {b[1]}" for b in bigrams_list]
    return bigrams_str

def main():
    parser = argparse.ArgumentParser(description="测试训练好的文本分类模型")
    parser.add_argument("--dataset", type=str, default="../../dataset/prompt_test_results.json", 
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
    
    # 加载测试数据集
    data_path = args.dataset
    dataset_name = os.path.splitext(os.path.basename(data_path))[0]
    print(f"使用数据集: {dataset_name}")
    
    tokenized_dataset = load_dataset(data_path, tokenizer)
    train_test_split = tokenized_dataset.train_test_split(test_size=0.2)
    test_dataset = train_test_split['test']
    
    columns = ['input_ids', 'attention_mask', 'label']
    if 'type' in test_dataset.features:
        columns.append('type')
    test_dataset.set_format(type='torch', columns=columns)

    # 为TF-IDF准备原始文本
    raw_texts = [tokenizer.decode(sample['input_ids'], skip_special_tokens=True) for sample in test_dataset]
    
    # 为单词准备TF-IDF
    tfidf_vectorizer = TfidfVectorizer()
    tfidf_matrix = tfidf_vectorizer.fit_transform(raw_texts)
    tfidf_vocab = tfidf_vectorizer.vocabulary_
    tfidf_feature_names = tfidf_vectorizer.get_feature_names_out()
    
    # 为词组（二元组）准备TF-IDF
    bigram_texts = [' '.join(get_text_bigrams(text)) for text in raw_texts]
    bigram_vectorizer = TfidfVectorizer()
    bigram_tfidf_matrix = bigram_vectorizer.fit_transform(bigram_texts)
    bigram_tfidf_vocab = bigram_vectorizer.vocabulary_
    bigram_tfidf_feature_names = bigram_vectorizer.get_feature_names_out()

    for config in model_configs:
        print(f"\n正在评估模型: {config['name']}")
        # 检查模型文件是否存在
        model_file_exists = True
        if config["type"] == "custom":
            model_path_check = f"./models/{config['name']}.bin"
            if not os.path.exists(model_path_check):
                print(f"警告: 未找到模型文件 '{model_path_check}'，跳过此模型评估")
                model_file_exists = False
        elif config["type"] == "transformer":
            model_dir_path = f"./models/{config['name']}"
            if not os.path.exists(model_dir_path) and not config.get("path"):
                 print(f"警告: 未找到 Transformer 模型目录 '{model_dir_path}' 或指定路径，跳过此模型评估")
                 model_file_exists = False
            elif not os.path.exists(model_dir_path):
                 print(f"警告: 未找到 Transformer 模型目录 '{model_dir_path}'，将尝试直接从 {config['path']} 加载")
        if not model_file_exists:
            continue

        # 1. 先用模型预测所有样本，筛选出恶意样本
        print("正在预测所有样本，筛选恶意样本...")
        model = None
        if config["type"] == "transformer":
            local_model_path = f"./models/{config['name']}"
            if os.path.exists(local_model_path):
                model = AutoModelForSequenceClassification.from_pretrained(local_model_path)
            else:
                model = AutoModelForSequenceClassification.from_pretrained(config["path"])
        else:
            model = config["class"](**custom_config)
            model_path = f"./models/{config['name']}.bin"
            model.load_state_dict(torch.load(model_path, map_location=device))
        model.to(device)
        model.eval()

        malicious_indices = []
        with torch.no_grad():
            for idx, batch in enumerate(DataLoader(test_dataset, batch_size=32)):
                input_ids = batch["input_ids"].to(device)
                if config["type"] == "transformer":
                    attention_mask = batch["attention_mask"].to(device)
                    outputs = model(input_ids=input_ids, attention_mask=attention_mask)
                    logits = outputs.logits
                else:
                    outputs = model(input_ids)
                    logits = outputs
                preds = torch.argmax(logits, dim=-1).cpu().numpy()
                labels = batch["label"].cpu().numpy()
                for i, pred in enumerate(preds):
                    if pred == 1:  # 只对恶意样本做解释
                        malicious_indices.append(idx * 32 + i)
        print(f"共检测到 {len(malicious_indices)} 个恶意样本")

        # 2. 对每个恶意样本用LIME解释，提取关键词和关键词组
        explainer = LimeTextExplainer(class_names=['正常', '恶意'])
        keyword_counter = Counter()
        keyword_tfidf = {}
        keyword_scores = {}  # 新增：记录每个单词的LIME得分
        
        # 新增：词组（二元组）统计
        bigram_counter = Counter()
        bigram_tfidf = {}
        bigram_scores = {}  # 新增：记录每个词组的LIME得分
        
        # 新增：为每个邮件输出单独的分析
        print("\n===== 单个邮件分析 =====")
        
        for i in tqdm(malicious_indices, desc="LIME解释恶意样本"):
            text = raw_texts[i]
            print(f"\n\n邮件 #{i} 内容:")
            print("-" * 80)
            print(text[:300] + "..." if len(text) > 300 else text)  # 只显示前300个字符
            print("-" * 80)
            
            def predict_proba(texts):
                preds = []
                for t in texts:
                    inputs = tokenizer(t, return_tensors="pt", truncation=True, padding="max_length", max_length=512)
                    input_ids = inputs["input_ids"].to(device)
                    if config["type"] == "transformer":
                        attention_mask = inputs["attention_mask"].to(device)
                        with torch.no_grad():
                            outputs = model(input_ids=input_ids, attention_mask=attention_mask)
                            logits = outputs.logits
                            prob = torch.softmax(logits, dim=-1).cpu().numpy()
                            preds.append(prob[0])
                    else:
                        with torch.no_grad():
                            outputs = model(input_ids)
                            prob = torch.softmax(outputs, dim=-1).cpu().numpy()
                            preds.append(prob[0])
                return np.array(preds)
                
            # 使用LIME解释单个样本
            exp = explainer.explain_instance(text, predict_proba, num_features=15)  # 增加到15个特征
            
            # 获取关键词及其影响概率
            lime_results = exp.as_list()
            
            # 新增：输出该邮件的关键词及其影响
            print(f"\n邮件 #{i} 关键特征分析 (使用 {config['name']} 模型):")
            print(f"{'关键词/词组':<30} {'影响概率':>10} {'对分类的影响'}")
            print("-" * 55)
            
            # 按照影响概率的绝对值排序，显示最重要的特征
            for word, score in sorted(lime_results, key=lambda x: abs(x[1]), reverse=True):
                influence = "恶意" if score > 0 else "正常"
                print(f"{word:<30} {score:>10.4f}  {influence}")
            
            # 单词及其影响概率
            top_keywords_with_scores = [(w, s) for w, s in lime_results if len(w.split()) == 1][:5]  # 只取单词
            top_keywords = [w for w, s in top_keywords_with_scores]  # 只要单词
            
            # 从文本中提取词组
            text_bigrams = get_text_bigrams(text)
            
            # 输出重要词组分析
            print("\n重要词组（二元组）分析:")
            print(f"{'词组':<30} {'估计影响概率':>10} {'对分类的影响'}")
            print("-" * 55)
            
            # 筛选与关键词相关的词组
            relevant_bigrams = []
            for bigram in text_bigrams:
                words_in_bigram = bigram.split()
                # 检查该词组是否包含任何LIME识别的关键词
                if any(word in top_keywords for word in words_in_bigram):
                    # 估算该词组的影响概率：使用组成词组的单词中最高的LIME得分
                    max_score = 0
                    for word, score in top_keywords_with_scores:
                        if word in words_in_bigram and abs(score) > abs(max_score):
                            max_score = score
                    
                    # 将词组和估计的影响概率添加到列表
                    relevant_bigrams.append((bigram, max_score))
            
            # 按照影响概率的绝对值排序，显示最重要的词组
            for bigram, score in sorted(relevant_bigrams, key=lambda x: abs(x[1]), reverse=True)[:10]:
                influence = "恶意" if score > 0 else "正常"
                print(f"{bigram:<30} {score:>10.4f}  {influence}")
            
            # 统计单词
            for word, score in top_keywords_with_scores:
                keyword_counter[word] += 1
                # 统计TF-IDF
                if word in tfidf_vocab:
                    idx = tfidf_vocab[word]
                    tfidf_val = tfidf_matrix[i, idx]
                    if word not in keyword_tfidf:
                        keyword_tfidf[word] = []
                    keyword_tfidf[word].append(tfidf_val)
                
                # 记录LIME得分
                if word not in keyword_scores:
                    keyword_scores[word] = []
                keyword_scores[word].append(score)
            
            # 从文本中提取词组并统计
            for bigram in text_bigrams:
                # 只统计那些在LIME结果中出现的单词组成的词组
                words_in_bigram = bigram.split()
                if any(word in top_keywords for word in words_in_bigram):
                    bigram_counter[bigram] += 1
                    # 统计词组的TF-IDF
                    if bigram in bigram_tfidf_vocab:
                        idx = bigram_tfidf_vocab[bigram]
                        tfidf_val = bigram_tfidf_matrix[i, idx]
                        if bigram not in bigram_tfidf:
                            bigram_tfidf[bigram] = []
                        bigram_tfidf[bigram].append(tfidf_val)
                        
                    # 记录词组的LIME得分：取组成词组的单词中最高的LIME得分
                    if bigram not in bigram_scores:
                        bigram_scores[bigram] = []
                    
                    # 查找组成该词组的单词的最高得分
                    max_score = 0
                    for word, score in top_keywords_with_scores:
                        if word in words_in_bigram and abs(score) > abs(max_score):
                            max_score = score
                    
                    bigram_scores[bigram].append(max_score)
            
            # 新增：显示该邮件的恶意和正常特征总结
            positive_features = [(w, s) for w, s in lime_results if s > 0]
            negative_features = [(w, s) for w, s in lime_results if s < 0]
            
            # 获取恶意和正常的重要词组
            positive_bigrams = [(b, s) for b, s in relevant_bigrams if s > 0][:5]
            negative_bigrams = [(b, s) for b, s in relevant_bigrams if s < 0][:5]
            
            print("\n特征总结:")
            print(f"  恶意指示词: {', '.join([w for w, _ in positive_features[:5]])}")
            print(f"  正常指示词: {', '.join([w for w, _ in negative_features[:5]])}")
            print(f"  恶意指示词组: {', '.join([b for b, _ in positive_bigrams])}")
            print(f"  正常指示词组: {', '.join([b for b, _ in negative_bigrams])}")
            
            # 输出分隔线，便于区分不同邮件
            print("\n" + "=" * 80)

        # 输出单词统计结果
        print(f"\n模型 {config['name']} 恶意样本关键词统计：")
        for word, count in keyword_counter.most_common(20):
            tfidf_vals = keyword_tfidf.get(word, [])
            avg_tfidf = np.mean(tfidf_vals) if tfidf_vals else 0.0
            score_vals = keyword_scores.get(word, [])
            avg_score = np.mean(score_vals) if score_vals else 0.0
            print(f"  关键词: {word:15s} 频次: {count:3d} 平均TF-IDF: {avg_tfidf:.4f} 平均影响概率: {avg_score:.4f}")
            
        # 输出词组统计结果
        print(f"\n模型 {config['name']} 恶意样本关键词组（二元组）统计：")
        for bigram, count in bigram_counter.most_common(20):
            tfidf_vals = bigram_tfidf.get(bigram, [])
            avg_tfidf = np.mean(tfidf_vals) if tfidf_vals else 0.0
            score_vals = bigram_scores.get(bigram, [])
            avg_score = np.mean(score_vals) if score_vals else 0.0
            print(f"  关键词组: {bigram:30s} 频次: {count:3d} 平均TF-IDF: {avg_tfidf:.4f} 平均影响概率: {avg_score:.4f}")
            
        # 分别按照正负影响概率排序词组
        print("\n按影响概率排序的关键词组 (至少在3个样本中出现):")
        # 提取所有有效词组的平均影响概率
        all_bigram_scores = [(bigram, np.mean(scores)) for bigram, scores in bigram_scores.items() if len(scores) >= 3]
        
        # 找出正向影响概率最高的5个词组 (对"恶意"类别贡献最大)
        positive_bigrams = sorted([(b, s) for b, s in all_bigram_scores if s > 0], 
                                key=lambda x: x[1], reverse=True)[:5]
        print("\n恶意类别的前5个最有影响力的词组:")
        print(f"{'词组':<30} {'影响概率':>10} {'频次':>6} {'方向'}")
        print("-" * 55)
        for bigram, avg_score in positive_bigrams:
            count = bigram_counter[bigram]
            print(f"  {bigram:<30} {avg_score:>8.4f} {count:>5d}  {'恶意'}")
        
        # 找出负向影响概率最高的5个词组 (对"正常"类别贡献最大)
        negative_bigrams = sorted([(b, s) for b, s in all_bigram_scores if s < 0], 
                                key=lambda x: x[1])[:5]
        print("\n正常类别的前5个最有影响力的词组:")
        print(f"{'词组':<30} {'影响概率':>10} {'频次':>6} {'方向'}")
        print("-" * 55)
        for bigram, avg_score in negative_bigrams:
            count = bigram_counter[bigram]
            print(f"  {bigram:<30} {avg_score:>8.4f} {count:>5d}  {'正常'}")
        
        print("\n简单分析：")
        print("  高频关键词多为: ", ', '.join([w for w, _ in keyword_counter.most_common(10)]))
        print("  高频关键词组多为: ", ', '.join([b for b, _ in bigram_counter.most_common(10)]))
        print("  单词和词组的TF-IDF值可以帮助区分通用词和特定上下文中的重要词组。")
        
        # 按照影响概率大小排序的关键词分析
        print("\n按影响概率排序的Top10关键词:")
        sorted_words = sorted([(word, np.mean(scores)) for word, scores in keyword_scores.items() 
                             if len(scores) >= 3],  # 至少在3个样本中出现
                             key=lambda x: abs(x[1]), reverse=True)[:10]
        for word, avg_score in sorted_words:
            count = keyword_counter[word]
            print(f"  {word:15s} 影响概率: {avg_score:>8.4f} 频次: {count:3d} 方向: {'恶意' if avg_score > 0 else '正常'}")
        
        # 分别按照正负影响概率排序关键词
        print("\n按影响概率排序的关键词 (至少在3个样本中出现):")
        # 提取所有有效词汇的平均影响概率
        all_word_scores = [(word, np.mean(scores)) for word, scores in keyword_scores.items() if len(scores) >= 3]
        
        # 找出正向影响概率最高的5个词 (对"恶意"类别贡献最大)
        positive_words = sorted([(w, s) for w, s in all_word_scores if s > 0], 
                               key=lambda x: x[1], reverse=True)[:5]
        print("\n恶意类别的前5个最有影响力的词:")
        print(f"{'词语':<15} {'影响概率':>10} {'频次':>6} {'方向'}")
        print("-" * 40)
        for word, avg_score in positive_words:
            count = keyword_counter[word]
            print(f"  {word:<15} {avg_score:>8.4f} {count:>5d}  {'恶意'}")
        
        # 找出负向影响概率最高的5个词 (对"正常"类别贡献最大)
        negative_words = sorted([(w, s) for w, s in all_word_scores if s < 0], 
                               key=lambda x: x[1])[:5]
        print("\n正常类别的前5个最有影响力的词:")
        print(f"{'词语':<15} {'影响概率':>10} {'频次':>6} {'方向'}")
        print("-" * 40)
        for word, avg_score in negative_words:
            count = keyword_counter[word]
            print(f"  {word:<15} {avg_score:>8.4f} {count:>5d}  {'正常'}")
        
        # 分析LIME得分和出现频率的关系
        print("\nLIME影响概率分析:")
        high_influence_words = [word for word, score in sorted_words[:5]]
        print(f"  影响力最大的词: {', '.join(high_influence_words)}")
        
        high_freq_words = [word for word, _ in keyword_counter.most_common(5)]
        print(f"  出现频率最高的词: {', '.join(high_freq_words)}")
        
        common_words = set(high_influence_words) & set(high_freq_words)
        if common_words:
            print(f"  既高频又高影响力的词: {', '.join(common_words)}")
        else:
            print("  高频词与高影响力词无交集，说明频率不等同于重要性。")

        print("\n总体特征分析：")
        print("  高频关键词多为: ", ', '.join([w for w, _ in keyword_counter.most_common(10)]))
        print("  高频关键词组多为: ", ', '.join([b for b, _ in bigram_counter.most_common(10)]))
        
        # 增强版LIME影响概率分析
        print("\n关键特征影响分析:")
        
        # 分析正负影响词
        positive_keywords = [word for word, score in positive_words]
        negative_keywords = [word for word, score in negative_words]
        print(f"  最具恶意指示性的词: {', '.join(positive_keywords)}")
        print(f"  最具正常指示性的词: {', '.join(negative_keywords)}")
        
        # 分析正负影响词组
        positive_bigram_words = [bigram for bigram, score in positive_bigrams]
        negative_bigram_words = [bigram for bigram, score in negative_bigrams]
        print(f"  最具恶意指示性的词组: {', '.join(positive_bigram_words)}")
        print(f"  最具正常指示性的词组: {', '.join(negative_bigram_words)}")
        
        # 分析高频词与高影响力词的关系
        high_freq_words = [word for word, _ in keyword_counter.most_common(10)]
        print("\n频率与影响力的关系:")
        
        # 检查恶意高影响力词中有多少是高频词
        common_positive = set(positive_keywords) & set(high_freq_words)
        if common_positive:
            print(f"  既高频又高恶意影响力的词: {', '.join(common_positive)}")
        else:
            print("  高频词与高恶意影响力词无交集，频率不等同于重要性。")
            
        # 检查正常高影响力词中有多少是高频词
        common_negative = set(negative_keywords) & set(high_freq_words)
        if common_negative:
            print(f"  既高频又高正常影响力的词: {', '.join(common_negative)}")
        else:
            print("  高频词与高正常影响力词无交集，频率不等同于重要性。")
            
        # 模型特征解释总结
        print("\n模型特征识别总结:")
        print(f"  模型 {config['name']} 主要通过以下特征识别恶意样本:")
        print(f"    1. 正向特征: {', '.join(positive_keywords[:3])}")
        print(f"    2. 正向词组: {', '.join(positive_bigram_words[:3])}")
        print(f"  而正常样本则倾向于包含: {', '.join(negative_keywords[:3])}")

if __name__ == "__main__":
    main() 
