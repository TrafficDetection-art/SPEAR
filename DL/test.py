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
import csv # Added: import csv module
import os.path # For extracting filenames
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

# Download necessary NLTK data
try:
    nltk.data.find('tokenizers/punkt')
except LookupError:
    nltk.download('punkt')

# Set up device
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

# Set up evaluation metrics
accuracy_metric = load_metric("accuracy")
precision_metric = load_metric("precision")
recall_metric = load_metric("recall")
f1_metric = load_metric("f1")

def evaluate_transformer_model(model_name, model_path, test_dataset):
    """Evaluate transformer model"""
    print(f"Loading model: {model_name}")
    
    # Modified: check if local path exists, otherwise load from original model path
    local_model_path = f"./models/{model_name}"
    try:
        # Try loading from local
        if os.path.exists(local_model_path):
            model = AutoModelForSequenceClassification.from_pretrained(local_model_path)
        else:
            # If not available locally, load directly from original model path
            model = AutoModelForSequenceClassification.from_pretrained(model_path)
            print(f"Loading model from {model_path}")
    except Exception as e:
        print(f"Failed to load locally, trying to load model from {model_path}")
        model = AutoModelForSequenceClassification.from_pretrained(model_path)
    
    model.to(device)
    model.eval()
    
    all_preds = []
    all_labels = []
    all_types = []  # Added: collect types of all samples
    
    # Create test data loader
    test_dataloader = DataLoader(test_dataset, batch_size=32)
    
    with torch.no_grad():
        for batch in tqdm(test_dataloader, desc=f"Evaluating {model_name}"):
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            labels = batch["label"]
            
            # Added: collect data type (if exists)
            if "type" in batch:
                types = batch["type"]
                all_types.extend(types)
            
            outputs = model(input_ids=input_ids, attention_mask=attention_mask)
            logits = outputs.logits
            preds = torch.argmax(logits, dim=-1).cpu().numpy()
            
            all_preds.extend(preds)
            all_labels.extend(labels.numpy())
    
    # Calculate overall metrics
    accuracy = accuracy_metric.compute(predictions=all_preds, references=all_labels)
    precision = precision_metric.compute(predictions=all_preds, references=all_labels, average="binary")
    recall = recall_metric.compute(predictions=all_preds, references=all_labels, average="binary")
    f1 = f1_metric.compute(predictions=all_preds, references=all_labels, average="binary")
    
    # Calculate adversarial success rate
    adv_success_rate = 1.0 - recall["recall"]
    
    results = {
        "overall": {
            "accuracy": accuracy["accuracy"],
            "precision": precision["precision"],
            "recall": recall["recall"],
            "f1": f1["f1"],
            "adv_success_rate": adv_success_rate  # Add adversarial success rate
        }
    }
    
    # If type information exists, calculate metrics for each type
    if all_types:
        unique_types = set(all_types)
        for data_type in unique_types:
            type_indices = [i for i, t in enumerate(all_types) if t == data_type]
            type_preds = [all_preds[i] for i in type_indices]
            type_labels = [all_labels[i] for i in type_indices]
            
            # Calculate metrics for each type
            type_accuracy = accuracy_metric.compute(predictions=type_preds, references=type_labels)
            type_precision = precision_metric.compute(predictions=type_preds, references=type_labels, average="binary")
            type_recall = recall_metric.compute(predictions=type_preds, references=type_labels, average="binary")
            type_f1 = f1_metric.compute(predictions=type_preds, references=type_labels, average="binary")
            
            # Calculate type-specific adversarial success rate
            type_adv_success_rate = 1.0 - type_recall["recall"]
            
            results[data_type] = {
                "accuracy": type_accuracy["accuracy"],
                "precision": type_precision["precision"],
                "recall": type_recall["recall"],
                "f1": type_f1["f1"],
                "adv_success_rate": type_adv_success_rate,  # Add adversarial success rate
                "count": len(type_indices)
            }
    
    return results

def evaluate_custom_model(model_class, model_name, test_dataloader, custom_config):
    """Evaluate custom model"""
    print(f"Loading model: {model_name}")
    
    # Check if model file exists - Modified: look for .bin files in ./models directory
    model_path = f"./models/{model_name}.bin" 
    if not os.path.exists(model_path):
        # Modified: updated error message
        print(f"Warning: model file '{model_path}' not found, skipping evaluation")
        return None # Return None or other indicator for failure
    
    # Initialize model
    model = model_class(**custom_config)
    
    try:
        # Modified: load .bin file
        model.load_state_dict(torch.load(model_path, map_location=device)) 
    except Exception as e:
        print(f"Error loading model '{model_path}': {e}")
        return None # Return None or other indicator for failure

    model.to(device)
    model.eval()
    
    all_preds = []
    all_labels = []
    all_types = []  # Added: collect types of all samples
    
    with torch.no_grad():
        for batch in tqdm(test_dataloader, desc=f"Evaluating {model_name}"):
            input_ids = batch["input_ids"].to(device)
            labels = batch["label"]
            
            # Added: collect data type (if exists)
            if "type" in batch:
                types = batch["type"]
                all_types.extend(types)
            
            outputs = model(input_ids)
            preds = torch.argmax(outputs, dim=-1).cpu().numpy()
            
            all_preds.extend(preds)
            all_labels.extend(labels.numpy())
    
    # Calculate overall metrics
    accuracy = accuracy_metric.compute(predictions=all_preds, references=all_labels)
    precision = precision_metric.compute(predictions=all_preds, references=all_labels, average="binary")
    recall = recall_metric.compute(predictions=all_preds, references=all_labels, average="binary")
    f1 = f1_metric.compute(predictions=all_preds, references=all_labels, average="binary")
    
    # Calculate adversarial success rate
    adv_success_rate = 1.0 - recall["recall"]
    
    results = {
        "overall": {
            "accuracy": accuracy["accuracy"],
            "precision": precision["precision"],
            "recall": recall["recall"],
            "f1": f1["f1"],
            "adv_success_rate": adv_success_rate  # Add adversarial success rate
        }
    }
    
    # If type information exists, calculate metrics for each type
    if all_types:
        unique_types = set(all_types)
        for data_type in unique_types:
            type_indices = [i for i, t in enumerate(all_types) if t == data_type]
            type_preds = [all_preds[i] for i in type_indices]
            type_labels = [all_labels[i] for i in type_indices]
            
            # Calculate metrics for each type
            type_accuracy = accuracy_metric.compute(predictions=type_preds, references=type_labels)
            type_precision = precision_metric.compute(predictions=type_preds, references=type_labels, average="binary")
            type_recall = recall_metric.compute(predictions=type_preds, references=type_labels, average="binary")
            type_f1 = f1_metric.compute(predictions=type_preds, references=type_labels, average="binary")
            
            # Calculate type-specific adversarial success rate
            type_adv_success_rate = 1.0 - type_recall["recall"]
            
            results[data_type] = {
                "accuracy": type_accuracy["accuracy"],
                "precision": type_precision["precision"],
                "recall": type_recall["recall"],
                "f1": type_f1["f1"],
                "adv_success_rate": type_adv_success_rate,  # Add adversarial success rate
                "count": len(type_indices)
            }
    
    return results

def predict_sample(text, model_name, model_type, tokenizer, model_path=None, custom_config=None):
    """Predict a single text sample using specified model"""
    print(f"Predicting with {model_name} model...")
    
    # Tokenize input text
    inputs = tokenizer(text, return_tensors="pt", truncation=True, padding="max_length", max_length=512)
    input_ids = inputs["input_ids"].to(device)
    
    if model_type == "transformer":
        attention_mask = inputs["attention_mask"].to(device)
        
        # Modified: check if local path exists, otherwise load from original model path
        local_model_path = f"./models/{model_name}" # Transformer models are still in models directory
        try:
            if os.path.exists(local_model_path):
                model = AutoModelForSequenceClassification.from_pretrained(local_model_path)
            else:
                model = AutoModelForSequenceClassification.from_pretrained(model_path)
        except Exception as e:
            print(f"Failed to load locally, trying to load model from {model_path}")
            model = AutoModelForSequenceClassification.from_pretrained(model_path)
            
        model.to(device)
        model.eval()
        
        with torch.no_grad():
            outputs = model(input_ids=input_ids, attention_mask=attention_mask)
            logits = outputs.logits
            pred = torch.argmax(logits, dim=-1).item()
    else: # custom model
        # Modified: initialize corresponding model class
        if model_name == "textcnn":
            model = TextCNNClassifier(**custom_config)
        elif model_name == "cnn_lstm":
            model = CNNLSTMClassifier(**custom_config)
        elif model_name == "dnn":
            model = DNNClassifier(**custom_config)
        elif model_name == "deeplog":
            # Ensure DeepLog exists in both main.py and test.py model configs or in neither
            # If DeepLog model exists, uncomment the line below
            # model = DeepLog(**custom_config) 
            pass # If no deeplog, keep as is or add error handling

        # Modified: load .bin file from ./models directory
        custom_model_path = f"./models/{model_name}.bin" 
        if not os.path.exists(custom_model_path):
            print(f"Error: model file '{custom_model_path}' not found during prediction")
            return None, "Error: model not found"

        try:
            # Modified: load .bin file
            model.load_state_dict(torch.load(custom_model_path, map_location=device)) 
        except Exception as e:
            print(f"Error loading model '{custom_model_path}': {e}")
            return None, f"Error: failed to load model"

        model.to(device)
        model.eval()
        
        with torch.no_grad():
            outputs = model(input_ids)
            pred = torch.argmax(outputs, dim=-1).item()
    
    return pred, "malicious" if pred == 1 else "benign"

def explain_prediction(text, model_name, model_type, tokenizer, model_path=None, custom_config=None):
    class_names = ['benign', 'malicious']
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
                # You can add similar logic for custom models
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
                    print(f"Model file {custom_model_path} not found")
                    return None
        return np.array(preds)
    
    exp = explainer.explain_instance(text, predict_proba, num_features=15)
    
    # Get and output influence probability for each word
    word_influences = exp.as_list()
    print("Word influence probability ranking:")
    print("-" * 50)
    print(f"{'Word':<20} {'Influence':<10} {'Classification Impact'}")
    print("-" * 50)
    for word, score in sorted(word_influences, key=lambda x: abs(x[1]), reverse=True):
        # Positive values indicate contribution to "malicious" class, negative values to "benign" class
        influence = "malicious" if score > 0 else "benign"
        print(f"{word:<20} {score:>10.4f} {influence}")
    print("-" * 50)
    
    # Return explanation object for further analysis
    return exp

def get_text_bigrams(text):
    """Extract word phrases (bigrams) from text"""
    # Clean text using regex, keeping only letters, numbers and spaces
    text = re.sub(r'[^\w\s]', ' ', text.lower())
    # Use NLTK tokenization
    tokens = nltk.word_tokenize(text)
    # Generate bigrams
    bigrams_list = list(ngrams(tokens, 2))
    # Convert bigrams to strings
    bigrams_str = [f"{b[0]} {b[1]}" for b in bigrams_list]
    return bigrams_str

def main():
    parser = argparse.ArgumentParser(description="Test trained text classification models")
    parser.add_argument("--dataset", type=str, default="../../dataset/prompt_test_results.json", 
                      help="Test dataset path, default is baseline_generated_emails.json")
    args = parser.parse_args()
    
    # Load tokenizer
    tokenizer = AutoTokenizer.from_pretrained("bert-base-uncased")
    
    # Set custom model configuration
    custom_config = {
        "vocab_size": tokenizer.vocab_size,
        "embed_size": 512,
        "num_classes": 2,
        "max_len": 512
    }
    
    # Define model configurations
    model_configs = [
        {"name": "textcnn", "path": None, "type": "custom", "class": TextCNNClassifier},
        {"name": "cnn_lstm", "path": None, "type": "custom", "class": CNNLSTMClassifier},
        {"name": "dnn", "path": None, "type": "custom", "class": DNNClassifier},
        {"name": "bert", "path": "bert-base-uncased", "type": "transformer"},
        {"name": "RoBERTa", "path": "roberta-base", "type": "transformer"},
        {"name": "DistilBERT", "path": "distilbert-base-uncased", "type": "transformer"}
    ]
    
    # Load test dataset
    data_path = args.dataset
    dataset_name = os.path.splitext(os.path.basename(data_path))[0]
    print(f"Using dataset: {dataset_name}")
    
    tokenized_dataset = load_dataset(data_path, tokenizer)
    train_test_split = tokenized_dataset.train_test_split(test_size=0.2)
    test_dataset = train_test_split['test']
    
    columns = ['input_ids', 'attention_mask', 'label']
    if 'type' in test_dataset.features:
        columns.append('type')
    test_dataset.set_format(type='torch', columns=columns)

    # Prepare raw texts for TF-IDF
    raw_texts = [tokenizer.decode(sample['input_ids'], skip_special_tokens=True) for sample in test_dataset]
    
    # Prepare TF-IDF for words
    tfidf_vectorizer = TfidfVectorizer()
    tfidf_matrix = tfidf_vectorizer.fit_transform(raw_texts)
    tfidf_vocab = tfidf_vectorizer.vocabulary_
    tfidf_feature_names = tfidf_vectorizer.get_feature_names_out()
    
    # Prepare TF-IDF for phrases (bigrams)
    bigram_texts = [' '.join(get_text_bigrams(text)) for text in raw_texts]
    bigram_vectorizer = TfidfVectorizer()
    bigram_tfidf_matrix = bigram_vectorizer.fit_transform(bigram_texts)
    bigram_tfidf_vocab = bigram_vectorizer.vocabulary_
    bigram_tfidf_feature_names = bigram_vectorizer.get_feature_names_out()

    for config in model_configs:
        print(f"\nEvaluating model: {config['name']}")
        # Check if model file exists
        model_file_exists = True
        if config["type"] == "custom":
            model_path_check = f"./models/{config['name']}.bin"
            if not os.path.exists(model_path_check):
                print(f"Warning: model file '{model_path_check}' not found, skipping evaluation")
                model_file_exists = False
        elif config["type"] == "transformer":
            model_dir_path = f"./models/{config['name']}"
            if not os.path.exists(model_dir_path) and not config.get("path"):
                 print(f"Warning: Transformer model directory '{model_dir_path}' or specified path not found, skipping evaluation")
                 model_file_exists = False
            elif not os.path.exists(model_dir_path):
                 print(f"Warning: Transformer model directory '{model_dir_path}' not found, will try loading directly from {config['path']}")
        if not model_file_exists:
            continue

        # 1. First predict all samples, filter malicious samples
        print("Predicting all samples, filtering malicious samples...")
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
                    if pred == 1:  # Only explain malicious samples
                        malicious_indices.append(idx * 32 + i)
        print(f"Detected {len(malicious_indices)} malicious samples")

        # 2. Use LIME to explain each malicious sample, extract keywords and key phrases
        explainer = LimeTextExplainer(class_names=['benign', 'malicious'])
        keyword_counter = Counter()
        keyword_tfidf = {}
        keyword_scores = {}  # Added: record LIME scores for each word
        
        # Added: phrase (bigram) statistics
        bigram_counter = Counter()
        bigram_tfidf = {}
        bigram_scores = {}  # Added: record LIME scores for each phrase
        
        # Added: output individual analysis for each email
        print("\n===== Individual Email Analysis =====")
        
        for i in tqdm(malicious_indices, desc="LIME explaining malicious samples"):
            text = raw_texts[i]
            print(f"\n\nEmail #{i} content:")
            print("-" * 80)
            print(text[:300] + "..." if len(text) > 300 else text)  # Only show first 300 characters
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
                
            # Use LIME to explain individual sample
            exp = explainer.explain_instance(text, predict_proba, num_features=15)  # Increased to 15 features
            
            # Get keywords and their influence probabilities
            lime_results = exp.as_list()
            
            # Added: output keywords and their influence for this email
            print(f"\nEmail #{i} key feature analysis (using {config['name']} model):")
            print(f"{'Keyword/Phrase':<30} {'Influence':>10} {'Classification Impact'}")
            print("-" * 55)
            
            # Sort by absolute influence probability, show most important features
            for word, score in sorted(lime_results, key=lambda x: abs(x[1]), reverse=True):
                influence = "malicious" if score > 0 else "benign"
                print(f"{word:<30} {score:>10.4f}  {influence}")
            
            # Words and their influence probabilities
            top_keywords_with_scores = [(w, s) for w, s in lime_results if len(w.split()) == 1][:5]  # Only take single words
            top_keywords = [w for w, s in top_keywords_with_scores]  # Only words
            
            # Extract phrases from text
            text_bigrams = get_text_bigrams(text)
            
            # Output important phrase analysis
            print("\nImportant phrase (bigram) analysis:")
            print(f"{'Phrase':<30} {'Est. Influence':>10} {'Classification Impact'}")
            print("-" * 55)
            
            # Filter phrases related to keywords
            relevant_bigrams = []
            for bigram in text_bigrams:
                words_in_bigram = bigram.split()
                # Check if this phrase contains any LIME-identified keywords
                if any(word in top_keywords for word in words_in_bigram):
                    # Estimate phrase influence probability: use highest LIME score among component words
                    max_score = 0
                    for word, score in top_keywords_with_scores:
                        if word in words_in_bigram and abs(score) > abs(max_score):
                            max_score = score
                    
                    # Add phrase and estimated influence probability to list
                    relevant_bigrams.append((bigram, max_score))
            
            # Sort by absolute influence probability, show most important phrases
            for bigram, score in sorted(relevant_bigrams, key=lambda x: abs(x[1]), reverse=True)[:10]:
                influence = "malicious" if score > 0 else "benign"
                print(f"{bigram:<30} {score:>10.4f}  {influence}")
            
            # Count words
            for word, score in top_keywords_with_scores:
                keyword_counter[word] += 1
                # Count TF-IDF
                if word in tfidf_vocab:
                    idx = tfidf_vocab[word]
                    tfidf_val = tfidf_matrix[i, idx]
                    if word not in keyword_tfidf:
                        keyword_tfidf[word] = []
                    keyword_tfidf[word].append(tfidf_val)
                
                # Record LIME scores
                if word not in keyword_scores:
                    keyword_scores[word] = []
                keyword_scores[word].append(score)
            
            # Extract phrases from text and count
            for bigram in text_bigrams:
                # Only count phrases composed of words appearing in LIME results
                words_in_bigram = bigram.split()
                if any(word in top_keywords for word in words_in_bigram):
                    bigram_counter[bigram] += 1
                    # Count phrase TF-IDF
                    if bigram in bigram_tfidf_vocab:
                        idx = bigram_tfidf_vocab[bigram]
                        tfidf_val = bigram_tfidf_matrix[i, idx]
                        if bigram not in bigram_tfidf:
                            bigram_tfidf[bigram] = []
                        bigram_tfidf[bigram].append(tfidf_val)
                        
                    # Record phrase LIME score: use highest LIME score among component words
                    if bigram not in bigram_scores:
                        bigram_scores[bigram] = []
                    
                    # Find the highest score among words composing this phrase
                    max_score = 0
                    for word, score in top_keywords_with_scores:
                        if word in words_in_bigram and abs(score) > abs(max_score):
                            max_score = score
                    
                    bigram_scores[bigram].append(max_score)
            
            # Added: show malicious and benign feature summary for this email
            positive_features = [(w, s) for w, s in lime_results if s > 0]
            negative_features = [(w, s) for w, s in lime_results if s < 0]
            
            # Get malicious and benign important phrases
            positive_bigrams = [(b, s) for b, s in relevant_bigrams if s > 0][:5]
            negative_bigrams = [(b, s) for b, s in relevant_bigrams if s < 0][:5]
            
            print("\nFeature summary:")
            print(f"  Malicious indicator words: {', '.join([w for w, _ in positive_features[:5]])}")
            print(f"  Benign indicator words: {', '.join([w for w, _ in negative_features[:5]])}")
            print(f"  Malicious indicator phrases: {', '.join([b for b, _ in positive_bigrams])}")
            print(f"  Benign indicator phrases: {', '.join([b for b, _ in negative_bigrams])}")
            
            # Output separator for distinguishing different emails
            print("\n" + "=" * 80)

        # Output word statistics results
        print(f"\nModel {config['name']} malicious sample keyword statistics:")
        for word, count in keyword_counter.most_common(20):
            tfidf_vals = keyword_tfidf.get(word, [])
            avg_tfidf = np.mean(tfidf_vals) if tfidf_vals else 0.0
            score_vals = keyword_scores.get(word, [])
            avg_score = np.mean(score_vals) if score_vals else 0.0
            print(f"  Keyword: {word:15s} Freq: {count:3d} Avg TF-IDF: {avg_tfidf:.4f} Avg Influence: {avg_score:.4f}")
            
        # Output phrase statistics results
        print(f"\nModel {config['name']} malicious sample key phrase (bigram) statistics:")
        for bigram, count in bigram_counter.most_common(20):
            tfidf_vals = bigram_tfidf.get(bigram, [])
            avg_tfidf = np.mean(tfidf_vals) if tfidf_vals else 0.0
            score_vals = bigram_scores.get(bigram, [])
            avg_score = np.mean(score_vals) if score_vals else 0.0
            print(f"  Key phrase: {bigram:30s} Freq: {count:3d} Avg TF-IDF: {avg_tfidf:.4f} Avg Influence: {avg_score:.4f}")
            
        # Sort phrases by positive and negative influence probabilities
        print("\nKey phrases sorted by influence probability (appearing in at least 3 samples):")
        # Extract average influence probability for all valid phrases
        all_bigram_scores = [(bigram, np.mean(scores)) for bigram, scores in bigram_scores.items() if len(scores) >= 3]
        
        # Find top 5 phrases with highest positive influence (greatest contribution to "malicious" class)
        positive_bigrams = sorted([(b, s) for b, s in all_bigram_scores if s > 0], 
                                key=lambda x: x[1], reverse=True)[:5]
        print("\nTop 5 most influential phrases for malicious class:")
        print(f"{'Phrase':<30} {'Influence':>10} {'Freq':>6} {'Direction'}")
        print("-" * 55)
        for bigram, avg_score in positive_bigrams:
            count = bigram_counter[bigram]
            print(f"  {bigram:<30} {avg_score:>8.4f} {count:>5d}  {'malicious'}")
        
        # Find top 5 phrases with highest negative influence (greatest contribution to "benign" class)
        negative_bigrams = sorted([(b, s) for b, s in all_bigram_scores if s < 0], 
                                key=lambda x: x[1])[:5]
        print("\nTop 5 most influential phrases for benign class:")
        print(f"{'Phrase':<30} {'Influence':>10} {'Freq':>6} {'Direction'}")
        print("-" * 55)
        for bigram, avg_score in negative_bigrams:
            count = bigram_counter[bigram]
            print(f"  {bigram:<30} {avg_score:>8.4f} {count:>5d}  {'benign'}")
        
        print("\nBrief analysis:")
        print("  High-frequency keywords are mostly: ", ', '.join([w for w, _ in keyword_counter.most_common(10)]))
        print("  High-frequency key phrases are mostly: ", ', '.join([b for b, _ in bigram_counter.most_common(10)]))
        print("  TF-IDF values for words and phrases help distinguish common words from important phrases in specific contexts.")
        
        # Keyword analysis sorted by influence probability
        print("\nTop 10 keywords sorted by influence probability:")
        sorted_words = sorted([(word, np.mean(scores)) for word, scores in keyword_scores.items() 
                             if len(scores) >= 3],  # Appearing in at least 3 samples
                             key=lambda x: abs(x[1]), reverse=True)[:10]
        for word, avg_score in sorted_words:
            count = keyword_counter[word]
            print(f"  {word:15s} Influence: {avg_score:>8.4f} Freq: {count:3d} Direction: {'malicious' if avg_score > 0 else 'benign'}")
        
        # Sort keywords by positive and negative influence probabilities
        print("\nKeywords sorted by influence probability (appearing in at least 3 samples):")
        # Extract average influence probability for all valid words
        all_word_scores = [(word, np.mean(scores)) for word, scores in keyword_scores.items() if len(scores) >= 3]
        
        # Find top 5 words with highest positive influence (greatest contribution to "malicious" class)
        positive_words = sorted([(w, s) for w, s in all_word_scores if s > 0], 
                               key=lambda x: x[1], reverse=True)[:5]
        print("\nTop 5 most influential words for malicious class:")
        print(f"{'Word':<15} {'Influence':>10} {'Freq':>6} {'Direction'}")
        print("-" * 40)
        for word, avg_score in positive_words:
            count = keyword_counter[word]
            print(f"  {word:<15} {avg_score:>8.4f} {count:>5d}  {'malicious'}")
        
        # Find top 5 words with highest negative influence (greatest contribution to "benign" class)
        negative_words = sorted([(w, s) for w, s in all_word_scores if s < 0], 
                               key=lambda x: x[1])[:5]
        print("\nTop 5 most influential words for benign class:")
        print(f"{'Word':<15} {'Influence':>10} {'Freq':>6} {'Direction'}")
        print("-" * 40)
        for word, avg_score in negative_words:
            count = keyword_counter[word]
            print(f"  {word:<15} {avg_score:>8.4f} {count:>5d}  {'benign'}")
        
        # Analyze relationship between LIME scores and frequency
        print("\nLIME influence probability analysis:")
        high_influence_words = [word for word, score in sorted_words[:5]]
        print(f"  Most influential words: {', '.join(high_influence_words)}")
        
        high_freq_words = [word for word, _ in keyword_counter.most_common(5)]
        print(f"  Most frequent words: {', '.join(high_freq_words)}")
        
        common_words = set(high_influence_words) & set(high_freq_words)
        if common_words:
            print(f"  Words that are both high-frequency and high-influence: {', '.join(common_words)}")
        else:
            print("  No overlap between high-frequency and high-influence words, indicating frequency does not equal importance.")

        print("\nOverall feature analysis:")
        print("  High-frequency keywords are mostly: ", ', '.join([w for w, _ in keyword_counter.most_common(10)]))
        print("  High-frequency key phrases are mostly: ", ', '.join([b for b, _ in bigram_counter.most_common(10)]))
        
        # Enhanced LIME influence probability analysis
        print("\nKey feature influence analysis:")
        
        # Analyze positive and negative influence words
        positive_keywords = [word for word, score in positive_words]
        negative_keywords = [word for word, score in negative_words]
        print(f"  Most indicative of malicious: {', '.join(positive_keywords)}")
        print(f"  Most indicative of benign: {', '.join(negative_keywords)}")
        
        # Analyze positive and negative influence phrases
        positive_bigram_words = [bigram for bigram, score in positive_bigrams]
        negative_bigram_words = [bigram for bigram, score in negative_bigrams]
        print(f"  Most malicious-indicative phrases: {', '.join(positive_bigram_words)}")
        print(f"  Most benign-indicative phrases: {', '.join(negative_bigram_words)}")
        
        # Analyze relationship between high-frequency words and high-influence words
        high_freq_words = [word for word, _ in keyword_counter.most_common(10)]
        print("\nRelationship between frequency and influence:")
        
        # Check how many malicious high-influence words are also high-frequency
        common_positive = set(positive_keywords) & set(high_freq_words)
        if common_positive:
            print(f"  Words that are both high-frequency and high malicious influence: {', '.join(common_positive)}")
        else:
            print("  No overlap between high-frequency and high malicious influence words, frequency does not equal importance.")
            
        # Check how many benign high-influence words are also high-frequency
        common_negative = set(negative_keywords) & set(high_freq_words)
        if common_negative:
            print(f"  Words that are both high-frequency and high benign influence: {', '.join(common_negative)}")
        else:
            print("  No overlap between high-frequency and high benign influence words, frequency does not equal importance.")
            
        # Model feature explanation summary
        print("\nModel feature identification summary:")
        print(f"  Model {config['name']} identifies malicious samples mainly through these features:")
        print(f"    1. Positive features: {', '.join(positive_keywords[:3])}")
        print(f"    2. Positive phrases: {', '.join(positive_bigram_words[:3])}")
        print(f"  While benign samples tend to contain: {', '.join(negative_keywords[:3])}")

if __name__ == "__main__":
    main() 
