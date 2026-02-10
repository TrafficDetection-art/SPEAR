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
    all_texts = []  # Added: collect original text of all samples
    all_indices = []  # Added: collect indices of all samples
    
    # Create test data loader
    test_dataloader = DataLoader(test_dataset, batch_size=32)
    
    with torch.no_grad():
        for batch_idx, batch in enumerate(tqdm(test_dataloader, desc=f"Evaluating {model_name}")):
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            labels = batch["label"]
            
            # Added: collect sample indices
            indices = [batch_idx * test_dataloader.batch_size + i for i in range(len(labels))]
            all_indices.extend(indices)
            
            # Added: collect original text (if exists)
            if "Text" in batch:
                texts = batch["Text"]
                all_texts.extend(texts)
            
            # Added: collect data type (if exists)
            if "type" in batch:
                types = batch["type"]
                all_types.extend(types)
            
            outputs = model(input_ids=input_ids, attention_mask=attention_mask)
            logits = outputs.logits
            preds = torch.argmax(logits, dim=-1).cpu().numpy()
            
            all_preds.extend(preds)
            all_labels.extend(labels.numpy())
    
    # Compute overall metrics
    accuracy = accuracy_metric.compute(predictions=all_preds, references=all_labels)
    precision = precision_metric.compute(predictions=all_preds, references=all_labels, average="binary")
    recall = recall_metric.compute(predictions=all_preds, references=all_labels, average="binary")
    f1 = f1_metric.compute(predictions=all_preds, references=all_labels, average="binary")
    
    # Compute adversarial success rate
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
    
    # If type information is available, compute metrics for each type
    if all_types:
        unique_types = set(all_types)
        for data_type in unique_types:
            type_indices = [i for i, t in enumerate(all_types) if t == data_type]
            type_preds = [all_preds[i] for i in type_indices]
            type_labels = [all_labels[i] for i in type_indices]
            
            # Compute metrics for each type
            type_accuracy = accuracy_metric.compute(predictions=type_preds, references=type_labels)
            type_precision = precision_metric.compute(predictions=type_preds, references=type_labels, average="binary")
            type_recall = recall_metric.compute(predictions=type_preds, references=type_labels, average="binary")
            type_f1 = f1_metric.compute(predictions=type_preds, references=type_labels, average="binary")
            
            # Compute type-specific adversarial success rate
            type_adv_success_rate = 1.0 - type_recall["recall"]
            
            results[data_type] = {
                "accuracy": type_accuracy["accuracy"],
                "precision": type_precision["precision"],
                "recall": type_recall["recall"],
                "f1": type_f1["f1"],
                "adv_success_rate": type_adv_success_rate,  # Add adversarial success rate
                "count": len(type_indices)
            }
    
    # Added: create detailed sample prediction results
    sample_predictions = []
    for i in range(len(all_preds)):
        sample_info = {
            "index": all_indices[i],
            "prediction": int(all_preds[i]),
            "prediction_label": "malicious" if all_preds[i] == 1 else "benign",
            "true_label": int(all_labels[i]),
            "true_label_text": "malicious" if all_labels[i] == 1 else "benign",
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
    """Evaluate custom model"""
    print(f"Loading model: {model_name}")
    
    # Check if model file exists - Modified: look for .bin file in ./models directory
    model_path = f"./models/{model_name}.bin" 
    if not os.path.exists(model_path):
        # Modified: update error message
        print(f"Warning: model file '{model_path}' not found, skipping this model evaluation")
        return None # Return None or other indicator of failure
    
    # Initialize model
    model = model_class(**custom_config)
    
    try:
        # Modified: load .bin file
        model.load_state_dict(torch.load(model_path, map_location=device)) 
    except Exception as e:
        print(f"Error loading model '{model_path}': {e}")
        return None # Return None or other indicator of failure

    model.to(device)
    model.eval()
    
    all_preds = []
    all_labels = []
    all_types = []  # Added: collect types of all samples
    all_texts = []  # Added: collect original text of all samples
    all_indices = []  # Added: collect indices of all samples
    
    with torch.no_grad():
        for batch_idx, batch in enumerate(tqdm(test_dataloader, desc=f"Evaluating {model_name}")):
            input_ids = batch["input_ids"].to(device)
            labels = batch["label"]
            
            # Added: collect sample indices
            indices = [batch_idx * test_dataloader.batch_size + i for i in range(len(labels))]
            all_indices.extend(indices)
            
            # Added: collect original text (if exists)
            if "Text" in batch:
                texts = batch["Text"]
                all_texts.extend(texts)
            
            # Added: collect data type (if exists)
            if "type" in batch:
                types = batch["type"]
                all_types.extend(types)
            
            outputs = model(input_ids)
            preds = torch.argmax(outputs, dim=-1).cpu().numpy()
            
            all_preds.extend(preds)
            all_labels.extend(labels.numpy())
    
    # Compute overall metrics
    accuracy = accuracy_metric.compute(predictions=all_preds, references=all_labels)
    precision = precision_metric.compute(predictions=all_preds, references=all_labels, average="binary")
    recall = recall_metric.compute(predictions=all_preds, references=all_labels, average="binary")
    f1 = f1_metric.compute(predictions=all_preds, references=all_labels, average="binary")
    
    # Compute adversarial success rate
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
    
    # If type information is available, compute metrics for each type
    if all_types:
        unique_types = set(all_types)
        for data_type in unique_types:
            type_indices = [i for i, t in enumerate(all_types) if t == data_type]
            type_preds = [all_preds[i] for i in type_indices]
            type_labels = [all_labels[i] for i in type_indices]
            
            # Compute metrics for each type
            type_accuracy = accuracy_metric.compute(predictions=type_preds, references=type_labels)
            type_precision = precision_metric.compute(predictions=type_preds, references=type_labels, average="binary")
            type_recall = recall_metric.compute(predictions=type_preds, references=type_labels, average="binary")
            type_f1 = f1_metric.compute(predictions=type_preds, references=type_labels, average="binary")
            
            # Compute type-specific adversarial success rate
            type_adv_success_rate = 1.0 - type_recall["recall"]
            
            results[data_type] = {
                "accuracy": type_accuracy["accuracy"],
                "precision": type_precision["precision"],
                "recall": type_recall["recall"],
                "f1": type_f1["f1"],
                "adv_success_rate": type_adv_success_rate,  # Add adversarial success rate
                "count": len(type_indices)
            }
    
    # Added: create detailed sample prediction results
    sample_predictions = []
    for i in range(len(all_preds)):
        sample_info = {
            "index": all_indices[i],
            "prediction": int(all_preds[i]),
            "prediction_label": "malicious" if all_preds[i] == 1 else "benign",
            "true_label": int(all_labels[i]),
            "true_label_text": "malicious" if all_labels[i] == 1 else "benign",
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
    """Predict a single text sample using specified model"""
    print(f"Predicting with {model_name} model...")
    
    # Tokenize the input text
    inputs = tokenizer(text, return_tensors="pt", truncation=True, padding="max_length", max_length=512)
    input_ids = inputs["input_ids"].to(device)
    
    if model_type == "transformer":
        attention_mask = inputs["attention_mask"].to(device)
        
        # Modified: check if local path exists, otherwise load from original model path
        local_model_path = f"./models/{model_name}" # Transformer models are still in the models directory
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
        # Modified: initialize the corresponding model class
        if model_name == "textcnn":
            model = TextCNNClassifier(**custom_config)
        elif model_name == "cnn_lstm":
            model = CNNLSTMClassifier(**custom_config)
        elif model_name == "dnn":
            model = DNNClassifier(**custom_config)
        elif model_name == "deeplog":
            # Ensure DeepLog exists in both main.py and test.py model configs, or neither
            # If the DeepLog model exists, uncomment the line below
            # model = DeepLog(**custom_config) 
            pass # If deeplog is not available, keep as is or add error handling

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

def main():
    parser = argparse.ArgumentParser(description="Test trained text classification models")
    parser.add_argument("--model", type=str, default=None, help="Specific model name to test (if not specified, all models will be tested)")
    parser.add_argument("--sample", type=str, default=None, help="Single sample text to predict")
    parser.add_argument("--dataset", type=str, default="../../dataset/generated_emails_20250523_160844.json", 
                      help="Test dataset path, defaults to baseline_generated_emails.json")
    args = parser.parse_args()
    
    # Load tokenizer
    tokenizer = AutoTokenizer.from_pretrained("bert-base-uncased")
    
    # Set up custom model configuration
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
    
    # Determine model configs to evaluate - default is to test all models
    if args.model is not None:
        selected_configs = [config for config in model_configs if config["name"] == args.model]
        if not selected_configs:
            print(f"Model not found: {args.model}")
            print(f"Available models: {', '.join([c['name'] for c in model_configs])}")
            return
    else:
        # Default: test all models
        selected_configs = model_configs
        print(f"Will test all models: {', '.join([c['name'] for c in model_configs])}")
    
    # If there is a single sample to predict
    if args.sample:
        predictions_made = False
        for config in selected_configs:
            # Modified: run prediction for specified model or all models
             pred_output = predict_sample(
                args.sample, 
                config["name"], 
                config["type"], 
                tokenizer,
                model_path=config.get("path"), # Use .get in case custom model has no path
                custom_config=custom_config if config["type"] == "custom" else None # Pass custom_config
            )
             # Check if predict_sample returned successfully
             if pred_output is not None:
                 pred_label, pred_text = pred_output
                 if pred_label is not None: # Further check if label is valid
                    print(f"{config['name']} prediction result: {pred_text} (label {pred_label})")
                    predictions_made = True
                 else:
                     print(f"{config['name']} prediction failed: {pred_text}") # Print error message

        if not predictions_made and args.model:
             print(f"Unable to predict for model '{args.model}' (may not be found or failed to load).")
        elif not predictions_made:
             print(f"Unable to predict for any selected model.")

        return
    
    # Otherwise, perform full test set evaluation
    # Load test dataset
    data_path = args.dataset
    
    # Extract dataset name (excluding path and extension)
    dataset_name = os.path.splitext(os.path.basename(data_path))[0]
    print(f"Using dataset: {dataset_name}")
    
    tokenized_dataset = load_dataset(data_path, tokenizer)
    # Use all data for testing, instead of splitting
    test_dataset = tokenized_dataset
    
    # Ensure the type field is included (if the dataset has this field)
    columns = ['input_ids', 'attention_mask', 'label']
    if 'type' in test_dataset.features:
        columns.append('type')
    if 'text' in test_dataset.features:  # Added: if original text field exists, include it too
        columns.append('text')
    
    test_dataset.set_format(type='torch', columns=columns)
    
    results = {}
    all_results_for_csv = [] # List for storing CSV data
    all_sample_predictions = {} # Added: for storing prediction results of each sample
    
    for config in selected_configs:
        print(f"\nEvaluating model: {config['name']}")
        
        # Check if model file exists
        model_file_exists = True
        if config["type"] == "custom":
            # Modified: check for .bin file in ./models directory
            model_path_check = f"./models/{config['name']}.bin"
            if not os.path.exists(model_path_check):
                print(f"Warning: model file '{model_path_check}' not found, skipping this model evaluation")
                model_file_exists = False
        elif config["type"] == "transformer":
            # Transformer model check remains unchanged (check ./models/ directory or huggingface path)
            model_dir_path = f"./models/{config['name']}"
            if not os.path.exists(model_dir_path) and not config.get("path"): # Check local and huggingface paths
                 print(f"Warning: Transformer model directory '{model_dir_path}' or specified path not found, skipping this model evaluation")
                 model_file_exists = False
            elif not os.path.exists(model_dir_path):
                 print(f"Warning: Transformer model directory '{model_dir_path}' not found, will try loading directly from {config['path']}")
                 # evaluate_transformer_model handles loading from config['path'] internally


        if not model_file_exists:
            continue

        metrics = None # Initialize metrics
        try:
            # Evaluate based on model type
            if config["type"] == "transformer":
                metrics = evaluate_transformer_model(
                    config["name"],
                    config["path"], # Pass original path
                    test_dataset
                )
            else: # Custom model
                test_dataloader = DataLoader(test_dataset, batch_size=32)
                
                # Modified: call updated evaluate_custom_model
                metrics = evaluate_custom_model(
                    config["class"],
                    config["name"],
                    test_dataloader,
                    custom_config
                )
            
            # Only process if metrics were successfully obtained
            if metrics is not None:
                results[config['name']] = metrics # Still keep the dict in case needed in the future

                # Added: save sample prediction results
                if "sample_predictions" in metrics:
                    all_sample_predictions[config['name']] = metrics["sample_predictions"]

                # Print and collect CSV results
                print(f"Model {config['name']} evaluation results on dataset {dataset_name}:")
                for data_type, type_metrics in metrics.items():
                    if data_type == "sample_predictions":  # Skip sample predictions, it's not a metric
                        continue
                        
                    is_overall = (data_type == "overall")
                    count = type_metrics.get("count", len(test_dataset) if is_overall else 'N/A') # Get sample count

                    # Print results
                    print(f"  Data type '{data_type}'" + (f" (sample count: {count})" if not is_overall else ""))
                    print(f"    Accuracy: {type_metrics['accuracy']:.4f}")
                    print(f"    Precision: {type_metrics['precision']:.4f}")
                    print(f"    Recall: {type_metrics['recall']:.4f}")
                    print(f"    F1 Score: {type_metrics['f1']:.4f}")
                    print(f"    Adversarial Success Rate: {type_metrics['adv_success_rate']:.4f}")  # Print adversarial success rate

                    # Add to CSV list
                    all_results_for_csv.append({
                        "model_name": config['name'],
                        "data_type": data_type,
                        "accuracy": f"{type_metrics['accuracy']:.4f}",
                        "precision": f"{type_metrics['precision']:.4f}",
                        "recall": f"{type_metrics['recall']:.4f}",
                        "f1": f"{type_metrics['f1']:.4f}",
                        "adv_success_rate": f"{type_metrics['adv_success_rate']:.4f}",
                        "count": count if not is_overall else len(test_dataset) # Ensure overall count is total sample count
                    })
            else:
                 print(f"Model {config['name']} evaluation failed or skipped.")

        except FileNotFoundError as e: # This FileNotFoundError should mainly be raised by evaluate_transformer_model
             print(f"Error evaluating Transformer model {config['name']}: {e}")
        except Exception as e:
             print(f"Unknown error evaluating model {config['name']}: {e}")
             import traceback
             traceback.print_exc() # Print detailed error stack trace


    # Ensure results directory exists
    results_dir = "./results"
    if not os.path.exists(results_dir):
        os.makedirs(results_dir)

    # Write all results to CSV file
    if all_results_for_csv:
        # Use dataset name as part of the filename
        csv_file_path = os.path.join(results_dir, f"{dataset_name}_model_results.csv")
        print(f"\nSaving all results to {csv_file_path}")
        
        fieldnames = ["dataset", "model_name", "data_type", "accuracy", "precision", "recall", "f1", "adv_success_rate", "count"]
        with open(csv_file_path, "w", newline='', encoding='utf-8') as csvfile:
            writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
            
            writer.writeheader()
            for row in all_results_for_csv:
                row["dataset"] = dataset_name  # Add dataset name
                writer.writerow(row)
        print("CSV file saved successfully.")
    else:
        print("\nNo evaluation results to save.")

    # Added: aggregate all model predictions and output per-sample json
    if all_sample_predictions:
        # 1. Get original text and true_label for all samples
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
            # 2. Add predictions from all models
            for model_name, preds in all_sample_predictions.items():
                if i < len(preds):
                    sample_obj[model_name] = preds[i]["prediction_label"]
            sample_json_list.append(sample_obj)
        # 3. Output to json file
        json_path = os.path.join(results_dir, f"{dataset_name}_all_model_predictions.json")
        with open(json_path, "w", encoding="utf-8") as f:
            json.dump(sample_json_list, f, ensure_ascii=False, indent=2)
        print(f"All model aggregated prediction results saved to: {json_path}")

if __name__ == "__main__":
    main() 
