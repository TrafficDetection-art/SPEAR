import sys; sys.path.insert(0, '..')
from project_settings import settings, get

import argparse
import joblib
import pandas as pd
import json
import numpy as np
import os
import ntpath
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score,
    f1_score, roc_auc_score, confusion_matrix
)


def load_models_and_vectorizer(models_dir=None, vectorizer_path=None, model_names=None):
    """Load vectorizer and trained models."""
    models_dir = models_dir or get("ml.models_dir", "./models")
    vectorizer_path = vectorizer_path or get("ml.vectorizer_file", os.path.join(models_dir, "vectorizer.joblib"))
    model_names = model_names or get("ml.model_names", ["LR", "RFC", "NB", "SVC"])
    
    vectorizer = joblib.load(vectorizer_path)
    models = {}
    for name in model_names:
        path = os.path.join(models_dir, f'{name}.joblib')
        try:
            models[name] = joblib.load(path)
            print(f"Successfully loaded {name} model")
        except FileNotFoundError:
            print(f"{name} model file not found at {path}")
    return vectorizer, models


def evaluate_models(test_df, vectorizer, models, dataset_name, results_dir=None):
    """Evaluate all models and save results."""
    results_dir = results_dir or get("ml.results_dir", "./results")
    os.makedirs(results_dir, exist_ok=True)
    
    X_test = test_df['Text']
    X_test_tfidf = vectorizer.transform(X_test)
    
    if 'Class' not in test_df.columns:
        # No labels - just predict
        for model_name, model in models.items():
            test_df[f'prediction_{model_name}'] = model.predict(X_test_tfidf)
            print(f'{model_name} prediction completed')
        pred_file = os.path.join(results_dir, f'prediction_results_{dataset_name}.csv')
        test_df.to_csv(pred_file, index=False)
        print(f"Prediction results saved to: {pred_file}")
        return
    
    # Remove NaN Class rows
    rows_before = len(test_df)
    test_df = test_df.dropna(subset=['Class'])
    if rows_before > len(test_df):
        print(f"Warning: removed {rows_before - len(test_df)} rows with NaN Class")
        X_test = test_df['Text']
        X_test_tfidf = vectorizer.transform(X_test)
    
    y_test = test_df['Class']
    unique_classes = sorted(y_test.unique())
    results_data = []
    
    print(f"Final data size: {len(test_df)}")
    print("Model performance evaluation:")
    print("-" * 50)
    
    for model_name, model in models.items():
        print(f"\n{model_name} model evaluation:")
        y_pred = model.predict(X_test_tfidf)
        
        accuracy = accuracy_score(y_test, y_pred)
        binary_precision = precision_score(y_test, y_pred, average='binary', zero_division=0)
        binary_recall = recall_score(y_test, y_pred, average='binary', zero_division=0)
        binary_f1 = f1_score(y_test, y_pred, average='binary', zero_division=0)
        asr = 1 - binary_recall
        
        print(f'  Accuracy: {accuracy:.4f}')
        print(f'  Precision: {binary_precision:.4f}, Recall: {binary_recall:.4f}, F1: {binary_f1:.4f}')
        print(f'  ASR: {asr:.4f}')
        
        result_row = {
            'Model': model_name, 'Dataset': dataset_name,
            'Accuracy': accuracy, 'Binary_Precision': binary_precision,
            'Binary_Recall': binary_recall, 'Binary_F1': binary_f1,
            'Adversarial_Success_Rate': asr, 'AUC': None,
        }
        
        if hasattr(model, 'predict_proba'):
            try:
                y_score = model.predict_proba(X_test_tfidf)[:, 1]
                result_row['AUC'] = roc_auc_score(y_test, y_score)
                print(f'  AUC: {result_row["AUC"]:.4f}')
            except Exception:
                print('  Unable to calculate AUC')
        
        # Per-class metrics
        for i, cl in enumerate(unique_classes):
            result_row[f'Precision_{cl}'] = precision_score(y_test, y_pred, labels=unique_classes, average=None, zero_division=0)[i]
            result_row[f'Recall_{cl}'] = recall_score(y_test, y_pred, labels=unique_classes, average=None, zero_division=0)[i]
            result_row[f'F1_{cl}'] = f1_score(y_test, y_pred, labels=unique_classes, average=None, zero_division=0)[i]
        
        results_data.append(result_row)
        
        # Confusion matrix
        cm = confusion_matrix(y_test, y_pred, labels=unique_classes)
        print("\n  Confusion matrix:")
        print(f'  {"True/Pred":<10}', end='')
        for cl in unique_classes:
            print(f'{cl:<8}', end='')
        print()
        for i, cl in enumerate(unique_classes):
            print(f'  {cl:<10}', end='')
            for j in range(len(unique_classes)):
                print(f'{cm[i, j]:<8d}', end='')
            print()
        
        # By-type evaluation
        if 'type' in test_df.columns:
            print("\n  By-type evaluation:")
            for dtype in sorted(test_df['type'].unique()):
                df_type = test_df[test_df['type'] == dtype]
                X_type = vectorizer.transform(df_type['Text'])
                y_type = df_type['Class']
                y_pred_type = model.predict(X_type)
                tp = precision_score(y_type, y_pred_type, average='binary', zero_division=0)
                tr = recall_score(y_type, y_pred_type, average='binary', zero_division=0)
                tf = f1_score(y_type, y_pred_type, average='binary', zero_division=0)
                print(f'    {dtype} ({len(df_type)}): P={tp:.4f} R={tr:.4f} F1={tf:.4f} ASR={1-tr:.4f}')
                results_data.append({
                    'Model': model_name, 'Dataset': dataset_name, 'Data_Type': dtype,
                    'Sample_Count': len(df_type), 'Accuracy': accuracy_score(y_type, y_pred_type),
                    'Binary_Precision': tp, 'Binary_Recall': tr, 'Binary_F1': tf,
                    'Adversarial_Success_Rate': 1 - tr,
                })
    
    results_df = pd.DataFrame(results_data)
    results_file = os.path.join(results_dir, f'model_evaluation_results_{dataset_name}.csv')
    results_df.to_csv(results_file, index=False)
    print(f"\nResults saved to: {results_file}")


def predict_text(text, vectorizer, models, model_name=None):
    """Predict a single text."""
    model_name = model_name or get("ml.model_names", ["LR"])[0]
    if model_name not in models:
        raise ValueError(f"Model {model_name} not found. Available: {list(models.keys())}")
    return models[model_name].predict(vectorizer.transform([text]))[0]


def predict_with_voting(text, vectorizer, models):
    """Multi-model voting prediction."""
    predictions = [m.predict(vectorizer.transform([text]))[0] for m in models.values()]
    unique_preds, counts = np.unique(predictions, return_counts=True)
    return unique_preds[np.argmax(counts)], predictions


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="ML Model Testing")
    parser.add_argument("--data_path", type=str, required=True, help="Test data JSON path")
    parser.add_argument("--models_dir", type=str, default=None, help="Models directory")
    parser.add_argument("--vectorizer_path", type=str, default=None, help="Vectorizer path")
    parser.add_argument("--results_dir", type=str, default=None, help="Results output directory")
    args = parser.parse_args()
    
    vectorizer, models = load_models_and_vectorizer(args.models_dir, args.vectorizer_path)
    
    with open(args.data_path, 'r', encoding='utf-8') as f:
        data = json.load(f)
    test_df = pd.DataFrame(data)
    test_df = test_df.dropna(subset=['Text'])
    
    dataset_name = os.path.splitext(ntpath.basename(args.data_path))[0]
    print(f"Dataset: {dataset_name}, size: {len(test_df)}")
    
    evaluate_models(test_df, vectorizer, models, dataset_name, args.results_dir)
