"""
ML model testing script (variant 2).
Functionally identical to test_body.py - use test_body.py with --data_path argument instead.
This file is kept for backward compatibility.
"""
import sys; sys.path.insert(0, '..')
from project_settings import get

import argparse
import os
import ntpath
import json
import pandas as pd

# Reuse test_body functions
from test_body import load_models_and_vectorizer, evaluate_models


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="ML Model Testing (variant 2)")
    parser.add_argument("--data_path", type=str, default=None, help="Test data JSON path")
    parser.add_argument("--models_dir", type=str, default=None, help="Models directory")
    parser.add_argument("--vectorizer_path", type=str, default=None, help="Vectorizer path")
    parser.add_argument("--results_dir", type=str, default=None, help="Results directory")
    args = parser.parse_args()
    
    data_path = args.data_path
    if data_path is None:
        print("Error: --data_path is required")
        print("Usage: python test_body2.py --data_path <path_to_test_data.json>")
        exit(1)
    
    vectorizer, models = load_models_and_vectorizer(args.models_dir, args.vectorizer_path)
    
    with open(data_path, 'r', encoding='utf-8') as f:
        data = json.load(f)
    test_df = pd.DataFrame(data)
    test_df = test_df.dropna(subset=['Text'])
    
    dataset_name = os.path.splitext(ntpath.basename(data_path))[0]
    print(f"Dataset: {dataset_name}, size: {len(test_df)}")
    
    evaluate_models(test_df, vectorizer, models, dataset_name, args.results_dir)
