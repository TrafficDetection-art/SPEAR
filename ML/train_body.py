import sys; sys.path.insert(0, '..')
from project_settings import settings, get

import argparse
import json
import joblib
import os
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.naive_bayes import MultinomialNB
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score


def main(args):
    # 1. Load JSON data
    data_path = args.data_path or get("paths.train_data_file", "../dataset/email_data.json")
    print(f"Loading data from: {data_path}")
    with open(data_path, 'r', encoding='utf-8') as f:
        data = json.load(f)

    data_df = pd.DataFrame(data)
    data_df = data_df.dropna(subset=['Text'])

    # 2. Split data
    X = data_df['Text']
    y = data_df['Class']

    test_size = args.test_size or get("ml.training.test_size", 0.2)
    random_state = args.random_state or get("ml.training.random_state", 42)
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=random_state
    )

    # 3. Create and train TfidfVectorizer
    max_features = args.max_features or get("ml.training.max_features", 5000)
    vectorizer = TfidfVectorizer(max_features=max_features)
    X_train_tfidf = vectorizer.fit_transform(X_train)
    X_test_tfidf = vectorizer.transform(X_test)

    # 4. Build model dict from config
    model_names = get("ml.model_names", ["LR", "RFC", "NB", "SVC"])
    model_params = get("ml.model_params", {})
    
    model_constructors = {
        "LR": lambda p: LogisticRegression(**p),
        "RFC": lambda p: RandomForestClassifier(**p),
        "NB": lambda p: MultinomialNB(**p),
        "SVC": lambda p: SVC(**p),
    }
    
    models_dir = args.models_dir or get("ml.models_dir", "./models")
    os.makedirs(models_dir, exist_ok=True)

    for name in model_names:
        params = model_params.get(name, {})
        if name not in model_constructors:
            print(f"Warning: Unknown model {name}, skipping")
            continue
        
        model = model_constructors[name](params)
        model.fit(X_train_tfidf, y_train)

        # 5. Evaluate
        y_pred = model.predict(X_test_tfidf)
        print(f'{name} accuracy: {accuracy_score(y_test, y_pred):.4f}')
        print(f'{name} precision: {precision_score(y_test, y_pred):.4f}')
        print(f'{name} recall: {recall_score(y_test, y_pred):.4f}')
        print(f'{name} F1: {f1_score(y_test, y_pred):.4f}')

        # 6. Save model
        model_path = os.path.join(models_dir, f'{name}.joblib')
        joblib.dump(model, model_path)
        print(f"Model saved to: {model_path}")

    # Save vectorizer
    vectorizer_path = args.vectorizer_path or get("ml.vectorizer_file", os.path.join(models_dir, "vectorizer.joblib"))
    joblib.dump(vectorizer, vectorizer_path)
    print(f"Vectorizer saved to: {vectorizer_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="ML Model Training")
    parser.add_argument("--data_path", type=str, default=None, help="Training data path")
    parser.add_argument("--models_dir", type=str, default=None, help="Models output directory")
    parser.add_argument("--vectorizer_path", type=str, default=None, help="Vectorizer save path")
    parser.add_argument("--test_size", type=float, default=None, help="Test split ratio")
    parser.add_argument("--random_state", type=int, default=None, help="Random state")
    parser.add_argument("--max_features", type=int, default=None, help="TF-IDF max features")
    args = parser.parse_args()
    main(args)
