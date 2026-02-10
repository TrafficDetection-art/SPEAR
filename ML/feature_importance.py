import sys; sys.path.insert(0, '..')
from project_settings import get

import joblib
import pandas as pd
import json
import numpy as np
import os
import ntpath
import matplotlib.pyplot as plt
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
import eli5
from eli5.lime import TextExplainer
from eli5.lime.samplers import MaskingTextSampler

# 1. Load trained vectorizer
vectorizer_path = get("ml.vectorizer_file", "./models/vectorizer.joblib")
vectorizer = joblib.load(vectorizer_path)

# 2. Load trained model - using LR model here as it is more interpretable
fi_model_name = get("ml.feature_importance.model_name", "LR")
model_path = os.path.join(get("ml.models_dir", "./models"), f"{fi_model_name}.joblib")
model = joblib.load(model_path)
print(f"Successfully loaded {fi_model_name} model")

# 3. Load dataset
data_path = get("ml.feature_importance.data_path", "../../dataset/responds.json")
dataset_filename = ntpath.basename(data_path)
dataset_name = os.path.splitext(dataset_filename)[0]
print(f"Using dataset: {dataset_name}")

with open(data_path, 'r', encoding='utf-8') as f:
    data = json.load(f)

# Convert JSON data to DataFrame
test_df = pd.DataFrame(data)

# Remove rows containing NaN
test_df = test_df.dropna(subset=['Text'])
if 'Class' in test_df.columns:
    test_df = test_df.dropna(subset=['Class'])

# 4. Get feature names (words)
feature_names = vectorizer.get_feature_names_out()

# 5. For LogisticRegression model, feature weights can be obtained directly
if isinstance(model, LogisticRegression):
    # Get coefficients (weights)
    coefficients = model.coef_[0]
    
    # Create DataFrame of feature names and weights
    features_df = pd.DataFrame({
        'feature': feature_names,
        'coefficient': coefficients
    })
    
    # Sort by contribution to class 1 (positive impact)
    positive_features = features_df.sort_values('coefficient', ascending=False)
    
    # Output top N words contributing most to class 1 prediction
    top_features_display = get("ml.feature_importance.top_features_display", 50)
    print("\nStrongest positive influence words for class 1 prediction:")
    print(positive_features.head(top_features_display).to_string(index=False))
    
    # Sort by contribution to class 0 (negative impact on class 1)
    negative_features = features_df.sort_values('coefficient', ascending=True)
    
    # Output top N words contributing most to class 0 (unfavorable for class 1)
    print("\nStrongest words for class 0 prediction (unfavorable for class 1):")
    print(negative_features.head(top_features_display).to_string(index=False))
    
    # Visualize contribution of top words
    fig_size = get("ml.feature_importance.figure_size", [12, 10])
    plt.figure(figsize=tuple(fig_size))
    
    # Select most important positive and negative features
    n_plot = get("ml.feature_importance.top_features_plot", 20)
    top_positive = positive_features.head(n_plot)
    top_negative = negative_features.head(n_plot)
    
    # Plot positive features
    plt.subplot(2, 1, 1)
    plt.barh(top_positive['feature'], top_positive['coefficient'], color='green')
    plt.title('Words with Highest Contribution to Class 1 Prediction')
    plt.xlabel('Coefficient Value (Weight)')
    plt.tight_layout()
    
    # Plot negative features
    plt.subplot(2, 1, 2)
    plt.barh(top_negative['feature'], top_negative['coefficient'], color='red')
    plt.title('Words with Highest Contribution to Class 0 Prediction')
    plt.xlabel('Coefficient Value (Weight)')
    plt.tight_layout()
    
    # Save visualization results
    os.makedirs('./results', exist_ok=True)
    plt.savefig(f'./results/feature_importance_{dataset_name}.png', dpi=get("ml.feature_importance.dpi", 300), bbox_inches='tight')
    print(f"\nFeature importance visualization saved to: ./results/feature_importance_{dataset_name}.png")
    
    # Save feature importance data to CSV
    features_df.to_csv(f'./results/feature_importance_{dataset_name}.csv', index=False)
    print(f"Feature importance data saved to: ./results/feature_importance_{dataset_name}.csv")

# 6. Explain prediction results for each specific text
def explain_prediction(text, true_class=None):
    """
    Explain model prediction for a specific text
    
    Args:
        text (str): Text to explain
        true_class: True class, if known
    """
    # Vectorize text
    text_tfidf = vectorizer.transform([text])
    
    # Make prediction
    prediction = model.predict(text_tfidf)[0]
    
    # Get prediction probabilities (if model supports)
    if hasattr(model, 'predict_proba'):
        proba = model.predict_proba(text_tfidf)[0]
        print(f"Predicted class: {prediction}, probability: class_0={proba[0]:.4f}, class_1={proba[1]:.4f}")
    else:
        print(f"Predicted class: {prediction}")
    
    if true_class is not None:
        print(f"True class: {true_class}")
    
    # For LogisticRegression model, we can directly use eli5 for explanation
    if isinstance(model, LogisticRegression):
        # Use eli5 to explain feature contributions
        explanation = eli5.explain_prediction(model, text, vec=vectorizer, 
                                              target_names=['class_0', 'class_1'])
        print(eli5.format_as_text(explanation))
    
    return prediction

# 7. Analyze samples predicted as class 1, find common features
def analyze_class_1_samples():
    """Analyze all samples predicted as class 1, find their common features"""
    # Prepare test data
    X_test = test_df['Text']
    X_test_tfidf = vectorizer.transform(X_test)
    
    # Make prediction
    predictions = model.predict(X_test_tfidf)
    
    # Get samples predicted as class 1
    class_1_indices = np.where(predictions == 1)[0]
    class_1_samples = test_df.iloc[class_1_indices]
    
    print(f"\nTotal {len(class_1_samples)} samples predicted as class 1")
    
    # If true labels exist, calculate accuracy
    if 'Class' in test_df.columns:
        true_positives = class_1_samples[class_1_samples['Class'] == 1]
        false_positives = class_1_samples[class_1_samples['Class'] == 0]
        print(f"True positives (truly class 1): {len(true_positives)}")
        print(f"False positives (truly class 0): {len(false_positives)}")
    
    # Randomly select samples for detailed explanation
    num_samples = get("ml.feature_importance.num_samples_to_explain", 10)
    if len(class_1_samples) > num_samples:
        samples_to_explain = class_1_samples.sample(num_samples)
    else:
        samples_to_explain = class_1_samples
    
    text_preview_length = get("ml.feature_importance.text_preview_length", 200)
    print("\nExplanation of randomly selected samples predicted as class 1:")
    for idx, row in samples_to_explain.iterrows():
        print("\n" + "="*50)
        print(f"Sample ID: {idx}")
        print(f"Text: {row['Text'][:text_preview_length]}...")  # Only show first N characters
        
        # Explain prediction
        true_class = row['Class'] if 'Class' in row else None
        explain_prediction(row['Text'], true_class)
    
    return class_1_samples

# 8. Use LIME to explain complex model predictions
def explain_with_lime(text, n_samples=None):
    """
    Use LIME to explain model prediction for a specific text
    
    Args:
        text (str): Text to explain
        n_samples (int): Number of LIME samples
    """
    if n_samples is None:
        n_samples = get("ml.feature_importance.lime_n_samples", 5000)
    
    # Create prediction function
    def predict_proba(texts):
        vectorized_texts = vectorizer.transform(texts)
        return model.predict_proba(vectorized_texts)
    
    # Initialize LIME text explainer
    sampler = MaskingTextSampler(
        replacement=get("ml.feature_importance.lime_replacement", "UNK"),
        max_replace=get("ml.feature_importance.lime_max_replace", 0.7),
        min_replace=get("ml.feature_importance.lime_min_replace", 0),
        token_pattern=get("ml.feature_importance.lime_token_pattern", r"(?u)\b\w\w+\b")
    )
    
    explainer = TextExplainer(
        sampler=sampler,
        n_samples=n_samples,
        random_state=get("general.random_seed", 42)
    )
    
    # Fit explainer
    explainer.fit(text, predict_proba)
    
    # Get explanation
    explanation = explainer.explain_prediction(target_names=['class_0', 'class_1'])
    
    # Display explanation results
    print("\nLIME explanation:")
    print(eli5.format_as_text(explanation))
    
    return explanation

# Main entry point
if __name__ == "__main__":
    # 1. Analyze global feature importance
    print("\nGlobal feature importance analysis completed")
    
    # 2. Analyze samples predicted as class 1
    class_1_samples = analyze_class_1_samples()
    
    # 3. Interactive interface for explaining specific texts
    print("\n" + "="*50)
    print("Do you want to explain a specific text? (y/n)")
    choice = input().strip().lower()
    
    if choice == 'y':
        print("Please enter the text to explain:")
        text = input()
        prediction = explain_prediction(text)
        
        if prediction == 1:
            print("\nUse LIME for more detailed explanation? (y/n)")
            lime_choice = input().strip().lower()
            if lime_choice == 'y':
                explain_with_lime(text)
