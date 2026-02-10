"""
Text preprocessing and validation utilities.

Provides text cleaning, validation, fallback analysis, and result validation
for LIME-based email analysis.
"""

import sys; sys.path.insert(0, '..')
from project_settings import get as pget

import re


def clean_text_for_tokenization(text):
    """Clean text to avoid tokenizer errors."""
    if not isinstance(text, str):
        text = str(text)

    # 1. Remove control characters
    text = re.sub(r'[\x00-\x08\x0B\x0C\x0E-\x1F\x7F-\x9F]', '', text)
    # 2. Handle special Unicode characters
    text = text.encode('utf-8', errors='ignore').decode('utf-8')
    # 3. Collapse multiple whitespace characters
    text = re.sub(r'\s+', ' ', text)
    # 4. Strip leading/trailing whitespace
    text = text.strip()
    # 5. Ensure text is not empty
    if not text:
        text = "empty text"

    return text


def validate_text_for_lime(text):
    """
    Validate whether text is suitable for LIME analysis.

    Returns:
        (is_valid: bool, message: str)
    """
    try:
        if not text or not isinstance(text, str):
            return False, "Text is empty or not a string"

        min_len = pget("spear.text_validation.min_text_length", 5)
        if len(text.strip()) < min_len:
            return False, "Text is too short"

        max_special = pget("spear.text_validation.max_special_char_ratio", 0.5)
        special_char_ratio = len(re.findall(r'[^\w\s]', text)) / len(text)
        if special_char_ratio > max_special:
            return False, f"Special character ratio too high: {special_char_ratio:.2f}"

        min_ascii = pget("spear.text_validation.min_ascii_ratio", 0.3)
        ascii_ratio = len(text.encode('ascii', errors='ignore')) / len(text.encode('utf-8'))
        if ascii_ratio < min_ascii:
            return False, f"ASCII character ratio too low: {ascii_ratio:.2f}"

        min_words = pget("spear.text_validation.min_word_count", 3)
        words = re.findall(r'\b\w+\b', text)
        if len(words) < min_words:
            return False, f"Too few words: {len(words)}"

        return True, "Text validation passed"

    except Exception as e:
        return False, f"Validation error: {e}"


def safe_tokenize_text(text, max_length=None):
    """Safely clean and optionally truncate text for tokenization."""
    try:
        if max_length is None:
            max_length = pget("spear.text_validation.max_token_length", 512)
        if text is None:
            print("Warning: Input text is None, using default text")
            text = "empty text content"

        cleaned_text = clean_text_for_tokenization(text)

        chars_per_token = pget("spear.text_validation.chars_per_token", 4)
        if len(cleaned_text) > max_length * chars_per_token:
            cleaned_text = cleaned_text[:max_length * chars_per_token]

        return cleaned_text
    except Exception as e:
        print(f"Text cleaning error: {e}")
        return "text cleaning failed"


def create_fallback_analysis_result(email_content):
    """Create a fallback analysis result using simple keyword rules."""
    if email_content is None:
        email_content = "empty email content"
        print("Warning: Email content is None, using default text")

    phishing_keywords = [
        'urgent', 'verify', 'click', 'account', 'suspended', 'expire',
        'confirm', 'update', 'security', 'alert', 'warning', 'action',
        'required', 'immediately', 'login', 'password', 'refund',
    ]
    legitimate_keywords = [
        'thank', 'welcome', 'information', 'service', 'company',
        'team', 'support', 'help', 'contact', 'regards', 'sincerely',
    ]

    text_lower = email_content.lower()

    phishing_score = sum(1 for kw in phishing_keywords if kw in text_lower)
    legitimate_score = sum(1 for kw in legitimate_keywords if kw in text_lower)

    is_phishing = phishing_score > legitimate_score
    confidence = max(phishing_score, legitimate_score) / max(
        len(phishing_keywords), len(legitimate_keywords)
    )

    found_phishing = [kw for kw in phishing_keywords if kw in text_lower]
    found_legitimate = [kw for kw in legitimate_keywords if kw in text_lower]

    return {
        "is_phishing": is_phishing,
        "confidence": confidence,
        "label": "phishing" if is_phishing else "legitimate",
        "high_weight_phishing_words": ', '.join(found_phishing),
        "legitimate_words": ', '.join(found_legitimate),
        "raw_explanation": {
            "phishing_words": [(word, 0.01) for word in found_phishing],
            "legitimate_words": [(word, 0.01) for word in found_legitimate],
        },
        "detailed_ngrams": {
            "phishing_unigrams": [(word, 0.01) for word in found_phishing],
            "normal_unigrams": [(word, 0.01) for word in found_legitimate],
            "phishing_bigrams": [],
            "normal_bigrams": [],
        },
        "bigrams": [],
        "fallback_used": True,
    }


def validate_analysis_result(analysis_result):
    """Validate and fill in missing fields of a LIME analysis result."""
    required_fields = {
        "is_phishing": False,
        "confidence": 0.0,
        "label": "unknown",
        "high_weight_phishing_words": "",
        "legitimate_words": "",
        "raw_explanation": None,
        "bigrams": [],
    }

    for field, default_value in required_fields.items():
        if field not in analysis_result:
            analysis_result[field] = default_value

    # Ensure detailed_ngrams structure is complete
    if "detailed_ngrams" not in analysis_result:
        analysis_result["detailed_ngrams"] = {}

    ngram_fields = [
        "phishing_unigrams", "normal_unigrams",
        "phishing_bigrams", "normal_bigrams",
    ]
    for field in ngram_fields:
        if field not in analysis_result["detailed_ngrams"]:
            analysis_result["detailed_ngrams"][field] = []

    # Ensure string fields are not None
    if analysis_result["high_weight_phishing_words"] is None:
        analysis_result["high_weight_phishing_words"] = ""
    if analysis_result["legitimate_words"] is None:
        analysis_result["legitimate_words"] = ""

    return analysis_result
