"""
LIME-based adversarial attack module.

Provides LIME email analysis, adversarial email generation,
LIME-based text preprocessing, and multi-model adversarial evasion.
"""

import sys; sys.path.insert(0, '..')
from project_settings import get as pget

import re
import random

from text_utils import (
    safe_tokenize_text,
    validate_text_for_lime,
    create_fallback_analysis_result,
    validate_analysis_result,
    clean_text_for_tokenization,
)
from memory import lime_memory, update_memory, get_memory_context
from llm_client import get_LLM_response_vllm
from email_ops import detect_email
from smart_mapping import get_smart_replacement_map
from lime_analyzer import LimeAnalyzer, get_all_model_configs
from my_prompt.lime_adversarial_prompt import lime_adversarial_prompt


# ========================== LIME Analysis ==========================

def lime_analyze_email(email_content, model_name="bert"):
    """
    Analyze email content with LIME to find key feature words.

    Args:
        email_content: Raw email text.
        model_name: Name of the ML model to use for LIME analysis.

    Returns:
        dict: Analysis result with phishing/legitimate words and confidence.
    """
    print(f"Running LIME analysis with {model_name} model...")

    # Preprocess email
    try:
        cleaned_email = safe_tokenize_text(email_content)
        print(f"Original email length: {len(email_content)}, cleaned length: {len(cleaned_email)}")

        is_valid, msg = validate_text_for_lime(cleaned_email)
        if not is_valid:
            print(f"Text validation failed: {msg}")
            return create_fallback_analysis_result(cleaned_email)
        print(f"Text validation passed: {msg}")

    except Exception as e:
        print(f"Email preprocessing failed: {e}")
        return create_fallback_analysis_result(email_content)

    # Get model configuration
    try:
        model_configs = get_all_model_configs()
    except Exception as e:
        print(f"Failed to get model configs: {e}")
        return create_fallback_analysis_result(cleaned_email)

    model_config = None
    for cfg in model_configs:
        if cfg["name"].lower() == model_name.lower():
            model_config = cfg
            break

    if not model_config:
        print(f"Warning: {model_name} config not found, using default TextCNN")
        model_config = {"name": "textcnn", "path": None, "type": "custom"}

    # Run LIME analysis
    try:
        analyzer = LimeAnalyzer(model_config)
        analysis_result = analyzer.analyze_email(cleaned_email)
        analysis_result = validate_analysis_result(analysis_result)

        print(f"LIME result: classified as '{analysis_result['label']}', "
              f"confidence: {analysis_result['confidence']:.4f}")
        print("High-weight phishing features:")
        print(analysis_result["high_weight_phishing_words"])
        print("High-weight legitimate features:")
        print(analysis_result["legitimate_words"])

        # Update memory
        update_memory(analysis_result, score_threshold=pget("spear.lime.score_threshold", 0.02))

        return analysis_result

    except Exception as e:
        print(f"LIME analysis error: {e}")
        import traceback
        traceback.print_exc()
        return create_fallback_analysis_result(cleaned_email)


# ========================== Adversarial Email Generation ==========================

def generate_lime_adversarial_email(generation_client, email_content, lime_analysis):
    """
    Generate an adversarial phishing email based on LIME analysis (attack side).

    Args:
        generation_client: Attack-side LLM client.
        email_content: Current email content.
        lime_analysis: LIME analysis result dict.

    Returns:
        str: Generated adversarial email.
    """
    print("Generating adversarial phishing email using LIME analysis...")

    memory_context = get_memory_context(top_k=pget("spear.lime.memory_top_k", 10))

    base_prompt = lime_adversarial_prompt.format(
        email_content=email_content,
        high_weight_phishing_words=lime_analysis["high_weight_phishing_words"],
        legitimate_words=lime_analysis["legitimate_words"],
    )

    if memory_context:
        enhanced_prompt = base_prompt + memory_context
        print("Added memory context to generation prompt")
    else:
        enhanced_prompt = base_prompt

    print("Prompt:-------------------\n", enhanced_prompt)

    adversarial_email = get_LLM_response_vllm(
        generation_client, enhanced_prompt,
        function_name="lime_adversarial_generation", role="attack",
    )

    return adversarial_email


# ========================== LIME Preprocessing ==========================

def preprocess_email_with_lime(email_content, analysis_result,
                               strategy="delete", threshold=0.02):
    """
    Preprocess email text based on LIME analysis before LLM generation.

    Args:
        email_content: Raw email content.
        analysis_result: LIME analysis result dict.
        strategy: Processing strategy - "replace", "delete", or "char_replace".
        threshold: Score threshold; only process words above this value.

    Returns:
        str: Preprocessed email content.
    """
    print(f"Starting LIME preprocessing, strategy: {strategy}, threshold: {threshold}")
    print("Goal: replace phishing words with legitimate words to reduce detection probability")

    try:
        cleaned_email = clean_text_for_tokenization(email_content)

        # Extract high-weight phishing feature words
        high_score_phishing_words = _extract_phishing_words(analysis_result, threshold)

        if not high_score_phishing_words:
            print("No phishing words found for preprocessing, returning original text")
            return cleaned_email

        print(f"High-weight phishing words to replace: {high_score_phishing_words}")

        processed_email = cleaned_email

        if strategy == "delete":
            processed_email = _strategy_delete(processed_email, high_score_phishing_words)
        elif strategy == "replace":
            processed_email = _strategy_replace(processed_email, high_score_phishing_words)
        elif strategy == "char_replace":
            processed_email = _strategy_char_replace(processed_email, high_score_phishing_words)

        print(f"Preprocessing complete. Original length: {len(cleaned_email)}, "
              f"new length: {len(processed_email)}")
        print(f"Preprocessed email snippet: {processed_email[:150]}...")

        return processed_email

    except Exception as e:
        print(f"Preprocessing error: {e}")
        import traceback
        traceback.print_exc()
        try:
            return clean_text_for_tokenization(email_content)
        except Exception:
            return email_content


def _extract_phishing_words(analysis_result, threshold):
    """Extract high-weight phishing words from analysis result."""
    words = []

    if not analysis_result or not isinstance(analysis_result, dict):
        print("Warning: Invalid analysis result, skipping preprocessing")
        return words

    # Try extracting from raw_explanation
    raw_explanation = analysis_result.get('raw_explanation', {})
    if isinstance(raw_explanation, dict):
        phishing_items = raw_explanation.get('phishing_words', [])
        if isinstance(phishing_items, list):
            for item in phishing_items:
                try:
                    if isinstance(item, (list, tuple)) and len(item) >= 2:
                        word, score = item[0], item[1]
                        if (isinstance(word, str)
                                and isinstance(score, (int, float))
                                and score > threshold):
                            words.append(word)
                            print(f"    Added high-weight phishing word: {word} (score: {score:.3f})")
                except (IndexError, TypeError, ValueError) as e:
                    print(f"Error processing word item: {item}, error: {e}")

    # Fallback: try from high_weight_phishing_words string
    if not words:
        phishing_str = analysis_result.get('high_weight_phishing_words', '')
        if isinstance(phishing_str, str) and phishing_str.strip():
            extracted = re.findall(r'\b\w+\b', phishing_str.lower())
            words = [w for w in extracted if len(w) > 2]

    return words


def _strategy_delete(email, phishing_words):
    """Delete phishing words from email."""
    print("Strategy: delete phishing words")
    for word in phishing_words:
        try:
            pattern = r'\b' + re.escape(word) + r'\b'
            email = re.sub(pattern, '', email, flags=re.IGNORECASE)
            print(f"  Deleted phishing word: {word}")
        except re.error as e:
            print(f"Error deleting word '{word}': {e}")
    return re.sub(r'\s+', ' ', email).strip()


def _strategy_replace(email, phishing_words):
    """Replace phishing words with legitimate words."""
    print("Strategy: replace phishing words with legitimate words")

    smart_mapping = get_smart_replacement_map()

    # Get legitimate words from memory for replacement
    memory_legitimate_words = []
    if lime_memory and lime_memory.get("legitimate_words"):
        sorted_legitimate = sorted(
            lime_memory["legitimate_words"].items(),
            key=lambda x: x[1], reverse=True,
        )
        memory_legitimate_words = [w for w, c in sorted_legitimate if c >= 1][:20]
        print(f"Legitimate words from memory: {memory_legitimate_words}")

    print(f"Available smart mappings: {len(smart_mapping)}")

    for phishing_word in phishing_words:
        try:
            replacement = None

            # 1. Try smart mapping table
            if phishing_word.lower() in smart_mapping:
                replacement = smart_mapping[phishing_word.lower()]
                print(f"  Smart mapping: {phishing_word} -> {replacement}")

            # 2. Try memory-based similar-length matching
            elif memory_legitimate_words:
                word_len = len(phishing_word)
                similar_words = [
                    w for w in memory_legitimate_words
                    if abs(len(w) - word_len) <= 2 and w.lower() != phishing_word.lower()
                ]
                if similar_words:
                    replacement = similar_words[0]
                    print(f"  Memory mapping: {phishing_word} -> {replacement}")
                elif memory_legitimate_words:
                    replacement = memory_legitimate_words[0]
                    print(f"  Default mapping: {phishing_word} -> {replacement}")

            # 3. Apply replacement
            if replacement:
                if phishing_word.isupper():
                    replacement = replacement.upper()
                elif phishing_word.istitle():
                    replacement = replacement.capitalize()
                pattern = r'\b' + re.escape(phishing_word) + r'\b'
                email = re.sub(pattern, replacement, email, flags=re.IGNORECASE)
                print(f"  Replaced: {phishing_word} -> {replacement}")
            else:
                print(f"  No replacement found, deleting: {phishing_word}")
                pattern = r'\b' + re.escape(phishing_word) + r'\b'
                email = re.sub(pattern, '', email, flags=re.IGNORECASE)

        except re.error as e:
            print(f"Error replacing word '{phishing_word}': {e}")

    return re.sub(r'\s+', ' ', email).strip()


def _strategy_char_replace(email, phishing_words):
    """Replace characters in phishing words with visually similar Unicode characters."""
    print("Strategy: character-level obfuscation for phishing words")

    char_substitutions = {
        'a': ['\u0430'],         # Cyrillic a
        'e': ['\u0435'],         # Cyrillic e
        'o': ['\u043e', '\u03bf'],  # Cyrillic o, Greek o
        'i': ['\u0456'],         # Cyrillic i
        'p': ['\u0440'],         # Cyrillic p
        'c': ['\u0441'],         # Cyrillic c
        'h': ['\u04bb'],         # Cyrillic h
        'x': ['\u0445'],         # Cyrillic x
        'y': ['\u0443'],         # Cyrillic y
        's': ['\u0455'],         # Cyrillic s
        'f': ['\u0192'],         # Latin f with hook
        'w': ['\u051d'],         # Cyrillic w
    }

    for phishing_word in phishing_words:
        try:
            confused_word = ""
            for char in phishing_word.lower():
                if char in char_substitutions:
                    confused_word += random.choice(char_substitutions[char])
                    print(f"    Char replace: {char} -> {confused_word[-1]}")
                else:
                    confused_word += char

            if phishing_word.isupper():
                confused_word = confused_word.upper()
            elif phishing_word.istitle():
                confused_word = confused_word.capitalize()

            pattern = r'\b' + re.escape(phishing_word) + r'\b'
            email = re.sub(pattern, confused_word, email, flags=re.IGNORECASE)
            print(f"  Char obfuscation: {phishing_word} -> {confused_word}")

        except Exception as e:
            print(f"Error obfuscating word '{phishing_word}': {e}")

    return email


# ========================== Multi-Model Adversarial Evasion ==========================

def adversarial_ml_model_evasion(generation_client, recheck_client, email_content,
                                 ml_model_names=None, max_iterations_per_model=5,
                                 use_lime_preprocessing=True):
    """
    Run adversarial attacks against multiple ML models.

    Args:
        generation_client: Attack-side LLM client.
        recheck_client: Defense-side LLM client.
        email_content: Original email content.
        ml_model_names: List of ML model names to attack.
        max_iterations_per_model: Max iterations per model.
        use_lime_preprocessing: Whether to preprocess with LIME before LLM generation.

    Returns:
        (current_email, final_prediction, detection_result, evasion_results, all_generated_emails)
    """
    if ml_model_names is None:
        ml_model_names = pget("spear.lime.default_models", ["textcnn"])

    print("Starting adversarial attack against ML models...")
    print(f"LIME preprocessing enabled: {use_lime_preprocessing}")

    all_analysis_results = {}
    current_email = email_content
    evasion_results = {}

    all_generated_emails = {
        "original_email": email_content,
        "models": {},
    }

    for model_name in ml_model_names:
        print(f"\n===== Adversarial attack on {model_name} model =====")

        all_generated_emails["models"][model_name] = {
            "iterations": {},
            "success_iteration": None,
            "final_email": None,
        }

        # Analyze current email
        analysis_result = lime_analyze_email(current_email, model_name)
        all_analysis_results[model_name] = analysis_result

        print("analysis_result:", analysis_result)

        # If already classified as benign, no need to attack
        if not analysis_result["is_phishing"]:
            print(f"{model_name} already classified email as benign, no attack needed")
            evasion_results[model_name] = {
                "success": True,
                "needed_evasion": False,
                "email": current_email,
                "iterations": 0,
            }
            for i in range(1, max_iterations_per_model + 1):
                all_generated_emails["models"][model_name]["iterations"][f"iteration_{i}"] = {
                    "email": current_email,
                    "preprocessed_email": None,
                    "input_email": current_email,
                    "success": True,
                    "preprocessing_used": False,
                    "is_original": True,
                }
            all_generated_emails["models"][model_name]["final_email"] = current_email
            all_generated_emails["models"][model_name]["success_iteration"] = 0
            continue

        # Iterative adversarial loop
        iteration = 0
        evasion_success = False
        working_email = current_email
        iteration_history = []
        successful_email = None

        while iteration < max_iterations_per_model and not evasion_success:
            print(f"Adversarial attempt {iteration + 1} against {model_name}...")

            # Step 1: LIME preprocessing (if enabled)
            if use_lime_preprocessing:
                strategies = pget("spear.lime.preprocessing_strategies", ["replace", "delete", "char_replace"])
                thresholds = pget("spear.lime.preprocessing_thresholds", [0.01, 0.02, 0.03])

                strategy = strategies[iteration % len(strategies)]
                threshold = thresholds[iteration % len(thresholds)]

                print(f"Step 1 - LIME preprocessing: strategy={strategy}, threshold={threshold}")
                preprocessed_email = preprocess_email_with_lime(
                    working_email, analysis_result,
                    strategy=strategy, threshold=threshold,
                )
                print(f"Preprocessed email: {preprocessed_email[:100]}...")
            else:
                preprocessed_email = working_email
                strategy = None
                threshold = None
                print("Skipping LIME preprocessing")

            # Step 2: LLM-based adversarial generation
            print("Step 2 - LLM adversarial email generation")
            adversarial_email = generate_lime_adversarial_email(
                generation_client, preprocessed_email, analysis_result,
            )

            print("adversarial_email:-------------------\n", adversarial_email)
            print("analysis_result:-------------------\n", analysis_result)

            # Re-analyze the adversarial email
            recheck_result = lime_analyze_email(adversarial_email, model_name)

            # Record this iteration
            iter_key = f"iteration_{iteration + 1}"
            all_generated_emails["models"][model_name]["iterations"][iter_key] = {
                "email": adversarial_email,
                "preprocessed_email": preprocessed_email if use_lime_preprocessing else None,
                "input_email": working_email,
                "success": not recheck_result["is_phishing"],
                "preprocessing_used": use_lime_preprocessing,
                "preprocessing_strategy": strategy,
                "preprocessing_threshold": threshold,
                "is_original": False,
                "is_replicated": False,
                "confidence_before": analysis_result.get("confidence", 0.0),
                "confidence_after": recheck_result.get("confidence", 0.0),
                "label_before": analysis_result.get("label", "unknown"),
                "label_after": recheck_result.get("label", "unknown"),
            }

            iteration_history.append({
                "iteration": iteration + 1,
                "input_email": working_email,
                "preprocessed_email": preprocessed_email if use_lime_preprocessing else None,
                "generated_email": adversarial_email,
                "success": not recheck_result["is_phishing"],
                "preprocessing_used": use_lime_preprocessing,
                "preprocessing_strategy": strategy if use_lime_preprocessing else None,
                "preprocessing_threshold": threshold if use_lime_preprocessing else None,
                "confidence_before": analysis_result.get("confidence", 0.0),
                "confidence_after": recheck_result.get("confidence", 0.0),
                "label_before": analysis_result.get("label", "unknown"),
                "label_after": recheck_result.get("label", "unknown"),
            })

            # Check if evasion succeeded
            if not recheck_result["is_phishing"]:
                print(f"Successfully evaded {model_name}! (attempt {iteration + 1})")
                evasion_success = True
                successful_email = adversarial_email
                working_email = adversarial_email
                all_generated_emails["models"][model_name]["success_iteration"] = iteration + 1
            else:
                print(f"Attempt {iteration + 1} against {model_name} failed, continuing...")
                working_email = adversarial_email
                analysis_result = recheck_result

            iteration += 1

        # Fill remaining iterations
        _fill_remaining_iterations(
            all_generated_emails["models"][model_name],
            iteration, max_iterations_per_model,
            evasion_success, successful_email, working_email, current_email,
        )

        if not evasion_success:
            print(f"Failed to evade {model_name} after {max_iterations_per_model} attempts")

        all_generated_emails["models"][model_name]["final_email"] = working_email

        evasion_results[model_name] = {
            "success": evasion_success,
            "needed_evasion": True,
            "email": working_email,
            "iterations": iteration,
            "iteration_history": iteration_history,
            "preprocessing_enabled": use_lime_preprocessing,
            "initial_confidence": all_analysis_results[model_name].get("confidence", 0.0),
            "initial_label": all_analysis_results[model_name].get("label", "unknown"),
            "final_confidence": recheck_result.get("confidence", 0.0) if 'recheck_result' in dir() else 0.0,
            "final_label": recheck_result.get("label", "unknown") if 'recheck_result' in dir() else "unknown",
        }

        if evasion_success:
            current_email = working_email

    # Final LLM detection
    final_prediction, detection_result = detect_email(recheck_client, current_email)

    return current_email, final_prediction, detection_result, evasion_results, all_generated_emails


def _fill_remaining_iterations(model_data, iteration, max_iterations,
                               evasion_success, successful_email,
                               working_email, original_email):
    """Fill remaining iteration slots with replicated data."""
    if evasion_success and successful_email:
        fill_email = successful_email
        fill_success = True
        replicated_from = model_data.get("success_iteration", iteration)
    else:
        fill_email = working_email if iteration > 0 else original_email
        fill_success = False
        replicated_from = iteration if iteration > 0 else 0

    for remaining in range(iteration + 1, max_iterations + 1):
        model_data["iterations"][f"iteration_{remaining}"] = {
            "email": fill_email,
            "preprocessed_email": None,
            "input_email": fill_email,
            "analysis_before": None,
            "analysis_after": None,
            "success": fill_success,
            "preprocessing_used": False,
            "preprocessing_strategy": None,
            "preprocessing_threshold": None,
            "is_original": False,
            "is_replicated": True,
            "replicated_from_iteration": replicated_from,
        }
