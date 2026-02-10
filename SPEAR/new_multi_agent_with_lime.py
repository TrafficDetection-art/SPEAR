"""
SPEAR: Multi-Agent Phishing Email Generation & Detection System

Main entry point for the SPEAR framework. Coordinates the overall pipeline:
  1. Load configuration and create attack/defense LLM clients
  2. Load or generate email data
  3. Run detection, adversarial attacks (LIME + LLM), and evaluation
  4. Save results, metrics, and cost analysis

Usage:
    python new_multi_agent_with_lime.py \\
        --data_source email_data \\
        --enable_lime_attack True \\
        --enable_llm_attack True \\
        --lime_models textcnn bert
"""

import sys
sys.path.insert(0, "..")
from project_settings import get as pget

import json
import time
import random
import os
import argparse
from datetime import datetime

# Internal modules
from config import config
from token_tracker import (
    get_token_usage_summary,
    get_detailed_cost_analysis,
    estimate_cost,
)
from serialization import NumpyEncoder
from metrics import evaluate_metrics_macro, calculate_asr
from llm_client import create_client, get_LLM_response_vllm
from email_ops import (
    detect_email,
    evaluate_email,
    parse_evaluation_result,
    check_quality_threshold,
    polish_email_with_evaluation,
    formatting_email,
    web_resource_email,
    attachment_resource_email,
    generate_iteration_list,
    generate_and_recheck_iteratively,
)
from lime_attack import adversarial_ml_model_evasion, lime_analyze_email
from output_manager import create_output_directory

# Prompt imports
from my_prompt.init_prompt import init_prompt, list_of_scenarios


# ========================== Data Loading ==========================

def load_email_data(data_path=None):
    """Load email data from JSON file."""
    if data_path is None:
        data_path = pget("spear.data_path", "../dataset/test_data.json")
    with open(data_path, 'r', encoding='utf-8') as f:
        return json.load(f)


def load_personal_info(data_path=None):
    """Load personal information templates."""
    if data_path is None:
        data_path = pget("spear.personal_info_path", "./personal_info.json")
    try:
        with open(data_path, 'r', encoding='utf-8') as f:
            return json.load(f)["templates"]
    except FileNotFoundError:
        print(f"Error: Personal info file {data_path} not found.")
        return []


# ========================== Email Generation ==========================

def generate_phishing_from_personal_info(client, personal_info,
                                         scenario="IT Security Alert"):
    """Generate a phishing email based on personal info and scenario (attack side)."""
    prompt = init_prompt.format(
        personal_info=json.dumps(personal_info, ensure_ascii=False, indent=2),
        scenario=random.choice(list_of_scenarios),
    )
    return get_LLM_response_vllm(client, prompt, role="attack")


def generate_emails_from_personal_info(generate_client, personal_info_data,
                                       scenarios=None):
    """Generate phishing emails for each personal info entry."""
    if scenarios is None:
        scenarios = [
            "IT Security Alert",
            "HR Benefits Update",
            "Financial Account Verification",
            "Company Policy Update",
            "System Maintenance Notification",
        ]

    generated_emails = []

    for index, personal_info in enumerate(personal_info_data):
        print(f"Processing personal info {index + 1}/{len(personal_info_data)}")
        scenario = scenarios[index % len(scenarios)]
        print(f"Using scenario: {scenario}")

        generated_email = generate_phishing_from_personal_info(
            generate_client, personal_info, scenario,
        )

        generated_emails.append({
            "Text": generated_email,
            "Class": 1,
            "type": "personal_info_generated",
            "personal_info": personal_info,
            "scenario": scenario,
        })

        print(f"Generated email with scenario '{scenario}':")
        print(generated_email)
        print("-" * 50)
        time.sleep(pget("spear.main.generation_delay_seconds", 2))

    return generated_emails


# ========================== Result Saving ==========================

def save_results(output_dir, data, evaluation_data, label_results,
                 first_results, final_results, type_results,
                 enable_lime_attack, enable_llm_attack, lime_models,
                 max_lime_iterations, use_lime_preprocessing,
                 data_source, emails_to_process):
    """Save all result files to the output directory."""
    # 1. Main results
    with open(os.path.join(output_dir, 'responds.json'), 'w', encoding='utf-8') as f:
        json.dump(data, f, ensure_ascii=False, indent=4, cls=NumpyEncoder)

    # 2. Evaluation results
    with open(os.path.join(output_dir, 'evaluation_results.json'), 'w', encoding='utf-8') as f:
        json.dump(evaluation_data, f, ensure_ascii=False, indent=4, cls=NumpyEncoder)

    # 3. Metrics summary (including ASR)
    metrics_summary = _compute_metrics_summary(
        label_results, first_results, final_results, type_results,
    )
    with open(os.path.join(output_dir, 'metrics_summary.json'), 'w', encoding='utf-8') as f:
        json.dump(metrics_summary, f, ensure_ascii=False, indent=4, cls=NumpyEncoder)
    print(f"Metrics summary (including ASR) saved to: {output_dir}/metrics_summary.json")

    # 4. Email histories
    _save_email_histories(output_dir, data)

    # 5. Type results
    with open(os.path.join(output_dir, 'type_results.json'), 'w', encoding='utf-8') as f:
        json.dump(type_results, f, ensure_ascii=False, indent=4, cls=NumpyEncoder)

    # 6. Attack configuration
    attack_config = {
        "attack_model": {
            "model": config['attack_model']['model'],
            "api_base_url": config['attack_model']['api_base_url'],
        },
        "defense_model": {
            "model": config['defense_model']['model'],
            "api_base_url": config['defense_model']['api_base_url'],
        },
        "lime_attack_enabled": enable_lime_attack,
        "llm_attack_enabled": enable_llm_attack,
        "lime_models": lime_models if enable_lime_attack else None,
        "max_lime_iterations": max_lime_iterations if enable_lime_attack else None,
        "use_lime_preprocessing": use_lime_preprocessing if enable_lime_attack else None,
        "data_source": data_source,
        "total_samples": len(emails_to_process),
    }
    with open(os.path.join(output_dir, 'attack_config.json'), 'w', encoding='utf-8') as f:
        json.dump(attack_config, f, ensure_ascii=False, indent=4, cls=NumpyEncoder)

    print(f"All results saved to directory: {output_dir}")


def _compute_metrics_summary(label_results, first_results, final_results, type_results):
    """Compute metrics summary dict."""
    summary = {"initial_detection": {}, "iterations": [], "by_type": {}}

    if first_results:
        initial_metrics = evaluate_metrics_macro(label_results, first_results)
        initial_asr = calculate_asr(label_results, first_results)
        summary["initial_detection"] = {
            "accuracy": initial_metrics.get("Accuracy", 0),
            "precision": initial_metrics.get("Precision", 0),
            "recall": initial_metrics.get("Recall", 0),
            "f1_score": initial_metrics.get("F1-score", 0),
            "asr": initial_asr["ASR"],
            "asr_percent": initial_asr["evasion_rate_percent"],
            "successful_evasions": initial_asr["successful_evasions"],
            "total_malicious": initial_asr["total_malicious"],
        }

    if final_results:
        for i, item_result in enumerate(zip(*final_results)):
            iter_metrics = evaluate_metrics_macro(label_results, list(item_result))
            iter_asr = calculate_asr(label_results, list(item_result))
            summary["iterations"].append({
                "iteration": i + 1,
                "accuracy": iter_metrics.get("Accuracy", 0),
                "precision": iter_metrics.get("Precision", 0),
                "recall": iter_metrics.get("Recall", 0),
                "f1_score": iter_metrics.get("F1-score", 0),
                "asr": iter_asr["ASR"],
                "asr_percent": iter_asr["evasion_rate_percent"],
                "successful_evasions": iter_asr["successful_evasions"],
                "total_malicious": iter_asr["total_malicious"],
            })

    for email_type, results in type_results.items():
        if results["first_predictions"]:
            type_metrics = evaluate_metrics_macro(results["labels"], results["first_predictions"])
            type_asr = calculate_asr(results["labels"], results["first_predictions"])
            summary["by_type"][email_type] = {
                "accuracy": type_metrics.get("Accuracy", 0),
                "precision": type_metrics.get("Precision", 0),
                "recall": type_metrics.get("Recall", 0),
                "f1_score": type_metrics.get("F1-score", 0),
                "asr": type_asr["ASR"],
                "asr_percent": type_asr["evasion_rate_percent"],
                "successful_evasions": type_asr["successful_evasions"],
                "total_malicious": type_asr["total_malicious"],
            }

    return summary


def _save_email_histories(output_dir, data):
    """Save all generated email histories and per-iteration files."""
    all_email_histories = []

    for sample_data in data:
        email_history = {
            "sample_info": {
                "original_text": sample_data.get("original_text", ""),
                "subject": sample_data.get("subject", ""),
                "type": sample_data.get("type", "Unknown"),
                "class": sample_data.get("Class", 0),
                "lime_attack_enabled": sample_data.get("lime_attack_enabled", False),
                "llm_attack_enabled": sample_data.get("llm_attack_enabled", False),
                "was_attacked": sample_data.get("was_attacked", False),
                "lime_attack_performed": sample_data.get("lime_attack_performed", False),
                "llm_attack_performed": sample_data.get("llm_attack_performed", False),
            },
        }

        if sample_data.get("all_generated_emails"):
            email_history["generated_emails"] = sample_data["all_generated_emails"]

        elif sample_data.get("llm_iteration_history"):
            email_history["generated_emails"] = _build_llm_history(sample_data)

        else:
            email_history["generated_emails"] = _build_no_attack_history(sample_data)

        all_email_histories.append(email_history)

    if not all_email_histories:
        return

    # Save combined history
    history_file = os.path.join(output_dir, 'all_generated_email_histories.json')
    with open(history_file, 'w', encoding='utf-8') as f:
        json.dump(all_email_histories, f, ensure_ascii=False, indent=4, cls=NumpyEncoder)
    print(f"Saved {len(all_email_histories)} sample email histories to: {history_file}")

    # Save per-iteration files
    _save_per_iteration_files(output_dir, all_email_histories)


def _build_llm_history(sample_data):
    """Build email history from LLM attack iteration history."""
    llm_hist = sample_data["llm_iteration_history"]
    original = sample_data.get("original_text", "")

    history = {
        "original_email": original,
        "models": {
            "llm_attack": {
                "iterations": {},
                "success_iteration": sample_data.get("iterations", 0),
                "final_email": sample_data.get("Text", original),
            },
        },
    }

    for iter_key, iter_data in llm_hist.items():
        history["models"]["llm_attack"]["iterations"][iter_key] = {
            "email": iter_data.get("email", ""),
            "preprocessed_email": None,
            "input_email": iter_data.get("email", ""),
            "success": iter_data.get("prediction", 1) == 0,
            "preprocessing_used": False,
            "preprocessing_strategy": None,
            "preprocessing_threshold": None,
            "is_original": False,
            "is_replicated": iter_data.get("is_replicated", False),
            "confidence_before": 0.0,
            "confidence_after": 0.0,
            "label_before": "unknown",
            "label_after": "ham" if iter_data.get("prediction", 1) == 0 else "spam",
            "detection_reason": iter_data.get("detection_reason", ""),
        }

    return history


def _build_no_attack_history(sample_data):
    """Build email history for samples that were not attacked."""
    current_email = sample_data.get("Text", sample_data.get("original_text", ""))

    history = {
        "original_email": sample_data.get("original_text", current_email),
        "models": {
            "no_attack": {
            "iterations": {},
            "success_iteration": None,
                "final_email": current_email,
            },
        },
    }

    for i in range(1, 6):
        history["models"]["no_attack"]["iterations"][f"iteration_{i}"] = {
                "email": current_email,
                    "preprocessed_email": None,
            "input_email": current_email,
            "success": sample_data.get("final_prediction", 0) == 0,
                    "preprocessing_used": False,
                    "preprocessing_strategy": None,
                    "preprocessing_threshold": None,
            "is_original": i == 1,
            "is_replicated": i > 1,
            "confidence_before": 0.0,
            "confidence_after": 0.0,
            "label_before": "unknown",
            "label_after": "unknown",
        }

    return history


def _save_per_iteration_files(output_dir, all_email_histories):
    """Save per-iteration email JSON files."""
    iteration_files = {"original": "iteration_0_original_emails.json"}
    for i in range(1, 6):
        iteration_files[f"iteration_{i}"] = f"iteration_{i}_emails.json"

    iteration_data = {name: [] for name in iteration_files}

    for email_history in all_email_histories:
        sample_info = email_history["sample_info"]
        generated = email_history["generated_emails"]

        iteration_data["original"].append({
            "Text": generated["original_email"],
            "Class": sample_info["class"],
            "type": sample_info["type"],
        })

        for model_name, model_data in generated["models"].items():
            for iter_num in range(1, 6):
                iter_key = f"iteration_{iter_num}"

                if iter_key in model_data["iterations"]:
                    email_text = model_data["iterations"][iter_key]["email"]
                else:
                    available = [
                        k for k in model_data["iterations"] if k.startswith("iteration_")
                    ]
                    if available:
                        last = max(available, key=lambda x: int(x.split("_")[1]))
                        email_text = model_data["iterations"][last]["email"]
                    else:
                        email_text = generated["original_email"]

                entry = {
                    "Text": email_text,
                    "Class": sample_info["class"],
                    "type": sample_info["type"],
                }
                if len(generated["models"]) > 1:
                    entry["model"] = model_name
                iteration_data[iter_key].append(entry)

    for iter_name, data_list in iteration_data.items():
        if data_list:
            path = os.path.join(output_dir, iteration_files[iter_name])
            with open(path, 'w', encoding='utf-8') as f:
                json.dump(data_list, f, ensure_ascii=False, indent=4, cls=NumpyEncoder)

    print("Saved emails to 6 per-iteration JSON files:")
    for iter_name, filename in iteration_files.items():
        count = len(iteration_data.get(iter_name, []))
        print(f"  - {iter_name}: {count} emails ({filename})")


# ========================== Reporting ==========================

def print_evaluation_report(label_results, first_results, final_results, type_results):
    """Print evaluation metrics to console."""
    print("\n" + "=" * 60)
    print("Evaluation Metrics")
    print("=" * 60)

    if first_results:
        print("\nInitial Detection Metrics:")
        initial_metrics = evaluate_metrics_macro(label_results, first_results)
        for metric, value in initial_metrics.items():
            print(f"  {metric}: {value:.4f}")

        initial_asr = calculate_asr(label_results, first_results)
        print(f"\nInitial ASR (Attack Success Rate):")
        print(f"  ASR: {initial_asr['ASR']:.4f} ({initial_asr['evasion_rate_percent']:.2f}%)")
        print(f"  Successful evasions: {initial_asr['successful_evasions']} / "
              f"{initial_asr['total_malicious']} malicious samples")
    else:
        print("No valid initial detection results to evaluate.")

    if final_results:
        for i, item_result in enumerate(zip(*final_results)):
            print(f"\nIteration {i + 1} Metrics:")
            iter_metrics = evaluate_metrics_macro(label_results, list(item_result))
            for metric, value in iter_metrics.items():
                print(f"  {metric}: {value:.4f}")

            iter_asr = calculate_asr(label_results, list(item_result))
            print(f"  ASR: {iter_asr['ASR']:.4f} ({iter_asr['evasion_rate_percent']:.2f}%)")
            print(f"  Successful evasions: {iter_asr['successful_evasions']} / "
                  f"{iter_asr['total_malicious']} malicious samples")
    else:
        print("No valid final detection results to evaluate.")

    print("\n" + "=" * 60)
    print("Metrics by Email Type (All Iterations)")
    print("=" * 60)

    for email_type, results in type_results.items():
        if results["final_predictions"]:
            print(f"\nType '{email_type}':")
            for i, iter_preds in enumerate(zip(*results["final_predictions"])):
                print(f"\n  Iteration {i + 1}:")
                type_metrics = evaluate_metrics_macro(results["labels"], list(iter_preds))
                for metric, value in type_metrics.items():
                    print(f"    {metric}: {value:.4f}")
                type_asr = calculate_asr(results["labels"], list(iter_preds))
                print(f"    ASR: {type_asr['ASR']:.4f} ({type_asr['evasion_rate_percent']:.2f}%)")
                print(f"    Successful evasions: {type_asr['successful_evasions']} / "
                      f"{type_asr['total_malicious']} malicious samples")
        else:
            print(f"No valid results for type '{email_type}'.")

    print("=" * 60)


def print_cost_report(start_time, end_time, output_dir):
    """Print and save token usage and cost analysis."""
    total_time = end_time - start_time
    print(f"\nEnd time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"Total execution time: {total_time:.2f}s ({total_time / 60:.2f}min)")

    print("\n" + "=" * 60)
    print("TOKEN USAGE & COST ANALYSIS")
    print("=" * 60)

    summary = get_token_usage_summary()
    print(f"Total tokens: {summary['total_tokens']:,}")
    print(f"  - Input tokens: {summary['total_prompt_tokens']:,}")
    print(f"  - Output tokens: {summary['total_completion_tokens']:,}")
    print(f"Total requests: {summary['total_requests']:,}")
    print(f"Average tokens/request: {summary['average_tokens_per_request']:.1f}")

    cost = estimate_cost(
        prompt_tokens=summary['total_prompt_tokens'],
        completion_tokens=summary['total_completion_tokens'],
    )
    print(f"\nCost breakdown:")
    print(f"  - Input cost: ${cost['input_cost']:.4f} ({cost['input_tokens']:,} tokens)")
    print(f"  - Output cost: ${cost['output_cost']:.4f} ({cost['output_tokens']:,} tokens)")
    print(f"  - Total cost: ${cost['total_cost']:.4f}")
    print(f"  - Model: {cost['model']}")

    detailed = get_detailed_cost_analysis()
    fn_costs = detailed['function_costs']

    if fn_costs:
        print(f"\nCost by function:")
        print(f"{'Function':<25} {'Requests':<8} {'In Tokens':<10} {'Out Tokens':<10} {'Cost':<10}")
        print("-" * 75)

        sorted_fns = sorted(fn_costs.items(), key=lambda x: x[1]['total_cost'], reverse=True)
        for fn_name, costs in sorted_fns:
            print(f"{fn_name:<25} {costs['request_count']:<8} "
                  f"{costs['prompt_tokens']:<10,} {costs['completion_tokens']:<10,} "
                  f"${costs['total_cost']:<9.4f}")

        print("-" * 75)

        total_cost_val = cost['total_cost']
        if total_cost_val > 0:
            print(f"\nTop cost contributors:")
            for i, (fn_name, costs) in enumerate(sorted_fns[:5]):
                pct = (costs['total_cost'] / total_cost_val) * 100
                print(f"  {i + 1}. {fn_name}: ${costs['total_cost']:.4f} ({pct:.1f}%)")

    print("=" * 60)

    # Save cost analysis
    detailed["execution_info"] = {
        "start_time": start_time,
        "end_time": end_time,
        "total_time_seconds": total_time,
        "total_time_minutes": total_time / 60,
    }
    cost_file = os.path.join(output_dir, 'cost_analysis.json')
    with open(cost_file, 'w', encoding='utf-8') as f:
        json.dump(detailed, f, ensure_ascii=False, indent=4, cls=NumpyEncoder)
    print(f"Cost analysis saved to: {cost_file}")


# ========================== Main Pipeline ==========================

def main(data_source="email_data", data_path=None, personal_info_path=None,
         scenarios=None, enable_lime_attack=False, enable_llm_attack=False,
         lime_models=None, max_lime_iterations=None, samples_per_type=None,
         use_lime_preprocessing=True):
    """
    Main pipeline function.

    Args:
        data_source: "email_data" or "personal_info".
        data_path: Path to email data JSON (for email_data source).
        personal_info_path: Path to personal info JSON (for personal_info mode).
        scenarios: List of phishing scenarios.
        enable_lime_attack: Enable LIME adversarial attacks against ML models.
        enable_llm_attack: Enable LLM-based adversarial attacks.
        lime_models: List of ML model names for LIME attacks.
        max_lime_iterations: Max iterations per LIME model.
        use_lime_preprocessing: Preprocess with LIME before LLM generation.
    """
    start_time = time.time()
    print(f"Start time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    
    if lime_models is None:
        lime_models = ["textcnn", "bert"]
    
    output_dir = create_output_directory()
    
    # ---- Create clients (attack/defense separation) ----
    print("\n" + "=" * 60)
    print("Initializing attack/defense clients")
    print("=" * 60)

    detection_client = create_client(role="defense")
    evaluation_client = create_client(role="defense")
    
    generate_client = create_client(role="attack")
    formatting_client = create_client(role="attack")
    web_resource_client = create_client(role="attack")
    attachment_resource_client = create_client(role="attack")
    lime_client = create_client(role="attack")

    print("=" * 60 + "\n")

    # ---- Load or generate data ----
    if max_lime_iterations is None:
        max_lime_iterations = pget("spear.main.max_iterations", 5)
    emails_to_process = _load_data(
        data_source, data_path, personal_info_path, scenarios,
        samples_per_type, generate_client,
    )

    # ---- Processing loop ----
    first_results = []
    final_results = []
    label_results = []
    data = []
    evaluation_data = []
    type_results = {}

    print(f"Attack config: LIME={enable_lime_attack}, LLM={enable_llm_attack}")
    if enable_lime_attack:
        print(f"LIME models: {lime_models}")

    for index, email_info in enumerate(emails_to_process):
        email_content = email_info["Text"]
        label = email_info["Class"]
        email_type = email_info.get("type", "Unknown")

        if email_type not in type_results:
            type_results[email_type] = {
                "labels": [], "first_predictions": [], "final_predictions": [],
            }

        # Initial LLM detection
        llm_prediction, llm_detection_result = detect_email(detection_client, email_content)
        current_email = email_content
        final_prediction = llm_prediction
        iterations = 0
        lime_evasion_results = {}
        all_generated_emails = None
        
        # Initialize all tracking variables
        evaluate_result = None
        recheck_result = llm_detection_result
        subject = ""
        url = ""
        attachment_name = ""
        attachment_type = ""
        has_url = "No"
        has_attachment = "No"
        web_resource = None
        attachment_resource = None
        formatted_email_dict = {}
        
        # ---- Determine attack strategy ----
        should_attack_with_lime = False
        should_attack_with_llm = False
        
        if enable_lime_attack and label == 1:
            print("Checking ML model classification result...")
            first_lime_model = lime_models[0] if lime_models else "textcnn"
            lime_analysis = lime_analyze_email(current_email, first_lime_model)
            if lime_analysis["is_phishing"]:
                should_attack_with_lime = True
                print(f"ML model ({first_lime_model}) classified as phishing, LIME attack needed")
            else:
                print(f"ML model ({first_lime_model}) classified as benign, no LIME attack needed")
        
        if enable_llm_attack and label == 1:
            if llm_prediction == 1:
                should_attack_with_llm = True
                print("LLM classified as phishing, LLM attack needed")
            else:
                print("LLM classified as benign, no LLM attack needed")
        
        should_attack = should_attack_with_lime or should_attack_with_llm

        # ---- Execute attacks ----
        llm_iteration_history = None
        
        if should_attack:
            print(f"\n=== Processing phishing email {index + 1} (attack needed) ===")
            print(f"LIME attack: {should_attack_with_lime}, LLM attack: {should_attack_with_llm}")
            
            if should_attack_with_lime:
                print("Starting LIME adversarial attack...")
                try:
                    (lime_email, lime_pred, lime_result,
                     lime_evasion_results, all_generated_emails) = adversarial_ml_model_evasion(
                        generate_client, detection_client, current_email,
                        lime_models, max_lime_iterations, use_lime_preprocessing,
                    )
                    current_email = lime_email
                    final_prediction = lime_pred
                    recheck_result = lime_result
                    print(f"LIME attack complete, detection result: {final_prediction}")
                except Exception as e:
                    print(f"LIME attack failed: {e}")
                    lime_evasion_results = {"error": str(e)}
                    all_generated_emails = None
                time.sleep(pget("spear.main.generation_delay_seconds", 2))
                
            if should_attack_with_llm:
                print("Starting LLM adversarial attack...")
                (generated_email, final_prediction, recheck_result,
                 iterations, llm_iteration_history) = generate_and_recheck_iteratively(
                    generate_client, detection_client, current_email, recheck_result,
                )
                current_email = generated_email
                print(f"LLM attack complete, iterations: {iterations}, "
                      f"result: {final_prediction}")
                time.sleep(pget("spear.main.generation_delay_seconds", 2))
            
            # ---- Quality Evaluation & Polish ----
            evaluate_result = evaluate_email(evaluation_client, current_email)
            print("evaluate_result:", evaluate_result)
            
            # Quality check and polish loop
            quality_enabled = pget("spear.main.quality_evaluation.enabled", True)
            max_polish_retries = pget("spear.main.quality_evaluation.max_polish_retries", 3)
            polish_attempt = 0
            quality_history = []
            
            if quality_enabled:
                parsed_eval = parse_evaluation_result(evaluate_result)
                quality_history.append({
                    "attempt": polish_attempt,
                    "evaluation": parsed_eval,
                })
                
                passed, reason = check_quality_threshold(parsed_eval)
                print(f"Quality check: {reason}")
                
                while not passed and polish_attempt < max_polish_retries:
                    polish_attempt += 1
                    print(f"\n--- Quality Polish Attempt {polish_attempt}/{max_polish_retries} ---")
                    print(f"Reason: {reason}")
                    
                    # Polish the email
                    polished_email = polish_email_with_evaluation(
                        generate_client, current_email, parsed_eval
                    )
                    current_email = polished_email
                    print(f"Email polished (attempt {polish_attempt})")
                    time.sleep(pget("spear.main.generation_delay_seconds", 2))
                    
                    # Re-evaluate
                    evaluate_result = evaluate_email(evaluation_client, current_email)
                    print("Re-evaluation result:", evaluate_result)
                    
                    parsed_eval = parse_evaluation_result(evaluate_result)
                    quality_history.append({
                        "attempt": polish_attempt,
                        "evaluation": parsed_eval,
                    })
                    
                    passed, reason = check_quality_threshold(parsed_eval)
                    print(f"Quality check: {reason}")
                    
                    if passed:
                        print(f"✓ Quality threshold met after {polish_attempt} polish attempt(s)")
                        break
                
                if not passed:
                    print(f"⚠ Quality threshold not met after {max_polish_retries} polish attempts")
        else:
            print(f"\n=== Processing email {index + 1} (no attack needed) ===")
            print(f"Reason: label={label}, llm_prediction={llm_prediction}")
            final_prediction = llm_prediction
            evaluate_result = None
            quality_history = []
        
        # ---- Prediction list ----
        if enable_llm_attack and should_attack_with_llm:
            final_prediction_list = generate_iteration_list(
                final_prediction, iterations + 1, max_iterations=5,
            )
        else:
            final_prediction_list = [final_prediction] * 5
        
        # ---- Formatting ----
        try:
            print("Formatting email...")
            formatted_email_dict = formatting_email(formatting_client, current_email)
            print("Formatted email:", formatted_email_dict.get('items', 'No items found'))

            subject = formatted_email_dict.get('subject', '')
            url = formatted_email_dict.get('url', '')
            attachment_name = formatted_email_dict.get('attchment', '')
            attachment_type = formatted_email_dict.get('attchment_type', '')

            format_prediction, _ = detect_email(detection_client, formatted_email_dict)
            final_prediction_list.append(format_prediction)

            try:
                has_url = formatted_email_dict.get('has_url', 'No')
                has_attachment = formatted_email_dict.get('has_attachment', 'No')
                
                if has_url == "Yes" and url:
                    web_resource = web_resource_email(web_resource_client, current_email, url)
                    print(f"Generated web resource: {url}")
                    
                if has_attachment == "Yes" and attachment_name:
                    attachment_resource = attachment_resource_email(
                        attachment_resource_client, current_email,
                        attachment_name, attachment_type,
                    )
                    print(f"Generated attachment: {attachment_name} ({attachment_type})")
                
            except Exception as e:
                print(f"Error processing URL/attachment: {e}")
                web_resource = None
                attachment_resource = None
                
        except Exception as e:
            print(f"Email formatting failed: {e}")
            format_prediction, _ = detect_email(detection_client, current_email)
            final_prediction_list.append(format_prediction)

        # ---- Record sample data ----
        sample_data = {
            "subject": subject,
            "original_text": email_content,
            "Text": current_email,
            "Class": label,
            "response": recheck_result,
            "prediction": final_prediction_list,
            "first_prediction": llm_prediction,
            "final_prediction": final_prediction,
            "iterations": iterations,
            "type": email_type,
            "has_url": has_url,
            "has_attachment": has_attachment,
            "attachment_name": attachment_name,
            "attachment_type": attachment_type,
            "url": url,
            "web_resource": web_resource,
            "attachment_resource": attachment_resource,
            "lime_attack_enabled": enable_lime_attack,
            "llm_attack_enabled": enable_llm_attack,
            "lime_evasion_results": (
                lime_evasion_results if enable_lime_attack and should_attack_with_lime else None
            ),
            "all_generated_emails": (
                all_generated_emails if enable_lime_attack and should_attack_with_lime else None
            ),
            "llm_iteration_history": llm_iteration_history if should_attack_with_llm else None,
            "was_attacked": should_attack,
            "lime_attack_performed": should_attack_with_lime,
            "llm_attack_performed": should_attack_with_llm,
            "attack_reason": (
                f"LIME: {should_attack_with_lime}, LLM: {should_attack_with_llm}"
                if should_attack
                else f"No attack needed: label={label}, llm_prediction={llm_prediction}"
            ),
            "formatted_email_success": bool(formatted_email_dict),
            "evaluation_performed": evaluate_result is not None,
            "quality_history": quality_history if 'quality_history' in locals() else [],
        }
        
        data.append(sample_data)
        
        if should_attack and evaluate_result:
            evaluation_data.append({
                "sample_index": index,
                "iteration": iterations,
                "evaluation_result": evaluate_result,
                "lime_attack_used": should_attack_with_lime,
                "llm_attack_used": should_attack_with_llm,
                "original_text": email_content,
                "final_text": current_email,
            })

        # Statistics
        if llm_prediction in (0, 1):
            label_results.append(label)
            first_results.append(llm_prediction)
            final_results.append(final_prediction_list)

            type_results[email_type]["labels"].append(label)
            type_results[email_type]["first_predictions"].append(llm_prediction)
            type_results[email_type]["final_predictions"].append(final_prediction_list)

        print(f"Sample {index + 1}: Label={label}, LLM={llm_prediction}, "
              f"Final={final_prediction_list[-1]}, Type={email_type}, "
              f"Attacked={should_attack}")

    # ---- Save results ----
    save_results(
        output_dir, data, evaluation_data, label_results,
        first_results, final_results, type_results,
        enable_lime_attack, enable_llm_attack, lime_models,
        max_lime_iterations, use_lime_preprocessing,
        data_source, emails_to_process,
    )

    # ---- Reports ----
    print_evaluation_report(label_results, first_results, final_results, type_results)
    print_cost_report(start_time, time.time(), output_dir)


def _load_data(data_source, data_path, personal_info_path, scenarios,
               samples_per_type, generate_client):
    """Load or generate email data based on the data source type."""
    if data_source == "email_data":
        emails_data = load_email_data(data_path)

        if samples_per_type is None:
            samples_per_type = pget("spear.main.samples_per_type", 10)
        emails_by_type = {}
        for email in emails_data:
            t = email.get("type", "Unknown")
            emails_by_type.setdefault(t, []).append(email)

        sampled = []
        for t, emails in emails_by_type.items():
            s = random.sample(emails, min(samples_per_type, len(emails)))
            sampled.extend(s)
            print(f"Sampled {len(s)} emails of type '{t}'")

        print(f"Total sampled emails: {len(sampled)}")
        return sampled

    elif data_source == "personal_info":
        if not personal_info_path:
            raise ValueError("personal_info_path required for personal_info mode")

        personal_info_data = load_personal_info(personal_info_path)
        if not personal_info_data:
            raise ValueError("No personal information data found")

        emails = generate_emails_from_personal_info(generate_client, personal_info_data, scenarios)
        print(f"Generated {len(emails)} emails from personal information")
        return emails

    else:
        raise ValueError(f"Unknown data source type: {data_source}")


# ========================== CLI Entry Point ==========================

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='SPEAR: Phishing Email Generation and Detection')
    
    parser.add_argument('--data_source', choices=['email_data', 'personal_info'],
                      default='email_data', help='Data source type')
    parser.add_argument('--data_path',
                        help='Path to email data JSON file (for email_data source)')
    parser.add_argument('--personal_info_path',
                        help='Path to personal information JSON file')
    parser.add_argument('--scenarios', nargs='+', 
                        default=None,
                        help='Scenarios for phishing email generation')

    # Attack control
    parser.add_argument('--enable_lime_attack',
                        type=lambda x: x.lower() == 'false', default=False,
                        help='Enable LIME adversarial attack (True/False)')
    parser.add_argument('--enable_llm_attack',
                        type=lambda x: x.lower() == 'false', default=False,
                      help='Enable LLM adversarial attack (True/False)')
    parser.add_argument('--lime_models', nargs='+', default=None,
                        help='Models for LIME attack (default: textcnn)')
    parser.add_argument('--max_lime_iterations', type=int, default=None,
                        help='Max iterations per LIME model (default: 5)')
    parser.add_argument('--max_iterations', type=int, default=None,
                        dest='max_lime_iterations',
                        help='Alias for --max_lime_iterations')
    parser.add_argument('--samples_per_type', type=int, default=None,
                        help='Number of samples per email type to process (default: 10)')
    parser.add_argument('--use_lime_preprocessing',
                        type=lambda x: x.lower() == 'true', default=True,
                      help='Use LIME preprocessing before LLM generation (True/False)')
    
    args = parser.parse_args()
    
    if args.data_source == 'personal_info' and not args.personal_info_path:
        parser.error("--personal_info_path is required for personal_info data source")
    
    main(
        data_source=args.data_source,
        data_path=args.data_path,
        personal_info_path=args.personal_info_path,
        scenarios=args.scenarios,
        enable_lime_attack=args.enable_lime_attack,
        enable_llm_attack=args.enable_llm_attack,
        lime_models=args.lime_models,
        max_lime_iterations=args.max_lime_iterations,
        samples_per_type=args.samples_per_type,
        use_lime_preprocessing=args.use_lime_preprocessing,
    )
