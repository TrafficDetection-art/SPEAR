"""
Email operations module.

Provides email detection, evaluation, formatting, web/attachment resource
generation, and iterative generation with re-checking logic.
"""

import json
import time

from llm_client import get_LLM_response_vllm
from my_prompt.detection_prompt import detection_prompt
from my_prompt.generation_prompt import generation_prompt
from my_prompt.evaluation_prompt import evaluation_prompt
from my_prompt.formatting import formatting_prompt
from my_prompt.web_generation_prompt import web_generation_prompt
from my_prompt.attachment_generation_prompt import attachment_generation_prompt


# ========================== Email Detection ==========================

def detect_email(client, email_content):
    """
    Use LLM to detect whether an email is phishing (defense side).

    Returns:
        (prediction, response): prediction is 0 (benign), 1 (phishing), or 2 (unknown).
    """
    prompt = detection_prompt.format(email_content=email_content)
    response = get_LLM_response_vllm(client, prompt, function_name="email_detection", role="defense")

    print("Detection response:", response)
    if "etermined as a phishing email" in response:
        return 1, response
    elif "etermined as another legitimate email" in response:
        return 0, response
    return 2, response  # Default: unknown


# ========================== Email Evaluation ==========================

def evaluate_email(client, email_content):
    """Use LLM to evaluate an email (defense side)."""
    prompt = evaluation_prompt.format(email_content=email_content)
    return get_LLM_response_vllm(client, prompt, role="defense")


# ========================== Email Formatting ==========================

def formatting_email(client, email_content):
    """Use LLM to format an email into structured JSON (attack side)."""
    prompt = formatting_prompt.format(email_content=email_content)
    response_str = get_LLM_response_vllm(client, prompt, role="attack")

    print("Formatting response:", response_str[:200] if response_str else "Empty response")

    if not response_str or response_str.strip() == "":
        print("Warning: LLM returned empty response, using default format")
        return create_default_formatted_email(email_content)

    start_index = response_str.find('{')
    end_index = response_str.rfind('}') + 1

    if start_index == -1 or end_index <= start_index:
        print("Warning: No JSON found in response, using default format")
        print(f"Response content: {response_str[:500]}")
        return create_default_formatted_email(email_content)

    json_str = response_str[start_index:end_index]

    try:
        response_dict = json.loads(json_str)

        for field in ['has_url', 'has_attachment']:
            if field not in response_dict:
                print(f"Warning: JSON missing required field '{field}', adding default")
                response_dict[field] = "No"

        return response_dict

    except json.JSONDecodeError as e:
        print(f"JSON parse error: {e}")
        print(f"Attempted JSON: {json_str[:200]}")
        return create_default_formatted_email(email_content)


def create_default_formatted_email(email_content):
    """Create default formatted email when formatting fails."""
    return {
        "subject": "Important Notice",
        "has_url": "No",
        "url": "",
        "has_attachment": "No",
        "attchment": "",
        "attchment_type": "",
        "items": {"Text": email_content},
    }


# ========================== Resource Generation ==========================

def web_resource_email(client, email_content, url):
    """Use LLM to generate web resource content (attack side)."""
    prompt = web_generation_prompt.format(email_content=email_content, url=url)
    response_str = get_LLM_response_vllm(client, prompt, role="attack")
    print("Web resource:", response_str)
    return response_str


def attachment_resource_email(client, email_content, attachment_name, attachment_type):
    """Use LLM to generate attachment resource content (attack side)."""
    prompt = attachment_generation_prompt.format(
        email_content=email_content,
        attachment_name=attachment_name,
        attachment_type=attachment_type,
    )
    response_str = get_LLM_response_vllm(client, prompt, role="attack")
    print("Attachment resource:", response_str)
    return response_str


# ========================== Iterative Generation & Re-check ==========================

def generate_and_recheck(generation_client, recheck_client, email_content, detection_reason):
    """
    Generate a new evasion phishing email and re-check with the detector.

    Args:
        generation_client: Attack-side client.
        recheck_client: Defense-side client.
        email_content: Original email body.
        detection_reason: Previous detection result.

    Returns:
        (new_email, prediction, detection_result)
    """
    prompt = generation_prompt.format(
        detection_reason=detection_reason,
        email_content=email_content,
    )
    new_email = get_LLM_response_vllm(generation_client, prompt, role="attack")
    print("New email:", new_email)
    time.sleep(pget("spear.llm.retry_delay", 1))

    final_prediction, detection_result = detect_email(recheck_client, new_email)
    return new_email, final_prediction, detection_result


def generate_iteration_list(final_prediction, iteration, max_iterations=None):
    """
    Generate a list of length max_iterations.

    Values before `iteration` are 1 (still iterating);
    from `iteration` onward, values are `final_prediction`.

    Args:
        final_prediction: Final prediction value (0 or 1).
        iteration: The iteration where evasion/detection succeeded (1-based).
        max_iterations: Maximum number of iterations.

    Returns:
        A list of length max_iterations.
    """
    if max_iterations is None:
        max_iterations = pget("spear.main.max_iterations", 5)
    if max_iterations is None:
        max_iterations = pget("spear.main.max_iterations", 5)
    return [1] * (iteration - 1) + [final_prediction] * (max_iterations - iteration + 1)


def generate_and_recheck_iteratively(generation_client, recheck_client,
                                     email_content, detection_reason, max_iterations=None):
    """
    Iteratively generate phishing emails until detection is evaded.

    Args:
        generation_client: Attack-side client.
        recheck_client: Defense-side client.
        email_content: Original email body.
        detection_reason: Previous detection result.
        max_iterations: Maximum iterations.

    Returns:
        (new_email, final_prediction, recheck_result, iteration, iteration_history)
    """
    if max_iterations is None:
        max_iterations = pget("spear.main.max_iterations", 5)
    iteration = 0
    new_email = email_content
    final_prediction = 0
    recheck_result = ""
    iteration_history = {}

    while iteration < max_iterations:
        iteration += 1
        print(f"Iteration {iteration}: Generating a new phishing email...")

        new_email, final_prediction, detection_reason = generate_and_recheck(
            generation_client, recheck_client, new_email, detection_reason
        )

        iteration_history[f"iteration_{iteration}"] = {
            "email": new_email,
            "prediction": final_prediction,
            "detection_reason": detection_reason,
        }

        if final_prediction == 0:
            print(f"Success! Evaded detection after {iteration} iterations.")
            break
        else:
            print(f"Failed to evade. Continuing iteration {iteration + 1}...")

    # Fill remaining iterations with the last email
    if iteration < max_iterations:
        for i in range(iteration + 1, max_iterations + 1):
            iteration_history[f"iteration_{i}"] = {
                "email": new_email,
                "prediction": final_prediction,
                "detection_reason": detection_reason,
                "is_replicated": True,
            }

    return new_email, final_prediction, recheck_result, iteration, iteration_history
