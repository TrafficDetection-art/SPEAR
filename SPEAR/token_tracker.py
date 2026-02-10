"""
Token usage statistics and cost estimation module.

Tracks API token consumption by function, estimates costs based on model pricing.
"""

from datetime import datetime
from config import config


# --------------- Global Token Usage Tracker ---------------

token_usage = {
    "total_prompt_tokens": 0,
    "total_completion_tokens": 0,
    "total_tokens": 0,
    "request_count": 0,
    "usage_by_function": {},
    "detailed_usage": [],
}


def reset_token_usage():
    """Reset the token usage statistics."""
    global token_usage
    token_usage = {
        "total_prompt_tokens": 0,
        "total_completion_tokens": 0,
        "total_tokens": 0,
        "request_count": 0,
        "usage_by_function": {},
        "detailed_usage": [],
    }


def add_token_usage(prompt_tokens, completion_tokens, total_tokens,
                    function_name="unknown", request_info=None):
    """Record a token usage entry."""
    global token_usage

    # Update totals
    token_usage["total_prompt_tokens"] += prompt_tokens
    token_usage["total_completion_tokens"] += completion_tokens
    token_usage["total_tokens"] += total_tokens
    token_usage["request_count"] += 1

    # Per-function breakdown
    if function_name not in token_usage["usage_by_function"]:
        token_usage["usage_by_function"][function_name] = {
            "prompt_tokens": 0,
            "completion_tokens": 0,
            "total_tokens": 0,
            "request_count": 0,
        }

    fn_usage = token_usage["usage_by_function"][function_name]
    fn_usage["prompt_tokens"] += prompt_tokens
    fn_usage["completion_tokens"] += completion_tokens
    fn_usage["total_tokens"] += total_tokens
    fn_usage["request_count"] += 1

    # Detailed log
    token_usage["detailed_usage"].append({
        "timestamp": datetime.now().isoformat(),
        "function": function_name,
        "prompt_tokens": prompt_tokens,
        "completion_tokens": completion_tokens,
        "total_tokens": total_tokens,
        "request_info": request_info or {},
    })


def get_token_usage_summary():
    """Return a summary of token usage."""
    return {
        "total_tokens": token_usage["total_tokens"],
        "total_prompt_tokens": token_usage["total_prompt_tokens"],
        "total_completion_tokens": token_usage["total_completion_tokens"],
        "total_requests": token_usage["request_count"],
        "average_tokens_per_request": (
            token_usage["total_tokens"] / max(token_usage["request_count"], 1)
        ),
        "usage_by_function": token_usage["usage_by_function"],
    }


def get_detailed_cost_analysis():
    """Return detailed cost analysis including per-function breakdown."""
    summary = get_token_usage_summary()

    total_cost = estimate_cost(
        prompt_tokens=summary["total_prompt_tokens"],
        completion_tokens=summary["total_completion_tokens"],
    )

    function_costs = {}
    for function_name, usage in summary["usage_by_function"].items():
        fn_cost = estimate_cost(
            prompt_tokens=usage["prompt_tokens"],
            completion_tokens=usage["completion_tokens"],
        )
        function_costs[function_name] = {
            "prompt_tokens": usage["prompt_tokens"],
            "completion_tokens": usage["completion_tokens"],
            "total_tokens": usage["total_tokens"],
            "request_count": usage["request_count"],
            "input_cost": fn_cost["input_cost"],
            "output_cost": fn_cost["output_cost"],
            "total_cost": fn_cost["total_cost"],
        }

    return {
        "total_cost_breakdown": total_cost,
        "function_costs": function_costs,
        "summary": summary,
    }


def estimate_cost(prompt_tokens=0, completion_tokens=0, model_name=None):
    """
    Estimate API usage cost in USD, with separate input/output token pricing.
    """
    pricing = {
        "gpt-4o": {
            "input": 0.005 / 1000,
            "output": 0.015 / 1000,
        },
        "gpt-4o-mini": {
            "input": 0.00015 / 1000,
            "output": 0.0006 / 1000,
        },
        "gpt-3.5-turbo": {
            "input": 0.0015 / 1000,
            "output": 0.002 / 1000,
        },
        "claude-3.5-sonnet": {
            "input": 0.003 / 1000,
            "output": 0.015 / 1000,
        },
        "gemini-2.5-flash": {
            "input": 0.00007 / 1000,
            "output": 0.00021 / 1000,
        },
        "deepseek": {
            "input": 0.00014 / 1000,
            "output": 0.00028 / 1000,
        },
    }

    if not model_name:
        # Try to get model name from config (compatible with both formats)
        if "attack_model" in config:
            model_name = config["attack_model"].get("model", "default")
        else:
            model_name = "default"

    # Default pricing if model not found
    default_pricing = {"input": 0.001 / 1000, "output": 0.003 / 1000}

    # Find matching model pricing
    model_pricing = default_pricing
    for model_key, price_config in pricing.items():
        if model_key.lower() in model_name.lower():
            model_pricing = price_config
            break

    input_cost = prompt_tokens * model_pricing["input"]
    output_cost = completion_tokens * model_pricing["output"]

    return {
        "input_cost": input_cost,
        "output_cost": output_cost,
        "total_cost": input_cost + output_cost,
        "input_tokens": prompt_tokens,
        "output_tokens": completion_tokens,
        "model": model_name,
    }
