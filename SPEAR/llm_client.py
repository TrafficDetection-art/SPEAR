"""
OpenAI client management and LLM interaction module.

Handles client creation (including local Qwen model support, SSL config)
and sending requests to the LLM with retry logic and token tracking.
"""

import sys
sys.path.insert(0, '..')
from project_settings import get as pget

import time
import httpx
from openai import OpenAI

from config import config
from token_tracker import add_token_usage


# ========================== Client Creation ==========================

def create_client(role="attack"):
    """
    Create an OpenAI client instance (supports local models and SSL config).

    Args:
        role: "attack" for attack-side client, "defense" for defense-side client.

    Returns:
        An OpenAI client instance or a local model adapter.
    """
    if role not in ("attack", "defense"):
        raise ValueError(f"Invalid role: {role}. Must be 'attack' or 'defense'")

    model_config = config[f"{role}_model"]
    base_url = model_config["api_base_url"].rstrip('/')

    # Detect local Qwen model (by localhost/127.0.0.1 in URL)
    is_local_qwen = "localhost" in base_url or "127.0.0.1" in base_url

    if is_local_qwen and "/api/infer" not in base_url:
        print("Detected local Qwen model, using adapter")
        try:
            from qwen_api_adapter import QwenClient
            client = QwenClient(
                api_key=model_config["api_key"],
                base_url=base_url,
            )
            print(f"Created {role} client (Qwen adapter): model={model_config.get('model', 'unknown')}")
            return client
        except ImportError:
            print("Warning: Cannot import Qwen adapter, trying standard OpenAI client")

    # Check if SSL verification should be disabled
    disable_ssl = model_config.get("disable_ssl_verify", False)

    if disable_ssl:
        print(f"Warning: SSL verification disabled for {role} client (not recommended for production)")
        http_client = httpx.Client(verify=False)
        client = OpenAI(
            api_key=model_config["api_key"],
            base_url=base_url,
            http_client=http_client,
        )
    else:
        client = OpenAI(
            api_key=model_config["api_key"],
            base_url=base_url,
        )

    print(f"Created {role} client: model={model_config.get('model', 'unknown')}")
    return client


# ========================== LLM Interaction ==========================

def get_LLM_response_vllm(client, prompt, need_sample=False,
                           function_name="unknown", role="attack"):
    """
    Send a request to the OpenAI-compatible LLM with retry logic.

    Args:
        client: OpenAI client instance.
        prompt: The prompt text.
        need_sample: Whether to return multiple sampled responses.
        function_name: Name for token-tracking purposes.
        role: "attack" or "defense" (determines model config).

    Returns:
        A string response, or a list of strings if need_sample is True.
    """
    sample_num = 3 if need_sample else 1
    conversation = [{"role": "user", "content": prompt}]
    model = config[f"{role}_model"]["model"]
    max_tokens_for_completion = 2048

    max_retries = 5
    delay = 2

    for attempt in range(max_retries):
        try:
            use_minimal_params = config[f"{role}_model"].get("use_minimal_params", False)

            api_params = {
                "model": model,
                "messages": conversation,
                "max_tokens": max_tokens_for_completion,
            }

            if not use_minimal_params:
                api_params["n"] = sample_num
                api_params["temperature"] = pget("spear.llm.temperature", 1.0)
                api_params["top_p"] = pget("spear.llm.top_p", 0.95)
            else:
                if sample_num > 1:
                    api_params["n"] = sample_num
                api_params["temperature"] = pget("spear.llm.temperature", 1.0)

            chat_response = client.chat.completions.create(**api_params)

            # Record token usage
            if hasattr(chat_response, 'usage') and chat_response.usage:
                usage = chat_response.usage
                add_token_usage(
                    prompt_tokens=getattr(usage, 'prompt_tokens', 0),
                    completion_tokens=getattr(usage, 'completion_tokens', 0),
                    total_tokens=getattr(usage, 'total_tokens', 0),
                    function_name=function_name,
                    request_info={
                        "model": model,
                        "sample_num": sample_num,
                        "attempt": attempt + 1,
                    },
                )
            else:
                # Estimate token counts when usage info is unavailable
                est_prompt = int(len(prompt.split()) * 1.3)
                est_completion = (
                    int(len(chat_response.choices[0].message.content.split()) * 1.3)
                    if chat_response.choices else 0
                )
                add_token_usage(
                    prompt_tokens=est_prompt,
                    completion_tokens=est_completion,
                    total_tokens=est_prompt + est_completion,
                    function_name=function_name + "_estimated",
                    request_info={
                        "model": model,
                        "sample_num": sample_num,
                        "attempt": attempt + 1,
                        "note": "estimated_usage",
                    },
                )

            if need_sample:
                return [choice.message.model_dump()['content'] for choice in chat_response.choices]
            return chat_response.choices[0].message.model_dump()['content']

        except Exception as e:
            print(f"Attempt {attempt + 1} failed: {e}. Retrying in {delay} seconds...")
            time.sleep(delay)

        time.sleep(delay)

    return "Error: LLM request failed."
