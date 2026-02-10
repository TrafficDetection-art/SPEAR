"""
Smart replacement mapping cache system.

Uses both LLM-based and rule-based approaches to map phishing words
to legitimate words, with a time-based and change-based cache.
"""

import sys; sys.path.insert(0, "..")
from project_settings import get as pget

import sys; sys.path.insert(0, "..")
from project_settings import get as pget

import hashlib
import json
import time

from memory import lime_memory
from llm_client import get_LLM_response_vllm


# --------------- Cache State ---------------

smart_replacement_cache = {
    "mapping": {},
    "last_update": 0,
    "update_interval": 300,  # updated at runtime from config
    "memory_hash": "",
    "update_count": 0,
}


# --------------- Cache Helpers ---------------

def get_memory_hash():
    """Compute a hash of the memory module to detect changes."""
    if not lime_memory:
        return ""

    memory_str = ""
    for key in sorted(lime_memory.keys()):
        memory_str += str(sorted(lime_memory[key].items()))

    return hashlib.md5(memory_str.encode()).hexdigest()


def should_update_smart_mapping():
    """Determine whether the smart mapping table should be updated."""
    current_time = time.time()

    time_condition = (current_time - smart_replacement_cache["last_update"]
                      >= smart_replacement_cache["update_interval"])
    memory_changed = get_memory_hash() != smart_replacement_cache["memory_hash"]
    never_updated = smart_replacement_cache["update_count"] == 0

    return time_condition or memory_changed or never_updated


# --------------- Mapping Generation ---------------

def generate_smart_replacement_mapping(client=None):
    """
    Generate a smart phishing-to-legitimate word mapping using LLM.

    Falls back to rule-based mapping if LLM is unavailable.
    """
    if (not lime_memory
            or not lime_memory.get("phishing_words")
            or not lime_memory.get("legitimate_words")):
        print("Memory data insufficient for smart mapping generation")
        return {}

    top_phishing = sorted(
        lime_memory["phishing_words"].items(), key=lambda x: x[1], reverse=True
    )[:pget("spear.smart_mapping.top_phishing_words", 20)]
    top_legitimate = sorted(
        lime_memory["legitimate_words"].items(), key=lambda x: x[1], reverse=True
    )[:pget("spear.smart_mapping.top_legitimate_words", 30)]

    if not top_phishing or not top_legitimate:
        print("Not enough high-frequency words for smart mapping")
        return {}

    phishing_words = [w for w, _ in top_phishing]
    legitimate_words = [w for w, _ in top_legitimate]

    if not client:
        print("No LLM client provided, using rule-based mapping")
        return generate_basic_smart_mapping(phishing_words, legitimate_words)

    # Build prompt for LLM
    prompt = f"""Based on the phishing email analysis, create a smart word replacement mapping to help evade detection.

High-frequency phishing words (to be replaced):
{', '.join(phishing_words)}

High-frequency legitimate words (for replacement):
{', '.join(legitimate_words)}

Create a JSON mapping where each phishing word is mapped to the most semantically appropriate legitimate word that would help evade phishing detection while maintaining email coherence.

Requirements:
1. Only map phishing words that appear in the provided list
2. Only use legitimate words from the provided list for replacements
3. Consider semantic similarity and context appropriateness
4. Prioritize replacements that maintain email readability
5. Return ONLY a valid JSON object in the format: {{"phishing_word": "legitimate_word", ...}}

Example format:
{{"urgent": "important", "verify": "check", "account": "profile"}}"""

    try:
        response = get_LLM_response_vllm(client, prompt, role="attack")

        start_idx = response.find('{')
        end_idx = response.rfind('}') + 1

        if start_idx != -1 and end_idx > start_idx:
            json_str = response[start_idx:end_idx]
            mapping = json.loads(json_str)

            # Validate mapping
            phishing_lower = {w.lower() for w in phishing_words}
            legitimate_lower = {w.lower() for w in legitimate_words}

            validated = {
                pw.lower(): lw.lower()
                for pw, lw in mapping.items()
                if pw.lower() in phishing_lower and lw.lower() in legitimate_lower
            }

            print(f"LLM generated smart mapping: {len(validated)} valid mappings")
            return validated
        else:
            print("LLM response format invalid, falling back to rule-based mapping")
            return generate_basic_smart_mapping(phishing_words, legitimate_words)

    except Exception as e:
        print(f"LLM smart mapping failed: {e}, falling back to rule-based mapping")
        return generate_basic_smart_mapping(phishing_words, legitimate_words)


def generate_basic_smart_mapping(phishing_words, legitimate_words):
    """Rule-based smart mapping generation (no LLM dependency)."""
    mapping = {}

    # Predefined semantic mapping rules
    semantic_rules = {
        'urgent': ['important', 'normal', 'regular'],
        'immediate': ['soon', 'quick', 'fast'],
        'verify': ['check', 'review', 'confirm'],
        'account': ['profile', 'information', 'details'],
        'suspended': ['updated', 'modified', 'changed'],
        'expire': ['change', 'update', 'renew'],
        'security': ['safety', 'protection', 'information'],
        'alert': ['notice', 'message', 'information'],
        'warning': ['notice', 'message', 'information'],
        'action': ['step', 'process', 'procedure'],
        'required': ['needed', 'necessary', 'important'],
        'confirm': ['check', 'review', 'verify'],
        'update': ['change', 'modify', 'adjust'],
        'login': ['access', 'entry', 'sign'],
        'click': ['visit', 'see', 'view'],
        'link': ['website', 'page', 'site'],
    }

    legitimate_lower = [w.lower() for w in legitimate_words]

    for phishing_word in phishing_words:
        pw_lower = phishing_word.lower()

        # 1. Try semantic rule matching
        if pw_lower in semantic_rules:
            for candidate in semantic_rules[pw_lower]:
                if candidate in legitimate_lower:
                    mapping[pw_lower] = candidate
                    break

        # 2. Try length + first-letter matching
        if pw_lower not in mapping:
            first_letter = phishing_word[0].lower()
            word_len = len(phishing_word)

            candidates = [
                w for w in legitimate_words
                if abs(len(w) - word_len) <= pget("spear.smart_mapping.length_similarity_threshold", 2)
                and w[0].lower() == first_letter
                and w.lower() != pw_lower
            ]

            if candidates:
                mapping[pw_lower] = candidates[0].lower()
            elif legitimate_words:
                mapping[pw_lower] = legitimate_words[0].lower()

    print(f"Rule-based mapping generated: {len(mapping)} mappings")
    return mapping


# --------------- Public API ---------------

def get_smart_replacement_map(client=None):
    """
    Get the smart replacement mapping table, using cache when possible.

    Args:
        client: Optional LLM client for generating LLM-based mappings.

    Returns:
        dict: {phishing_word: legitimate_word} mapping.
    """
    # Return cache if still valid
    if not should_update_smart_mapping() and smart_replacement_cache["mapping"]:
        print(f"Using cached smart mapping ({len(smart_replacement_cache['mapping'])} mappings)")
        return smart_replacement_cache["mapping"]

    # Need to update
    if should_update_smart_mapping():
        print(f"Triggering smart mapping update "
              f"(update #{smart_replacement_cache['update_count'] + 1})")
        new_mapping = generate_smart_replacement_mapping(client)

        if new_mapping:
            smart_replacement_cache["mapping"] = new_mapping
            smart_replacement_cache["last_update"] = time.time()
            smart_replacement_cache["memory_hash"] = get_memory_hash()
            smart_replacement_cache["update_count"] += 1
            print(f"Smart mapping updated: {len(new_mapping)} mappings")
        else:
            print("Smart mapping update failed, keeping existing cache")

    if smart_replacement_cache["mapping"]:
        return smart_replacement_cache["mapping"]

    print("Smart mapping is empty, returning empty dict")
    return {}
