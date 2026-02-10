"""
LIME memory module.

Stores and retrieves high-frequency phishing/legitimate words and bigrams
discovered through LIME analysis across iterations.
"""

import sys; sys.path.insert(0, '..')
from project_settings import get as pget

import re
from text_utils import clean_text_for_tokenization


# --------------- Global Memory Store ---------------

lime_memory = {
    "phishing_words": {},      # {word: count}
    "legitimate_words": {},    # {word: count}
    "phishing_bigrams": {},    # {bigram: count}
    "legitimate_bigrams": {},  # {bigram: count}
}


# --------------- Word Similarity ---------------

def calculate_word_similarity(word1, word2):
    """Calculate a simple similarity score between two words."""
    if word1 == word2:
        return 0.0  # Identical words are not suitable for replacement

    # Length similarity
    len_similarity = 1.0 - abs(len(word1) - len(word2)) / max(len(word1), len(word2))
    # First-letter similarity
    first_letter_similarity = 1.0 if word1[0].lower() == word2[0].lower() else 0.0
    # Character overlap similarity
    common_chars = set(word1.lower()) & set(word2.lower())
    char_similarity = len(common_chars) / max(len(set(word1.lower())), len(set(word2.lower())))

    weights = pget("spear.memory.similarity_weights", [0.4, 0.3, 0.3])
    return len_similarity * weights[0] + first_letter_similarity * weights[1] + char_similarity * weights[2]


# --------------- Dynamic Replacement Map ---------------

def generate_dynamic_replacement_map(min_phishing_count=None, min_legitimate_count=None,
                                     similarity_threshold=None):
    """
    Dynamically generate a phishing-to-legitimate word replacement mapping
    based on the memory module.
    """
    if min_phishing_count is None:
        min_phishing_count = pget("spear.memory.min_phishing_count", 2)
    if min_legitimate_count is None:
        min_legitimate_count = pget("spear.memory.min_legitimate_count", 2)
    if similarity_threshold is None:
        similarity_threshold = pget("spear.memory.similarity_threshold", 0.3)

    min_word_len = pget("spear.memory.min_word_length", 3)

    if (not lime_memory
            or not lime_memory.get("phishing_words")
            or not lime_memory.get("legitimate_words")):
        return {}

    high_freq_phishing = {
        w: c for w, c in lime_memory["phishing_words"].items()
        if c >= min_phishing_count and len(w) >= min_word_len
    }
    high_freq_legitimate = {
        w: c for w, c in lime_memory["legitimate_words"].items()
        if c >= min_legitimate_count and len(w) >= min_word_len
    }

    if not high_freq_phishing or not high_freq_legitimate:
        return {}

    legitimate_words = list(high_freq_legitimate.keys())
    replacement_map = {}

    print(f"Generating dynamic replacement map: "
          f"{len(high_freq_phishing)} phishing words -> {len(legitimate_words)} legitimate words")

    for phishing_word in high_freq_phishing:
        best_match = None
        best_similarity = 0.0

        for legitimate_word in legitimate_words:
            similarity = calculate_word_similarity(phishing_word, legitimate_word)
            if similarity > best_similarity and similarity >= similarity_threshold:
                best_similarity = similarity
                best_match = legitimate_word

        if best_match:
            replacement_map[phishing_word.lower()] = best_match
            print(f"  Mapping: {phishing_word} -> {best_match} (similarity: {best_similarity:.3f})")
        else:
            most_frequent = max(high_freq_legitimate.items(), key=lambda x: x[1])[0]
            replacement_map[phishing_word.lower()] = most_frequent
            print(f"  Mapping: {phishing_word} -> {most_frequent} (default high-freq)")

    print(f"Generated {len(replacement_map)} dynamic mappings")
    return replacement_map


# --------------- Memory Update ---------------

def _update_from_detailed_ngrams(detailed_ngrams, score_threshold):
    """Update memory from the new detailed_ngrams data structure."""
    min_word_len = pget("spear.memory.min_word_length", 3)

    categories = [
        ("phishing_unigrams", "phishing_words", True),
        ("phishing_bigrams", "phishing_bigrams", False),
        ("normal_unigrams", "legitimate_words", True),
        ("normal_bigrams", "legitimate_bigrams", False),
    ]

    for ngram_key, memory_key, is_unigram in categories:
        items = detailed_ngrams.get(ngram_key, [])
        if not isinstance(items, list):
            continue

        added_count = 0
        for item in items:
            if not (isinstance(item, (list, tuple)) and len(item) >= 2):
                continue
            word, score = item[0], item[1]
            if not isinstance(word, str) or not isinstance(score, (int, float)):
                continue
            if abs(score) <= score_threshold:
                continue

            if is_unigram:
                if len(word) < min_word_len:
                    continue
            else:
                if len(word.split()) != 2:
                    continue

            lime_memory[memory_key][word] = lime_memory[memory_key].get(word, 0) + 1
            added_count += 1

        label = ngram_key.replace("_", " ").capitalize()
        print(f"  {label}: added {added_count}/{len(items)} "
              f"high-weight features (threshold: {score_threshold})")


def _update_from_legacy_format(lime_analysis, score_threshold):
    """Update memory from the legacy LIME analysis format."""
    min_word_len = pget("spear.memory.min_word_length", 3)

    for field, memory_key, label in [
        ("high_weight_phishing_words", "phishing_words", "phishing"),
        ("legitimate_words", "legitimate_words", "legitimate"),
    ]:
        words_str = lime_analysis.get(field, "")
        if not isinstance(words_str, str) or not words_str.strip():
            continue

        cleaned = clean_text_for_tokenization(words_str)
        words = re.findall(r'\b\w+\b', cleaned.lower())
        added = 0
        for word in words:
            if len(word) >= min_word_len:
                lime_memory[memory_key][word] = lime_memory[memory_key].get(word, 0) + 1
                added += 1
        print(f"  Legacy {label} words: added {added} words (legacy format, assumed high-weight)")


def update_memory(lime_analysis, score_threshold=None):
    """Update memory with LIME analysis results."""
    try:
        if score_threshold is None:
            score_threshold = pget("spear.lime.score_threshold", 0.02)

        if not lime_analysis or not isinstance(lime_analysis, dict):
            print("Warning: LIME analysis result is empty or invalid")
            return

        print(f"Updating memory module, score threshold: {score_threshold}")

        if "detailed_ngrams" in lime_analysis and isinstance(lime_analysis["detailed_ngrams"], dict):
            _update_from_detailed_ngrams(lime_analysis["detailed_ngrams"], score_threshold)
        else:
            _update_from_legacy_format(lime_analysis, score_threshold)

        total_phishing = len(lime_memory["phishing_words"]) + len(lime_memory["phishing_bigrams"])
        total_legitimate = len(lime_memory["legitimate_words"]) + len(lime_memory["legitimate_bigrams"])
        print(f"Memory update complete: {total_phishing} phishing features, "
              f"{total_legitimate} legitimate features")

    except Exception as e:
        print(f"Error updating memory: {e}")
        import traceback
        traceback.print_exc()


# --------------- Memory Context ---------------

def get_memory_context(top_k=None):
    """Retrieve high-frequency words/n-grams from memory for prompt enhancement."""
    if top_k is None:
        top_k = pget("spear.lime.memory_top_k", 10)

    if not any(lime_memory.values()):
        return ""

    half_k = top_k // 2

    top_phishing_words = sorted(
        lime_memory["phishing_words"].items(), key=lambda x: x[1], reverse=True
    )[:half_k]
    top_phishing_bigrams = sorted(
        lime_memory["phishing_bigrams"].items(), key=lambda x: x[1], reverse=True
    )[:half_k]

    top_legitimate_words = sorted(
        lime_memory["legitimate_words"].items(), key=lambda x: x[1], reverse=True
    )[:half_k]
    top_legitimate_bigrams = sorted(
        lime_memory["legitimate_bigrams"].items(), key=lambda x: x[1], reverse=True
    )[:half_k]

    phishing_features = (
        [word for word, _ in top_phishing_words]
        + [f'"{phrase}"' for phrase, _ in top_phishing_bigrams]
    )
    legitimate_features = (
        [word for word, _ in top_legitimate_words]
        + [f'"{phrase}"' for phrase, _ in top_legitimate_bigrams]
    )

    context = "\nPrevious LIME analysis memory:\n"
    context += f"- Frequent phishing features to avoid: {', '.join(phishing_features[:top_k])}\n"
    context += f"- Frequent legitimate features to use: {', '.join(legitimate_features[:top_k])}\n"
    context += "Use this knowledge to improve evasion."
    return context
