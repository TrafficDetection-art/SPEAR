"""Temporary script to update SPEAR files with config imports."""

def update_file(filepath, replacements):
    with open(filepath, 'r', encoding='utf-8') as f:
        content = f.read()
    for old, new in replacements:
        if old in content:
            content = content.replace(old, new)
            print(f"  OK: {old[:50]}...")
        else:
            print(f"  SKIP (not found): {old[:50]}...")
    with open(filepath, 'w', encoding='utf-8') as f:
        f.write(content)

# lime_attack.py
print("=== lime_attack.py ===")
update_file("lime_attack.py", [
    ('"""\n\nimport re\nimport random',
     '"""\n\nimport sys; sys.path.insert(0, \'..\')\nfrom project_settings import get as pget\n\nimport re\nimport random'),
    ('update_memory(analysis_result, score_threshold=0.02)',
     'update_memory(analysis_result, score_threshold=pget("spear.lime.score_threshold", 0.02))'),
    ('memory_context = get_memory_context(top_k=10)',
     'memory_context = get_memory_context(top_k=pget("spear.lime.memory_top_k", 10))'),
    ('        ml_model_names = ["textcnn"]',
     '        ml_model_names = pget("spear.lime.default_models", ["textcnn"])'),
    ('                strategies = ["replace", "delete", "char_replace"]',
     '                strategies = pget("spear.lime.preprocessing_strategies", ["replace", "delete", "char_replace"])'),
    ('                thresholds = [0.01, 0.02, 0.03]',
     '                thresholds = pget("spear.lime.preprocessing_thresholds", [0.01, 0.02, 0.03])'),
])

# generate_LLM.py
print("\n=== generate_LLM.py ===")
update_file("generate_LLM.py", [
    ('import json\nimport time\nimport os',
     'import sys; sys.path.insert(0, \'..\')\nfrom project_settings import get as pget\n\nimport json\nimport time\nimport os'),
    ('def load_config(config_path="config.json"):',
     'def load_config(config_path=None):'),
    ('    with open(config_path, "r") as config_file:',
     '    if config_path is None:\n        config_path = pget("spear.config_path", "config.json")\n    with open(config_path, "r") as config_file:'),
    ('    base_output_dir = "outputs"',
     '    base_output_dir = pget("spear.output_base_dir", "outputs")'),
    ('    emails_per_type = 5',
     '    emails_per_type = pget("spear.generate_llm.emails_per_type", 5)'),
    ('            time.sleep(1)  # Avoid API rate limits',
     '            time.sleep(pget("spear.generate_llm.api_delay_seconds", 1))  # Avoid API rate limits'),
])

# config.py
print("\n=== config.py ===")
update_file("config.py", [
    ('import json\nimport os',
     'import sys; sys.path.insert(0, \'..\')\nfrom project_settings import get as pget\n\nimport json\nimport os'),
    ('def load_config(config_path="./config.json"):',
     'def load_config(config_path=None):'),
])

# Add None-check to config.py load_config
with open("config.py", 'r') as f:
    c = f.read()
old = '    """Load configuration file and return config data."""\n    with open'
new = '    """Load configuration file and return config data."""\n    if config_path is None:\n        config_path = pget("spear.config_path", "./config.json")\n    with open'
if old in c:
    c = c.replace(old, new)
    with open("config.py", 'w') as f:
        f.write(c)
    print("  config.py body updated")
else:
    print("  config.py body not found")

print("\nDone!")
