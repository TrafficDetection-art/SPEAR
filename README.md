# SPEAR: System for Personalized Email Adversarial Refinement

**Title:** *Emerging Threats of LLM-Powered Phishing: Personalized and Adversarial Email Attacks via Multi-Agent Systems*  

---

## 🔍 Overview

**SPEAR** is a fully automated, modular system designed to simulate end-to-end spear-phishing attacks using large language models (LLMs). It integrates real-time profiling, personalized email generation, adversarial refinement, and multi-layered evaluation to highlight the escalating threat landscape of LLM-assisted phishing in adversarial contexts.

> ⚠️ *This project is currently under peer review. Certain implementation details may be anonymized or withheld in accordance with double-blind submission guidelines.*
>
> ⚠️ *Ethical scope: The released artifacts are intended for research-only reproduction on public phishing datasets. They support text-only generation/rewriting and evaluation against detectors, while excluding operational components such as email delivery, automated sending, commercial gateway probing, and any target-specific profiling.
---


## Environment Setup

### System Requirements
	•	OS: Linux / macOS / Windows (Linux recommended)
	•	Python: 3.10+ (recommended: 3.11)
	•	Optional GPU: recommended for BERT/Transformer models (CPU-only is supported but slower)


### Install Dependencies
If you have a top-level requirements file:

pip install -r requirements.txt

Or if dependencies are split by modules (recommended for clarity):

```
pip install -r requirements.txt
pip install -r SPEAR/requirements.txt
```

## Experimental Replication

### Train Traditional ML Models
```
cd ML
python train_body.py --data_path=../dataset/email_data.json
```

### Train Deep Learning Models (e.g., TextCNN / BERT)
```
cd DL
python main.py --data_path=../dataset/email_data.json
```

### Attack/Refinement Against ML/DL/LLM Detectors

*LIME attacks*
```
python new_multi_agent_with_lime.py     --data_path ../dataset/test_data.json     --lime_models textcnn bert     --max_iterations 5     --samples_per_type 100 --enable_lime_attack true
```

*LLM attacks*
```
python new_multi_agent_with_lime.py     --data_path ../dataset/test_data.json  --max_iterations 5     --samples_per_type 100 --enable_llm_attack true
```

### Tips

> ⚠️ **LLM API Configuration Required**  
> To run the LLM-dependent modules (e.g., controlled evaluation), you must configure your own LLM API credentials in `./SPEAR/config.json`.

**Steps**
1. Open `./SPEAR/config.json`
2. Fill in `api_key` and `api_base_url` for the required models.
3. **Do NOT commit** your API key to any public repository.

**Example (`./SPEAR/config.json`)**
```json
{
  "attack_model": {
    "model": "gpt-4o",
    "api_key": "YOUR_API_KEY",
    "api_base_url": "YOUR_BASE_URL",
    "use_minimal_params": true
  },
  "defense_model": {
    "model": "YOUR_MODEL_NAME",
    "api_key": "YOUR_API_KEY",
    "api_base_url": "YOUR_BASE_URL",
    "use_minimal_params": true
  }
}
```

Apart from ./SPEAR/config.json, this project centralizes most paths and hyperparameters in `project_config.json`.
- **Hyperparameters & module configs:** see **[CONFIG.md](CONFIG.md)**.


## 💡 Feature Highlights

- 🎯 **Customization via LLM-Driven Profiling**  **(Not yet open-sourced)**
  The system uses search engine results and LLMs to dynamically construct rich **user profiles**, which guide the generation of **highly targeted** phishing emails.

- 🤖 **Adversarial Phishing Generation (AI vs AI)**  
  SPEAR integrates adversarial feedback loops combining **LIME interpretability** and **LLM feedback** to evolve emails that evade both **traditional ML/DL detectors** and **LLM-based classifiers**.

- 📊 **LLM-Based Quality Scoring**  
  Emails are evaluated using in-context LLM critics, which assess clarity, persuasiveness, and realism to promote high-quality attack samples.

- 🛡️ **Comprehensive Evaluation & Defense Evasion**  
  Our pipeline is tested against a **wide spectrum of defenses**, including:
  - Machine Learning classifiers (e.g., SVM, Random Forest)
  - Deep Learning models (e.g., TextCNN, BERT)
  - LLM-based phishing detectors
  - Commercial-grade **Email Gateways**
  - **Human Evaluation** (via human-in-the-loop experiments)

Results demonstrating SPEAR’s evasion capability are discussed in the paper.

---

## 🧠 Key Capabilities

- 🔎 **Personalized Phishing Email Generation** from real-world profiling data  
- 🧩 **Multi-Agent Pipeline** with modular agents for profiling, generation, refinement, and evaluation  
- ⚔️ **Adversarial Optimization Loop** combining LIME-based interpretability and LLM-guided feedback  
- 🛡️ **Evaluation Against Defenses** including ML/DL-based detectors and LLM-based classifiers  
- 🔁 **End-to-End Automation** from web-based target modeling to simulated email delivery

---

## 🧭 Ethical Use & Disclosure

This project is developed **solely for security research**, to demonstrate potential misuse of LLMs in automated phishing generation and to **encourage the development of stronger defenses**.

- ⚠️ **Strictly No Malicious Use**: SPEAR is **not intended** for deployment or malicious use in any real-world setting.
- 🔐 **Partial Disclosure**: In compliance with responsible disclosure practices and ethical considerations, certain sensitive components (e.g., full delivery modules, deployment scripts) are intentionally **withheld or redacted**.
- 📢 **Call to the Community**: We encourage the broader security and ML communities to use this work as a benchmark and cautionary case for building **robust anti-LLM phishing defenses**.

---

## 📌 Notes for Reviewers

- **Reproducibility:** All experiments are conducted in controlled environments, including evaluations against a variety of detectors and human annotators.
- **Ethical Disclosure:** This system is intended strictly for research and awareness under responsible disclosure and red-teaming frameworks.
- **Blindness Compliance:** All identifying information (e.g., author names, institutions, repositories) has been redacted in accordance with double-blind review requirements.
