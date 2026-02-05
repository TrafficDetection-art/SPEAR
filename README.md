# SPEAR: System for Personalized Email Adversarial Refinement

📄 **S&P 2026 Submission**  
**Title:** *Emerging Threats of LLM-Powered Phishing: Personalized and Adversarial Email Attacks via Multi-Agent Systems*  
**Status:** Under Review (Double-Blind Submission)  
**Conference:** IEEE Symposium on Security and Privacy (S&P) Symposium 2026  

---

## 🔍 Overview

**SPEAR** is a fully automated, modular system designed to simulate end-to-end spear-phishing attacks using large language models (LLMs). It integrates real-time profiling, personalized email generation, adversarial refinement, and multi-layered evaluation to highlight the escalating threat landscape of LLM-assisted phishing in adversarial contexts.

> ⚠️ *This project is currently under peer review. Certain implementation details may be anonymized or withheld in accordance with double-blind submission guidelines.*

---


## 💡 Feature Highlights

- ✅ **One-Line Input to Full Attack**  
  Given just a single input — a tuple of **{organization, name}** — SPEAR automatically performs profiling, custom email generation, adversarial refinement, and simulated delivery without human intervention.

- 🎯 **Customization via LLM-Driven Profiling**  
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
