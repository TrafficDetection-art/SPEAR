# 📧 SPEAR

This module is part of the SPEAR framework and is used to generate email samples for phishing likelihood, semantic quality, realism, and personalization level.

**Note**: Datasets are temporarily not uploaded to avoid interference with the double-blind review process.

---

## 🚀 Quick Start

### 1. Install Dependencies

Ensure you are using Python 3.10+, then install the required packages:

```bash
pip install -r requirements.txt
```

---

### 2. Configure API Key

You can set your OpenAI API key using one of the following methods:

```bash
# Edit config.json and replace the placeholder with your actual API key
```

---

### 3. Run the Evaluation Script

From the project root directory, run the following command:

```bash
cd SPEAR/
python new_multi_agent_with_lime.py \
  --data_source personal_info \
  --personal_info_path ./input.json
```

To enable the adversarial module (LIME + LLM attacks), use the following:

⚠️ **Note**: For LIME attacks to work, corresponding deep learning models must be pre-trained.

```bash
python new_multi_agent_with_lime.py \
  --data_source personal_info \
  --personal_info_path ./input.json \
  --enable_llm_attack true \
  --enable_lime_attack true
```

---

## 📁 Input Format

The script expects a JSON file (not CSV). Each email object should contain:

| Field | Description |
|-------|-------------|
| id | Unique identifier for each email |
| Text | Raw email content |
| Class | Label (0 = legitimate, 1 = phishing) |
| type | Email type or category |
| source_file | Source file name of the sample |
| Phishing | (To be filled) LLM-detected phishing label |

---

## 📤 Output Files

After processing, a timestamped results folder will be created, containing:

### ✅ completed_results.csv

Includes the original columns plus the following:

- **Phishing (0 / 1 / 2)**:
  - 0: Legitimate
  - 1: Phishing
  - 2: Uncertain
- **Semantic Quality**:
  - High / Medium / Low
- **Realism**:
  - Real / Fake / Uncertain
- **Personalization**:
  - High / Medium / Low

---


## 🔒 Ethical Use Notice

This tool is developed strictly for research purposes under the SPEAR project, and must not be used in any real-world attack, deployment, or detection product.

- ❌ This is not a commercial product
- ⚠️ Any misuse or abuse is strictly discouraged
- 🔐 Certain sensitive components (e.g., email delivery code) are intentionally omitted to prevent malicious use

---
