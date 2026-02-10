## Hyperparameters & Central Configuration

All important hyperparameters and paths are centralized in `project_config.json` to improve reproducibility and reduce hard-coded values across modules.

> **Note:** For fair comparison and reproducibility, we recommend keeping `random_seed`, dataset splits, and model/training settings unchanged unless explicitly stated.

### 1) File Location
- Central config: `project_config.json`
- LLM API config (separate file): `./SPEAR/config.json`

### 2) Configuration Overview

#### 2.1 `general`
Controls global runtime behavior.
- `device`: compute device (e.g., `cuda:0`, `cpu`)
- `gpu_id`: GPU index (if using CUDA)
- `random_seed`: global seed used across modules (recommended to keep fixed)
- `proxy.http` / `proxy.https`: optional proxy settings (leave empty if not needed)

#### 2.2 `paths`
Defines dataset and split file paths.
- `dataset_dir`: dataset directory
- `dataset_file`: main dataset JSON (often used for training)
- `train_data_file`, `test_data_file`: explicit split files if pre-generated

> Recommended: If you generate splits dynamically, document the split ratios and the seed. If you provide fixed split files, ensure they are consistent across runs.

---

### 3) Deep Learning Module (`dl`)

#### 3.1 Directories
- `models_dir`: where DL checkpoints are stored
- `results_dir`: where evaluation outputs are written
- `tokenizer_name`: tokenizer backbone (default: `bert-base-uncased`)

#### 3.2 Shared Model Settings (`dl.model`)
Common settings used across architectures:
- `embed_size`: embedding dimension for non-transformer models
- `num_classes`: classification label count (e.g., 2 for binary)
- `max_len`: maximum sequence length

#### 3.3 Architecture-Specific Hyperparameters (`dl.model_architectures`)
Defines per-model hyperparameters:
- `cnn_lstm`: CNN + LSTM hybrid settings (`conv_*`, `lstm_hidden_size`, `dropout`)
- `textcnn`: filter count, kernel sizes, dropout
- `dnn`: hidden layer sizes and dropout
- `deeplog`: hidden size and number of layers

#### 3.4 Training Hyperparameters (`dl.training`)
- `learning_rate`: optimizer learning rate
- `train_batch_size`, `eval_batch_size`
- `num_epochs`
- `weight_decay`
- `test_split_ratio`, `val_split_ratio`: dataset split ratios (keep constant for reproducibility)

#### 3.5 Testing / LIME-Related Settings (`dl.testing`)
- `batch_size`: inference batch size
- `num_lime_features`: number of features used for interpretability output
- `max_length`: max token length during evaluation

#### 3.6 Transformer Model Registry (`dl.transformer_models`)
Shortcut names for transformer backbones:
- `bert`, `RoBERTa`, `DistilBERT`

---

### 4) Traditional ML Module (`ml`)

#### 4.1 Directories and Vectorizers
- `models_dir`, `results_dir`
- `vectorizer_file`: serialized feature extractor (e.g., TF-IDF)

#### 4.2 Training Settings (`ml.training`)
- `test_size`: test set proportion
- `random_state`: seed for splitting
- `max_features`: vocabulary size / feature limit

#### 4.3 Model Selection (`ml.model_names` and `ml.model_params`)
- `model_names`: enabled ML baselines (e.g., `LR`, `RFC`, `NB`, `SVC`)
- `model_params`: per-model configuration (e.g., linear SVM kernel)

#### 4.4 Optional Pipeline/Processing (`ml.process`)
Used for certain preprocessing and combined scoring workflows:
- model/vectorizer paths
- input/output directories
- `combination_weights`, `classification_threshold`: scoring/threshold parameters (document if used in your experiments)

#### 4.5 Feature Importance / Interpretability (`ml.feature_importance`)
Parameters for interpretability and analysis:
- `model_name`: which baseline to explain (default: `LR`)
- `top_features_display`, `top_features_plot`
- plot settings (`figure_size`, `dpi`)
- LIME-related settings (`lime_n_samples`, replacement rules, token patterns)
- number of samples to explain and preview lengths

---

### 5) SPEAR Module (`spear`)

#### 5.1 Paths and Outputs
- `config_path`: points to `./SPEAR/config.json` (LLM API configuration)
- `data_path`: internal processed data path (if used)
- `personal_info_path`: optional profile/persona file (if applicable)
- `output_base_dir`: base directory for outputs (recommended structure: `outputs/`)

#### 5.2 LLM Runtime Settings (`spear.llm`)
These parameters affect LLM usage and stability:
- `max_tokens`: response token limit
- `max_retries`, `retry_delay`: retry policy on transient failures
- `sample_count`: number of candidates generated per step (if used)
- `temperature`, `top_p`: sampling parameters
- `token_estimation_multiplier`: safety factor for token estimation

#### 5.3 Interpretability / Analysis Settings (`spear.lime`)
- `default_models`: default detector models used in analysis (e.g., `textcnn`, `bert`)
- `score_threshold`: threshold for selecting salient features
- `memory_top_k`: how many prior items to keep in memory buffer
- `preprocessing_strategies`: available preprocessing operations
- `preprocessing_thresholds`: thresholds for applying preprocessing

#### 5.4 Memory and Similarity (`spear.memory`)
- minimum examples per class
- similarity thresholds and token constraints
- weights used for similarity aggregation

#### 5.5 Smart Mapping (`spear.smart_mapping`)
- update interval
- number of top words to track per class
- length similarity threshold

#### 5.6 Text Validation (`spear.text_validation`)
Sanity checks / filtering rules for generated text:
- minimum length, word count
- special character ratio
- ASCII ratio
- token length limits and estimation parameters

#### 5.7 Main Loop Defaults (`spear.main`)
- `samples_per_type`: sample budget per scenario/type
- `max_iterations`: number of refinement rounds
- `generation_delay_seconds`: delay between generations (rate limiting / stability)
- `scenarios`: scenario templates used for controlled experiments

#### 5.8 Generation Defaults (`spear.generate_llm`)
- `emails_per_type`
- `api_delay_seconds`: rate limiting between API calls
- `email_types`: category list used during generation

---

### 6) Recommended Reproducibility Checklist
- Keep these fixed unless you explicitly study sensitivity:
  - `general.random_seed`
  - dataset split parameters (`dl.training.*_split_ratio`, `ml.training.test_size`, `random_state`)
  - model architecture hyperparameters and training settings
- Log and report:
  - `pip freeze` output
  - hardware (CPU/GPU) and CUDA version (if applicable)
  - exact config file used for each run (commit hash + config snapshot)
