import torch
import numpy as np
import re
import nltk
from nltk.util import ngrams
from nltk.tokenize import word_tokenize
from lime.lime_text import LimeTextExplainer
from transformers import AutoTokenizer, AutoModelForSequenceClassification
import os
import sys
import string

# Add DL directory to path for imports
sys.path.append(os.path.join(os.path.dirname(os.path.dirname(__file__)), 'DL'))
from models import TextCNNClassifier, CNNLSTMClassifier, DNNClassifier

# Download necessary NLTK data if not already present
try:
    nltk.data.find('tokenizers/punkt')
except LookupError:
    nltk.download('punkt')

class LimeAnalyzer:
    def __init__(self, model_config, device='cuda' if torch.cuda.is_available() else 'cpu'):
        """
        Initialize the LIME analyzer with a specific model configuration
        
        Args:
            model_config: Dictionary containing model name, path, type, and class
            device: Computing device (CPU or CUDA)
        """
        self.model_config = model_config
        self.device = device
        self.model_name = model_config['name']
        self.model_type = model_config['type']
        self.tokenizer = AutoTokenizer.from_pretrained("bert-base-uncased")
        
        # Custom config for non-transformer models - moved before _load_model()
        self.custom_config = {
            "vocab_size": self.tokenizer.vocab_size,
            "embed_size": 512,
            "num_classes": 2,
            "max_len": 512
        }
        
        # Load the model
        self.model = self._load_model()
        self.model.to(self.device)
        self.model.eval()
        
        # Initialize the LIME explainer with proper configuration
        self._initialize_lime_explainer()
    
    def _initialize_lime_explainer(self):
        """安全地初始化LIME解释器，处理版本兼容性问题"""
        print("正在初始化LIME解释器...")
        
        # 获取LimeTextExplainer的构造函数参数
        import inspect
        try:
            sig = inspect.signature(LimeTextExplainer.__init__)
            supported_params = list(sig.parameters.keys())
            print(f"支持的LimeTextExplainer参数: {supported_params}")
        except Exception as e:
            print(f"无法获取LimeTextExplainer参数信息: {e}")
            supported_params = ['class_names']  # 最基本的参数
        
        # 构建初始化参数字典
        init_params = {
            'class_names': ['normal', 'phishing']
        }
        
        # 逐步添加可选参数，如果支持的话
        optional_params = [
            ('char_level', False),
            ('bow', False),
            ('split_expression', self._custom_split_expression),
            ('random_state', 42)
        ]
        
        for param_name, param_value in optional_params:
            if param_name in supported_params:
                init_params[param_name] = param_value
                print(f"添加参数: {param_name}")
            else:
                print(f"跳过不支持的参数: {param_name}")
        
        # 尝试初始化
        try:
            self.explainer = LimeTextExplainer(**init_params)
            print("LIME解释器初始化成功")
        except Exception as e:
            print(f"使用参数 {init_params} 初始化失败: {e}")
            # 回退到最基本的初始化
            try:
                self.explainer = LimeTextExplainer(class_names=['normal', 'phishing'])
                print("使用基本参数初始化LIME解释器成功")
                # 重新定义分词函数为None，避免自定义分词器的问题
                self.use_custom_tokenizer = False
            except Exception as e2:
                print(f"基本初始化也失败: {e2}")
                raise RuntimeError(f"无法初始化LIME解释器: {e2}")
        
        # 设置标志，表示是否使用自定义分词器
        self.use_custom_tokenizer = 'split_expression' in init_params
    
    def _preprocess_text_for_lime(self, text):
        """预处理文本以避免LIME分词问题"""
        if not text or not isinstance(text, str):
            return "empty text"
        
        # 1. 移除非打印字符
        text = ''.join(char for char in text if char.isprintable())
        
        # 2. 替换连续的空白字符
        text = re.sub(r'\s+', ' ', text)
        
        # 3. 移除或替换可能导致分词问题的特殊字符
        # 保留基本标点，但移除其他特殊字符
        allowed_chars = string.ascii_letters + string.digits + string.punctuation + ' '
        text = ''.join(char if char in allowed_chars else ' ' for char in text)
        
        # 4. 再次清理多余空格
        text = re.sub(r'\s+', ' ', text).strip()
        
        # 5. 确保文本不为空
        if not text:
            return "empty text"
        
        # 6. 限制文本长度，避免过长文本导致的问题
        if len(text) > 5000:
            text = text[:5000] + "..."
        
        return text
    
    def _load_model(self):
        """Load the model based on its type and configuration"""
        if self.model_type == "transformer":
            local_model_path = f"../DL/models/{self.model_name}"
            print(f"Loading model from local path: {local_model_path}")
            try:
                if os.path.exists(local_model_path):
                    return AutoModelForSequenceClassification.from_pretrained(local_model_path)
                else:
                    return AutoModelForSequenceClassification.from_pretrained(self.model_config['path'])
            except Exception as e:
                print(f"Failed to load model from local path, trying original path: {e}")
                return AutoModelForSequenceClassification.from_pretrained(self.model_config['path'])
        else:  # custom model
            model_path = f"../DL/models/{self.model_name}.bin"
            print(f"Loading model from local path: {model_path}")
            if not os.path.exists(model_path):
                raise FileNotFoundError(f"Model file not found: {model_path}")
            
            # Initialize the appropriate model class
            if self.model_name == "textcnn":
                model = TextCNNClassifier(**self.custom_config)
            elif self.model_name == "cnn_lstm":
                model = CNNLSTMClassifier(**self.custom_config)
            elif self.model_name == "dnn":
                model = DNNClassifier(**self.custom_config)
            else:
                raise ValueError(f"Unknown model type: {self.model_name}")
            
            # Load model weights
            model.load_state_dict(torch.load(model_path, map_location=self.device))
            return model
    
    def predict(self, text):
        """Predict a single text sample"""
        # 预处理文本
        text = self._preprocess_text_for_lime(text)
        
        try:
            inputs = self.tokenizer(text, return_tensors="pt", truncation=True, padding="max_length", max_length=512)
            input_ids = inputs["input_ids"].to(self.device)
            
            with torch.no_grad():
                if self.model_type == "transformer":
                    attention_mask = inputs["attention_mask"].to(self.device)
                    outputs = self.model(input_ids=input_ids, attention_mask=attention_mask)
                    logits = outputs.logits
                else:
                    outputs = self.model(input_ids)
                    logits = outputs
                    
                probs = torch.softmax(logits, dim=-1)
                pred = torch.argmax(probs, dim=-1).item()
                
            return pred, probs.cpu().numpy()[0]
        except Exception as e:
            print(f"预测过程出错: {e}")
            # 返回默认值：正常邮件
            return 0, np.array([0.8, 0.2])
    
    def _predict_proba_wrapper(self, texts):
        """Wrapper for LIME to predict probabilities for multiple texts"""
        probs = []
        for text in texts:
            try:
                # 确保每个文本都经过预处理
                if not text or not isinstance(text, str):
                    text = "empty text"
                _, prob = self.predict(text)
                probs.append(prob)
            except Exception as e:
                print(f"批量预测中出错: {e}")
                # 返回默认概率
                probs.append(np.array([0.8, 0.2]))
        return np.array(probs)
    
    def explain(self, text, num_features=15):
        """Explain the prediction using LIME and extract n-gram features"""
        try:
            # 预处理输入文本
            preprocessed_text = self._preprocess_text_for_lime(text)
            print(f"原始文本长度: {len(text)}, 预处理后长度: {len(preprocessed_text)}")
            
            # Use LIME to explain the prediction with standard tokenization
            try:
                # 根据是否使用自定义分词器来调整调用方式
                if hasattr(self, 'use_custom_tokenizer') and self.use_custom_tokenizer:
                    print("使用自定义分词器进行LIME解释")
                    explanation = self.explainer.explain_instance(
                        preprocessed_text,
                        self._predict_proba_wrapper,
                        num_features=num_features,
                        num_samples=100
                    )
                else:
                    print("使用默认分词器进行LIME解释")
                    # 对于使用默认分词器的情况，尝试更保守的参数
                    explanation = self.explainer.explain_instance(
                        preprocessed_text,
                        self._predict_proba_wrapper,
                        num_features=min(num_features, 10),  # 减少特征数量
                        num_samples=50  # 减少样本数量
                    )
            except Exception as lime_error:
                print(f"LIME explain_instance调用失败: {lime_error}")
                # 尝试最基本的调用
                try:
                    explanation = self.explainer.explain_instance(
                        preprocessed_text,
                        self._predict_proba_wrapper
                    )
                    print("使用最基本参数成功调用LIME")
                except Exception as basic_error:
                    print(f"基本LIME调用也失败: {basic_error}")
                    raise basic_error
            
            # Get word influence scores from LIME
            word_influences = explanation.as_list()
            
            # Extract n-grams from the original text (使用原始文本来保持语义)
            text_ngrams = self._extract_text_ngrams(text)
            
            # Categorize LIME features
            unigrams = []
            for feature, score in word_influences:
                unigrams.append((feature, score))
            
            # Calculate n-gram scores based on constituent words
            bigrams = self._calculate_ngram_scores(text_ngrams['bigrams'], word_influences)
            
            # Split into positive (phishing) and negative (normal) influences
            phishing_unigrams = [(word, score) for word, score in unigrams if score > 0]
            normal_unigrams = [(word, score) for word, score in unigrams if score < 0]
            phishing_bigrams = [(phrase, score) for phrase, score in bigrams if score > 0]
            normal_bigrams = [(phrase, score) for phrase, score in bigrams if score < 0]
            
            # Sort by absolute influence
            phishing_unigrams = sorted(phishing_unigrams, key=lambda x: x[1], reverse=True)
            normal_unigrams = sorted(normal_unigrams, key=lambda x: abs(x[1]), reverse=True)
            phishing_bigrams = sorted(phishing_bigrams, key=lambda x: x[1], reverse=True)
            normal_bigrams = sorted(normal_bigrams, key=lambda x: abs(x[1]), reverse=True)
            
            return {
                "phishing_words": phishing_unigrams,
                "normal_words": normal_unigrams,
                "phishing_bigrams": phishing_bigrams,
                "normal_bigrams": normal_bigrams,
                "raw_explanation": explanation
            }
            
        except Exception as e:
            print(f"LIME解释过程出错: {e}")
            import traceback
            traceback.print_exc()
            # 返回一个空的解释结果，避免程序崩溃
            return {
                "phishing_words": [],
                "normal_words": [],
                "phishing_bigrams": [],
                "normal_bigrams": [],
                "raw_explanation": None
            }
    
    def _extract_text_ngrams(self, text):
        """Extract bigrams from text"""
        try:
            # 预处理文本
            preprocessed_text = self._preprocess_text_for_lime(text)
            
            # 使用自定义分词器
            tokens = self._custom_split_expression(preprocessed_text)
            
            # 进一步过滤tokens
            valid_tokens = []
            for token in tokens:
                # 只保留有意义的token
                if (len(token) > 1 and 
                    not token.isspace() and 
                    any(c.isalnum() for c in token) and  # 包含字母或数字
                    len(token) < 50):  # 避免过长的token
                    valid_tokens.append(token.lower())
            
            bigrams = []
            
            # 生成bigrams
            if len(valid_tokens) > 1:
                for i in range(len(valid_tokens) - 1):
                    bigram = ' '.join(valid_tokens[i:i+2])
                    if len(bigram.strip()) > 3:  # 只包含有意义的bigrams
                        bigrams.append(bigram)
            
            print(f"提取了 {len(bigrams)} 个bigrams")
            
            return {
                'bigrams': bigrams[:50]  # 限制数量以避免过多处理
            }
            
        except Exception as e:
            print(f"N-gram提取出错: {e}")
            import traceback
            traceback.print_exc()
            return {
                'bigrams': []
            }
    
    def _calculate_ngram_scores(self, ngrams, word_influences):
        """Calculate n-gram scores based on constituent word scores"""
        try:
            word_scores = dict(word_influences)
            ngram_scores = []
            
            for ngram in ngrams:
                words = ngram.split()
                # Calculate average score of constituent words
                scores = []
                for word in words:
                    # Try exact match first, then lowercase match
                    if word in word_scores:
                        scores.append(word_scores[word])
                    elif word.lower() in word_scores:
                        scores.append(word_scores[word.lower()])
                    else:
                        # If word not found in LIME explanation, assign neutral score
                        scores.append(0.0)
                
                if scores:
                    avg_score = sum(scores) / len(scores)
                    # Only include n-grams with non-zero average scores
                    if abs(avg_score) > 0.001:  # Small threshold to filter noise
                        ngram_scores.append((ngram, avg_score))
            
            return ngram_scores
            
        except Exception as e:
            print(f"N-gram分数计算出错: {e}")
            return []
    
    def extract_ngrams(self, text, n=2):
        """Extract n-grams from text"""
        # Clean text
        text = re.sub(r'[^\w\s]', ' ', text.lower())
        # Tokenize
        tokens = word_tokenize(text)
        # Generate n-grams
        n_grams = list(ngrams(tokens, n))
        # Convert n-grams to strings
        n_gram_strs = [' '.join(g) for g in n_grams]
        return n_gram_strs
    
    def analyze_email(self, email_content):
        """Analyze an email and return insights for adversarial generation"""
        try:
            # 预处理邮件内容
            preprocessed_content = self._preprocess_text_for_lime(email_content)
            
            # Predict
            label, probs = self.predict(preprocessed_content)
            print(label, probs,"probs-----------------------\n")
            is_phishing = (label == 1)
            confidence = probs[label]
            
            # Get explanation
            explanation = self.explain(email_content, num_features=30)
            
            # Format results for the prompt - including n-grams
            high_weight_phishing = []
            
            # Add unigrams
            for word, score in explanation["phishing_words"][:5]:
                high_weight_phishing.append(f"- Word: '{word}' (score: {score:.4f})")
            
            # Add bigrams
            for phrase, score in explanation["phishing_bigrams"][:5]:
                high_weight_phishing.append(f"- Phrase: '{phrase}' (score: {score:.4f})")
            
            legitimate_words = []
            
            # Add unigrams
            for word, score in explanation["normal_words"][:5]:
                legitimate_words.append(f"- Word: '{word}' (score: {score:.4f})")
            
            # Add bigrams
            for phrase, score in explanation["normal_bigrams"][:5]:
                legitimate_words.append(f"- Phrase: '{phrase}' (score: {score:.4f})")
            
            # Get bigrams for additional context (legacy support)
            bigrams = self.extract_ngrams(email_content, n=2)
            
            return {
                "is_phishing": is_phishing,
                "confidence": confidence,
                "label": "phishing" if is_phishing else "normal",
                "high_weight_phishing_words": "\n".join(high_weight_phishing),
                "legitimate_words": "\n".join(legitimate_words),
                "raw_explanation": explanation,
                "bigrams": bigrams[:20],  # Legacy support
                "detailed_ngrams": {
                    "phishing_unigrams": explanation["phishing_words"][:10],
                    "phishing_bigrams": explanation["phishing_bigrams"][:10],
                    "normal_unigrams": explanation["normal_words"][:10],
                    "normal_bigrams": explanation["normal_bigrams"][:10]
                }
            }
        except Exception as e:
            print(f"邮件分析过程出错: {e}")
            import traceback
            traceback.print_exc()
            # 返回默认分析结果
            return {
                "is_phishing": False,
                "confidence": 0.5,
                "label": "error",
                "high_weight_phishing_words": "分析失败",
                "legitimate_words": "分析失败",
                "raw_explanation": None,
                "bigrams": [],
                "detailed_ngrams": {
                    "phishing_unigrams": [],
                    "phishing_bigrams": [],
                    "normal_unigrams": [],
                    "normal_bigrams": []
                }
            }
    
    def _custom_split_expression(self, text):
        """自定义分词函数，确保分词结果的一致性"""
        try:
            # 检查输入
            if not text or not isinstance(text, str):
                return ["empty"]
            
            # 预处理文本
            text = self._preprocess_text_for_lime(text)
            
            # 确保文本不为空
            if not text.strip():
                return ["empty"]
            
            # 使用nltk进行分词（如果可用）
            try:
                from nltk.tokenize import word_tokenize
                tokens = word_tokenize(text)
            except ImportError:
                print("NLTK不可用，使用基本分词")
                # 回退到基本的正则表达式分词
                import re
                tokens = re.findall(r'\b\w+\b', text)
            except Exception as e:
                print(f"NLTK分词失败: {e}")
                # 回退到基本的正则表达式分词
                import re
                tokens = re.findall(r'\b\w+\b', text)
            
            # 过滤掉空白和非常短的token
            filtered_tokens = []
            for token in tokens:
                if token and len(token) > 0 and not token.isspace():
                    # 确保token是字符串
                    if isinstance(token, str):
                        filtered_tokens.append(token)
                    else:
                        filtered_tokens.append(str(token))
            
            # 确保至少有一个token
            if not filtered_tokens:
                return ["empty"]
            
            return filtered_tokens
            
        except Exception as e:
            print(f"分词过程出错: {e}")
            # 回退到最简单的空格分词
            try:
                if text and isinstance(text, str):
                    result = text.split()
                    return result if result else ["empty"]
                else:
                    return ["empty"]
            except:
                return ["empty"]
    
def get_all_model_configs():
    """Return a list of available model configurations"""
    return [
        {"name": "textcnn", "path": None, "type": "custom", "class": TextCNNClassifier},
        {"name": "cnn_lstm", "path": None, "type": "custom", "class": CNNLSTMClassifier},
        {"name": "dnn", "path": None, "type": "custom", "class": DNNClassifier},
        {"name": "bert", "path": "bert-base-uncased", "type": "transformer"},
        {"name": "RoBERTa", "path": "roberta-base", "type": "transformer"},
        {"name": "DistilBERT", "path": "distilbert-base-uncased", "type": "transformer"},
        # 重新训练的对抗性模型（用于测试防御能力）
        # {"name": "RoBERTa_adversarial", "path": "roberta-base", "type": "transformer"},
    ] 