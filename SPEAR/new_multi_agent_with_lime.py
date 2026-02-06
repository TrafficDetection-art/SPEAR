import json
import time
import random
import os
import re
from datetime import datetime
from openai import OpenAI
import httpx
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
import inspect
import numpy as np
from my_prompt.detection_prompt import detection_prompt
from my_prompt.generation_prompt import generation_prompt
from my_prompt.evaluation_prompt import evaluation_prompt
from my_prompt.formatting import formatting_prompt
from my_prompt.web_generation_prompt import web_generation_prompt
from my_prompt.attachment_generation_prompt import attachment_generation_prompt
from my_prompt.lime_adversarial_prompt import lime_adversarial_prompt
from my_prompt.init_prompt import init_prompt, list_of_scenarios
from lime_analyzer import LimeAnalyzer, get_all_model_configs

# from my_prompt import evaluation_prompt, title_extraction_prompt


# ========================== 配置加载 ==========================

def load_config(config_path="./config.json"):
    """
    加载配置文件，支持攻防分离的模型配置
    
    配置文件格式示例：
    {
        "attack_model": {
            "api_key": "...",
            "api_base_url": "...",
            "model": "gpt-4o"
        },
        "defense_model": {
            "api_key": "...",
            "api_base_url": "...",
            "model": "gpt-4o-mini"
        }
    }
    
    或者保持兼容旧格式：
    {
        "api_key": "...",
        "api_base_url": "...",
        "model": "gpt-4o"
    }
    """
    with open(config_path, "r") as config_file:
        config_data = json.load(config_file)
    
    # 检查是否使用新格式（攻防分离）
    if "attack_model" in config_data and "defense_model" in config_data:
        print("检测到攻防分离配置模式")
        print(f"  攻击方模型: {config_data['attack_model'].get('model', 'unknown')}")
        print(f"  防御方模型: {config_data['defense_model'].get('model', 'unknown')}")
        return config_data
    else:
        # 兼容旧格式，攻防使用相同配置
        print("使用统一配置模式（攻防共用同一模型）")
        print(f"  模型: {config_data.get('model', 'unknown')}")
        return {
            "attack_model": config_data,
            "defense_model": config_data
        }


config = load_config()

# ========================== Token使用量统计 ==========================

# 全局token使用量统计器
token_usage = {
    "total_prompt_tokens": 0,
    "total_completion_tokens": 0,
    "total_tokens": 0,
    "request_count": 0,
    "usage_by_function": {},  # 按功能分类的token使用量
    "detailed_usage": []  # 详细的使用记录
}

def reset_token_usage():
    """重置token使用量统计"""
    global token_usage
    token_usage = {
        "total_prompt_tokens": 0,
        "total_completion_tokens": 0,
        "total_tokens": 0,
        "request_count": 0,
        "usage_by_function": {},
        "detailed_usage": []
    }

def add_token_usage(prompt_tokens, completion_tokens, total_tokens, function_name="unknown", request_info=None):
    """添加token使用量记录"""
    global token_usage
    
    # 更新总计
    token_usage["total_prompt_tokens"] += prompt_tokens
    token_usage["total_completion_tokens"] += completion_tokens
    token_usage["total_tokens"] += total_tokens
    token_usage["request_count"] += 1
    
    # 按功能分类统计
    if function_name not in token_usage["usage_by_function"]:
        token_usage["usage_by_function"][function_name] = {
            "prompt_tokens": 0,
            "completion_tokens": 0,
            "total_tokens": 0,
            "request_count": 0
        }
    
    token_usage["usage_by_function"][function_name]["prompt_tokens"] += prompt_tokens
    token_usage["usage_by_function"][function_name]["completion_tokens"] += completion_tokens
    token_usage["usage_by_function"][function_name]["total_tokens"] += total_tokens
    token_usage["usage_by_function"][function_name]["request_count"] += 1
    
    # 详细记录
    detailed_record = {
        "timestamp": datetime.now().isoformat(),
        "function": function_name,
        "prompt_tokens": prompt_tokens,
        "completion_tokens": completion_tokens,
        "total_tokens": total_tokens,
        "request_info": request_info or {}
    }
    token_usage["detailed_usage"].append(detailed_record)

def get_token_usage_summary():
    """获取token使用量摘要"""
    return {
        "total_tokens": token_usage["total_tokens"],
        "total_prompt_tokens": token_usage["total_prompt_tokens"],
        "total_completion_tokens": token_usage["total_completion_tokens"],
        "total_requests": token_usage["request_count"],
        "average_tokens_per_request": token_usage["total_tokens"] / max(token_usage["request_count"], 1),
        "usage_by_function": token_usage["usage_by_function"]
    }

def get_detailed_cost_analysis():
    """获取详细的费用分析，包括按功能分类的费用"""
    summary = get_token_usage_summary()
    
    # 总费用
    total_cost = estimate_cost(
        prompt_tokens=summary['total_prompt_tokens'],
        completion_tokens=summary['total_completion_tokens']
    )
    
    # 按功能分类的费用
    function_costs = {}
    for function_name, usage in summary['usage_by_function'].items():
        function_cost = estimate_cost(
            prompt_tokens=usage['prompt_tokens'],
            completion_tokens=usage['completion_tokens']
        )
        function_costs[function_name] = {
            "prompt_tokens": usage['prompt_tokens'],
            "completion_tokens": usage['completion_tokens'],
            "total_tokens": usage['total_tokens'],
            "request_count": usage['request_count'],
            "input_cost": function_cost['input_cost'],
            "output_cost": function_cost['output_cost'],
            "total_cost": function_cost['total_cost']
        }
    
    return {
        "total_cost_breakdown": total_cost,
        "function_costs": function_costs,
        "summary": summary
    }

def estimate_cost(prompt_tokens=0, completion_tokens=0, model_name=None):
    """估算API使用成本（以美元为单位），区分输入和输出tokens的不同定价"""
    # 基于OpenAI的定价，区分输入（prompt）和输出（completion）
    pricing = {
        "gpt-4o": {
            "input": 0.005 / 1000,   # $0.005 per 1K input tokens
            "output": 0.015 / 1000   # $0.015 per 1K output tokens
        },
        "gpt-4o-mini": {
            "input": 0.00015 / 1000,  # $0.00015 per 1K input tokens
            "output": 0.0006 / 1000   # $0.0006 per 1K output tokens
        },
        "gpt-3.5-turbo": {
            "input": 0.0015 / 1000,   # $0.0015 per 1K input tokens
            "output": 0.002 / 1000    # $0.002 per 1K output tokens
        },
        "claude-3.5-sonnet": {
            "input": 0.003 / 1000,    # $0.003 per 1K input tokens
            "output": 0.015 / 1000    # $0.015 per 1K output tokens
        },
        "gemini-2.5-flash": {
            "input": 0.00007 / 1000,  # $0.00007 per 1K input tokens
            "output": 0.00021 / 1000  # $0.00021 per 1K output tokens
        },
        "deepseek": {
            "input": 0.00014 / 1000,  # $0.00014 per 1K input tokens
            "output": 0.00028 / 1000  # $0.00028 per 1K output tokens
        }
    }
    
    if not model_name:
        # 尝试从配置中获取模型名称（兼容新旧格式）
        if "attack_model" in config:
            model_name = config["attack_model"].get("model", "default")
        else:
            model_name = "default"
    
    # 默认定价（如果模型未找到）
    default_pricing = {
        "input": 0.001 / 1000,    # 默认输入价格
        "output": 0.003 / 1000    # 默认输出价格
    }
    
    # 查找匹配的模型定价
    model_pricing = default_pricing
    for model_key, price_config in pricing.items():
        if model_key.lower() in model_name.lower():
            model_pricing = price_config
            break
    
    # 分别计算输入和输出成本
    input_cost = prompt_tokens * model_pricing["input"]
    output_cost = completion_tokens * model_pricing["output"]
    total_cost = input_cost + output_cost
    
    return {
        "input_cost": input_cost,
        "output_cost": output_cost,
        "total_cost": total_cost,
        "input_tokens": prompt_tokens,
        "output_tokens": completion_tokens,
        "model": model_name
    }

# ========================== 记忆模块 ==========================
lime_memory = {
    "phishing_words": {},  # {word: count}
    "legitimate_words": {},  # {word: count}
    "phishing_bigrams": {},  # {bigram: count}
    "legitimate_bigrams": {}  # {bigram: count}
}

def calculate_word_similarity(word1, word2):
    """计算两个单词的相似度"""
    # 简单的相似度计算：基于长度差异、首字母、字符重叠等
    if word1 == word2:
        return 0.0  # 完全相同的词不适合替换
    
    # 长度相似度
    len_similarity = 1.0 - abs(len(word1) - len(word2)) / max(len(word1), len(word2))
    
    # 首字母相似度
    first_letter_similarity = 1.0 if word1[0].lower() == word2[0].lower() else 0.0
    
    # 字符重叠相似度
    common_chars = set(word1.lower()) & set(word2.lower())
    char_similarity = len(common_chars) / max(len(set(word1.lower())), len(set(word2.lower())))
    
    # 加权平均
    similarity = (len_similarity * 0.4 + first_letter_similarity * 0.3 + char_similarity * 0.3)
    
    return similarity

def generate_dynamic_replacement_map(min_phishing_count=2, min_legitimate_count=2, similarity_threshold=0.3):
    """
    根据记忆模块动态生成钓鱼词汇到正常词汇的替换映射表
    
    Args:
        min_phishing_count: 钓鱼词汇的最小出现次数
        min_legitimate_count: 正常词汇的最小出现次数
        similarity_threshold: 相似度阈值
    
    Returns:
        dict: {phishing_word: legitimate_word} 的映射表
    """
    if not lime_memory or not lime_memory.get("phishing_words") or not lime_memory.get("legitimate_words"):
        return {}
    
    # 获取高频钓鱼词汇和正常词汇
    high_freq_phishing = {word: count for word, count in lime_memory["phishing_words"].items() 
                         if count >= min_phishing_count and len(word) > 2}
    high_freq_legitimate = {word: count for word, count in lime_memory["legitimate_words"].items() 
                           if count >= min_legitimate_count and len(word) > 2}
    
    if not high_freq_phishing or not high_freq_legitimate:
        return {}
    
    replacement_map = {}
    legitimate_words = list(high_freq_legitimate.keys())
    
    print(f"动态生成替换映射: {len(high_freq_phishing)} 个钓鱼词汇 -> {len(legitimate_words)} 个正常词汇")
    
    # 为每个钓鱼词汇找到最相似的正常词汇
    for phishing_word in high_freq_phishing.keys():
        best_match = None
        best_similarity = 0.0
        
        for legitimate_word in legitimate_words:
            similarity = calculate_word_similarity(phishing_word, legitimate_word)
            if similarity > best_similarity and similarity >= similarity_threshold:
                best_similarity = similarity
                best_match = legitimate_word
        
        if best_match:
            replacement_map[phishing_word.lower()] = best_match
            print(f"  映射: {phishing_word} -> {best_match} (相似度: {best_similarity:.3f})")
        else:
            # 如果没有找到相似的，选择频次最高的正常词汇
            most_frequent_legitimate = max(high_freq_legitimate.items(), key=lambda x: x[1])[0]
            replacement_map[phishing_word.lower()] = most_frequent_legitimate
            print(f"  映射: {phishing_word} -> {most_frequent_legitimate} (默认高频)")
    
    print(f"生成了 {len(replacement_map)} 个动态映射")
    return replacement_map

def update_memory(lime_analysis, score_threshold=0.02):
    """
    更新记忆，记录LIME分析结果中的高频词汇和n-grams
    只有分数高于threshold的特征才会被记录到记忆中
    
    Args:
        lime_analysis: LIME分析结果
        score_threshold: 分数阈值，默认0.02，只有高于此值的特征才会被记录
    """
    try:
        # 检查lime_analysis是否有效
        if not lime_analysis or not isinstance(lime_analysis, dict):
            print("警告：LIME分析结果为空或格式无效")
            return
        
        print(f"更新记忆模块，使用分数阈值: {score_threshold}")
        
        # 使用新的detailed_ngrams数据结构（如果可用）
        if "detailed_ngrams" in lime_analysis and isinstance(lime_analysis["detailed_ngrams"], dict):
            detailed_ngrams = lime_analysis["detailed_ngrams"]
            
            # 安全地更新钓鱼unigrams - 添加阈值筛选
            phishing_unigrams = detailed_ngrams.get("phishing_unigrams", [])
            if isinstance(phishing_unigrams, list):
                added_count = 0
                for item in phishing_unigrams:
                    if isinstance(item, (list, tuple)) and len(item) >= 2:
                        word, score = item[0], item[1]
                        if isinstance(word, str) and len(word) > 2 and isinstance(score, (int, float)) and abs(score) > score_threshold:
                            lime_memory["phishing_words"][word] = lime_memory["phishing_words"].get(word, 0) + 1
                            added_count += 1
                print(f"  钓鱼unigrams: 添加 {added_count}/{len(phishing_unigrams)} 个高权重特征 (阈值: {score_threshold})")
            
            # 安全地更新钓鱼bigrams - 添加阈值筛选
            phishing_bigrams = detailed_ngrams.get("phishing_bigrams", [])
            if isinstance(phishing_bigrams, list):
                added_count = 0
                for item in phishing_bigrams:
                    if isinstance(item, (list, tuple)) and len(item) >= 2:
                        phrase, score = item[0], item[1]
                        if isinstance(phrase, str) and len(phrase.split()) == 2 and isinstance(score, (int, float)) and abs(score) > score_threshold:
                            lime_memory["phishing_bigrams"][phrase] = lime_memory["phishing_bigrams"].get(phrase, 0) + 1
                            added_count += 1
                print(f"  钓鱼bigrams: 添加 {added_count}/{len(phishing_bigrams)} 个高权重特征 (阈值: {score_threshold})")
            
            # 安全地更新正常unigrams - 添加阈值筛选
            normal_unigrams = detailed_ngrams.get("normal_unigrams", [])
            if isinstance(normal_unigrams, list):
                added_count = 0
                for item in normal_unigrams:
                    if isinstance(item, (list, tuple)) and len(item) >= 2:
                        word, score = item[0], item[1]
                        if isinstance(word, str) and len(word) > 2 and isinstance(score, (int, float)) and abs(score) > score_threshold:
                            lime_memory["legitimate_words"][word] = lime_memory["legitimate_words"].get(word, 0) + 1
                            added_count += 1
                print(f"  正常unigrams: 添加 {added_count}/{len(normal_unigrams)} 个高权重特征 (阈值: {score_threshold})")
            
            # 安全地更新正常bigrams - 添加阈值筛选
            normal_bigrams = detailed_ngrams.get("normal_bigrams", [])
            if isinstance(normal_bigrams, list):
                added_count = 0
                for item in normal_bigrams:
                    if isinstance(item, (list, tuple)) and len(item) >= 2:
                        phrase, score = item[0], item[1]
                        if isinstance(phrase, str) and len(phrase.split()) == 2 and isinstance(score, (int, float)) and abs(score) > score_threshold:
                            lime_memory["legitimate_bigrams"][phrase] = lime_memory["legitimate_bigrams"].get(phrase, 0) + 1
                            added_count += 1
                print(f"  正常bigrams: 添加 {added_count}/{len(normal_bigrams)} 个高权重特征 (阈值: {score_threshold})")
        
        else:
            # 向后兼容：处理旧格式的钓鱼词汇
            phishing_words = lime_analysis.get("high_weight_phishing_words", "")
            if isinstance(phishing_words, str) and phishing_words.strip():
                # 清理和分词
                cleaned_text = clean_text_for_tokenization(phishing_words)
                words = re.findall(r'\b\w+\b', cleaned_text.lower())
                added_count = 0
                for word in words:
                    if len(word) > 2:  # 过滤太短的词
                        # 对于旧格式，无法获取具体分数，假设都是高权重的
                        lime_memory["phishing_words"][word] = lime_memory["phishing_words"].get(word, 0) + 1
                        added_count += 1
                print(f"  旧格式钓鱼词汇: 添加 {added_count} 个词汇 (旧格式默认为高权重)")
            
            # 向后兼容：处理旧格式的正常词汇
            legitimate_words = lime_analysis.get("legitimate_words", "")
            if isinstance(legitimate_words, str) and legitimate_words.strip():
                # 清理和分词
                cleaned_text = clean_text_for_tokenization(legitimate_words)
                words = re.findall(r'\b\w+\b', cleaned_text.lower())
                added_count = 0
                for word in words:
                    if len(word) > 2:  # 过滤太短的词
                        # 对于旧格式，无法获取具体分数，假设都是高权重的
                        lime_memory["legitimate_words"][word] = lime_memory["legitimate_words"].get(word, 0) + 1
                        added_count += 1
                print(f"  旧格式正常词汇: 添加 {added_count} 个词汇 (旧格式默认为高权重)")
                        
        total_phishing = len(lime_memory['phishing_words']) + len(lime_memory['phishing_bigrams'])
        total_legitimate = len(lime_memory['legitimate_words']) + len(lime_memory['legitimate_bigrams'])
        print(f"记忆更新完成: 钓鱼特征总计 {total_phishing} 个, 正常特征总计 {total_legitimate} 个")
        
    except Exception as e:
        print(f"更新记忆时出错: {e}")
        import traceback
        traceback.print_exc()

def get_memory_context(top_k=10):
    """获取记忆中的高频词汇和n-grams，用于增强prompt"""
    if not any(lime_memory.values()):
        return ""
    
    # 获取高频钓鱼特征 - 修改为平均分配unigrams和bigrams
    top_phishing_words = sorted(lime_memory["phishing_words"].items(), key=lambda x: x[1], reverse=True)[:top_k//2]
    top_phishing_bigrams = sorted(lime_memory["phishing_bigrams"].items(), key=lambda x: x[1], reverse=True)[:top_k//2]
    
    # 获取高频正常特征 - 修改为平均分配unigrams和bigrams
    top_legitimate_words = sorted(lime_memory["legitimate_words"].items(), key=lambda x: x[1], reverse=True)[:top_k//2]
    top_legitimate_bigrams = sorted(lime_memory["legitimate_bigrams"].items(), key=lambda x: x[1], reverse=True)[:top_k//2]
    
    # 构建特征列表
    phishing_features = []
    if top_phishing_words:
        phishing_features.extend([word for word, _ in top_phishing_words])
    if top_phishing_bigrams:
        phishing_features.extend([f'"{phrase}"' for phrase, _ in top_phishing_bigrams])
    
    legitimate_features = []
    if top_legitimate_words:
        legitimate_features.extend([word for word, _ in top_legitimate_words])
    if top_legitimate_bigrams:
        legitimate_features.extend([f'"{phrase}"' for phrase, _ in top_legitimate_bigrams])
    
    memory_context = f"""
Previous LIME analysis memory:
- Frequent phishing features to avoid: {', '.join(phishing_features[:top_k])}
- Frequent legitimate features to use: {', '.join(legitimate_features[:top_k])}
Use this knowledge to improve evasion."""
    
    return memory_context

# ========================== 输出目录管理 ==========================

def create_output_directory():
    """创建输出目录结构"""
    # 基础输出目录
    output_base_dir = "outputs"
    
    # 获取攻防模型名称
    attack_model = config['attack_model']['model']
    defense_model = config['defense_model']['model']
    
    # 如果攻防模型相同，使用单一模型名称；否则显示攻防对比
    if attack_model == defense_model:
        model_name = f"agent-{attack_model}"
    else:
        model_name = f"agent-attack_{attack_model}_vs_defense_{defense_model}"
    
    # 创建时间戳，精确到秒
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    
    # 完整的输出目录路径
    output_dir = os.path.join(output_base_dir, model_name, timestamp)
    
    # 创建目录（如果不存在）
    os.makedirs(output_dir, exist_ok=True)
    
    return output_dir


# ========================== OpenAI 客户端管理 ==========================

def create_client(role="attack"):
    """
    创建 OpenAI 客户端实例（支持本地模型和SSL配置）
    
    Args:
        role: "attack" 表示攻击方客户端，"defense" 表示防御方客户端
    
    Returns:
        OpenAI 客户端实例或本地模型适配器
    """
    if role not in ["attack", "defense"]:
        raise ValueError(f"Invalid role: {role}. Must be 'attack' or 'defense'")
    
    model_config = config[f"{role}_model"]
    base_url = model_config["api_base_url"].rstrip('/')
    
    # 检测是否为本地Qwen模型（通过URL中的/api/infer判断）
    is_local_qwen = "localhost" in base_url or "127.0.0.1" in base_url
    
    if is_local_qwen and "/api/infer" not in base_url:
        # 如果是本地模型但URL格式特殊，使用Qwen适配器
        print(f"检测到本地Qwen模型，使用适配器")
        try:
            from qwen_api_adapter import QwenClient
            client = QwenClient(
                api_key=model_config["api_key"],
                base_url=base_url
            )
            print(f"创建{role}客户端(Qwen适配器): 模型={model_config.get('model', 'unknown')}")
            return client
        except ImportError:
            print("警告: 无法导入Qwen适配器，尝试使用标准OpenAI客户端")
    
    # 检查是否需要禁用SSL验证
    disable_ssl = model_config.get("disable_ssl_verify", False)
    
    # 使用标准OpenAI客户端（适用于OpenAI兼容的API）
    if disable_ssl:
        print(f"⚠️  警告: 为{role}客户端禁用SSL验证（不推荐用于生产环境）")
        # 创建禁用SSL验证的HTTP客户端
        http_client = httpx.Client(verify=False)
        client = OpenAI(
            api_key=model_config["api_key"], 
            base_url=base_url,
            http_client=http_client
        )
    else:
        client = OpenAI(
            api_key=model_config["api_key"], 
            base_url=base_url
        )
    
    print(f"创建{role}客户端: 模型={model_config.get('model', 'unknown')}")
    return client


# ========================== LLM 交互 ==========================

def get_LLM_response_vllm(client, prompt, need_sample=False, function_name="unknown", role="attack"):
    """
    发送请求到 OpenAI LLM，并处理异常情况
    
    Args:
        client: OpenAI 客户端
        prompt: 提示词
        need_sample: 是否需要采样
        function_name: 函数名称（用于统计）
        role: "attack" 或 "defense"，用于确定使用哪个模型配置
    """
    sample_num = 3 if need_sample else 1
    conversation = [{"role": "user", "content": prompt}]
    model = config[f"{role}_model"]["model"]
    max_tokens_for_completion = 2048  # 预留 Token 限制，避免超限

    max_retries = 5
    delay = 2

    for attempt in range(max_retries):
        try:
            # 检查是否使用最小参数集（用于不支持所有参数的模型）
            use_minimal_params = config[f"{role}_model"].get("use_minimal_params", False)
            
            # 构建API调用参数
            api_params = {
                "model": model,
                "messages": conversation,
                "max_tokens": max_tokens_for_completion,
            }
            
            # 如果不是最小参数模式，添加额外参数
            if not use_minimal_params:
                api_params["n"] = sample_num
                api_params["temperature"] = 1.0
                api_params["top_p"] = 0.95
            else:
                # 最小参数模式：只添加必要参数
                if sample_num > 1:
                    api_params["n"] = sample_num
                # 某些模型可能支持temperature但不支持top_p
                api_params["temperature"] = 1.0
            
            chat_response = client.chat.completions.create(**api_params)

            # 记录token使用量
            if hasattr(chat_response, 'usage') and chat_response.usage:
                usage = chat_response.usage
                prompt_tokens = getattr(usage, 'prompt_tokens', 0)
                completion_tokens = getattr(usage, 'completion_tokens', 0)
                total_tokens = getattr(usage, 'total_tokens', 0)
                
                # 添加token使用量记录
                add_token_usage(
                    prompt_tokens=prompt_tokens,
                    completion_tokens=completion_tokens, 
                    total_tokens=total_tokens,
                    function_name=function_name,
                    request_info={
                        "model": model,
                        "sample_num": sample_num,
                        "attempt": attempt + 1
                    }
                )
            else:
                # 如果没有usage信息，估算token数量
                estimated_prompt_tokens = len(prompt.split()) * 1.3  # 粗略估算
                estimated_completion_tokens = len(chat_response.choices[0].message.content.split()) * 1.3 if chat_response.choices else 0
                estimated_total = estimated_prompt_tokens + estimated_completion_tokens
                
                add_token_usage(
                    prompt_tokens=int(estimated_prompt_tokens),
                    completion_tokens=int(estimated_completion_tokens),
                    total_tokens=int(estimated_total),
                    function_name=function_name + "_estimated",
                    request_info={
                        "model": model,
                        "sample_num": sample_num,
                        "attempt": attempt + 1,
                        "note": "estimated_usage"
                    }
                )

            if need_sample:
                return [choice.message.model_dump()['content'] for choice in chat_response.choices]
            return chat_response.choices[0].message.model_dump()['content']

        except Exception as e:
            print(f"Attempt {attempt + 1} failed: {e}. Retrying in {delay} seconds...prompt: {prompt}")
            time.sleep(delay)

        time.sleep(delay)  # Sleep between attempts

    return "Error: LLM request failed."


# ========================== 评估指标 ==========================

def evaluate_metrics_macro(y_true, y_pred, average='macro'):
    """计算分类模型的 ACC、F1、精确率 和 召回率（多分类）"""
    return {
        "Accuracy": accuracy_score(y_true, y_pred),
        "Precision": precision_score(y_true, y_pred, average=average, zero_division=0),
        "Recall": recall_score(y_true, y_pred, average=average, zero_division=0),
        "F1-score": f1_score(y_true, y_pred, average=average, zero_division=0)
    }

def evaluate_metrics(y_true, y_pred):
    """二分类下，精准计算ACC、F1、精确率、召回率（关注钓鱼=1）"""
    # 过滤掉无效预测（值为2的情况），将其视为正常邮件（0）
    y_pred_filtered = [0 if pred == 2 else pred for pred in y_pred]
    
    return {
        "Accuracy": accuracy_score(y_true, y_pred_filtered),
        "Precision": precision_score(y_true, y_pred_filtered, average='binary', pos_label=1, zero_division=0),
        "Recall": recall_score(y_true, y_pred_filtered, average='binary', pos_label=1, zero_division=0),
        "F1-score": f1_score(y_true, y_pred_filtered, average='binary', pos_label=1, zero_division=0)
    }

def calculate_asr(y_true, y_pred):
    """
    计算ASR (Attack Success Rate) - 攻击成功率
    ASR = 成功绕过检测的恶意样本数 / 总恶意样本数
    也就是恶意样本的漏报率 (False Negative Rate)
    
    在钓鱼邮件场景中：
    - 真实标签为1（恶意钓鱼邮件）
    - 模型预测为0（被误判为正常邮件）
    - 表示攻击成功绕过了检测
    
    Args:
        y_true: 真实标签列表 (1=恶意, 0=正常)
        y_pred: 预测标签列表 (1=检测为恶意, 0=检测为正常)
    
    Returns:
        dict: 包含ASR及相关统计信息
    """
    # 过滤掉无效预测（值为2的情况），将其视为正常邮件（0）
    y_pred_filtered = [0 if pred == 2 else pred for pred in y_pred]
    
    # 统计恶意样本总数
    total_malicious = sum(1 for label in y_true if label == 1)
    
    # 如果没有恶意样本，返回空结果
    if total_malicious == 0:
        return {
            "ASR": 0.0,
            "successful_evasions": 0,
            "total_malicious": 0,
            "evasion_rate_percent": 0.0,
            "note": "No malicious samples in dataset"
        }
    
    # 统计成功绕过检测的恶意样本数量
    # 即：真实标签为1（恶意），但预测为0（正常）的样本数
    successful_evasions = sum(1 for true_label, pred_label in zip(y_true, y_pred_filtered) 
                             if true_label == 1 and pred_label == 0)
    
    # 计算ASR
    asr = successful_evasions / total_malicious
    
    return {
        "ASR": asr,
        "successful_evasions": successful_evasions,
        "total_malicious": total_malicious,
        "evasion_rate_percent": asr * 100
    }

# ========================== 文本预处理工具函数 ==========================

def clean_text_for_tokenization(text):
    """清理文本以避免分词器错误"""
    if not isinstance(text, str):
        text = str(text)
    
    # 移除或替换可能导致分词器错误的字符
    # 1. 移除控制字符
    text = re.sub(r'[\x00-\x08\x0B\x0C\x0E-\x1F\x7F-\x9F]', '', text)
    
    # 2. 处理特殊的Unicode字符
    text = text.encode('utf-8', errors='ignore').decode('utf-8')
    
    # 3. 替换多个连续的空白字符
    text = re.sub(r'\s+', ' ', text)
    
    # 4. 移除首尾空白
    text = text.strip()
    
    # 5. 确保文本不为空
    if not text:
        text = "empty text"
    
    return text

def validate_text_for_lime(text):
    """验证文本是否适合LIME分析"""
    try:
        # 基本验证
        if not text or not isinstance(text, str):
            return False, "文本为空或不是字符串类型"
        
        # 长度验证
        if len(text.strip()) < 5:
            return False, "文本过短"
        
        # 检查是否包含过多的特殊字符
        special_char_ratio = len(re.findall(r'[^\w\s]', text)) / len(text)
        if special_char_ratio > 0.5:
            return False, f"特殊字符比例过高: {special_char_ratio:.2f}"
        
        # 检查是否主要由ASCII字符组成
        ascii_ratio = len(text.encode('ascii', errors='ignore')) / len(text.encode('utf-8'))
        if ascii_ratio < 0.3:
            return False, f"ASCII字符比例过低: {ascii_ratio:.2f}"
        
        # 检查是否包含足够的单词
        words = re.findall(r'\b\w+\b', text)
        if len(words) < 3:
            return False, f"单词数量过少: {len(words)}"
        
        return True, "文本验证通过"
        
    except Exception as e:
        return False, f"验证过程出错: {e}"

def safe_tokenize_text(text, max_length=512):
    """安全的文本分词处理"""
    try:
        # 检查text是否为None
        if text is None:
            print("警告：输入文本为None，使用默认文本")
            text = "empty text content"
        
        # 清理文本
        cleaned_text = clean_text_for_tokenization(text)
        
        # 如果文本太长，截断
        if len(cleaned_text) > max_length * 4:  # 粗略估算，一般一个token约对应4个字符
            cleaned_text = cleaned_text[:max_length * 4]
        
        return cleaned_text
    except Exception as e:
        print(f"文本清理错误: {e}")
        return "text cleaning failed"

def create_fallback_analysis_result(email_content):
    """创建备用的分析结果"""
    # 检查email_content是否为None
    if email_content is None:
        email_content = "empty email content"
        print("警告：邮件内容为None，使用默认文本")
    
    # 使用简单的规则进行基础分析
    phishing_keywords = [
        'urgent', 'verify', 'click', 'account', 'suspended', 'expire',
        'confirm', 'update', 'security', 'alert', 'warning', 'action',
        'required', 'immediately', 'login', 'password', 'refund'
    ]
    
    legitimate_keywords = [
        'thank', 'welcome', 'information', 'service', 'company',
        'team', 'support', 'help', 'contact', 'regards', 'sincerely'
    ]
    
    text_lower = email_content.lower()
    
    # 计算关键词出现次数
    phishing_score = sum(1 for keyword in phishing_keywords if keyword in text_lower)
    legitimate_score = sum(1 for keyword in legitimate_keywords if keyword in text_lower)
    
    # 简单判断
    is_phishing = phishing_score > legitimate_score
    confidence = max(phishing_score, legitimate_score) / max(len(phishing_keywords), len(legitimate_keywords))
    
    # 提取出现的关键词
    found_phishing = [kw for kw in phishing_keywords if kw in text_lower]
    found_legitimate = [kw for kw in legitimate_keywords if kw in text_lower]
    
    return {
        "is_phishing": is_phishing,
        "confidence": confidence,
        "label": "phishing" if is_phishing else "legitimate",
        "high_weight_phishing_words": ', '.join(found_phishing),
        "legitimate_words": ', '.join(found_legitimate),
        "raw_explanation": {
            "phishing_words": [(word, 0.01) for word in found_phishing],
            "legitimate_words": [(word, 0.01) for word in found_legitimate]
        },
        "detailed_ngrams": {
            "phishing_unigrams": [(word, 0.01) for word in found_phishing],
            "normal_unigrams": [(word, 0.01) for word in found_legitimate],
            "phishing_bigrams": [],
            "normal_bigrams": []
        },
        "bigrams": [],
        "fallback_used": True
    }

def validate_analysis_result(analysis_result):
    """验证和修复分析结果的数据结构"""
    # 确保必需的字段存在
    required_fields = {
        "is_phishing": False,
        "confidence": 0.0,
        "label": "unknown",
        "high_weight_phishing_words": "",
        "legitimate_words": "",
        "raw_explanation": None,
        "bigrams": []
    }
    
    for field, default_value in required_fields.items():
        if field not in analysis_result:
            analysis_result[field] = default_value
    
    # 确保detailed_ngrams结构完整 - 移除trigrams
    if "detailed_ngrams" not in analysis_result:
        analysis_result["detailed_ngrams"] = {}
    
    ngram_fields = [
        "phishing_unigrams", "normal_unigrams",
        "phishing_bigrams", "normal_bigrams"
    ]
    
    for field in ngram_fields:
        if field not in analysis_result["detailed_ngrams"]:
            analysis_result["detailed_ngrams"][field] = []
    
    # 确保字符串字段不为None
    if analysis_result["high_weight_phishing_words"] is None:
        analysis_result["high_weight_phishing_words"] = ""
    if analysis_result["legitimate_words"] is None:
        analysis_result["legitimate_words"] = ""
    
    return analysis_result

# ========================== 邮件检测 ==========================

def detect_email(client, email_content):
    """使用 LLM 检测电子邮件是否为钓鱼邮件（防御方）"""
    prompt = detection_prompt.format(email_content=email_content)
    response = get_LLM_response_vllm(client, prompt, function_name="email_detection", role="defense")

    print("detection response：",response)
    if "etermined as a phishing email" in response:
        return 1, response
    elif "etermined as another legitimate email" in response:
        return 0, response
    return 2, response  # 默认返回正常邮件


# ========================== LIME-based 邮件对抗 ==========================

def lime_analyze_email(email_content, model_name="bert"):
    """使用 LIME 分析邮件内容，寻找关键特征词"""
    print(f"使用 {model_name} 模型进行 LIME 分析...")
    
    # 首先清理和预处理邮件内容
    try:
        cleaned_email = safe_tokenize_text(email_content)
        print(f"原始邮件长度: {len(email_content)}, 清理后长度: {len(cleaned_email)}")
        
        # 验证清理后的文本是否适合LIME分析
        is_valid, validation_message = validate_text_for_lime(cleaned_email)
        if not is_valid:
            print(f"文本验证失败: {validation_message}")
            return create_fallback_analysis_result(cleaned_email)
        else:
            print(f"文本验证成功: {validation_message}")
            
    except Exception as e:
        print(f"邮件预处理失败: {e}")
        return create_fallback_analysis_result(email_content)
    
    # 获取所有模型配置
    try:
        model_configs = get_all_model_configs()
    except Exception as e:
        print(f"获取模型配置失败: {e}")
        return create_fallback_analysis_result(cleaned_email)
    
    # 找到指定的模型配置
    model_config = None
    for config in model_configs:
        if config["name"].lower() == model_name.lower():
            model_config = config
            break
    
    if not model_config:
        print(f"警告：未找到 {model_name} 模型配置，使用默认的 TextCNN 模型")
        # 使用默认的 TextCNN 模型
        model_config = {"name": "textcnn", "path": None, "type": "custom"}
    
    try:
        # 初始化 LIME 分析器
        analyzer = LimeAnalyzer(model_config)
        
        # 分析邮件，使用清理后的内容
        analysis_result = analyzer.analyze_email(cleaned_email)
        
        # 验证分析结果的完整性
        analysis_result = validate_analysis_result(analysis_result)
        
        print(f"LIME 分析结果: 分类为 '{analysis_result['label']}', 置信度: {analysis_result['confidence']:.4f}")
        print("高权重钓鱼特征词:")
        print(analysis_result["high_weight_phishing_words"])
        print("高权重正常特征词:")
        print(analysis_result["legitimate_words"])
        
        # 更新记忆
        update_memory(analysis_result, score_threshold=0.02)
        
        return analysis_result
    
    except Exception as e:
        print(f"LIME 分析时出错: {e}")
        import traceback
        traceback.print_exc()
        # 返回一个备用结果
        return create_fallback_analysis_result(cleaned_email)

def generate_lime_adversarial_email(generation_client, email_content, lime_analysis):
    """根据 LIME 分析结果生成对抗性钓鱼邮件（攻击方）"""
    print("使用 LIME 分析结果生成对抗性钓鱼邮件...")
    
    # 获取记忆上下文
    memory_context = get_memory_context(top_k=10)
    
    # 使用 LIME 对抗性生成 prompt
    base_prompt = lime_adversarial_prompt.format(
        email_content=email_content,
        high_weight_phishing_words=lime_analysis["high_weight_phishing_words"],
        legitimate_words=lime_analysis["legitimate_words"]
    )
    
    # 如果有记忆上下文，添加到prompt中
    if memory_context:
        enhanced_prompt = base_prompt + memory_context
        print("已添加记忆上下文到生成prompt")
    else:
        enhanced_prompt = base_prompt

    print("prompt:-------------------\n ", enhanced_prompt)
    
    # 生成对抗性邮件（攻击方）
    adversarial_email = get_LLM_response_vllm(generation_client, enhanced_prompt, function_name="lime_adversarial_generation", role="attack")
    
    return adversarial_email

# ========================== LIME预处理功能 ==========================

def preprocess_email_with_lime(email_content, analysis_result, strategy="delete", threshold=0.02):
    """
    根据LIME分析结果预处理邮件文本，为后续大模型生成做准备
    
    Args:
        email_content: 原始邮件内容
        analysis_result: LIME分析结果
        strategy: 处理策略，可选 "replace", "delete", "char_replace"
        threshold: 分数阈值，只处理分数高于此值的词
    
    Returns:
        预处理后的邮件内容
    """
    print(f"开始LIME预处理，策略: {strategy}, 阈值: {threshold}")
    print("预处理目标：用正常词汇替换钓鱼词汇，以降低钓鱼检测概率")
    
    try:
        # 首先清理邮件文本
        cleaned_email = clean_text_for_tokenization(email_content)
        
        # 提取高权重钓鱼特征词（这些词有正权重，支持钓鱼分类）
        high_score_phishing_words = []
        
        # 检查analysis_result结构
        if not analysis_result or not isinstance(analysis_result, dict):
            print("警告：分析结果无效，跳过预处理")
            return cleaned_email
        
        # 尝试从raw_explanation中提取钓鱼词汇（正权重）
        raw_explanation = analysis_result.get('raw_explanation', {})
        if isinstance(raw_explanation, dict):
            phishing_words = raw_explanation.get('phishing_words', [])
            if isinstance(phishing_words, list):
                for item in phishing_words:
                    try:
                        if isinstance(item, (list, tuple)) and len(item) >= 2:
                            word, score = item[0], item[1]
                            # 钓鱼词汇应该有正权重，且分数需要高于阈值
                            if isinstance(word, str) and isinstance(score, (int, float)) and score > threshold:
                                high_score_phishing_words.append(word)
                                print(f"    添加高权重钓鱼词: {word} (分数: {score:.3f})")
                    except (IndexError, TypeError, ValueError) as e:
                        print(f"处理词汇项时出错: {item}, 错误: {e}")
                        continue
        
        # 如果没有从raw_explanation获取到词汇，尝试从高级字段获取
        if not high_score_phishing_words:
            phishing_words_str = analysis_result.get('high_weight_phishing_words', '')
            if isinstance(phishing_words_str, str) and phishing_words_str.strip():
                # 从字符串中提取词汇
                words = re.findall(r'\b\w+\b', phishing_words_str.lower())
                high_score_phishing_words = [word for word in words if len(word) > 2]
        
        print(f"需要替换的高权重钓鱼词汇: {high_score_phishing_words}")
        
        if not high_score_phishing_words:
            print("没有找到需要预处理的钓鱼词汇，返回原始文本")
            return cleaned_email
        
        processed_email = cleaned_email
        
        if strategy == "delete":
            # 直接删除高权重钓鱼词
            print("策略：删除钓鱼词汇")
            for word in high_score_phishing_words:
                try:
                    pattern = r'\b' + re.escape(word) + r'\b'
                    processed_email = re.sub(pattern, '', processed_email, flags=re.IGNORECASE)
                    print(f"  删除钓鱼词: {word}")
                except re.error as e:
                    print(f"删除词汇 '{word}' 时出错: {e}")
                    continue
            processed_email = re.sub(r'\s+', ' ', processed_email).strip()
            
        elif strategy == "replace":
            print("策略：用正常词汇替换钓鱼词汇")
            
            # 获取智能映射表
            smart_mapping = get_smart_replacement_map()
            
            # 获取记忆模块中的正常特征词汇用于替换
            memory_legitimate_words = []
            if lime_memory and lime_memory.get("legitimate_words"):
                # 按频次排序，获取高频的正常词汇
                sorted_legitimate = sorted(lime_memory["legitimate_words"].items(), 
                                         key=lambda x: x[1], reverse=True)
                memory_legitimate_words = [word for word, count in sorted_legitimate if count >= 1][:20]
                print(f"从记忆中获取的正常词汇: {memory_legitimate_words}")
            
            print(f"可用智能映射表: {len(smart_mapping)} 个映射")
            
            # 为钓鱼词汇寻找正常词汇进行替换
            for phishing_word in high_score_phishing_words:
                try:
                    replacement = None
                    
                    # 1. 首先尝试从智能映射表中找替换词
                    if phishing_word.lower() in smart_mapping:
                        replacement = smart_mapping[phishing_word.lower()]
                        print(f"  智能映射: {phishing_word} -> {replacement}")
                    
                    # 2. 如果智能映射表没有，尝试从记忆模块中找相似的正常词汇
                    elif memory_legitimate_words:
                        # 寻找长度相近的正常词汇作为替换词
                        word_len = len(phishing_word)
                        similar_length_words = [w for w in memory_legitimate_words 
                                              if abs(len(w) - word_len) <= 2 and w.lower() != phishing_word.lower()]
                        
                        if similar_length_words:
                            replacement = similar_length_words[0]
                            print(f"  记忆映射: {phishing_word} -> {replacement}")
                        else:
                            # 如果没有相似长度的，选择一个高频的正常词汇
                            if memory_legitimate_words:
                                replacement = memory_legitimate_words[0]
                                print(f"  默认映射: {phishing_word} -> {replacement}")
                    
                    # 3. 应用替换
                    if replacement:
                        # 保持原词的大小写格式
                        if phishing_word.isupper():
                            replacement = replacement.upper()
                        elif phishing_word.istitle():
                            replacement = replacement.capitalize()
                        pattern = r'\b' + re.escape(phishing_word) + r'\b'
                        processed_email = re.sub(pattern, replacement, processed_email, flags=re.IGNORECASE)
                        print(f"  ✓ 替换完成: {phishing_word} -> {replacement}")
                    else:
                        # 如果没有找到替换词，就删除钓鱼词
                        print(f"  未找到替换词，删除钓鱼词: {phishing_word}")
                        pattern = r'\b' + re.escape(phishing_word) + r'\b'
                        processed_email = re.sub(pattern, '', processed_email, flags=re.IGNORECASE)
                        
                except re.error as e:
                    print(f"替换词汇 '{phishing_word}' 时出错: {e}")
                    continue
            processed_email = re.sub(r'\s+', ' ', processed_email).strip()
        
        elif strategy == "char_replace":
            print("策略：字符级混淆替换钓鱼词汇")
            
            # 定义视觉相似的字符映射表
            char_substitutions = {
                'a': ['а'],  # 希腊字母α, 西里尔字母а等
                'e': ['е'],       # 西里尔字母е, 希腊字母ε等
                'o': ['о', 'ο'], # 西里尔字母о, 希腊字母ο等
                'i': ['і'],       # 西里尔字母і, 希腊字母ι等
                'p': ['р'],                                # 西里尔字母р, 希腊字母ρ
                'c': ['с'],                     # 西里尔字母с等
                'h': ['һ'],                                # 西里尔字母һ等
                'x': ['х'],                                # 西里尔字母х, 希腊字母χ
                'y': ['у'],                          # 西里尔字母у等
                's': ['ѕ'],                          # 西里尔字母ѕ等
                'f': ['ƒ'],                                     # 特殊f等
                'w': ['ԝ']                                     # 特殊w等
            }
            
            # 为每个钓鱼词汇进行字符级替换
            for phishing_word in high_score_phishing_words:
                try:
                    # 对钓鱼词汇进行字符级混淆
                    confused_word = ""
                    for char in phishing_word.lower():
                        if char in char_substitutions and len(char_substitutions[char]) > 0:
                            # 随机选择一个相似字符进行替换
                            import random
                            confused_char = random.choice(char_substitutions[char])
                            confused_word += confused_char
                            print(f"    字符替换: {char} -> {confused_char}")
                        else:
                            confused_word += char
                    
                    # 保持原词的大小写格式
                    if phishing_word.isupper():
                        confused_word = confused_word.upper()
                    elif phishing_word.istitle():
                        confused_word = confused_word.capitalize()
                    
                    # 应用字符级替换
                    pattern = r'\b' + re.escape(phishing_word) + r'\b'
                    processed_email = re.sub(pattern, confused_word, processed_email, flags=re.IGNORECASE)
                    print(f"  ✓ 字符混淆完成: {phishing_word} -> {confused_word}")
                    
                except Exception as e:
                    print(f"字符混淆词汇 '{phishing_word}' 时出错: {e}")
                    continue
        
        print(f"预处理完成。原长度: {len(cleaned_email)}, 新长度: {len(processed_email)}")
        print(f"预处理后的邮件片段: {processed_email[:150]}...")
        
        return processed_email
        
    except Exception as e:
        print(f"预处理过程中出现错误: {e}")
        import traceback
        traceback.print_exc()
        # 出错时返回清理后的原始文本
        try:
            return clean_text_for_tokenization(email_content)
        except:
            return email_content

def adversarial_ml_model_evasion(generation_client, recheck_client, email_content, ml_model_names=["textcnn"], max_iterations_per_model=5, use_lime_preprocessing=True):
    """对多个机器学习模型进行对抗性攻击"""
    print("开始对机器学习模型进行对抗性攻击...")
    print(f"使用LIME预处理: {use_lime_preprocessing}")
    
    # 初始记录原始邮件的分析结果
    all_analysis_results = {}
    current_email = email_content
    evasion_results = {}
    
    # 保存所有生成的邮件历史 - 新增功能
    all_generated_emails = {
        "original_email": email_content,  # 保存原始邮件
        "models": {}
    }
    
    # 逐个模型进行对抗
    for model_name in ml_model_names:
        print(f"\n===== 对 {model_name} 模型进行对抗 =====")
        
        # 初始化该模型的邮件历史
        all_generated_emails["models"][model_name] = {
            "iterations": {},
            "success_iteration": None,
            "final_email": None
        }
        
        # 分析当前邮件
        analysis_result = lime_analyze_email(current_email, model_name)
        all_analysis_results[model_name] = analysis_result
        
        print("analysis_result: ",analysis_result)
        # 如果邮件已经被分类为正常，则不需要对抗
        if not analysis_result["is_phishing"]:
            print(f"{model_name} 模型已将邮件分类为正常，无需对抗")
            evasion_results[model_name] = {
                "success": True,
                "needed_evasion": False,
                "email": current_email,
                "iterations": 0
            }
            
            # 对于不需要对抗的模型，所有轮次都使用原始邮件
            for i in range(1, max_iterations_per_model + 1):
                all_generated_emails["models"][model_name]["iterations"][f"iteration_{i}"] = {
                    "email": current_email,
                    "is_original": True,
                    "success": True,
                    "preprocessing_used": False
                }
            all_generated_emails["models"][model_name]["final_email"] = current_email
            all_generated_emails["models"][model_name]["success_iteration"] = 0
            continue
        
        # 迭代对抗逻辑
        iteration = 0
        evasion_success = False
        working_email = current_email
        iteration_history = []
        successful_email = None  # 记录成功对抗的邮件
        
        while iteration < max_iterations_per_model and not evasion_success:
            print(f"第 {iteration + 1} 次对抗 {model_name} 模型...")
            
            # 步骤1: LIME预处理（如果启用）
            if use_lime_preprocessing:
                # 尝试不同的预处理策略
                strategies = ["replace", "delete", "char_replace"]
                thresholds = [0.01, 0.02, 0.03]
                
                strategy_idx = iteration % len(strategies)
                threshold_idx = iteration % len(thresholds)
                
                strategy = strategies[strategy_idx]
                threshold = thresholds[threshold_idx]
                
                print(f"步骤1 - LIME预处理: 策略={strategy}, 阈值={threshold}")
                preprocessed_email = preprocess_email_with_lime(
                    working_email, 
                    analysis_result, 
                    strategy=strategy, 
                    threshold=threshold
                )
                print(f"预处理后邮件: {preprocessed_email[:100]}...")
            else:
                preprocessed_email = working_email
                strategy = None
                threshold = None
                print("跳过LIME预处理")
            
            # 步骤2: 使用大模型进一步生成对抗性邮件
            print("步骤2 - 大模型生成对抗性邮件")
            adversarial_email = generate_lime_adversarial_email(
                generation_client, 
                preprocessed_email,  # 使用预处理后的邮件
                analysis_result
            )

            print("adversarial_email:-------------------\n ",adversarial_email)
            print("analysis_result:-------------------\n ",analysis_result)
            
            # 再次分析生成的对抗性邮件
            recheck_result = lime_analyze_email(adversarial_email, model_name)
            
            # 保存这一轮生成的邮件 - 新增功能
            iteration_key = f"iteration_{iteration + 1}"
            all_generated_emails["models"][model_name]["iterations"][iteration_key] = {
                "email": adversarial_email,
                "preprocessed_email": preprocessed_email if use_lime_preprocessing else None,
                "input_email": working_email,
                "success": not recheck_result["is_phishing"],
                "preprocessing_used": use_lime_preprocessing,
                "preprocessing_strategy": strategy,
                "preprocessing_threshold": threshold,
                "is_original": False,
                "is_replicated": False,
                "confidence_before": analysis_result.get("confidence", 0.0),
                "confidence_after": recheck_result.get("confidence", 0.0),
                "label_before": analysis_result.get("label", "unknown"),
                "label_after": recheck_result.get("label", "unknown")
            }
            
            # 记录这次迭代的历史
            iteration_history.append({
                "iteration": iteration + 1,
                "input_email": working_email,
                "preprocessed_email": preprocessed_email if use_lime_preprocessing else None,
                "generated_email": adversarial_email,
                "success": not recheck_result["is_phishing"],
                "preprocessing_used": use_lime_preprocessing,
                "preprocessing_strategy": strategy if use_lime_preprocessing else None,
                "preprocessing_threshold": threshold if use_lime_preprocessing else None,
                "confidence_before": analysis_result.get("confidence", 0.0),
                "confidence_after": recheck_result.get("confidence", 0.0),
                "label_before": analysis_result.get("label", "unknown"),
                "label_after": recheck_result.get("label", "unknown")
            })
            
            # 检查对抗是否成功
            if not recheck_result["is_phishing"]:
                print(f"成功对抗 {model_name} 模型! (第 {iteration + 1} 次尝试)")
                evasion_success = True
                successful_email = adversarial_email
                working_email = adversarial_email
                all_generated_emails["models"][model_name]["success_iteration"] = iteration + 1
            else:
                print(f"第 {iteration + 1} 次对抗 {model_name} 模型失败，继续尝试...")
                # 更新工作邮件和分析结果，为下一次迭代做准备
                working_email = adversarial_email
                analysis_result = recheck_result
                
            iteration += 1
        
        # 如果成功对抗，复制成功的邮件到剩余的轮次 - 新增功能
        if evasion_success and successful_email:
            for remaining_iter in range(iteration + 1, max_iterations_per_model + 1):
                iteration_key = f"iteration_{remaining_iter}"
                all_generated_emails["models"][model_name]["iterations"][iteration_key] = {
                    "email": successful_email,
                    "preprocessed_email": None,
                    "input_email": successful_email,
                    "analysis_before": None,
                    "analysis_after": None,
                    "success": True,
                    "preprocessing_used": False,
                    "preprocessing_strategy": None,
                    "preprocessing_threshold": None,
                    "is_original": False,
                    "is_replicated": True,  # 标记为复制的邮件
                    "replicated_from_iteration": all_generated_emails["models"][model_name]["success_iteration"]
                }
        else:
            # 如果没有成功，使用最后一次生成的邮件填充剩余轮次
            last_email = working_email if iteration > 0 else current_email
            for remaining_iter in range(iteration + 1, max_iterations_per_model + 1):
                iteration_key = f"iteration_{remaining_iter}"
                all_generated_emails["models"][model_name]["iterations"][iteration_key] = {
                    "email": last_email,
                    "preprocessed_email": None,
                    "input_email": last_email,
                    "analysis_before": None,
                    "analysis_after": None,
                    "success": False,
                    "preprocessing_used": False,
                    "preprocessing_strategy": None,
                    "preprocessing_threshold": None,
                    "is_original": False,
                    "is_replicated": True,  # 标记为复制的邮件
                    "replicated_from_iteration": iteration if iteration > 0 else 0
                }
        
        if not evasion_success:
            print(f"经过 {max_iterations_per_model} 次尝试，未能成功对抗 {model_name} 模型")
        
        # 设置最终邮件
        all_generated_emails["models"][model_name]["final_email"] = working_email
        
        # 记录结果
        evasion_results[model_name] = {
            "success": evasion_success,
            "needed_evasion": True,
            "email": working_email,
            "iterations": iteration,
            "iteration_history": iteration_history,
            "preprocessing_enabled": use_lime_preprocessing,
            "initial_confidence": all_analysis_results[model_name].get("confidence", 0.0),
            "initial_label": all_analysis_results[model_name].get("label", "unknown"),
            "final_confidence": recheck_result.get("confidence", 0.0) if 'recheck_result' in locals() else 0.0,
            "final_label": recheck_result.get("label", "unknown") if 'recheck_result' in locals() else "unknown"
        }
        
        # 更新当前邮件为对抗成功的邮件（如果成功）
        if evasion_success:
            current_email = working_email
    
    # 使用 LLM 检测最终生成的邮件
    final_prediction, detection_result = detect_email(recheck_client, current_email)
    
    return current_email, final_prediction, detection_result, evasion_results, all_generated_emails


# ========================== 邮件评估 ==========================

def evaluate_email(client, email_content):
    """使用 LLM 评估电子邮件是否为钓鱼邮件（防御方）"""
    prompt = evaluation_prompt.format(email_content=email_content)
    response = get_LLM_response_vllm(client, prompt, role="defense")
    return response

def formatting_email(client, email_content):
    """使用 LLM 格式化电子邮件（攻击方）"""
    prompt = formatting_prompt.format(email_content=email_content)
    response_str = get_LLM_response_vllm(client, prompt, role="attack")

    print("formatting response: ", response_str[:200] if response_str else "Empty response")

    # 检查响应是否为空
    if not response_str or response_str.strip() == "":
        print("⚠️ 警告: LLM返回空响应，使用默认格式")
        return create_default_formatted_email(email_content)
    
    # 查找JSON内容
    start_index = response_str.find('{')
    end_index = response_str.rfind('}') + 1
    
    # 检查是否找到JSON
    if start_index == -1 or end_index <= start_index:
        print("⚠️ 警告: 响应中未找到JSON格式，使用默认格式")
        print(f"响应内容: {response_str[:500]}")
        return create_default_formatted_email(email_content)
    
    json_str = response_str[start_index:end_index]
    
    try:
        # 将JSON字符串解析为Python字典
        response_dict = json.loads(json_str)
        
        # 验证必要字段
        required_fields = ['has_url', 'has_attachment']
        for field in required_fields:
            if field not in response_dict:
                print(f"⚠️ 警告: JSON缺少必要字段'{field}'，添加默认值")
                response_dict[field] = "No"
        
        return response_dict
        
    except json.JSONDecodeError as e:
        print(f"⚠️ JSON解析错误: {e}")
        print(f"尝试解析的JSON: {json_str[:200]}")
        return create_default_formatted_email(email_content)


def create_default_formatted_email(email_content):
    """
    当formatting失败时，创建默认的格式化结果
    """
    return {
        "subject": "Important Notice",
        "has_url": "No",
        "url": "",
        "has_attachment": "No",
        "attchment": "",
        "attchment_type": "",
        "items": {
            "Text": email_content
        }
    }


def web_resource_email(client, email_content, url):
    """使用 LLM 生成Web资源（攻击方）"""
    prompt = web_generation_prompt.format(email_content=email_content, url=url)
    response_str = get_LLM_response_vllm(client, prompt, role="attack")

    print("web_resource: ",response_str)
    return response_str

def attachment_resource_email(client, email_content, attachment_name, attachment_type):
    """使用 LLM 生成附件资源（攻击方）"""
    prompt = attachment_generation_prompt.format(email_content=email_content, attachment_name=attachment_name, attachment_type=attachment_type)
    response_str = get_LLM_response_vllm(client, prompt, role="attack")

    print("attachment_resource: ",response_str)
    return response_str

# ========================== 迭代样本生成 & 二次检测 ==========================

def generate_and_recheck(generation_client, recheck_client, email_content, detection_reason):
    """
    生成规避检测的新钓鱼邮件，并进行二次检测
    :param generation_client: 攻击方客户端
    :param recheck_client: 防御方客户端
    :param email_content: 原始邮件正文
    :param detection_reason: 之前的检测结果
    :return: 新生成的邮件, 二次检测结果
    """
    prompt = generation_prompt.format(detection_reason=detection_reason, email_content=email_content)
    new_email = get_LLM_response_vllm(generation_client, prompt, role="attack")
    print("new_email",new_email)
    time.sleep(1)
    
    final_prediction, detection_result = detect_email(recheck_client, new_email)
    # recheck_prompt = detection_prompt.format(email_content=new_email)

    # recheck_result = get_LLM_response_vllm(recheck_client, recheck_prompt,"recheck")
    # time.sleep(1)

    # if "etermined as a phishing email" in recheck_result:
    #     print("recheck_result",recheck_result)
    #     final_prediction = 1
    # elif "etermined as another legitimate email" in recheck_result:
    #     final_prediction = 0
    # else:
    #     final_prediction = 0

    return new_email, final_prediction, detection_result

def generate_iteration_list(final_prediction, iteration, max_iterations=5):
    """
    生成一个长度为 max_iterations 的列表。
    - 在 iteration 之前的值设为 1（表示仍在迭代）。
    - 从 iteration 开始（包括 iteration 位置）之后的值设为 final_prediction。

    :param final_prediction: 最终预测值 (0 或 1)
    :param iteration: 逃逸或检测成功的轮次 (1-based index)
    :param max_iterations: 最大迭代次数
    :return: 长度为 max_iterations 的列表
    """
    result = [1] * (iteration - 1) + [final_prediction] * (max_iterations - iteration + 1)
    return result

def generate_and_recheck_iteratively(generation_client,recheck_client, email_content, detection_reason, max_iterations=5):
    """
    迭代生成钓鱼邮件，直到被检测为钓鱼邮件，最多迭代 max_iterations 轮
    :param client: OpenAI 客户端
    :param email_content: 原始邮件正文
    :param detection_reason: 之前的检测结果
    :param max_iterations: 最大迭代次数
    :return: 生成的邮件, 最终检测结果, 迭代次数, 所有迭代的邮件历史
    """
    iteration = 0
    new_email = email_content
    final_prediction = 0
    recheck_result = ""
    
    # 🔧 修复: 保存每轮迭代的邮件
    iteration_history = {}

    while iteration < max_iterations:
        iteration += 1
        print(f"Iteration {iteration}: Generating a new phishing email...")

        new_email, final_prediction, detection_reason = generate_and_recheck(generation_client, recheck_client, new_email, detection_reason)
        
        # 🔧 修复: 记录这一轮的邮件
        iteration_history[f"iteration_{iteration}"] = {
            "email": new_email,
            "prediction": final_prediction,
            "detection_reason": detection_reason
        }

        if final_prediction == 0:
            print(f"Success! The new phishing email successfully evaded detection after {iteration} iterations.")
            break  # 生成的样本已被检测为钓鱼邮件，结束迭代
        else:
            print(f"Failed to detect. Continuing iteration {iteration + 1}...")

    # 🔧 修复: 如果没有完成5轮，用最后一轮的邮件填充剩余轮次
    if iteration < max_iterations:
        for i in range(iteration + 1, max_iterations + 1):
            iteration_history[f"iteration_{i}"] = {
                "email": new_email,
                "prediction": final_prediction,
                "detection_reason": detection_reason,
                "is_replicated": True  # 标记为复制的
            }

    return new_email, final_prediction, recheck_result, iteration, iteration_history


# ========================== 主执行逻辑 ==========================

def load_email_data(data_path="./data/filtered_data.json"):
    """加载邮件数据"""
    with open(data_path, 'r', encoding='utf-8') as f:
        return json.load(f)

def load_personal_info(data_path="./personal_info.json"):
    """加载个人信息数据"""
    try:
        with open(data_path, 'r', encoding='utf-8') as f:
            return json.load(f)["templates"]
    except FileNotFoundError:
        print(f"Error: Personal info file {data_path} not found.")
        return []

def generate_phishing_from_personal_info(client, personal_info, scenario="IT Security Alert"):
    """基于个人信息和指定场景生成钓鱼邮件（攻击方）"""
    prompt = init_prompt.format(
        personal_info=json.dumps(personal_info, ensure_ascii=False, indent=2),
        scenario=random.choice(list_of_scenarios)
    )
    
    response = get_LLM_response_vllm(client, prompt, role="attack")
    return response

def generate_emails_from_personal_info(generate_client, personal_info_data, scenarios=None):
    """基于个人信息生成钓鱼邮件列表"""
    # 默认场景列表
    if scenarios is None:
        scenarios = [
            "IT Security Alert", 
            "HR Benefits Update", 
            "Financial Account Verification",
            "Company Policy Update",
            "System Maintenance Notification"
        ]
    
    generated_emails = []
    
    for index, personal_info in enumerate(personal_info_data):
        print(f"Processing personal info {index + 1}/{len(personal_info_data)}")
        
        # 为每个人选择一个场景（循环使用）
        scenario = scenarios[index % len(scenarios)]
        print(f"Using scenario: {scenario}")
        
        # 生成钓鱼邮件
        generated_email = generate_phishing_from_personal_info(generate_client, personal_info, scenario)
        
        # 保存生成的邮件数据
        email_data = {
            "Text": generated_email,
            "Class": 1,  # 标记为钓鱼邮件
            "type": "personal_info_generated",  # 标记来源
            "personal_info": personal_info,  # 保存原始个人信息
            "scenario": scenario,  # 保存使用的场景
        }
        
        generated_emails.append(email_data)
        
        print(f"Generated email with scenario '{scenario}':")
        print(generated_email)
        print("-" * 50)
        
        # 添加延时以避免API限制
        time.sleep(2)
    
    return generated_emails

def main(data_source="email_data", personal_info_path=None, scenarios=None, enable_lime_attack=False, enable_llm_attack=False, lime_models=None, max_lime_iterations=5, use_lime_preprocessing=True):
    """
    主函数
    :param data_source: 数据源类型，可选 "email_data" 或 "personal_info"
    :param personal_info_path: 个人信息数据文件路径（仅在 personal_info 模式下使用）
    :param scenarios: 场景列表，用于生成钓鱼邮件
    :param enable_lime_attack: 是否启用LIME对抗攻击
    :param enable_llm_attack: 是否启用大模型对抗攻击
    :param lime_models: 用于LIME对抗的模型列表
    :param max_lime_iterations: 每个LIME模型的最大对抗迭代次数
    :param use_lime_preprocessing: 是否在调用大模型前先进行LIME预处理
    """
    # 记录开始时间
    start_time = time.time()
    print(f"程序开始时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    
    # 默认LIME模型
    if lime_models is None:
        lime_models = ["textcnn", "bert"]
    
    # 创建输出目录
    output_dir = create_output_directory()
    
    # 创建客户端 - 攻防分离
    print("\n" + "="*60)
    print("初始化攻防客户端")
    print("="*60)
    
    # 防御方客户端（用于检测和评估）
    detection_client = create_client(role="defense")
    evaluation_client = create_client(role="defense")
    
    # 攻击方客户端（用于生成钓鱼邮件）
    generate_client = create_client(role="attack")
    formatting_client = create_client(role="attack")
    web_resource_client = create_client(role="attack")
    attachment_resource_client = create_client(role="attack")
    lime_client = create_client(role="attack")  # LIME对抗专用客户端（攻击方）
    
    print("="*60 + "\n")

    # 根据数据源类型加载或生成数据
    if data_source == "email_data":
        # 原有的邮件数据模式
        emails_data = load_email_data()
        
        # 按类型采样
        samples_per_type = 10
        sampled_data = []
        emails_by_type = {}
        
        for email in emails_data:
            email_type = email.get("type", "Unknown")
            if email_type not in emails_by_type:
                emails_by_type[email_type] = []
            emails_by_type[email_type].append(email)
        
        for email_type, emails in emails_by_type.items():
            type_samples = random.sample(emails, min(samples_per_type, len(emails)))
            sampled_data.extend(type_samples)
            print(f"Sampled {len(type_samples)} emails of type '{email_type}'")
        
        print(f"Total sampled emails: {len(sampled_data)}")
        emails_to_process = sampled_data
        
    elif data_source == "personal_info":
        # 基于个人信息生成邮件的模式
        if not personal_info_path:
            raise ValueError("personal_info_path must be provided for personal_info mode")
        
        personal_info_data = load_personal_info(personal_info_path)
        if not personal_info_data:
            raise ValueError("No personal information data found")
        
        emails_to_process = generate_emails_from_personal_info(generate_client, personal_info_data, scenarios)
        print(f"Generated {len(emails_to_process)} emails from personal information")
    
    else:
        raise ValueError(f"Unknown data source type: {data_source}")

    first_results = []
    final_results = []
    label_results = []
    data = []
    evaluation_data = []
    type_results = {}

    print(f"Attack configuration: LIME Attack: {enable_lime_attack}, LLM Attack: {enable_llm_attack}")
    if enable_lime_attack:
        print(f"LIME models to use: {lime_models}")

    # 处理邮件数据
    for index, email_info in enumerate(emails_to_process):
        email_content = email_info["Text"]
        label = email_info["Class"]
        email_type = email_info.get("type", "Unknown")

        if email_type not in type_results:
            type_results[email_type] = {
                "labels": [],
                "first_predictions": [],
                "final_predictions": []
            }

        # 第一次检测 - 使用大模型检测器
        llm_prediction, llm_detection_result = detect_email(detection_client, email_content)
        current_email = email_content
        final_prediction = llm_prediction
        iterations = 0
        lime_evasion_results = {}
        all_generated_emails = None  # 初始化邮件历史变量
        
        # 初始化所有需要的变量，确保每个样本都有完整的数据结构
        evaluate_result = None
        recheck_result = llm_detection_result
        subject = ""
        url = ""
        attachment_name = ""
        attachment_type = ""
        has_url = "No"
        has_attachment = "No"
        web_resource = None
        attachment_resource = None
        formatted_email_dict = {}
        
        # 根据攻击类型判断是否需要进行攻击
        should_attack_with_lime = False
        should_attack_with_llm = False
        
        # 对于LIME攻击：检查ML模型是否将邮件识别为钓鱼邮件
        if enable_lime_attack and label == 1:  # 只对真实的钓鱼邮件进行LIME攻击
            print("检查ML模型对邮件的分类结果...")
            # 使用LIME分析第一个模型来判断是否需要攻击
            first_lime_model = lime_models[0] if lime_models else "textcnn"
            lime_analysis = lime_analyze_email(current_email, first_lime_model)
            
            # 如果ML模型将邮件分类为钓鱼邮件，则需要进行LIME攻击
            if lime_analysis["is_phishing"]:
                should_attack_with_lime = True
                print(f"ML模型({first_lime_model})将邮件分类为钓鱼邮件，需要进行LIME攻击")
            else:
                print(f"ML模型({first_lime_model})已将邮件分类为正常邮件，无需LIME攻击")
        
        # 对于LLM攻击：检查大模型是否将邮件识别为钓鱼邮件
        if enable_llm_attack and label == 1:  # 只对真实的钓鱼邮件进行LLM攻击
            if llm_prediction == 1:
                should_attack_with_llm = True
                print("大模型将邮件分类为钓鱼邮件，需要进行LLM攻击")
            else:
                print("大模型已将邮件分类为正常邮件，无需LLM攻击")
        
        # 决定是否需要攻击
        should_attack = should_attack_with_lime or should_attack_with_llm
        
        if should_attack:
            print(f"\n=== Processing phishing email {index + 1} (需要攻击) ===")
            print(f"LIME攻击: {should_attack_with_lime}, LLM攻击: {should_attack_with_llm}")
            
            # LIME对抗攻击
            if should_attack_with_lime:
                print("启动LIME对抗攻击...")
                try:
                    lime_adversarial_email, lime_prediction, lime_detection_result, lime_evasion_results, all_generated_emails = adversarial_ml_model_evasion(
                        generate_client, detection_client, current_email, lime_models, max_lime_iterations, use_lime_preprocessing
                    )
                    current_email = lime_adversarial_email
                    final_prediction = lime_prediction
                    recheck_result = lime_detection_result
                    print(f"LIME对抗完成，当前检测结果: {final_prediction}")
                except Exception as e:
                    print(f"LIME对抗攻击失败: {e}")
                    lime_evasion_results = {"error": str(e)}
                    all_generated_emails = None
                
                time.sleep(2)
            
            # 大模型对抗攻击
            llm_iteration_history = None
            if should_attack_with_llm:
                print("启动大模型对抗攻击...")
                generated_email, final_prediction, recheck_result, iterations, llm_iteration_history = generate_and_recheck_iteratively(
                    generate_client, detection_client, current_email, recheck_result
                )
                current_email = generated_email
                print(f"大模型对抗完成，迭代次数: {iterations}, 最终检测结果: {final_prediction}")
                time.sleep(2)
            
            # 评估生成的邮件
            evaluate_result = evaluate_email(evaluation_client, current_email)
            print("evaluate_result: ", evaluate_result)
        else:
            print(f"\n=== Processing email {index + 1} (无需攻击) ===")
            print(f"原因: label={label}, llm_prediction={llm_prediction}")
            if enable_lime_attack and label == 1:
                print("ML模型已将邮件分类为正常邮件")
            if enable_llm_attack and label == 1:
                print("大模型已将邮件分类为正常邮件")
            final_prediction = llm_prediction
        
        # 计算预测结果列表
        if enable_llm_attack and should_attack_with_llm:
            final_prediction_list = generate_iteration_list(final_prediction, iterations+1, max_iterations=5)
        else:
            final_prediction_list = [final_prediction] * 5
        
        # 对所有邮件进行格式化处理（不只是攻击后的邮件）
        try:
            print("正在格式化邮件...")
            formatted_email_dict = formatting_email(formatting_client, current_email)
            print("formatted_email: ", formatted_email_dict.get('items', 'No items found'))

            subject = formatted_email_dict.get('subject', '')
            url = formatted_email_dict.get('url', '')
            attachment_name = formatted_email_dict.get('attchment', '')
            attachment_type = formatted_email_dict.get('attchment_type', '')

            # 对格式化后的邮件再次进行检测
            format_prediction, format_detection_result = detect_email(detection_client, formatted_email_dict)
            final_prediction_list.append(format_prediction)

            # 处理URL和附件资源
            try:
                has_url = formatted_email_dict.get('has_url', 'No')
                has_attachment = formatted_email_dict.get('has_attachment', 'No')
                
                if has_url == "Yes" and url:
                    web_resource = web_resource_email(web_resource_client, current_email, url)
                    print(f"生成了Web资源: {url}")
                else:
                    web_resource = None
                    
                if has_attachment == "Yes" and attachment_name:
                    attachment_resource = attachment_resource_email(attachment_resource_client, current_email, attachment_name, attachment_type)
                    print(f"生成了附件资源: {attachment_name} ({attachment_type})")
                else:
                    attachment_resource = None
                
            except Exception as e:
                print(f"处理URL/附件时出错: {e}")
                web_resource = None
                attachment_resource = None
                
        except Exception as e:
            print(f"邮件格式化失败: {e}")
            # 格式化失败时，使用原始邮件进行最后一次检测
            format_prediction, format_detection_result = detect_email(detection_client, current_email)
            final_prediction_list.append(format_prediction)

        # 为所有样本保存完整的数据结构
        sample_data = {
            "subject": subject,
            "original_text": email_content,
            "Text": current_email,
            "Class": label,
            "response": recheck_result,
            "prediction": final_prediction_list,
            "first_prediction": llm_prediction,  # 大模型的初始预测
            "final_prediction": final_prediction,
            "iterations": iterations,
            "type": email_type,
            "has_url": has_url,
            "has_attachment": has_attachment,
            "attachment_name": attachment_name,
            "attachment_type": attachment_type,
            "url": url,
            "web_resource": web_resource,
            "attachment_resource": attachment_resource,
            "lime_attack_enabled": enable_lime_attack,
            "llm_attack_enabled": enable_llm_attack,
            "lime_evasion_results": lime_evasion_results if enable_lime_attack and should_attack_with_lime else None,
            "all_generated_emails": all_generated_emails if enable_lime_attack and should_attack_with_lime else None,
            "llm_iteration_history": llm_iteration_history if should_attack_with_llm else None,  # 🔧 修复: 保存LLM攻击的迭代历史
            "was_attacked": should_attack,
            "lime_attack_performed": should_attack_with_lime,
            "llm_attack_performed": should_attack_with_llm,
            "attack_reason": f"LIME: {should_attack_with_lime}, LLM: {should_attack_with_llm}" if should_attack else f"无需攻击: label={label}, llm_prediction={llm_prediction}",
            "formatted_email_success": bool(formatted_email_dict),
            "evaluation_performed": evaluate_result is not None
        }
        
        data.append(sample_data)
        
        # 如果进行了攻击和评估，保存评估数据
        if should_attack and evaluate_result:
            evaluation_data.append({
                "sample_index": index,
                "iteration": iterations,
                "evaluation_result": evaluate_result,
                "lime_attack_used": should_attack_with_lime,
                "llm_attack_used": should_attack_with_llm,
                "original_text": email_content,
                "final_text": current_email
            })

        # 统计数据（保持原有逻辑）
        if llm_prediction in [0, 1]:
            label_results.append(label)
            first_results.append(llm_prediction)
            final_results.append(final_prediction_list)

            type_results[email_type]["labels"].append(label)
            type_results[email_type]["first_predictions"].append(llm_prediction)
            type_results[email_type]["final_predictions"].append(final_prediction_list)

        print(f"Sample {index + 1}: Label={label}, LLM Detection={llm_prediction}, Final Detection={final_prediction_list[-1]}, Type={email_type}, Attacked={should_attack}")

    # 保存结果
    results_file = os.path.join(output_dir, 'responds.json')
    with open(results_file, 'w', encoding='utf-8') as f:
        json.dump(data, f, ensure_ascii=False, indent=4, cls=NumpyEncoder)

    eval_file = os.path.join(output_dir, 'evaluation_results.json')
    with open(eval_file, 'w', encoding='utf-8') as f:
        json.dump(evaluation_data, f, ensure_ascii=False, indent=4, cls=NumpyEncoder)
    
    # 计算并保存ASR及其他指标到单独的文件
    metrics_summary = {
        "initial_detection": {},
        "iterations": [],
        "by_type": {}
    }
    
    if first_results:
        initial_metrics = evaluate_metrics_macro(label_results, first_results)
        initial_asr = calculate_asr(label_results, first_results)
        metrics_summary["initial_detection"] = {
            "accuracy": initial_metrics.get("Accuracy", 0),
            "precision": initial_metrics.get("Precision", 0),
            "recall": initial_metrics.get("Recall", 0),
            "f1_score": initial_metrics.get("F1-score", 0),
            "asr": initial_asr["ASR"],
            "asr_percent": initial_asr["evasion_rate_percent"],
            "successful_evasions": initial_asr["successful_evasions"],
            "total_malicious": initial_asr["total_malicious"]
        }
    
    if final_results:
        final_results_transposed = list(zip(*final_results))
        for i, item_result in enumerate(final_results_transposed):
            iteration_metrics = evaluate_metrics_macro(label_results, list(item_result))
            iteration_asr = calculate_asr(label_results, list(item_result))
            metrics_summary["iterations"].append({
                "iteration": i + 1,
                "accuracy": iteration_metrics.get("Accuracy", 0),
                "precision": iteration_metrics.get("Precision", 0),
                "recall": iteration_metrics.get("Recall", 0),
                "f1_score": iteration_metrics.get("F1-score", 0),
                "asr": iteration_asr["ASR"],
                "asr_percent": iteration_asr["evasion_rate_percent"],
                "successful_evasions": iteration_asr["successful_evasions"],
                "total_malicious": iteration_asr["total_malicious"]
            })
    
    for email_type, results in type_results.items():
        if results["first_predictions"]:
            type_metrics = evaluate_metrics_macro(results["labels"], results["first_predictions"])
            type_asr = calculate_asr(results["labels"], results["first_predictions"])
            metrics_summary["by_type"][email_type] = {
                "accuracy": type_metrics.get("Accuracy", 0),
                "precision": type_metrics.get("Precision", 0),
                "recall": type_metrics.get("Recall", 0),
                "f1_score": type_metrics.get("F1-score", 0),
                "asr": type_asr["ASR"],
                "asr_percent": type_asr["evasion_rate_percent"],
                "successful_evasions": type_asr["successful_evasions"],
                "total_malicious": type_asr["total_malicious"]
            }
    
    # 保存指标摘要
    metrics_file = os.path.join(output_dir, 'metrics_summary.json')
    with open(metrics_file, 'w', encoding='utf-8') as f:
        json.dump(metrics_summary, f, ensure_ascii=False, indent=4, cls=NumpyEncoder)
    print(f"指标摘要（包含ASR）已保存到: {metrics_file}")
    
    # 保存所有生成的邮件历史到单独的文件 - 修复：包含所有样本
    all_email_histories = []
    for sample_data in data:
        # 为所有样本创建邮件历史，不仅仅是有 all_generated_emails 的样本
        email_history = {
            "sample_info": {
                "original_text": sample_data.get("original_text", ""),
                "subject": sample_data.get("subject", ""),
                "type": sample_data.get("type", "Unknown"),
                "class": sample_data.get("Class", 0),
                "lime_attack_enabled": sample_data.get("lime_attack_enabled", False),
                "llm_attack_enabled": sample_data.get("llm_attack_enabled", False),
                "was_attacked": sample_data.get("was_attacked", False),
                "lime_attack_performed": sample_data.get("lime_attack_performed", False),
                "llm_attack_performed": sample_data.get("llm_attack_performed", False)
            }
        }
        
        # 如果有 LIME 生成的邮件历史，使用它；否则创建基础的邮件历史
        if sample_data.get("all_generated_emails"):
            email_history["generated_emails"] = sample_data["all_generated_emails"]
        elif sample_data.get("llm_iteration_history"):
            # 🔧 修复: 使用LLM攻击的迭代历史
            llm_hist = sample_data["llm_iteration_history"]
            original_email = sample_data.get("original_text", "")
            
            email_history["generated_emails"] = {
                "original_email": original_email,
                "models": {
                    "llm_attack": {
                        "iterations": {},
                        "success_iteration": sample_data.get("iterations", 0),
                        "final_email": sample_data.get("Text", original_email)
                    }
                }
            }
            
            # 使用LLM攻击的实际迭代邮件
            for iter_key, iter_data in llm_hist.items():
                email_history["generated_emails"]["models"]["llm_attack"]["iterations"][iter_key] = {
                    "email": iter_data.get("email", ""),
                    "preprocessed_email": None,
                    "input_email": iter_data.get("email", ""),
                    "success": iter_data.get("prediction", 1) == 0,
                    "preprocessing_used": False,
                    "preprocessing_strategy": None,
                    "preprocessing_threshold": None,
                    "is_original": False,
                    "is_replicated": iter_data.get("is_replicated", False),
                    "confidence_before": 0.0,
                    "confidence_after": 0.0,
                    "label_before": "unknown",
                    "label_after": "ham" if iter_data.get("prediction", 1) == 0 else "spam",
                    "detection_reason": iter_data.get("detection_reason", "")
                }
        else:
            # 为没有进行任何攻击的样本创建基础的邮件历史结构
            current_email = sample_data.get("Text", sample_data.get("original_text", ""))
            email_history["generated_emails"] = {
                "original_email": sample_data.get("original_text", current_email),
                "models": {
                    "no_attack": {
                        "iterations": {},
                        "success_iteration": None,
                        "final_email": current_email
                    }
                }
            }
            
            # 为所有5轮迭代填充相同的邮件内容（因为没有进行攻击）
            for i in range(1, 6):
                email_history["generated_emails"]["models"]["no_attack"]["iterations"][f"iteration_{i}"] = {
                    "email": current_email,
                    "preprocessed_email": None,
                    "input_email": current_email,
                    "success": sample_data.get("final_prediction", 0) == 0,
                    "preprocessing_used": False,
                    "preprocessing_strategy": None,
                    "preprocessing_threshold": None,
                    "is_original": i == 1,  # 第一轮标记为原始
                    "is_replicated": i > 1,  # 后续轮次标记为复制
                    "confidence_before": 0.0,
                    "confidence_after": 0.0,
                    "label_before": "unknown",
                    "label_after": "unknown"
                }
            
            email_history["generated_emails"]["models"]["no_attack"]["final_email"] = current_email
        
        all_email_histories.append(email_history)
    
    if all_email_histories:
        email_histories_file = os.path.join(output_dir, 'all_generated_email_histories.json')
        with open(email_histories_file, 'w', encoding='utf-8') as f:
            json.dump(all_email_histories, f, ensure_ascii=False, indent=4, cls=NumpyEncoder)
        print(f"保存了 {len(all_email_histories)} 个样本的邮件生成历史到: {email_histories_file}")
        
        # 直接保存原始邮件和5轮生成的邮件到JSON文件 - 简化版本
        iteration_data = {}
        iteration_files = {}
        
        # 定义各轮次的文件名
        iteration_files["original"] = "iteration_0_original_emails.json"
        for i in range(1, 6):
            iteration_files[f"iteration_{i}"] = f"iteration_{i}_emails.json"
        
        # 初始化每个轮次的数据
        for iter_name in iteration_files.keys():
            iteration_data[iter_name] = []
        
        # 为每个样本收集数据
        for sample_idx, email_history in enumerate(all_email_histories):
            sample_info = email_history["sample_info"]
            generated_emails = email_history["generated_emails"]
            
            # 原始邮件数据
            original_data = {
                "Text": generated_emails["original_email"],
                "Class": sample_info["class"],
                "type": sample_info["type"]
            }
            iteration_data["original"].append(original_data)
            
            # 为每个模型的每轮迭代收集邮件
            for model_name, model_data in generated_emails["models"].items():
                for iter_num in range(1, 6):  # 5轮迭代
                    iter_key = f"iteration_{iter_num}"
                    
                    if iter_key in model_data["iterations"]:
                        email_content = model_data["iterations"][iter_key]["email"]
                    else:
                        # 如果该轮次不存在，使用最后一个可用的邮件
                        available_iters = [k for k in model_data["iterations"].keys() if k.startswith("iteration_")]
                        if available_iters:
                            last_iter = max(available_iters, key=lambda x: int(x.split("_")[1]))
                            email_content = model_data["iterations"][last_iter]["email"]
                        else:
                            email_content = generated_emails["original_email"]
                    
                    iteration_entry = {
                        "Text": email_content,
                        "Class": sample_info["class"],
                        "type": sample_info["type"]
                    }
                    
                    # 如果有多个模型，需要区分不同模型的结果
                    if len(generated_emails["models"]) > 1:
                        iteration_entry["model"] = model_name
                    
                    iteration_data[iter_key].append(iteration_entry)
        
        # 直接保存每个轮次的数据到JSON文件
        for iter_name, data_list in iteration_data.items():
            if data_list:  # 只保存非空的数据
                iter_file = os.path.join(output_dir, iteration_files[iter_name])
                with open(iter_file, 'w', encoding='utf-8') as f:
                    json.dump(data_list, f, ensure_ascii=False, indent=4, cls=NumpyEncoder)
        
        print(f"已将邮件分别保存到6个JSON文件:")
        for iter_name, filename in iteration_files.items():
            if iteration_data[iter_name]:
                email_count = len(iteration_data[iter_name])
                file_path = os.path.join(output_dir, filename)
                print(f"  - {iter_name}: {email_count} 条邮件 ({filename})")
            else:
                print(f"  - {iter_name}: 0 条邮件")
    
    type_results_file = os.path.join(output_dir, 'type_results.json')
    with open(type_results_file, 'w', encoding='utf-8') as f:
        json.dump(type_results, f, ensure_ascii=False, indent=4, cls=NumpyEncoder)
    
    # 保存攻击配置信息
    attack_config = {
        "attack_model": {
            "model": config['attack_model']['model'],
            "api_base_url": config['attack_model']['api_base_url']
        },
        "defense_model": {
            "model": config['defense_model']['model'],
            "api_base_url": config['defense_model']['api_base_url']
        },
        "lime_attack_enabled": enable_lime_attack,
        "llm_attack_enabled": enable_llm_attack,
        "lime_models": lime_models if enable_lime_attack else None,
        "max_lime_iterations": max_lime_iterations if enable_lime_attack else None,
        "use_lime_preprocessing": use_lime_preprocessing if enable_lime_attack else None,
        "data_source": data_source,
        "total_samples": len(emails_to_process)
    }
    config_file = os.path.join(output_dir, 'attack_config.json')
    with open(config_file, 'w', encoding='utf-8') as f:
        json.dump(attack_config, f, ensure_ascii=False, indent=4, cls=NumpyEncoder)
    
    print(f"All results saved to directory: {output_dir}")

    # 计算评估指标
    print("\n" + "="*60)
    print("评估指标结果")
    print("="*60)
    
    if first_results:
        print("\n初始检测指标 (Initial Detection Metrics):")
        initial_metrics = evaluate_metrics_macro(label_results, first_results)
        for metric, value in initial_metrics.items():
            print(f"  {metric}: {value:.4f}")
        
        # 计算初始ASR
        initial_asr = calculate_asr(label_results, first_results)
        print(f"\n初始ASR (Attack Success Rate - 攻击成功率/漏报率):")
        print(f"  ASR: {initial_asr['ASR']:.4f} ({initial_asr['evasion_rate_percent']:.2f}%)")
        print(f"  成功绕过检测: {initial_asr['successful_evasions']} / {initial_asr['total_malicious']} 恶意样本")
    else:
        print("No valid initial detection results to evaluate.")

    if final_results:
        final_results_transposed = list(zip(*final_results))
        for i, item_result in enumerate(final_results_transposed):
            print(f"\n第 {i + 1} 轮迭代检测指标 (Iteration {i + 1}):")
            iteration_metrics = evaluate_metrics_macro(label_results, list(item_result))
            for metric, value in iteration_metrics.items():
                print(f"  {metric}: {value:.4f}")
            
            # 计算该轮次的ASR
            iteration_asr = calculate_asr(label_results, list(item_result))
            print(f"  ASR (攻击成功率): {iteration_asr['ASR']:.4f} ({iteration_asr['evasion_rate_percent']:.2f}%)")
            print(f"  成功绕过检测: {iteration_asr['successful_evasions']} / {iteration_asr['total_malicious']} 恶意样本")
    else:
        print("No valid final detection results to evaluate.")

    print("\n" + "="*60)
    print("按邮件类型分类的指标（所有轮次）")
    print("="*60)
    for email_type, results in type_results.items():
        if results["final_predictions"]:
            print(f"\n类型 '{email_type}':")
            
            # 转置final_predictions，得到每一轮的预测结果
            type_final_transposed = list(zip(*results["final_predictions"]))
            
            for i, iteration_preds in enumerate(type_final_transposed):
                print(f"\n  第 {i + 1} 轮迭代检测指标 (Iteration {i + 1}):")
                type_metrics = evaluate_metrics_macro(results["labels"], list(iteration_preds))
                for metric, value in type_metrics.items():
                    print(f"    {metric}: {value:.4f}")
                
                # 计算该类型该轮次的ASR
                type_asr = calculate_asr(results["labels"], list(iteration_preds))
                print(f"    ASR (攻击成功率): {type_asr['ASR']:.4f} ({type_asr['evasion_rate_percent']:.2f}%)")
                print(f"    成功绕过检测: {type_asr['successful_evasions']} / {type_asr['total_malicious']} 恶意样本")
        else:
            print(f"No valid detection results to evaluate for type '{email_type}'.")
    
    print("="*60)

    # 计算并输出执行时间和token统计
    end_time = time.time()
    total_time = end_time - start_time
    print(f"\n程序结束时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"总执行时间: {total_time:.2f} 秒 ({total_time/60:.2f} 分钟)")
    
    # 输出token使用统计和费用分析
    print("\n" + "="*60)
    print("TOKEN使用统计和费用分析")
    print("="*60)
    
    token_summary = get_token_usage_summary()
    print(f"总Token使用量: {token_summary['total_tokens']:,}")
    print(f"  - 输入Tokens: {token_summary['total_prompt_tokens']:,}")
    print(f"  - 输出Tokens: {token_summary['total_completion_tokens']:,}")
    print(f"总请求次数: {token_summary['total_requests']:,}")
    print(f"平均每次请求Token数: {token_summary['average_tokens_per_request']:.1f}")
    
    # 计算详细费用
    cost_breakdown = estimate_cost(
        prompt_tokens=token_summary['total_prompt_tokens'],
        completion_tokens=token_summary['total_completion_tokens']
    )
    print(f"\n总费用明细:")
    print(f"  - 输入费用: ${cost_breakdown['input_cost']:.4f} 美元 ({cost_breakdown['input_tokens']:,} tokens)")
    print(f"  - 输出费用: ${cost_breakdown['output_cost']:.4f} 美元 ({cost_breakdown['output_tokens']:,} tokens)")
    print(f"  - 总费用: ${cost_breakdown['total_cost']:.4f} 美元")
    print(f"  - 模型: {cost_breakdown['model']}")
    
    # 按功能分类的费用分析
    detailed_analysis = get_detailed_cost_analysis()
    function_costs = detailed_analysis['function_costs']
    
    if function_costs:
        print(f"\n按功能分类的费用分析:")
        print(f"{'功能名称':<25} {'请求次数':<8} {'输入Token':<10} {'输出Token':<10} {'总费用':<10}")
        print("-" * 75)
        
        # 按总费用排序
        sorted_functions = sorted(function_costs.items(), key=lambda x: x[1]['total_cost'], reverse=True)
        
        for function_name, costs in sorted_functions:
            print(f"{function_name:<25} {costs['request_count']:<8} {costs['prompt_tokens']:<10,} {costs['completion_tokens']:<10,} ${costs['total_cost']:<9.4f}")
        
        print("-" * 75)
        
        # 显示占比最高的几个功能
        total_cost = cost_breakdown['total_cost']
        if total_cost > 0:
            print(f"\n费用占比最高的功能:")
            for i, (function_name, costs) in enumerate(sorted_functions[:5]):
                percentage = (costs['total_cost'] / total_cost) * 100
                print(f"  {i+1}. {function_name}: ${costs['total_cost']:.4f} ({percentage:.1f}%)")
    
    print("="*60)
    
    # 保存详细的费用分析结果
    detailed_cost_analysis = get_detailed_cost_analysis()
    # 添加执行时间信息
    detailed_cost_analysis["execution_info"] = {
        "start_time": start_time,
        "end_time": end_time,
        "total_time_seconds": total_time,
        "total_time_minutes": total_time / 60
    }
    cost_analysis_file = os.path.join(output_dir, 'cost_analysis.json')
    with open(cost_analysis_file, 'w', encoding='utf-8') as f:
        json.dump(detailed_cost_analysis, f, ensure_ascii=False, indent=4, cls=NumpyEncoder)
    print(f"费用分析结果已保存到: {cost_analysis_file}")


# ========================== JSON序列化处理 ==========================

class NumpyEncoder(json.JSONEncoder):
    """自定义JSON编码器，处理numpy数据类型和其他不可序列化对象"""
    def default(self, obj):
        if isinstance(obj, np.integer):
            return int(obj)
        elif isinstance(obj, np.floating):
            return float(obj)
        elif isinstance(obj, np.ndarray):
            return obj.tolist()
        elif hasattr(obj, 'item'):  # 处理numpy标量
            return obj.item()
        elif hasattr(obj, '__dict__'):  # 对于复杂对象，尝试转换为字符串
            return f"<{obj.__class__.__name__} object - not serializable>"
        else:
            try:
                return str(obj)
            except:
                return f"<{type(obj).__name__} - cannot serialize>"

# ========================== 智能映射缓存系统 ==========================

# 智能映射表的缓存机制
smart_replacement_cache = {
    "mapping": {},  # 实际的映射表
    "last_update": 0,  # 上次更新时间戳
    "update_interval": 300,  # 更新间隔（秒），5分钟
    "memory_hash": "",  # 记忆模块的哈希值，用于检测变化
    "update_count": 0  # 更新次数
}

def get_memory_hash():
    """计算记忆模块的哈希值，用于检测是否有变化"""
    import hashlib
    
    if not lime_memory:
        return ""
    
    # 将记忆模块的内容转换为字符串进行哈希
    memory_str = ""
    for key in sorted(lime_memory.keys()):
        memory_str += str(sorted(lime_memory[key].items()))
    
    return hashlib.md5(memory_str.encode()).hexdigest()

def should_update_smart_mapping():
    """判断是否应该更新智能映射表"""
    current_time = time.time()
    
    # 时间条件：距离上次更新超过指定间隔
    time_passed = current_time - smart_replacement_cache["last_update"]
    time_condition = time_passed >= smart_replacement_cache["update_interval"]
    
    # 记忆变化条件：记忆模块内容发生变化
    current_hash = get_memory_hash()
    memory_changed = current_hash != smart_replacement_cache["memory_hash"]
    
    # 首次更新条件：从未更新过
    never_updated = smart_replacement_cache["update_count"] == 0
    
    return time_condition or memory_changed or never_updated

def generate_smart_replacement_mapping(client=None):
    """使用大模型生成智能的钓鱼词汇到正常词汇的映射表"""
    if not lime_memory or not lime_memory.get("phishing_words") or not lime_memory.get("legitimate_words"):
        print("记忆模块数据不足，无法生成智能映射")
        return {}
    
    # 获取高频词汇
    top_phishing = sorted(lime_memory["phishing_words"].items(), key=lambda x: x[1], reverse=True)[:20]
    top_legitimate = sorted(lime_memory["legitimate_words"].items(), key=lambda x: x[1], reverse=True)[:30]
    
    if not top_phishing or not top_legitimate:
        print("高频词汇数量不足，无法生成智能映射")
        return {}
    
    phishing_words = [word for word, count in top_phishing]
    legitimate_words = [word for word, count in top_legitimate]
    
    # 如果没有客户端，返回空映射
    if not client:
        print("没有提供LLM客户端，使用基础映射生成")
        return generate_basic_smart_mapping(phishing_words, legitimate_words)
    
    # 构建用于大模型的prompt
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
        
        # 尝试解析JSON响应
        import json
        start_idx = response.find('{')
        end_idx = response.rfind('}') + 1
        
        if start_idx != -1 and end_idx > start_idx:
            json_str = response[start_idx:end_idx]
            mapping = json.loads(json_str)
            
            # 验证映射的有效性
            validated_mapping = {}
            for phishing_word, legitimate_word in mapping.items():
                if (phishing_word.lower() in [w.lower() for w in phishing_words] and 
                    legitimate_word.lower() in [w.lower() for w in legitimate_words]):
                    validated_mapping[phishing_word.lower()] = legitimate_word.lower()
            
            print(f"LLM生成智能映射: {len(validated_mapping)} 个有效映射")
            return validated_mapping
        else:
            print("LLM响应格式无效，使用基础映射生成")
            return generate_basic_smart_mapping(phishing_words, legitimate_words)
            
    except Exception as e:
        print(f"LLM智能映射生成失败: {e}，使用基础映射生成")
        return generate_basic_smart_mapping(phishing_words, legitimate_words)

def generate_basic_smart_mapping(phishing_words, legitimate_words):
    """基于规则的基础智能映射生成（不依赖LLM）"""
    mapping = {}
    
    # 预定义的语义映射规则
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
        'link': ['website', 'page', 'site']
    }
    
    for phishing_word in phishing_words:
        phishing_lower = phishing_word.lower()
        
        # 首先尝试语义规则匹配
        if phishing_lower in semantic_rules:
            for candidate in semantic_rules[phishing_lower]:
                if candidate in [w.lower() for w in legitimate_words]:
                    mapping[phishing_lower] = candidate
                    break
        
        # 如果语义规则没有匹配，尝试长度和首字母匹配
        if phishing_lower not in mapping:
            word_len = len(phishing_word)
            first_letter = phishing_word[0].lower()
            
            # 寻找长度相近且首字母相同的词
            candidates = [w for w in legitimate_words 
                         if abs(len(w) - word_len) <= 2 and 
                         w[0].lower() == first_letter and 
                         w.lower() != phishing_lower]
            
            if candidates:
                mapping[phishing_lower] = candidates[0].lower()
            elif legitimate_words:
                # 最后选择一个高频的正常词汇
                mapping[phishing_lower] = legitimate_words[0].lower()
    
    print(f"基础规则生成映射: {len(mapping)} 个映射")
    return mapping

def get_smart_replacement_map(client=None):
    """获取智能替换映射表，使用缓存机制"""
    # 如果不需要更新且有缓存，直接返回缓存
    if not should_update_smart_mapping():
        if smart_replacement_cache["mapping"]:
            print(f"使用缓存的智能映射表 ({len(smart_replacement_cache['mapping'])} 个映射)")
            return smart_replacement_cache["mapping"]
    
    # 需要更新映射表
    if should_update_smart_mapping():
        print(f"触发智能映射表更新 (第 {smart_replacement_cache['update_count'] + 1} 次)")
        new_mapping = generate_smart_replacement_mapping(client)
        
        if new_mapping:
            smart_replacement_cache["mapping"] = new_mapping
            smart_replacement_cache["last_update"] = time.time()
            smart_replacement_cache["memory_hash"] = get_memory_hash()
            smart_replacement_cache["update_count"] += 1
            print(f"智能映射表更新完成: {len(new_mapping)} 个映射")
        else:
            print("智能映射表更新失败，保持原有缓存")
    
    # 返回映射表（可能是缓存的或新生成的）
    if smart_replacement_cache["mapping"]:
        return smart_replacement_cache["mapping"]
    else:
        print("智能映射表为空，返回空字典")
        return {}

if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description='Phishing Email Generation and Detection')
    parser.add_argument('--data_source', choices=['email_data', 'personal_info'],
                      default='email_data', help='Data source type')
    parser.add_argument('--personal_info_path', help='Path to personal information JSON file')
    parser.add_argument('--scenarios', nargs='+', 
                      default=['IT Security Alert', 'HR Benefits Update', 'Financial Account Verification', 'Company Policy Update', 'System Maintenance Notification'],
                      help='List of scenarios for phishing email generation (default: IT Security Alert, HR Benefits Update, etc.)')
    
    # 攻击控制参数
    parser.add_argument('--enable_lime_attack', type=lambda x: x.lower() == 'true', default=False,
                      help='Enable LIME adversarial attack against ML models (True/False)')
    parser.add_argument('--enable_llm_attack', type=lambda x: x.lower() == 'true', default=False,
                      help='Enable LLM adversarial attack (True/False)')
    parser.add_argument('--lime_models', nargs='+', default=['textcnn'],
                      help='List of models to use for LIME attack (default: textcnn)')
    parser.add_argument('--max_lime_iterations', type=int, default=5,
                      help='Maximum iterations per LIME model for adversarial attack (default: 5)')
    parser.add_argument('--use_lime_preprocessing', type=lambda x: x.lower() == 'true', default=True,
                      help='Use LIME preprocessing before LLM generation (True/False)')
    
    args = parser.parse_args()
    
    if args.data_source == 'personal_info' and not args.personal_info_path:
        parser.error("--personal_info_path is required for personal_info data source")
    
    main(
        data_source=args.data_source, 
        personal_info_path=args.personal_info_path,
        scenarios=args.scenarios,
        enable_lime_attack=args.enable_lime_attack,
        enable_llm_attack=args.enable_llm_attack,
        lime_models=args.lime_models,
        max_lime_iterations=args.max_lime_iterations,
        use_lime_preprocessing=args.use_lime_preprocessing
    )

