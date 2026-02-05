import joblib
import pandas as pd
import json
import numpy as np
import os
import ntpath
import matplotlib.pyplot as plt
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
import eli5
from eli5.lime import TextExplainer
from eli5.lime.samplers import MaskingTextSampler

# 1. 加载训练好的向量化器
vectorizer_path = './models/vectorizer.joblib'
vectorizer = joblib.load(vectorizer_path)

# 2. 加载训练好的模型 - 这里主要使用LR模型，因为它更容易解释
model_path = './models/LR.joblib'
model = joblib.load(model_path)
print("成功加载LR模型")

# 3. 加载数据集
data_path = '../../dataset/responds.json'
dataset_filename = ntpath.basename(data_path)
dataset_name = os.path.splitext(dataset_filename)[0]
print(f"使用数据集: {dataset_name}")

with open(data_path, 'r', encoding='utf-8') as f:
    data = json.load(f)

# 将 JSON 数据转换为 DataFrame
test_df = pd.DataFrame(data)

# 删除包含 NaN 的行
test_df = test_df.dropna(subset=['Text'])
if 'Class' in test_df.columns:
    test_df = test_df.dropna(subset=['Class'])

# 4. 获取特征名称（单词）
feature_names = vectorizer.get_feature_names_out()

# 5. 如果是LogisticRegression模型，可以直接获取特征权重
if isinstance(model, LogisticRegression):
    # 获取系数（权重）
    coefficients = model.coef_[0]
    
    # 创建特征名称和权重的DataFrame
    features_df = pd.DataFrame({
        'feature': feature_names,
        'coefficient': coefficients
    })
    
    # 按照对类别1的贡献程度（正面影响）排序
    positive_features = features_df.sort_values('coefficient', ascending=False)
    
    # 输出对预测为类别1贡献最大的前50个词
    print("\n预测为类别1的最强正面影响词:")
    print(positive_features.head(50).to_string(index=False))
    
    # 按照对类别0的贡献程度（负面影响类别1）排序
    negative_features = features_df.sort_values('coefficient', ascending=True)
    
    # 输出对预测为类别0（即不利于预测为类别1）贡献最大的前50个词
    print("\n预测为类别0的最强词（不利于预测为类别1）:")
    print(negative_features.head(50).to_string(index=False))
    
    # 可视化顶部词汇的贡献度
    plt.figure(figsize=(12, 10))
    
    # 选择最重要的正面和负面特征
    top_positive = positive_features.head(20)
    top_negative = negative_features.head(20)
    
    # 绘制正面特征
    plt.subplot(2, 1, 1)
    plt.barh(top_positive['feature'], top_positive['coefficient'], color='green')
    plt.title('对预测为类别1贡献最大的词汇')
    plt.xlabel('系数值（权重）')
    plt.tight_layout()
    
    # 绘制负面特征
    plt.subplot(2, 1, 2)
    plt.barh(top_negative['feature'], top_negative['coefficient'], color='red')
    plt.title('对预测为类别0贡献最大的词汇')
    plt.xlabel('系数值（权重）')
    plt.tight_layout()
    
    # 保存可视化结果
    os.makedirs('./results', exist_ok=True)
    plt.savefig(f'./results/feature_importance_{dataset_name}.png', dpi=300, bbox_inches='tight')
    print(f"\n特征重要性可视化已保存至: ./results/feature_importance_{dataset_name}.png")
    
    # 保存特征重要性数据到CSV
    features_df.to_csv(f'./results/feature_importance_{dataset_name}.csv', index=False)
    print(f"特征重要性数据已保存至: ./results/feature_importance_{dataset_name}.csv")

# 6. 为每个具体的文本解释预测结果
def explain_prediction(text, true_class=None):
    """
    解释模型对特定文本的预测
    
    参数:
        text (str): 要解释的文本
        true_class: 真实类别，如果已知
    """
    # 向量化文本
    text_tfidf = vectorizer.transform([text])
    
    # 进行预测
    prediction = model.predict(text_tfidf)[0]
    
    # 获取预测概率（如果模型支持）
    if hasattr(model, 'predict_proba'):
        proba = model.predict_proba(text_tfidf)[0]
        print(f"预测类别: {prediction}, 概率: 类别0={proba[0]:.4f}, 类别1={proba[1]:.4f}")
    else:
        print(f"预测类别: {prediction}")
    
    if true_class is not None:
        print(f"真实类别: {true_class}")
    
    # 只有针对LogisticRegression模型，我们可以直接使用eli5解释
    if isinstance(model, LogisticRegression):
        # 使用eli5解释特征贡献
        explanation = eli5.explain_prediction(model, text, vec=vectorizer, 
                                              target_names=['类别0', '类别1'])
        print(eli5.format_as_text(explanation))
    
    return prediction

# 7. 分析被预测为类别1的样本，找出共同特征
def analyze_class_1_samples():
    """分析所有被预测为类别1的样本，找出它们的共同特征"""
    # 准备测试数据
    X_test = test_df['Text']
    X_test_tfidf = vectorizer.transform(X_test)
    
    # 进行预测
    predictions = model.predict(X_test_tfidf)
    
    # 获取预测为类别1的样本
    class_1_indices = np.where(predictions == 1)[0]
    class_1_samples = test_df.iloc[class_1_indices]
    
    print(f"\n总共有 {len(class_1_samples)} 个样本被预测为类别1")
    
    # 如果有真实标签，计算准确性
    if 'Class' in test_df.columns:
        true_positives = class_1_samples[class_1_samples['Class'] == 1]
        false_positives = class_1_samples[class_1_samples['Class'] == 0]
        print(f"其中真正例（真实为1）: {len(true_positives)}")
        print(f"假正例（真实为0）: {len(false_positives)}")
    
    # 随机选择10个样本进行详细解释
    if len(class_1_samples) > 10:
        samples_to_explain = class_1_samples.sample(10)
    else:
        samples_to_explain = class_1_samples
    
    print("\n随机选择的被预测为类别1的样本解释:")
    for idx, row in samples_to_explain.iterrows():
        print("\n" + "="*50)
        print(f"样本ID: {idx}")
        print(f"文本: {row['Text'][:200]}...")  # 只显示前200个字符
        
        # 解释预测
        true_class = row['Class'] if 'Class' in row else None
        explain_prediction(row['Text'], true_class)
    
    return class_1_samples

# 8. 使用LIME解释复杂模型的预测结果
def explain_with_lime(text, n_samples=5000):
    """
    使用LIME解释模型对特定文本的预测
    
    参数:
        text (str): 要解释的文本
        n_samples (int): LIME采样数量
    """
    # 创建预测函数
    def predict_proba(texts):
        vectorized_texts = vectorizer.transform(texts)
        return model.predict_proba(vectorized_texts)
    
    # 初始化LIME文本解释器
    sampler = MaskingTextSampler(
        replacement="UNK",
        max_replace=0.7,
        min_replace=0,
        token_pattern=r"(?u)\b\w\w+\b"
    )
    
    explainer = TextExplainer(
        sampler=sampler,
        n_samples=n_samples,
        random_state=42
    )
    
    # 拟合解释器
    explainer.fit(text, predict_proba)
    
    # 获取解释
    explanation = explainer.explain_prediction(target_names=['类别0', '类别1'])
    
    # 显示解释结果
    print("\nLIME解释:")
    print(eli5.format_as_text(explanation))
    
    return explanation

# 主要功能入口
if __name__ == "__main__":
    # 1. 分析全局特征重要性
    print("\n全局特征重要性分析已完成")
    
    # 2. 分析被预测为类别1的样本
    class_1_samples = analyze_class_1_samples()
    
    # 3. 如果用户想要解释特定文本，可以提供交互式界面
    print("\n" + "="*50)
    print("是否要解释特定文本? (y/n)")
    choice = input().strip().lower()
    
    if choice == 'y':
        print("请输入要解释的文本:")
        text = input()
        prediction = explain_prediction(text)
        
        if prediction == 1:
            print("\n是否使用LIME进行更详细的解释? (y/n)")
            lime_choice = input().strip().lower()
            if lime_choice == 'y':
                explain_with_lime(text) 