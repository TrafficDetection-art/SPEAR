import joblib
import pandas as pd
import json
import numpy as np
import os
import ntpath
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, 
    f1_score, roc_auc_score, confusion_matrix
)

# 1. 加载训练好的向量化器
vectorizer_path = './models/vectorizer.joblib'
vectorizer = joblib.load(vectorizer_path)

# 2. 加载训练好的模型
model_names = ["LR", "RFC", "NB", "SVC"]
models = {}

for model_name in model_names:
    model_path = f'./models/{model_name}.joblib'
    try:
        models[model_name] = joblib.load(model_path)
        print(f"成功加载{model_name}模型")
    except FileNotFoundError:
        print(f"未找到{model_name}模型文件")

# 3. 加载测试数据
# 可以使用与训练相同的数据集，或专门的测试数据集
data_path = '../email_ccs/outputs/agent-gpt-4o/20250707_213314/iteration_0_original_emails.json'  # 根据实际情况修改路径

# 提取数据集名称，用于输出文件命名
dataset_filename = ntpath.basename(data_path)
dataset_name = os.path.splitext(dataset_filename)[0]  # 去掉文件扩展名
print(f"使用数据集: {dataset_name}")

with open(data_path, 'r', encoding='utf-8') as f:
    data = json.load(f)

# 将 JSON 数据转换为 DataFrame
test_df = pd.DataFrame(data)

# 删除包含 Text 为 NaN 的行
initial_rows = len(test_df)
test_df = test_df.dropna(subset=['Text'])
dropped_text_rows = initial_rows - len(test_df)
if dropped_text_rows > 0:
    print(f"警告: 删除了 {dropped_text_rows} 行因为 'Text' 列包含 NaN 值")

# 4. 准备测试数据
X_test = test_df['Text']
X_test_tfidf = vectorizer.transform(X_test)

# 5. 如果测试数据有标签，可以评估模型性能
if 'Class' in test_df.columns:
    # 删除 Class 为 NaN 的行
    rows_before = len(test_df)
    test_df = test_df.dropna(subset=['Class'])
    rows_after = len(test_df)
    
    if rows_before > rows_after:
        print(f"警告: 删除了 {rows_before - rows_after} 行因为 'Class' 列包含 NaN 值")
        # 因为删除了行，所以需要重新准备测试数据
        X_test = test_df['Text']
        X_test_tfidf = vectorizer.transform(X_test)
    
    y_test = test_df['Class']
    
    print("模型性能评估：")
    print("-" * 50)
    
    # 获取所有唯一的类别
    unique_classes = sorted(y_test.unique())
    
    # 创建结果存储容器
    results_data = []
    
    for model_name, model in models.items():
        print(f"\n{model_name}模型评估结果:")
        
        y_pred = model.predict(X_test_tfidf)
        
        # 计算整体评分
        accuracy = accuracy_score(y_test, y_pred)
        print(f'整体准确率: {accuracy:.4f}')
        
        # 计算每个类别的评分
        precision_per_class = precision_score(y_test, y_pred, labels=unique_classes, average=None, zero_division=0)
        recall_per_class = recall_score(y_test, y_pred, labels=unique_classes, average=None, zero_division=0)
        f1_per_class = f1_score(y_test, y_pred, labels=unique_classes, average=None, zero_division=0)
        
        # 二分类评估指标
        binary_precision = precision_score(y_test, y_pred, average='binary', zero_division=0)
        binary_recall = recall_score(y_test, y_pred, average='binary', zero_division=0)
        binary_f1 = f1_score(y_test, y_pred, average='binary', zero_division=0)
        
        # 计算对抗成功率 = 1 - 召回率
        adversarial_success_rate = 1 - binary_recall
        
        print(f'二分类精确率: {binary_precision:.4f}, 召回率: {binary_recall:.4f}, F1值: {binary_f1:.4f}')
        print(f'对抗成功率: {adversarial_success_rate:.4f}')
        
        # AUC计算
        auc = None
        if hasattr(model, 'predict_proba'):
            try:
                y_score = model.predict_proba(X_test_tfidf)[:, 1]  # 假设第二列是正类的概率
                auc = roc_auc_score(y_test, y_score)
                print(f'ROC曲线下面积(AUC): {auc:.4f}')
            except:
                print('无法计算AUC')
        
        # 将主要评估指标添加到结果中
        result_row = {
            'Model': model_name,
            'Dataset': dataset_name,
            'Accuracy': accuracy,
            'Binary_Precision': binary_precision,
            'Binary_Recall': binary_recall,
            'Binary_F1': binary_f1,
            'Adversarial_Success_Rate': adversarial_success_rate,
            'AUC': auc
        }
        
        # 添加每个类别的评分
        for i, class_label in enumerate(unique_classes):
            result_row[f'Precision_{class_label}'] = precision_per_class[i]
            result_row[f'Recall_{class_label}'] = recall_per_class[i]
            result_row[f'F1_{class_label}'] = f1_per_class[i]
        
        results_data.append(result_row)
        
        # 计算混淆矩阵
        cm = confusion_matrix(y_test, y_pred, labels=unique_classes)
        
        print("\n混淆矩阵:")
        # 打印列标签
        print(f'{"真实/预测":<10}', end='')
        for class_label in unique_classes:
            print(f'{class_label:<8}', end='')
        print()
        
        # 打印混淆矩阵的每一行
        for i, class_label in enumerate(unique_classes):
            print(f'{class_label:<10}', end='')
            for j in range(len(unique_classes)):
                print(f'{cm[i, j]:<8d}', end='')
            print()
        
        # 按数据集类型评估
        if 'type' in test_df.columns:
            print("\n按数据集类型(type)的评估结果:")
            print("-" * 50)
            
            # 获取所有唯一的数据集类型
            unique_types = sorted(test_df['type'].unique())
            
            for data_type in unique_types:
                # 获取当前类型的数据
                # 修复: 使用布尔索引而不是位置索引来避免IndexError
                test_df_type = test_df[test_df['type'] == data_type]
                # 只选择那些行索引存在于test_df中的部分
                X_test_type = test_df_type['Text']
                X_test_type_tfidf = vectorizer.transform(X_test_type)
                y_test_type = test_df_type['Class']
                
                # 对该类型的数据进行预测
                y_pred_type = model.predict(X_test_type_tfidf)
                
                # 计算该类型数据的样本数
                type_count = len(test_df_type)
                
                print(f"\n类型: {data_type} (样本数: {type_count})")
                
                # 计算该类型的评分
                type_accuracy = accuracy_score(y_test_type, y_pred_type)
                
                # 二分类评估指标
                type_binary_precision = precision_score(y_test_type, y_pred_type, average='binary', zero_division=0)
                type_binary_recall = recall_score(y_test_type, y_pred_type, average='binary', zero_division=0)
                type_binary_f1 = f1_score(y_test_type, y_pred_type, average='binary', zero_division=0)
                
                # 计算对抗成功率
                type_adversarial_success_rate = 1 - type_binary_recall
                
                print(f'二分类精确率: {type_binary_precision:.4f}, 召回率: {type_binary_recall:.4f}, F1值: {type_binary_f1:.4f}')
                print(f'对抗成功率: {type_adversarial_success_rate:.4f}')
                
                # 将数据集类型评估结果添加到结果中
                type_result_row = {
                    'Model': model_name,
                    'Dataset': dataset_name,
                    'Data_Type': data_type,
                    'Sample_Count': type_count,
                    'Accuracy': type_accuracy,
                    'Binary_Precision': type_binary_precision,
                    'Binary_Recall': type_binary_recall,
                    'Binary_F1': type_binary_f1,
                    'Adversarial_Success_Rate': type_adversarial_success_rate
                }
                results_data.append(type_result_row)
                
                # 计算该类型的混淆矩阵
                type_cm = confusion_matrix(y_test_type, y_pred_type, labels=unique_classes)
                
                print("\n该类型的混淆矩阵:")
                # 打印列标签
                print(f'{"真实/预测":<10}', end='')
                for class_label in unique_classes:
                    print(f'{class_label:<8}', end='')
                print()
                
                # 打印混淆矩阵的每一行
                for i, class_label in enumerate(unique_classes):
                    print(f'{class_label:<10}', end='')
                    for j in range(len(unique_classes)):
                        print(f'{type_cm[i, j]:<8d}', end='')
                    print()
        
        print("-" * 50)
    
    # 创建结果数据框并保存到CSV
    results_df = pd.DataFrame(results_data)
    
    # 确保结果目录存在
    os.makedirs('./results', exist_ok=True)
    
    # 保存评估结果，文件名中包含数据集名称
    results_file = f'./results/model_evaluation_results_{dataset_name}.csv'
    results_df.to_csv(results_file, index=False)
    print(f"\n评估结果已保存至: {results_file}")
else:
    # 如果没有标签，只进行预测
    for model_name, model in models.items():
        predictions = model.predict(X_test_tfidf)
        test_df[f'prediction_{model_name}'] = predictions
        print(f'{model_name}预测完成')
    
    # 保存预测结果，文件名中包含数据集名称
    prediction_file = f'./results/prediction_results_{dataset_name}.csv'
    test_df.to_csv(prediction_file, index=False)
    print(f"预测结果已保存至: {prediction_file}")

# 6. 单个文本预测函数
def predict_text(text, model_name="LR"):
    """
    对单个文本进行分类预测
    
    参数:
        text (str): 要预测的文本
        model_name (str): 要使用的模型名称，默认为LR
    
    返回:
        预测的类别
    """
    if model_name not in models:
        raise ValueError(f"模型 {model_name} 未找到。可用模型: {list(models.keys())}")
    
    # 向量化文本
    text_tfidf = vectorizer.transform([text])
    
    # 预测
    prediction = models[model_name].predict(text_tfidf)[0]
    
    return prediction

# 7. 多模型投票预测函数
def predict_with_voting(text):
    """
    使用所有模型进行投票预测
    
    参数:
        text (str): 要预测的文本
    
    返回:
        投票后的预测类别
    """
    predictions = []
    for model_name, model in models.items():
        text_tfidf = vectorizer.transform([text])
        prediction = model.predict(text_tfidf)[0]
        predictions.append(prediction)
    
    # 使用多数投票确定最终预测结果
    unique_preds, counts = np.unique(predictions, return_counts=True)
    final_prediction = unique_preds[np.argmax(counts)]
    
    return final_prediction, predictions
