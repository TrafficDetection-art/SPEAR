from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.naive_bayes import MultinomialNB

from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

import joblib
import pandas as pd
import json


# 1. 加载 JSON 数据
data_path = '../../dataset/filtered_train_data.json'
with open(data_path, 'r', encoding='utf-8') as f:
    data = json.load(f)

# 将 JSON 数据转换为 DataFrame
data_df = pd.DataFrame(data)

# 删除包含 NaN 的行
data_df = data_df.dropna(subset=['Text'])  # 假设文本字段名为 'Text'

# 2. 切分数据为训练集和测试集
X = data_df['Text']  # 文本特征
y = data_df['Class']  # 假设标签字段名为 'Class'

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# 3. 创建并训练 TfidfVectorizer
vectorizer = TfidfVectorizer(max_features=5000)  # 设置 max_features 可以限制词汇表大小
X_train_tfidf = vectorizer.fit_transform(X_train)  # 先 fit 再 transform
X_test_tfidf = vectorizer.transform(X_test)  # 对测试集只需要 transform

# 4. 训练一个机器学习模型（Logistic Regression）
model_dict = {
    "LR": LogisticRegression(),
    "RFC": RandomForestClassifier(),
    "NB": MultinomialNB(),
    "SVC": SVC(kernel='linear'),
}
for model_name in model_dict:
    model = model_dict[model_name]
    model.fit(X_train_tfidf, y_train)

    # 5. 评估模型在测试集上的性能
    y_pred = model.predict(X_test_tfidf)
    accuracy = accuracy_score(y_test, y_pred)
    precision = precision_score(y_test, y_pred)
    recall = recall_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred)
    print(f'{model_name}在测试集上的准确率: {accuracy:.4f}')
    print(f'{model_name}在测试集上的精确率: {precision:.4f}')
    print(f'{model_name}在测试集上的召回率: {recall:.4f}')
    print(f'{model_name}在测试集上的F1值: {f1:.4f}')

    # 6. 保存训练好的模型和vectorizer
    model_path = f'./models/{model_name}.joblib'  # 注意：路径改为 models 目录
    joblib.dump(model, model_path)

vectorizer_path = './models/vectorizer.joblib'  # 注意：路径改为 models 目录
joblib.dump(vectorizer, vectorizer_path)

print(f"模型已保存至: {model_path}")
print(f"向量化器已保存至: {vectorizer_path}")
