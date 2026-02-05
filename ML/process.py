# -*- coding: utf-8 -*-

import time
import numpy as np
import pandas as pd
import glob
import os
import joblib
import fasttext
import argparse
import nltk
import jieba
from typing import *
from urllib.parse import urlparse
from nltk.tokenize import RegexpTokenizer
import tldextract

from concurrent.futures import ThreadPoolExecutor
from sklearn.pipeline import Pipeline

from sklearn.preprocessing import OneHotEncoder, StandardScaler, MinMaxScaler
from sklearn.compose import ColumnTransformer
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.pipeline import FeatureUnion, Pipeline
from sklearn.base import BaseEstimator, TransformerMixin


from nltk.corpus import stopwords
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.feature_extraction.text import CountVectorizer

def parse_url(url: str) -> Optional[Dict[str, str]]:
    try:
        no_scheme = not url.startswith('https://') and not url.startswith('http://')
        if no_scheme:
            parsed_url = urlparse(f"http://{url}")
            return {
                "scheme": None, # not established a value for this
                "netloc": parsed_url.netloc,
                "path": parsed_url.path,
                "params": parsed_url.params,
                "query": parsed_url.query,
                "fragment": parsed_url.fragment,
            }
        else:
            parsed_url = urlparse(url)
            return {
                "scheme": parsed_url.scheme,
                "netloc": parsed_url.netloc,
                "path": parsed_url.path,
                "params": parsed_url.params,
                "query": parsed_url.query,
                "fragment": parsed_url.fragment,
            }
    except:
        return None
def get_num_subdomains(netloc: str) -> int:
    subdomain = tldextract.extract(netloc).subdomain
    if subdomain == "":
        return 0
    return subdomain.count('.') + 1

def tokenize_domain(netloc: str, tokenizer) -> str:
    split_domain = tldextract.extract(netloc)
    no_tld = str(split_domain.subdomain +'.'+ split_domain.domain)
    return " ".join(map(str,tokenizer.tokenize(no_tld)))


def process_email_csv_url(url_data):
    tokenizer = RegexpTokenizer(r'[A-Za-z]+')
    url_data["parsed_url"] = url_data.url.apply(parse_url)
    url_data = pd.concat([url_data.drop(['parsed_url'], axis=1), url_data['parsed_url'].apply(pd.Series)], axis=1)
    url_data = url_data[~url_data.netloc.isnull()]
    url_data["length"] = url_data.url.str.len()
    url_data["tld"] = url_data.netloc.apply(lambda nl: tldextract.extract(nl).suffix)
    url_data['tld'] = url_data['tld'].replace('', 'None')
    url_data["is_ip"] = url_data.netloc.str.fullmatch(r"\d+\.\d+\.\d+\.\d+")
    url_data['domain_hyphens'] = url_data.netloc.str.count('-')
    url_data['domain_underscores'] = url_data.netloc.str.count('_')
    url_data['path_hyphens'] = url_data.path.str.count('-')
    url_data['path_underscores'] = url_data.path.str.count('_')
    url_data['slashes'] = url_data.path.str.count('/')
    url_data['full_stops'] = url_data.path.str.count('.')
    url_data['num_subdomains'] = url_data['netloc'].apply(lambda net: get_num_subdomains(net))
    url_data['domain_tokens'] = url_data['netloc'].apply(lambda net: tokenize_domain(net, tokenizer))
    url_data['path_tokens'] = url_data['path'].apply(lambda path: " ".join(map(str, tokenizer.tokenize(path))))

    url_data.drop('url', axis=1, inplace=True)
    url_data.drop('scheme', axis=1, inplace=True)
    url_data.drop('netloc', axis=1, inplace=True)
    url_data.drop('path', axis=1, inplace=True)
    url_data.drop('params', axis=1, inplace=True)
    url_data.drop('query', axis=1, inplace=True)
    url_data.drop('fragment', axis=1, inplace=True)
    # url_data.to_csv('../data/url/PhiUSIIL_Phishing_URL_Dataset_2024_clean.csv', index=False)
    return url_data

class Converter(BaseEstimator, TransformerMixin):
    def fit(self, x, y=None):
        return self

    def transform(self, data_frame):
        return data_frame.values.ravel()

def results(X_val, model: BaseEstimator) -> None:
    t0 = time.perf_counter()
    preds = model.predict_proba(X_val)
    print('detect time: ', time.perf_counter() - t0, ', test total: ', len(X_val))
    return pd.DataFrame(preds, columns=['url_n', 'url_p', 'url_s'])

def detect_email_url(data_df):
    url_data = process_email_csv_url(data_df)
    numeric_features = ['length', 'domain_hyphens', 'domain_underscores', 'path_hyphens', 'path_underscores', 'slashes',
                        'full_stops', 'num_subdomains']
    numeric_transformer = Pipeline(steps=[('scaler', MinMaxScaler())])
    categorical_features = ['tld', 'is_ip']
    categorical_transformer = Pipeline(steps=[('onehot', OneHotEncoder(handle_unknown='ignore'))])
    vectorizer_features = ['domain_tokens', 'path_tokens']
    vectorizer_transformer = Pipeline(steps=[('con', Converter()), ('tf', TfidfVectorizer())])

    preprocessor = ColumnTransformer(transformers=[('num', numeric_transformer, numeric_features),
                                                   ('cat', categorical_transformer, categorical_features),
                                                   ('domvec', vectorizer_transformer, ['domain_tokens']),
                                                   ('pathvec', vectorizer_transformer, ['path_tokens'])])

    pickle_svc = './pickle/svc_model_url.pkl'
    svc_val = joblib.load(pickle_svc)
    res_url_df = results(url_data, svc_val)
    return res_url_df

def detect_email_body(data_df):
    loaded_vectorizer = joblib.load('./model/vectorizer.joblib')
    t0 = time.perf_counter()
    val_X = loaded_vectorizer.transform(data_df['Text'])
    process_data_time = round(time.perf_counter() - t0, 3)
    print('process_testdata_time:', process_data_time)

    data_df.head()
    data_df.info()
    data_df.describe()

    pickle_mlp = './pickle/lg_model_v1.pkl'
    t1 = time.perf_counter()
    mlp_test = joblib.load(pickle_mlp)
    preds = mlp_test.predict_proba(val_X)
    # test_auc = classification_report(val_Y, mlp_test.predict(val_X))
    test_time = round(time.perf_counter() - t1, 3)
    print("lg time:", test_time)
    return pd.DataFrame(preds, columns=['body_n', 'body_p', 'body_s'])

def combined_probabilities(row, weights=[0.4, 0.4]):
    # 计算加权平均概率
    combined_probs = (
        weights[0] * np.array([row[f'body_{i}'] if row['Text'] != 'NaN' else 0 for i in ['n','p','s']]) +
        weights[1] * np.array([row[f'url_{i}'] if row['url'] != 'NaN' else 0 for i in ['n','p','s']])
    )
    return combined_probs

def classify_email(row, threshold=0.5):
    probs = combined_probabilities(row)
    predicted_class = np.argmax(probs)
    return predicted_class

def detect_email_csv(file_path,file_name):
    data_df_ori = pd.read_csv(file_path, encoding='utf-8')
    data_df_ori = data_df_ori.fillna('NaN')
    res_data_df = data_df_ori[['Text', 'URL', 'Class']]
    res_data_df.rename(columns={'URL': 'url'}, inplace=True)

    with ThreadPoolExecutor() as executor:  #max_workers=10
        future_body = executor.submit(detect_email_body, res_data_df[['Text']])
        future_url = executor.submit(detect_email_url, res_data_df[['url']])

        res_body_df = future_body.result()
        res_url_df = future_url.result()

    for i in ['n','p','s']:
        res_data_df[f'body_{i}'] = res_body_df[f'body_{i}']
        res_data_df[f'url_{i}'] = res_url_df[f'url_{i}']

    res_data_df['combined_probs'] = res_data_df.apply(combined_probabilities, axis=1)
    res_data_df['predicted_class'] = res_data_df.apply(classify_email, axis=1)
    print(classification_report(res_data_df['Class'], res_data_df['predicted_class']))
    res_data_df.to_csv(f'./res/res_{file_name}', index=False)

def main(input_path):
    for root, dirs, files in os.walk(input_path):
        for file_name in files:
            if file_name.endswith('.csv'):
                # print(file_name,root)
                file_path = os.path.join(root, file_name)
                detect_email_csv(file_path, file_name)



if __name__ == "__main__":
    # parser = argparse.ArgumentParser(description='Process a test CSV file path for phishing email detection.')
    # parser.add_argument('input_path', type=str, help='The input CSV file path')
    #
    # args = parser.parse_args()
    input_path = './test_data/'
    main(input_path)
