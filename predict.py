import logging
import os
from pathlib import Path

import numpy as np
import pandas as pd
import yaml
from nltk.corpus import stopwords

import mlflow
import src.get_comments as get_comments
import src.preprocessing_text as preprocess

config_path = os.path.join(Path(__file__).parent, 'config/params_all.yaml')
config = yaml.safe_load(open(config_path))['predict']
config_train = yaml.safe_load(open(config_path))['train']
SEED = config['SEED']

logging.basicConfig(filename='log/app.log', filemode='w+', format='%(asctime)s : %(levelname)s : %(message)s',
                    level=logging.DEBUG)


def main():
    """
    Получение тематик из текста и сохранение их в файл
    """
    comments_linux = get_comments.get_all_comments(**config['comments_linux'])
    comments_windows = get_comments.get_all_comments(**config['comments_windows'])

    mlflow.set_tracking_uri("http://localhost:5000")
    model_name = config_train['model_name'][config_train['clf']]
    model_uri_clf = f'models:/{model_name}/{config[f"version_{model_name}"]}'
    model_uri_tf = f'models:/{config["model_vec"]}/{config["version_vec"]}'

    model_clf = mlflow.sklearn.load_model(model_uri_clf)
    tfidf = mlflow.sklearn.load_model(model_uri_tf)

    comments_clean_linux = preprocess.get_clean_text(comments_linux, stopwords.words(config['stopwords']))
    comments_clean_windows = preprocess.get_clean_text(comments_windows, stopwords.words(config['stopwords']))
    comments_clean = comments_clean_linux + comments_clean_windows

    X_matrix = preprocess.vectorize_text(comments_clean, tfidf)

    predictions = pd.Series(model_clf.predict(X_matrix), name='predictions')
    predictions = predictions.apply(lambda x: 'linux' if x == 0 else 'windows')
    predictions.to_csv(config['name_file'])


if __name__ == "__main__":
    main()