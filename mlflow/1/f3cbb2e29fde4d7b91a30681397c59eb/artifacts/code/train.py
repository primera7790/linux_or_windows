import logging
import os
from pathlib import Path

import numpy as np
import yaml
import nltk
from nltk.corpus import stopwords
from mlflow.tracking import MlflowClient
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier

from sklearn.model_selection import train_test_split
from sklearn.model_selection import cross_val_score
from sklearn.metrics import accuracy_score

import mlflow
import src.get_comments as get_comments
import src.preprocessing_text as preprocess


config_path = os.path.join(Path(__file__).parent, 'config/params_all.yaml')
config = yaml.safe_load(open(config_path))['train']
config_predict = yaml.safe_load(open(config_path))['predict']
os.chdir(config['dir_folder'])
SEED = config['SEED']

logging.basicConfig(filename='log/app.log', filemode='w+', format='%(asctime)s : %(levelname)s : %(message)s',
                    level=logging.DEBUG)


def get_version_model(config_name, client):
    """
    Получение последней версии модели из MLFlow
    """
    dict_push = {}
    for count, value in enumerate(client.search_model_versions(f'name="{config_name}"')):
        dict_push[count] = value
    return dict(list(dict_push.items())[0][1])['version']

def main():
    """
    Получение тематик из текста и сохранение моделиmodel_name
    """
    comments_linux = get_comments.get_all_comments(**config['comments_linux'])
    comments_windows = get_comments.get_all_comments(**config['comments_windows'])

    comments_clean_linux = preprocess.get_clean_text(comments_linux, stopwords.words(config['stopwords']))
    comments_clean_windows = preprocess.get_clean_text(comments_windows, stopwords.words(config['stopwords']))
    comments_clean = comments_clean_linux + comments_clean_windows

    vectorizer = TfidfVectorizer(**config['tf_model'])
    tfidf = vectorizer.fit(comments_clean)
    X_matrix_linux = preprocess.vectorize_text(comments_clean_linux, tfidf)
    X_matrix_windows = preprocess.vectorize_text(comments_clean_windows, tfidf)

    X = np.vstack((X_matrix_linux, X_matrix_windows))

    y_linux = [0] * X_matrix_linux.shape[0]
    y_windows = [1] * X_matrix_windows.shape[0]
    y = y_linux + y_windows

    X_train, X_test, y_train, y_test = train_test_split(X,
                                                        y,
                                                        **config['cross_val'],
                                                        random_state=SEED)

    lr_clf = LogisticRegression()      # clf: 0
    dt_clf = DecisionTreeClassifier()  # clf: 1
    rf_clf = RandomForestClassifier()  # clf: 2
    classifiers = np.array([lr_clf, dt_clf, rf_clf], dtype=object)

    clf = classifiers[config['clf']]

    # scores = cross_val_score(clf, X, y, cv=5, n_jobs=-1)
    # print(scores.mean())

    mlflow.set_tracking_uri('http://localhost:5000')
    mlflow.set_experiment(config['name_experiment'])

    with mlflow.start_run(run_name=f'{config_predict["version_vec"]}_{config["model_name"][config["clf"]]}_run'):
        clf.fit(X_train, y_train)

        mlflow.log_param('accuracy', accuracy_score(y_test, clf.predict(X_test)))
        mlflow.sklearn.log_model(tfidf,
                                 artifact_path='vector',
                                 registered_model_name=f'{config["model_vec"]}')
        mlflow.sklearn.log_model(clf,
                                 artifact_path=config['model_name'][config['clf']],
                                 registered_model_name=f'{config["model_name"][config["clf"]]}')
        mlflow.log_artifact(local_path='./train.py',
                            artifact_path='code')
        mlflow.end_run()

    client = MlflowClient()
    last_version_clf = get_version_model(config['model_name'][config['clf']], client)
    last_version_vec = get_version_model(config['model_vec'], client)

    yaml_file = yaml.safe_load(open(config_path))
    yaml_file['predict'][f'version_{config["model_name"][config["clf"]]}'] = int(last_version_clf)
    yaml_file['predict']['version_vec'] = int(last_version_vec)

    with open(config_path, 'w') as file:
        yaml.dump(yaml_file, file, encoding='UTF-8', allow_unicode=True)
    return


if __name__ == '__main__':
    nltk.download('stopwords')
    main()
