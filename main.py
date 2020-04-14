import json 
import argparse

import tqdm
import numpy as np
from sklearn.linear_model import LogisticRegression, SGDClassifier
from sklearn.metrics import roc_auc_score, log_loss
import warnings; warnings.simplefilter('ignore')
from dataset import Dataset

if __name__ == "__main__":
    
    parser = argparse.ArgumentParser()
    parser.add_argument('--config_path', type=str, required = True, help='path to the config_file')
    args = parser.parse_args()
    config_path = args.config_path

    with open(config_path, 'r') as f: 
        config = json.load(f)
    SEED = config['seed']
    # we load the dataset and perform edge extraction to get the train test split
    print('loading dataset ...')
    dataset = Dataset(**config["dataset"])
    print('embedding network ...')
    dataset.embed_network(**config['embedding'])

    # we do the classification
    clf = LogisticRegression(random_state=SEED)

    dataset.embed_edges()
    clf.fit(dataset.x_train, dataset.y_train)
    test_pred_prob = clf.predict_proba(dataset.x_test)[:, 1]
    print('auc score on test set', roc_auc_score(dataset.y_test, test_pred_prob))