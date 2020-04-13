import json 

import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from dataset import Dataset

import argparse



DEBUG = False


if __name__ == "__main__":



    parser = argparse.ArgumentParser()
    parser.add_argument('--config_path', type=str, required = True, help='path to the config_file')
    args = parser.parse_args()
    config_path = args.config_path

    with open(config_path, 'r') as f: 
        config = json.load(f)
    SEED = config['seed']
    # we load the dataset and perform edge extraction to get the train test split
    dataset = Dataset(**config["dataset"])



    # we perform the embedding of the nodes in the network using the residual network 
    dataset.embed_network(**config['embedding'])

    if config["classifier"]["n_batch"] <= 1 :
        dataset.embed_edges()


    if DEBUG : 
        n_train = len(dataset.x_train)
        dataset.x_train = np.random.random((n_train,128))
        n_test = len(dataset.x_test)
        x_test_edges = np.random.random((n_test,128))

    # we do the classification
    clf = LogisticRegression(random_state=SEED)

    clf.fit(dataset.x_train, dataset.y_train)
    test_pred_prob = clf.predict_proba(dataset.x_test)
    print('auc score on test set', roc_auc_score(dataset.y_test, test_pred_prob))
