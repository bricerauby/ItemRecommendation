import json 
import argparse

import numpy as np
from sklearn.linear_model import LogisticRegression, SGDClassifier
from sklearn.metrics import roc_auc_score, log_loss
import warnings; warnings.simplefilter('ignore')
from dataset import Dataset

DEBUG = False

def train_by_batch(clf, dataset, nb_epochs) :
    index_train = np.arange(len(dataset.x_train))
    y_train = dataset.y_train
    loss = []
    for epoch in range(nb_epochs) :
        for i in tqdm.tqdm(range((len(index_train) // batch_size_train) + 1)):
            b_indexes = index_train[i * batch_size_train:(i + 1) * batch_size_train]
            y_batch = y_train[b_indexes]
            x_batch = dataset.embed_edges(b_indexes, test = False)
            clf.partial_fit(x_batch, y_batch, classes = np.unique(y_train)) 
            y_pred =  clf.predict_proba(x_batch)[:, 0]
            loss.append(log_loss(y_batch, y_pred))
    return clf, loss

def test_by_batch(clf, dataset):
    index_test = np.arange(len(dataset.x_test))
    y_pred = np.zeros(len(index_test))
    auc = []
    for i in tqdm.tqdm(range((len(index_test)// batch_size_test) + 1)):
        b_indexes = index_test[i * batch_size_test:(i + 1) * batch_size_test]
        x_batch = dataset.embed_edges(b_indexes, train=False)
        test_pred_prob = clf.predict_proba(x_batch)[:, 0]
        y_pred[b_indexes] = test_pred_prob
    return y_pred

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

    dataset.embed_network(**config['embedding'])

# we do the classification
# clf = LogisticRegression(random_state=SEED)

# clf.fit(dataset.x_train, dataset.y_train)
# test_pred_prob = clf.predict_proba(dataset.x_test)
# print('auc score on test set', roc_auc_score(dataset.y_test, test_pred_prob))

    clf = SGDClassifier(loss ='log')
    batch_size_train = 10000
    batch_size_test = 10000


    clf, _ = train_by_batch(clf, dataset, nb_epochs=10)
    y_pred = test_by_batch(clf, dataset)
    print("Auc score on test set", roc_auc_score(dataset.y_test, y_pred))
