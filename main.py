import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from dataset import Dataset
from graph import Graph


DEBUG = True
SEED = 0
# we load the dataset and perform edge extraction to get the train test split
dataset = Dataset()
x_train, y_train, x_test, y_test = dataset.get_split()

# we perform the embedding of the nodes in the network using the residual network 
graph = Graph(dataset.residual_network)

# we get the embeddings for both set
if DEBUG : 
    n_train = len(x_train)
    x_train_edges = np.random.random((n_train,128))
    n_test = len(x_test)
    x_test_edges = np.random.random((n_test,128))

else: 
    x_train_edges = graph.get_embeddings(x_train)
    x_test_edges = graph.get_embeddings(x_test)

# we do the classification
clf = LogisticRegression(random_state=SEED)

clf.fit(x_train_edges, y_train)
test_pred_prob = clf.predict_proba(x_test_edges)
print('auc score on test set', roc_auc_score(y_test, test_pred_prob))