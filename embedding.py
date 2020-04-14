import tempfile
import os
import json
import gensim
import networkx
import node2vec
import time
import tqdm

from gensim.models import Word2Vec
from gensim.models.word2vec import LineSentence
from gensim.models.callbacks import CallbackAny2Vec
DEBUG = True

class callback(CallbackAny2Vec):
    '''Callback to print loss after each epoch.'''

    def __init__(self, all_loss):
        self.epoch = 0
        self.loss_to_be_subed = 0
        self.all_loss = all_loss

    def on_epoch_end(self, model):
        loss = model.get_latest_training_loss()
        loss_now = loss - self.loss_to_be_subed
        self.loss_to_be_subed = loss
        self.all_loss.append(loss_now)
        print('Loss after epoch {}: {}'.format(self.epoch, loss_now))
        self.epoch += 1
        
class Embedding():
    def __init__(self, graph=None):
        self.graph = graph
        self.num_nodes = 410236
        self.word_vectors = None
        self.vectors_path = None
        self.walks = None
        self.walks_path = None

    def set_paths(self, name):

        if not os.path.exists(name):
            os.makedirs(name)
        self.vectors_path = '{}/embedding.json'.format(name)
        self.walks_path = '{}/walks.json'.format(name)

    def get_embedding(self):
        return self.embedding

    def save_embedding(self):
        with open(self.vectors_path, 'w') as f:
            json.dump(self.word_vectors, f)

    def load_embedding(self):
        with open(self.vectors_path, 'r') as f:
            self.word_vectors = json.load(f)

    def save_walks(self):
        with open(self.walks_path, 'w') as f:
            json.dump(self.walks, f)

    def load_walks(self):
        with open(self.walks_path, 'r') as f:
            self.walks = json.load(f)


class Node2vec(Embedding):
    def __init__(self, graph=None, n_batch=8, path='models/node2vec'):
        Embedding.__init__(self, graph)
        self.graph = graph
        self.walks = None
        self.embedding = None
        self.path = path
        self.set_paths(path)
        self.n_batch = n_batch

    def compute_walks(self, n_batch=0, walk_length=10, num_walks=80, p=1, q=1, workers=1):

        if n_batch == 0:
            self.walks = node2vec.Node2Vec(
                self.graph, walk_length=walk_length, num_walks=num_walks, p=p, q=q, workers=workers).walks
#             import pdb; pdb.set_trace()
            self.save_walks()
            self.walks = None
        else:
            # not very efficient for computation but requieres less memory
            for batch_index in range(n_batch):
                self.set_paths(self.path + '_batch_{}'.format(batch_index))
                self.compute_walks(walk_length=walk_length,
                                   num_walks=num_walks//n_batch, p=p, q=q)
        # need to be done at the end otherwise intermediate call will set n_batch to 0
        self.n_batch = n_batch
        self.set_paths(self.path)

    def load_walks(self, batch_index=0):
        if self.n_batch == 0:
            super().load_walks()
        else:
            self.set_paths(self.path + '_batch_{}'.format(batch_index))
            super().load_walks()
            self.set_paths(self.path)

    def fit(self, window=10, embedding_size=128, workers=8, iter=50):
        kwargs = {}
        kwargs["min_count"] = kwargs.get("min_count", 0)
        kwargs["size"] = embedding_size
        kwargs["sg"] = 1  # skip gram
        kwargs["workers"] = workers
        kwargs["window"] = window
        kwargs["iter"] = iter
        kwargs["compute_loss"] = True
        kwargs["min_alpha"] = 1e-2
        call_back = callback(all_loss=[])
        kwargs['callbacks'] = [call_back]
        if self.n_batch == 0:
            self.load_walks()
            kwargs["sentences"] = self.walks
            self.word_vectors = gensim.models.Word2Vec(**kwargs).wv

        else:
            for batch_index in tqdm.tqdm(range(self.n_batch)):
                self.load_walks(batch_index=batch_index)
                kwargs["sentences"] = self.walks
                if batch_index == 0:
                    print("Learning embedding vectors...")
                    model = Word2Vec(**kwargs)
#                     print("Learning embedding vectors done!")
#                     model = gensim.models.Word2Vec(
#                         self.walks, size=embedding_size,
#                         min_count=0, sg=1, workers=4,
#                         window=window)
                    import matplotlib.pyplot as plt
                    fig = plt.figure()
                    plt.plot(call_back.all_loss)
                    fig.savefig('node2vec_loss.png')
                else:
                    model.train(self.walks, epochs=model.iter,
                                total_words=self.num_nodes)
            self.word_vectors = model.wv
        self.word_vectors = dict(
            zip(self.word_vectors.index2word, self.word_vectors.vectors.tolist()))


if __name__ == "__main__":
    from dataset import Dataset
    DEBUG = False
    if DEBUG:
        embedding = Node2vec(n_batch=8)
        # embedding = Node2vec(Dataset().residual_network)
        # embedding.compute_walks(n_batch=8)
        # embedding.load_walks()

    else:
        dataset = Dataset()
        embedding = Node2vec(Dataset.residual_network, n_batch=1)
        embedding.compute_walks(walk_length=10, num_walks=80,
                                workers=8, n_batch=1)
        embedding.save_walks()

    embedding.fit()
    embedding.save_embedding()