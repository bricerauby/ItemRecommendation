import tempfile
import os
import json
import gensim
import networkx
import node2vec
import time
import tqdm
DEBUG = True


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

    def compute_walks(self, n_batch=0, walk_length=10, num_walks=80, p=1, q=1):

        if n_batch == 0:
            self.walks = node2vec.Node2Vec(
                self.graph, walk_length=walk_length, num_walks=num_walks, p=p, q=q, workers=1).walks
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

    def fit(self, window=10, embedding_size=128):
        if self.n_batch == 0:
            self.load_walks()
            self.word_vectors = gensim.models.Word2Vec(
                self.walks, size=embedding_size, min_count=0, sg=1, workers=4, window=window).wv

        else:
            for batch_index in tqdm.tqdm(range(self.n_batch)):
                self.load_walks(batch_index=batch_index)
                if batch_index == 0:
                    model = gensim.models.Word2Vec(
                        self.walks, size=embedding_size, min_count=0, sg=1, workers=4, window=window)
                else:
                    model.train(self.walks, epochs=model.iter,
                                total_words=self.num_nodes)
            self.word_vectors = model.wv
        self.word_vectors = dict(
            zip(self.word_vectors.index2word, self.word_vectors.vectors.tolist()))


if __name__ == "__main__":
    from dataset import Dataset
    if DEBUG:
        embedding = Node2vec(n_batch=8)
        # embedding = Node2vec(Dataset().residual_network)
        # embedding.compute_walks(n_batch=8)
        # embedding.load_walks()

    else:
        dataset = Dataset()
        embedding = Node2vec(Dataset().residual_network)
        embedding.compute_walks()
        embedding.save_walks()

    embedding.fit()
    embedding.save_embedding()