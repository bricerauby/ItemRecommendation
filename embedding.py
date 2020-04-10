import tempfile
import os
import json
import gensim
import networkx
import node2vec
import time 

DEBUG = True

class Embedding():
    def __init__(self, graph=None):
        self.graph = graph
        self.word_vectors = None
        self.vectors_path = None
        self.walks = None
        self.walks_path = None

    def set_paths(self, name):
        self.vectors_path = '{}_embedding.json'.format(name)
        self.walks_path = '{}_walks.json'.format(name)

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
    def __init__(self, graph=None):
        Embedding.__init__(self,graph)
        self.graph = graph
        self.walks = None
        self.embedding = None
        self.set_paths('node2vec')

    def compute_walks(self, n_batch=0, walk_length=10, num_walks=80, p=1, q=1):

        if n_batch == 0 : 
            self.walks = node2vec.Node2Vec(self.graph, walk_length=walk_length, num_walks=num_walks, p=p, q=q, workers=1).walks
            self.save_walks()
            self.walks = None
        else: 
            #not very efficient for computation but requieres less memory
            for batch_index in range(n_batch):
                self.walks_path = 'node2vec_embedding_batch_{}.json'.format(batch_index)
                self.compute_walks(walk_length=walk_length, num_walks=num_walks//n_batch, p=p, q=q)
        # need to be done at the end
        self.n_batch=n_batch
    
    def load_walks(self):
        if self.n_batch==0: 
            super().load_walks()
        else: 
            walks=[]
            for batch_index in range(self.n_batch):
                self.walks_path = 'node2vec_embedding_batch_{}.json'.format(batch_index)
                super().load_walks()
                walks += self.walks
            self.walks = walks


    def fit(self, window=10, embedding_size=128):
        if self.walks is None: 
            raise ValueError('walks is none') 
        self.word_vectors = gensim.models.Word2Vec(self.walks, size=embedding_size, min_count=0, sg=1, workers=4, window=window).wv
        self.word_vectors = dict(zip(self.word_vectors.index2word, self.word_vectors.vectors.tolist()))


    


if __name__ == "__main__":
    from dataset import Dataset
    if DEBUG:
        embedding = Node2vec(Dataset().residual_network)
        embedding.compute_walks(n_batch=8)
        embedding.load_walks()
        
    else:
        dataset = Dataset()
        embedding = Node2vec(Dataset().residual_network)
        embedding.compute_walks()
        embedding.save_walks()
    embedding.fit()
    embedding.save_embedding()
    
