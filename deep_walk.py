import itertools
import math
import random
import numpy as np
from joblib import Parallel, delayed
from gensim.models import Word2Vec
from tqdm import trange
from dataset import Dataset
import tempfile
import networkx

def partition_num(num, workers):
    if num % workers == 0:
        return [num//workers]*workers
    else:
        return [num//workers]*workers + [num % workers]
    
class RandomWalker:
    def __init__(self, G):
        self.G = G

    def deepwalk_walk(self, walk_length, 
                      start_node):

        walk = [start_node]

        while len(walk) < walk_length:
            cur = walk[-1]
            cur_nbrs = list(self.G.neighbors(cur))
            if len(cur_nbrs) > 0:
                walk.append(random.choice(cur_nbrs))
            else:
                break
        return walk

    def simulate_walks(self, num_walks, walk_length, 
                       workers=1, verbose=0):

        G = self.G

        nodes = list(G.nodes())

        results = Parallel(n_jobs=workers, verbose=verbose, )(
            delayed(self._simulate_walks)(nodes, num, walk_length) for num in
            partition_num(num_walks, workers))

        walks = list(itertools.chain(*results))

        return walks

    def _simulate_walks(self, nodes, num_walks, walk_length,):
        walks = []
        for _ in range(num_walks):
            random.shuffle(nodes)
            for v in nodes:
                walks.append(self.deepwalk_walk(
                        walk_length=walk_length, start_node=v))
        return walks
    
class DeepWalk:
    def __init__(self, graph, walk_length, 
                 num_walks, workers=1):

        self.graph = graph
        self.w2v_model = None
        self._embeddings = {}
        self.num_walks = num_walks
        self.walk_length = walk_length
        self.workers = workers
        self.walker = RandomWalker(
            graph)
    def generate_walks(self, verbose=10) :
        self.sentences = self.walker.simulate_walks(
            num_walks=self.num_walks, walk_length=self.walk_length, 
            workers=self.workers, verbose=verbose)

    def train(self, embed_size=128, window_size=5, 
              workers=4, iter=5, **kwargs):

        kwargs["sentences"] = self.sentences
        kwargs["min_count"] = kwargs.get("min_count", 0)
        kwargs["size"] = embed_size
        kwargs["sg"] = 1  # skip gram
        kwargs["hs"] = 1  # deepwalk use Hierarchical Softmax
        kwargs["workers"] = workers
        kwargs["window"] = window_size
        kwargs["iter"] = iter

        print("Learning embedding vectors...")
        model = Word2Vec(**kwargs)
        print("Learning embedding vectors done!")

        self.w2v_model = model
        return model

    def get_embeddings(self, edges):
        result = np.zeros(len(edges), 128)
        for i, (elt1, elt2) in enumerate(edges) :
            emb_1 = self.w2v_model.wv[elt1]
            emb_2 = self.w2v_model.wv[elt2]
            emb = np.multiply(emb_1, emb_2)
            result[i] = emb
            
        return result
    
if __name__ == '__main__' :
    graph_nx = Dataset().residual_network
    deep_walk = DeepWalk(graph=graph_nx, walk_length=10, 
                         num_walks=80, workers=4)
    deep_walk.generate_walks()
    deep_walk.train(iter = 5, workers=4)
    deep_walk.w2v_model.wv.save_word2vec_format('random_walk')
    #edges_emb = deep_walk.get_embeddings(graph_nx.edges())