import itertools
import math
import random
import numpy as np
from joblib import Parallel, delayed
from gensim.models import Word2Vec
from tqdm import trange
import tempfile
import networkx
import json
from embedding import Embedding

from gensim.models.word2vec import LineSentence
from gensim.models.callbacks import CallbackAny2Vec

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
                       workers=8, verbose=10):
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
    
class DeepWalk(Embedding):
    def __init__(self, graph, walk_length,
                 num_walks, path='models/deep_walk', workers=8):

        self.graph = graph
        self.w2v_model = None
        self.set_paths(path)
        self._embeddings = {}
        self.num_walks = num_walks
        self.walk_length = walk_length
        self.workers = workers
        self.walker = RandomWalker(
            graph)
    def generate_walks(self, verbose=10) :
        print('Generating walks...')
        self.walks = self.walker.simulate_walks(
            num_walks=self.num_walks, walk_length=self.walk_length, 
            workers=self.workers, verbose=verbose)
        self.save_walks()
    def train(self, embed_size=128, window_size=5, 
              workers=8, iter=50, **kwargs):
        
        kwargs["sentences"] = self.walks
        kwargs["min_count"] = kwargs.get("min_count", 0)
        kwargs["size"] = 128
        kwargs["sg"] = 1  # skip gram
        kwargs["hs"] = 1  # deepwalk use Hierarchical Softmax
        kwargs["workers"] = workers
        kwargs["window"] = window_size
        kwargs["iter"] = iter
        kwargs["compute_loss"] = True
        kwargs["min_alpha"] = 1e-2
        call_back = callback(all_loss=[])
        kwargs['callbacks'] = [call_back]
        print("Learning embedding vectors...")
        model = Word2Vec(**kwargs)
        print("Learning embedding vectors done!")
        self.w2v_model = model
        self.word_vectors = {}
        for elt in self.w2v_model.wv.vocab :
            self.word_vectors[elt] = list(self.w2v_model.wv[elt].astype('float'))
            
        import matplotlib.pyplot as plt
        fig = plt.figure()
        plt.plot(call_back.all_loss)
        fig.savefig('deep_walk_loss.png')
        return model
        
    def save_embedding(self):
        with open(self.vectors_path, 'w') as f:
            json.dump(self.word_vectors, f)
            
    def load_embedding(self):
        with open(self.vectors_path, 'r') as f:
            self.word_vectors = json.load(f)
            
    def get_embeddings(self, edges):
        result = np.zeros((len(edges), 128))
        for i, (elt1, elt2) in enumerate(edges) :
            emb_1 = self.w2v_model.wv[str(elt1)]
            emb_2 = self.w2v_model.wv[str(elt2)]
            emb = np.multiply(emb_1, emb_2)            
        return result
    
if __name__ == '__main__' :
#     graph_nx = Dataset().residual_network
    from dataset import Dataset
    deep_walk = DeepWalk(graph=graph, walk_length=2, 
                         num_walks=1, workers=2)
    deep_walk.generate_walks()
    deep_walk.train(iter = 10, workers=2)
    deep_walk.save_embedding()
#     deep_walk.w2v_model.wv.save_word2vec_format('random_walk')
#     edges_emb = deep_walk.get_embeddings(graph_nx.edges())