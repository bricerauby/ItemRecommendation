import itertools
import random
import json
from joblib import Parallel, delayed

from gensim.models import Word2Vec

from models import Embedding, callback



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
    def __init__(self, graph, save_path):
        Embedding.__init__(self, graph)
        self.set_paths(save_path)
        self.walker = RandomWalker(
            graph)

    def get_walks(self, num_walks_per_node, walk_length, precomputed, workers) :
        
        if precomputed:
            self.load_walks()
        else:
            print('Generating walks...')
            self.walks = self.walker.simulate_walks(
                num_walks=num_walks_per_node, walk_length=walk_length, 
                workers=workers, verbose=10)
            self.save_walks()

    def train(self, embedding_size=128, window_size=5, 
              workers=8, nb_epochs=50, **kwargs):
        
        kwargs["sentences"] = self.walks
        kwargs["min_count"] = kwargs.get("min_count", 0)
        kwargs["size"] = embedding_size
        kwargs["sg"] = 1  # skip gram
        kwargs["hs"] = 1  # deepwalk use Hierarchical Softmax
        kwargs["workers"] = workers
        kwargs["window"] = window_size
        kwargs["iter"] = nb_epochs
        kwargs["compute_loss"] = True
        kwargs["min_alpha"] = 1e-2
        call_back = callback(all_loss=[])
        kwargs['callbacks'] = [call_back]

        print("Learning embedding vectors...")
        self.word_vectors = Word2Vec(**kwargs).wv
        print("Learning embedding vectors done!")
        
        self.word_vectors = dict(
            zip(self.word_vectors.index2word, self.word_vectors.vectors.tolist()))