
import tqdm 
from gensim.models import Word2Vec
import node2vec
from models import Embedding, callback

class Node2vec(Embedding):

    def __init__(self, graph, path):
        Embedding.__init__(self, graph)
        self.graph = graph
        self.walks = None
        self.embedding = None
        self.path = path
        self.set_paths(path)

    def get_walks(self, walk_length, num_walks_per_node, p, q, workers, precomputed):
        if precomputed:
            self.load_walks()
        else:
            self.walks = node2vec.Node2Vec(
                self.graph, walk_length=walk_length, num_walks=num_walks_per_node, p=p, q=q, workers=workers).walks
            self.save_walks()

    def train(self, window=10, embedding_size=128, workers=8, nb_epochs=50):
        kwargs = {}
        kwargs["min_count"] = kwargs.get("min_count", 0)
        kwargs["size"] = embedding_size
        kwargs["sg"] = 1  # skip gram
        kwargs["workers"] = workers
        kwargs["window"] = window
        kwargs["iter"] = nb_epochs
        kwargs["compute_loss"] = True
        kwargs["min_alpha"] = 1e-2
        call_back = callback(all_loss=[])
        kwargs['callbacks'] = [call_back]

        self.load_walks()
        kwargs["sentences"] = self.walks
        
        print("Learning embedding vectors...")
        self.word_vectors = Word2Vec(**kwargs).wv
        print("Learning embedding vectors done!")

        self.word_vectors = dict(
            zip(self.word_vectors.index2word, self.word_vectors.vectors.tolist()))