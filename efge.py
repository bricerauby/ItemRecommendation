import os

import numpy as np

from embedding import Embedding

def sigmoid(x): 
    return 1 / (1 + np.exp(-x))


class Efge(Embedding):
    def __init__(self, graph=None, model='efge-bern', embedding_dimension=128, n_negative_sampling=5):
        Embedding.__init__(self, graph)
        self.model_type = model
        self.nodes = np.asarray(list(graph.nodes()))
        self.embedding_dimension = embedding_dimension
        self.n_negative_sampling = n_negative_sampling
        self.context_embedding = np.random.random((self.num_nodes, self.embedding_dimension))
        self.center_embedding = np.random.random((self.num_nodes, self.embedding_dimension))
        self.walks=[]
        
        self.set_paths()

    def learning_rate(self):
        return 1e-2

    def set_paths(self):

        if not os.path.exists(name):
            os.makedirs(name)
        self.vectors_path = '{}/embedding.json'.format(name)
        # efge and node2vec share the same walks
        self.walks_path = 'model/node2vec/walks.json'

    def update_efge_bern(self, learning_rate, center_id, context_ids, in_neighborhood):
        context_embedding = self.context_embedding[context_ids]
        center_embedding = np.expand_dims(self.center_embedding[center_id], axis=0)
        eta = np.sum(center_embedding * context_embedding, axis=-1)
        grad = np.expand_dims(sigmoid(eta) - in_neighborhood,axis=-1)
        # update the context_embedding
        self.context_embedding[context_ids] += - learning_rate * grad * center_embedding
        # update the center embedding 
        self.center_embedding[center_id] += - learning_rate * np.sum(grad * context_embedding, axis=0) 

    def fit_one_epoch(self, walks=None):
        if walks is None : 
            self.load_walks()
            walks = self.walks
        
        for walk in walks: 
            for i, center_id in enumerate(walk):
                for j, context_id in enumerate(walk):
                    if j==i: 
                        pass 
                    else:
                        negative_context = self.nodes[np.random.randint(0,self,num_nodes,self.n_negative_sampling)]
                        context_ids = np.concatenate([[context_ids], negative_context.tolist()])
                        in_neighborhood = np.asarray([1]+ self.n_negative_sampling[0])
                        if self.model_type == 'efge-bern':
                            self.update_efge_bern(self.learning_rate(), center_id, context_ids, in_neighborhood)
                        elif self.model_type =='efge-pois':
                            raise NotImplementedError
                        elif self.model_type == 'efge-norm':
                            raise NotImplementedError
                        else: 
                            raise ValueError('the model should be efge-bern, efge-pois or efge-norm')



from dataset import Dataset
dataset = Dataset()

efge = Efge(dataset.residual_network)
efge.load_walks()
walks = efge.walks[:100]
efge.fit_one_epoch(walks)