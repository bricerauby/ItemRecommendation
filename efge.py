import os
import numpy as np
from embedding import Embedding
import node2vec


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
        self.walks = []
        
        self.set_paths(name='model/efge')

    def learning_rate(self):
        return 1e-2

    def set_paths(self, name):

        if not os.path.exists(name):
            os.makedirs(name)
        self.vectors_path = '{}/embedding.json'.format(name)
        # efge and node2vec share the same walks
        self.walks_path = 'model/efge/walks.json'
        
    def compute_walks(self, walk_length=10, num_walks=80,
                     p=1, q=1, workers=4) :
        self.walks = node2vec.Node2Vec(
                self.graph, walk_length=walk_length, num_walks=num_walks,
                p=p, q=q, workers=workers).walks
        self.save_walks()
        
    def update_efge_bern(self, learning_rate, center_id, context_ids, in_neighborhood):
        context_embedding = self.context_embedding[context_ids]
        center_embedding = np.expand_dims(self.center_embedding[center_id], axis=0)
        eta = np.sum(center_embedding * context_embedding, axis=-1)
        grad = np.expand_dims(sigmoid(eta) - in_neighborhood,axis=-1)
        # update the context_embedding
        self.context_embedding[context_ids] += - learning_rate * grad * center_embedding
        # update the center embedding 
        self.center_embedding[center_id] += - learning_rate * np.sum(grad * context_embedding, axis=0) 

    def update_efge_poisson(self, learning_rate, center_id, context_ids, in_neighborhood):
        context_embedding = self.context_embedding[context_ids]
        center_embedding = np.expand_dims(self.center_embedding[center_id], axis=0)
        eta = np.sum(center_embedding * context_embedding, axis=-1)
        grad = np.expand_dims(np.exp(eta) - in_neighborhood,axis=-1)
        # update the context_embedding
        self.context_embedding[context_ids] += - learning_rate * grad * center_embedding
        # update the center embedding 
        self.center_embedding[center_id] += - learning_rate * np.sum(grad * context_embedding, axis=0) 
        
    def update_efge_norm(self, learning_rate, center_id, context_ids, in_neighborhood, std_dev=1.):
        context_embedding = self.context_embedding[context_ids]
        center_embedding = np.expand_dims(self.center_embedding[center_id], axis=0)
        eta = np.sum(center_embedding * context_embedding, axis=-1)
        grad = np.expand_dims(-np.exp(-2*eta) + np.exp(-eta)*in_neighborhood/std_dev, axis=-1)
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
                        context_id = int(context_id)
                        center_id = int(center_id)
                        negative_context = self.nodes[np.random.randint(0, self.num_nodes,
                                                                        self.n_negative_sampling)]
                        negative_context = np.array([int(i) for i in negative_context])
                        context_ids = np.concatenate([[context_id], negative_context.tolist()])
#                         print(context_ids.shape, context_ids)
                        in_neighborhood = np.asarray([1]+ self.n_negative_sampling*[0])
                        if self.model_type == 'efge-bern':
                            self.update_efge_bern(self.learning_rate(), center_id, list(context_ids), in_neighborhood)
                        elif self.model_type =='efge-pois':
                            self.update_efge_poisson(self.learning_rate(), center_id, context_ids, in_neighborhood)
                        elif self.model_type == 'efge-norm':
                            self.update_efge_norm(self.learning_rate(), center_id, context_ids, in_neighborhood, 
                                                  std_dev=1.)
                        else: 
                            raise ValueError('the model should be efge-bern, efge-pois or efge-norm')
    def get_word_vectors(self,):
        self.word_vectors = {str(elt) : list(value.astype('float')) for (elt, value) in\
                 enumerate(self.center_embedding)}   
         
if __name__ == '__main__' :
    from dataset import Dataset
    nb_epochs = 10
    dataset = Dataset().residual_network
    efge = Efge(dataset)
    import gc; gc.collect()
    efge.compute_walks(walk_length=10, num_walks=80, workers=4)
    
    for epochs in range(nb_epochs) :
        efge.fit_one_epoch(efge.walks)
    efge.get_word_vectors()
    efge.save_embedding()
#     efge.load_walks()