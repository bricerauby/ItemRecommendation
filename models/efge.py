import os
import numpy as np
from models import Embedding
import node2vec
import multiprocessing

import tqdm


def sigmoid(x):
    return 1 / (1 + np.exp(-x))


class Efge(Embedding):
    def __init__(self, graph, path):
        Embedding.__init__(self, graph)
        self.nodes = np.asarray(list(graph.nodes()))
        self.context_embedding = None
        self.center_embedding = None
        self.set_paths(path)

    def learning_rate(self):
        if self.epoch > 20: 
            return 1e-6
        if self.epoch > 10: 
            return 1e-5
        return 0.0025

    def get_walks(self, walk_length, num_walks_per_node, p, q, workers, precomputed):
        if precomputed:
            self.load_walks()
        else:
            self.walks = node2vec.Node2Vec(
                self.graph, walk_length=walk_length, num_walks=num_walks_per_node, p=p, q=q, workers=workers).walks
            self.save_walks()

    def build_negative_sampling(self, power) :
        nb_edges = len(self.graph.edges())
        self.nodes = []
        self.probas = []
        for node, outdegree in self.graph.degree():
            self.probas.append(np.power(outdegree, power))
            self.nodes.append(node)
        self.probas = self.probas / np.sum(self.probas)

    def grad_efge_bern(self, learning_rate, center_embedding, context_embedding, in_neighborhood):
        eta = np.sum(center_embedding * context_embedding, axis=-1)
        grad = - learning_rate * \
            np.expand_dims(sigmoid(eta) - in_neighborhood, axis=-1)
        update = (grad * center_embedding,
                  np.sum(grad * context_embedding, axis=0))
        return update

    def grad_efge_poisson(self, learning_rate, center_embedding, context_embedding, in_neighborhood):
        eta = np.sum(center_embedding * context_embedding, axis=-1)
        grad = - learning_rate * \
            np.expand_dims(np.exp(eta) - in_neighborhood, axis=-1)
        update = (grad * center_embedding,
                  np.sum(grad * context_embedding, axis=0))
        return update

    def grad_efge_norm(self, learning_rate, center_embedding, context_embedding, in_neighborhood, std_dev=1.):
        eta = np.sum(center_embedding * context_embedding, axis=-1)
        grad = - learning_rate * \
            np.expand_dims(-np.exp(-2*eta) + np.exp(-eta) *
                           in_neighborhood/std_dev, axis=-1)
        update = (grad * center_embedding,
                  np.sum(grad * context_embedding, axis=0))
        return update

    def train(self, nb_epochs, workers, embedding_size, n_negative_sampling, update_type, batch_size, power):
        if self.context_embedding is None or self.center_embedding is None:
            self.context_embedding = np.random.random(
                (self.num_nodes, embedding_size))
            self.center_embedding = np.random.random(
                (self.num_nodes, embedding_size))
        self.build_negative_sampling(power)
        for epoch in range(nb_epochs):
            print(15 * '#' + 'epoch_' + str(epoch)+ '_' + 15 * '#')
            self.epoch = epoch
            if update_type == 'fast_bern':
                for i, walk in enumerate(self.walks):
                    walk = np.asarray(walk).astype('int')
                    center_embeddings = self.center_embedding[walk].reshape(len(walk),1,1,embedding_size)
                    positive_context_embedding = self.context_embedding[walk].reshape(1,len(walk),1,embedding_size)
                    negative_context = self.nodes[np.random.randint(0, self.num_nodes,
                                                n_negative_sampling)].astype('int')
                    negative_context_embedding = self.context_embedding[negative_context]
                    negative_context_embedding = negative_context_embedding.reshape(1, 1, 
                                                                                    n_negative_sampling, 
                                                                                    embedding_size)
                    update_center_vec = np.zeros((len(walk),128))
                    update_context_vec = np.zeros((len(walk),128))
                    if update_type == 'fast_bern':
                        eta_positive = np.sum(center_embeddings * positive_context_embedding, axis=(-1)) - 1
                        eta_negative = np.sum(center_embeddings * negative_context_embedding, axis=(-1), keepdims=True) 
                        grad_positive = np.squeeze(sigmoid(eta_positive))
                        np.fill_diagonal(grad_positive,0)

                        grad_positive = grad_positive.reshape(len(walk),len(walk),1 , 1)
                        print(45 * '#')
                        print(grad_positive)
                        grad_negative = sigmoid(eta_negative)
                        print(45 * '#')
                        print(grad_negative)
                        # self.center_embedding[walk] += - self.learning_rate() *   np.squeeze(np.sum(grad_positive * positive_context_embedding, axis=1))
                        # self.center_embedding[walk] += - self.learning_rate() * np.squeeze(np.sum(grad_negative * negative_context_embedding, axis=2))
                        
                        # self.context_embedding[walk] += - self.learning_rate() * np.squeeze(np.sum(grad_positive * center_embeddings, axis=0))
                        # self.context_embedding[negative_context] += - self.learning_rate() * np.squeeze(np.sum(grad_negative * center_embeddings, axis=0))

                        update_center_vec += - self.learning_rate() *   np.squeeze(np.sum(grad_positive * positive_context_embedding, axis=1))
                        update_center_vec += - self.learning_rate() * np.squeeze(np.sum(grad_negative * negative_context_embedding, axis=2))
                        
                        # update_context_vec += - self.learning_rate() * np.squeeze(np.sum(grad_positive * center_embeddings, axis=0))
                        # update_context_vec += - self.learning_rate() * np.squeeze(np.sum(grad_negative * center_embeddings, axis=0))
                        # if i % 1000 == 0:
                        #     print(np.sum(grad_negative**2))
                        #     print(np.sum(grad_positive**2))

                    update_center_for = np.zeros((len(walk),128))
                    # update_context_for = np.zeros((len(walk),128))
                    for i, center_id in enumerate(walk):
                        for j, context_id in enumerate(walk):
                            if j == i:
                                pass
                            else:
                                context_id = int(context_id)
                                center_id = int(center_id)
                                # negative_context = self.nodes[np.random.randint(0, self.num_nodes,
                                #                                                 n_negative_sampling)]
                                negative_context = np.array(
                                    [int(i) for i in negative_context])
                                context_ids = [(context_id)] + negative_context.tolist()
                                in_neighborhood = np.asarray(
                                    [1] + n_negative_sampling*[0])
                                args = [self.learning_rate(), 
                                        np.expand_dims(self.center_embedding[center_id], axis=0), 
                                        self.context_embedding[context_ids], in_neighborhood]    
                                # if update_type == 'bern':

                                print(45 * '#')
                                update_context, update_center = self.grad_efge_bern(*args)
                                # elif update_type == 'pois':
                                #     update_context, update_center = self.grad_efge_poisson(*args)
                                # elif update_type == 'norm':
                                #     update_context, update_center = self.grad_efge_norm(*args)
                                # else:
                                #     raise ValueError(
                                #         'the update_type should be bern, pois or norm')
                                # # update the context_embedding
                                # self.context_embedding[context_ids] += update_context
                                # # update the center embedding
                                # self.center_embedding[center_id] += update_center
                                update_center_for += update_center
                                # update_context_for += update_context
                    print(np.sum(np.abs(update_center_for - update_center_vec)))
                    # print(np.sum(np.abs(update_context_for - update_context_vec)))
                    break
            else:
                for walk in tqdm.tqdm(self.walks):
                    for i, center_id in enumerate(walk):
                        for j, context_id in enumerate(walk):
                            if j == i:
                                pass
                            else:
                                context_id = int(context_id)
                                center_id = int(center_id)
                                negative_context = np.random.choice(self.nodes, 
                                                                        n_negative_sampling, 
                                                                        p=self.probas)
                                negative_context = np.array(
                                    [int(i) for i in negative_context])
                                context_ids = [(context_id)] + negative_context.tolist()
                                in_neighborhood = np.asarray(
                                    [1] + n_negative_sampling*[0])
                                args = [self.learning_rate(), 
                                        np.expand_dims(self.center_embedding[center_id], axis=0), 
                                        self.context_embedding[context_ids], in_neighborhood]    
                                if update_type == 'bern':
                                    update_context, update_center = self.grad_efge_bern(*args)
                                elif update_type == 'pois':
                                    update_context, update_center = self.grad_efge_poisson(*args)
                                elif update_type == 'norm':
                                    update_context, update_center = self.grad_efge_norm(*args)
                                else:
                                    raise ValueError(
                                        'the update_type should be bern, pois or norm')
                                # update the context_embedding
                                self.context_embedding[context_ids] += update_context
                                # update the center embedding
                                self.center_embedding[center_id] += update_center
                                
        self.word_vectors = {str(elt): list(value.astype('float')) for (
            elt, value) in enumerate(self.center_embedding)}
