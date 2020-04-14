import urllib.request
import gzip
import shutil
import os
import time
import json 

import tqdm
import numpy as np
import networkx
import tempfile
import embedding
from deep_walk import DeepWalk
from line import Line_model
from efge import Efge

DEBUG = True


class Dataset(object):
    '''
    Dataset contains the network, the residual network, the train set and the test set

    Parameters
    ----------
    data_path : str
        The path to the dataset (defaulted to the path set up by the main execution).
    residual_ratio : float
        The proportion of edges to keep in the residual network 
    seed : int
        Seed for reproductibility of the dataset generation

    Attributes
    ----------
    residual_network : networkx graph representation of the residual network
    removed_edges : list of the edges that have been removed to obtain the residual network
    kept_edges : list of the edges that have been kept in the residual network
    x_train : the edges in the train set 
    y_train : the labels for the train set 0 is fictive edge 1 is existing edge
    x_test : the edges in the test set 
    y_test : the labels for the test set 0 is fictive edge 1 is existing edge
    '''

    def __init__(self, data_path='data/amazon-meta.txt', residual_ratio=0.5, seed=0, precompute=False):
        if precompute:
            networkx.read_edgelist(self.residual_network, 'data/residual_network.txt')
            with open('data/train.json', 'r') as f:
                json.dump({'x_train':self.x_train.tolist(), 'y_train': self.y_train.tolist()}, f)
            with open('data/test.json', 'r') as f:
                json.load({'x_test':self.x_test.tolist(), 'y_test': self.y_test.tolist()}, f)


        else:
            network = networkx.read_edgelist(data_path)
            removed_edges = set()
            kept_edges = set()
            # get the number of the edges to remove
            n_edges_to_keep = int(
                (residual_ratio) * network.number_of_edges())
            n_edges_to_remove = network.number_of_edges() - n_edges_to_keep

            # set the seed
            np.random.seed(seed)

            print('removing edges randomly')
            start = time.time()

            # taking the minimal spanning tree  and adding edges is a way to enforce the connectivity of the  residual graph
            print('searching the minimal spanning tree')
            residual_network = networkx.algorithms.tree.minimum_spanning_edges(
                network, data=False, keys=False)
            residual_network = list(residual_network)
            n_edges = len(residual_network)
            is_acceptable = n_edges < network.number_of_edges() * residual_ratio

            if not is_acceptable:
                print(
                    'minimum spanning tree has more edge than required by the residual ratio')

            print('removing unessential edges from the network')
            # we remove the edges that have already been added to the residual network
            network.remove_edges_from(residual_network)
            network = list(network.edges())
            network = np.random.permutation(network)
            removed_edges = network[:n_edges_to_remove]

            residual_network = networkx.Graph(residual_network)
            # we add to the residual network the edges left in network (ie not in the spanning tree not in remove)
            residual_network.add_edges_from(network[n_edges_to_remove:])
            kept_edges = set(residual_network.edges())

            network = networkx.Graph(removed_edges.tolist())

            print('the network is connected : {}'.format(
                networkx.is_connected(residual_network)))

            print('time taken {} to generate the residual network'.format(
                time.time()-start))

            n_train = 2 * len(kept_edges)
            n_test = 2 * len(removed_edges)

            fictive_edges = []

            print('generating random fictive edges')
            nodes = list(residual_network.nodes())
            for i in tqdm.tqdm(range(n_train//2 + n_test//2)):
                Id_src = nodes[np.random.randint(len(nodes))]
                Id_dst = nodes[np.random.randint(len(nodes))]
                not_acceptable = Id_dst == Id_src
                not_acceptable = not_acceptable or residual_network.has_edge(
                    Id_src, Id_dst)
                not_acceptable = not_acceptable or network.has_edge(Id_src, Id_dst)
                while not_acceptable:
                    Id_src = nodes[np.random.randint(len(nodes))]
                    Id_dst = nodes[np.random.randint(len(nodes))]
                    not_acceptable = Id_dst == Id_src
                    not_acceptable = not_acceptable or residual_network.has_edge(
                        Id_src, Id_dst)
                    not_acceptable = not_acceptable or network.has_edge(
                        Id_src, Id_dst)

                fictive_edges.append((Id_src, Id_dst))

            self.x_train, self.y_train = [], []
            self.x_test, self.y_test = [], []

            self.x_test += list(removed_edges)
            self.y_test += len(self.x_test) * [1]
            self.x_test += fictive_edges[:n_test//2]
            self.y_test += n_test//2 * [0]
            self.x_test = np.asarray(self.x_test)
            self.y_test = np.asarray(self.y_test)

            self.x_train += list(kept_edges)
            self.y_train += len(self.x_train) * [1]
            self.x_train += fictive_edges[n_test//2:]
            self.y_train += n_train//2 * [0]

            self.x_train, self.y_train = np.asarray(
                self.x_train), np.asarray(self.y_train)
            shuffled_indexes = np.random.permutation(n_train)
            self.x_train = self.x_train[shuffled_indexes]
            self.y_train = self.y_train[shuffled_indexes]
            self.residual_network = residual_network
            #we save the graph for other runs 
            networkx.write_edgelist(self.residual_network, 'data/residual_network.txt')
            with open('data/train.json', 'w') as f:
                json.dump({'x_train':self.x_train.tolist(), 'y_train': self.y_train.tolist()}, f)
            with open('data/test.json', 'w') as f:
                json.dump({'x_test':self.x_test.tolist(), 'y_test': self.y_test.tolist()}, f)

    def get_split(self):
        return self.x_train, self.y_train, self.x_test, self.y_test

    def embed_network(self, seed, algorithm_name, num_walks_per_node, walk_length,
                      n_batch, precompute, path, workers, nb_epochs):
        '''
        methods that embed the network with a given algorithm, the network is replaced by its embedding to save memory
        '''
        if algorithm_name == 'node2vec':
            # if precompute we try to load the stored embedding
            # or the stored random walks
            if precompute:
                try:
                    model = embedding.Node2vec(path, n_batch)
                    model.load_embedding()
                except:
                    try:  # try to load the walk
                        model = embedding.Node2vec(path, n_batch)
                        model.fit()
                    except:
                        model = embedding.Node2vec(
                            self.residual_network, n_batch=n_batch, path=path)
                        model.compute_walks(
                            walk_length=walk_length, num_walks=num_walks_per_node,
                            n_batch=n_batch, workers=workers)
                        model.fit()
            else:
                model = embedding.Node2vec(
                    self.residual_network, n_batch=n_batch, path=path)
#                 model.compute_walks(walk_length=walk_length,
#                                     num_walks=num_walks_per_node, 
#                                     n_batch=n_batch, workers=workers)
                model.load_walks()
                model.fit()
                
        elif algorithm_name == 'deep_walk':
            model = DeepWalk(graph=self.residual_network, walk_length=walk_length, 
                             num_walks=num_walks_per_node, workers=workers)
            model.generate_walks()
#             model.load_walks()
            model.train(iter = nb_epochs, workers=workers)
            
        elif algorithm_name == 'line' :
            model = Line_model(graph = self.residual_network)
            model.train()
            
        elif algorithm_name == 'efge' :
            model = Efge(self.residual_network)
#             model.compute_walks(walk_length=walk_length,
#                                 num_walks=num_walks_per_node, 
#                                 workers=workers)
            model.load_walks()
            for epochs in range(nb_epochs) :
                model.fit_one_epoch(model.walks)
            model.get_word_vectors()
        else:
            raise NotImplementedError('embedding is not implemented')
            
        model.save_embedding()
        import gc; gc.collect()
        # we replace the network by its embedding to save memory
        self.residual_network = model.word_vectors
        self.embedded = True
        
    def load_embeddings(self, algorithm_name):
        vectors_path = 'models/' + algorithm_name + '/embedding.json'
        with open(vectors_path, 'r') as f:
            self.residual_network = json.load(f)
            
    def embed_edges(self, index=None, train=True, test=True):
        if index is None :
            if not self.embedded:
                raise ValueError('First embed the network')
            else:
                self.x_train = self.x_train.tolist()
                self.x_test = self.x_test.tolist()
                for i in tqdm.tqdm(range(len(self.x_train))):
                    edge = self.x_train[i]
                    self.x_train[i] = np.asarray(self.residual_network[str(edge[0])]) * np.asarray(self.residual_network[str(edge[1])])
                for i in tqdm.tqdm(range(len(self.x_test))):
                    edge = self.x_test[i]
                    self.x_test[i] = np.asarray(self.residual_network[str(edge[0])]) * np.asarray(self.residual_network[str(edge[1])])
        else :
            if train :
                edges = self.x_train[index]
            if test :
                edges = self.x_test[index]
            index_embeddings = np.zeros((len(index), 128))
            for i, edge in enumerate(edges) :
                index_embeddings[i] = np.asarray(self.residual_network[str(edge[0])]) * np.asarray(self.residual_network[str(edge[1])])
            return index_embeddings 
        
if __name__ == '__main__':

    if not DEBUG:
        url = 'https://snap.stanford.edu/data/amazon0505.txt.gz'

        try:
            os.mkdir('data')
        except FileExistsError:
            print('data folder is already existing')

        # download the dataset from the snap url
        urllib.request.urlretrieve(url, "data/amazon-meta.txt.gz")

        # decompress the file
        with open('data/amazon-meta.txt', 'wb') as decompressed_file:
            with gzip.open('data/amazon-meta.txt.gz', 'rb') as compressed_file:
                shutil.copyfileobj(compressed_file, decompressed_file)
    else:
        dataset = Dataset()