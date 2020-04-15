import os
import time
import json

import tqdm
import numpy as np
import networkx
import tempfile
from models import Embedding
from models.node2vec import Node2vec
from models.efge import Efge
from models.line import Line_model
from models.deep_walk import DeepWalk

DEBUG = False


class Dataset(object):
    '''
    Dataset contains the residual network, the train set and the test set. It can be used to create a new ones or to load existent ones from a directory

    Parameters
    ----------
    data_path : str
        The path to the dataset.
    residual_ratio : float
        The proportion of edges to keep in the residual network 
    seed : int
        Seed for reproductibility of the dataset generation
    precomputed : bool
        wether a precomputed residual graph and train/test split should be used or not
    save_path : str 
        the path to the dir where the precomputed should be stored to or loaded from (it depends on `precomputed` value)
    reduce_dataset: int :
        the number of node to take in case we want to reduce the dataset
    Attributes
    ----------
    residual_network : networkx graph representation of the residual network
    x_train : the edges in the train set 
    y_train : the labels for the train set 0 is fictive edge 1 is existing edge
    x_test : the edges in the test set 
    y_test : the labels for the test set 0 is fictive edge 1 is existing edge
    '''

    def __init__(self, data_path, residual_ratio, seed, precomputed, save_path, reduce_dataset=None):
        save_path = os.path.join(save_path, 'seed_{}'.format(seed))
        if not reduce_dataset is None:
            save_path += '_reduced_{}'.format(reduce_dataset)
        try :
            os.makedirs(save_path)
        except: 
            pass

        save_graph_path = os.path.join(
            save_path, 'residual_network.txt')
        save_train_path = os.path.join(
            save_path, 'train.json')
        save_test_path = os.path.join(
            save_path, 'test.json')


        if precomputed:
            self.residual_network = networkx.read_edgelist(save_graph_path)
            with open(save_train_path, 'r') as f:
                train_dict = json.load(f)
                self.x_train = np.asarray(train_dict['x_train'])
                self.y_train = np.asarray(train_dict['y_train'])

            with open(save_test_path, 'r') as f:
                test_dict = json.load(f)
                self.x_test = np.asarray(test_dict['x_test'])
                self.y_test = np.asarray(test_dict['y_test'])

        else:
            network = networkx.read_edgelist(data_path)

            if not reduce_dataset is None:
                network = network.subgraph(list(network.nodes())[:reduce_dataset]).copy()
                network = networkx.relabel.convert_node_labels_to_integers(network)
                mapping = dict(zip(list(network.nodes()), [str(node) for node in list(network.nodes())]))
                network = networkx.relabel.relabel_nodes(network, mapping)
            
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
                not_acceptable = not_acceptable or network.has_edge(
                    Id_src, Id_dst)
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

            # we save the graph  and the train/test set for other runs
            networkx.write_edgelist( self.residual_network, save_graph_path)
            with open(save_train_path, 'w') as f:
                json.dump({'x_train': self.x_train.tolist(),
                           'y_train': self.y_train.tolist()}, f)
            with open(save_test_path, 'w') as f:
                json.dump({'x_test': self.x_test.tolist(),
                           'y_test': self.y_test.tolist()}, f)

    def embed_network(self, seed, save_path, algorithm_name, precomputed, training, walks):
        '''
        methods that embed the network with a given algorithm, the network is replaced by its embedding to save memory
        '''
        save_path = os.path.join(save_path, 'seed_{}'.format(seed))

        if DEBUG:
            save_path += '_debug'
        try :
            os.makedirs(save_path)
        except: 
            pass

        if precomputed:
            model = Embedding(self.residual_network)
            model.set_paths(save_path)
            model.load_embedding()
            self.residual_network = model.word_vectors
            self.embedded = True

        elif algorithm_name == 'node2vec':
            model = Node2vec(self.residual_network, save_path)

        elif algorithm_name == 'deep_walk':
            model = DeepWalk(self.residual_network, save_path)

        elif algorithm_name == 'efge':
            model = Efge(self.residual_network, save_path)

        else:
            raise NotImplementedError('embedding is not implemented')

        model.get_walks(**walks)
        model.train(**training)
        model.save_embedding()
        # we replace the network by its embedding to save memory
        self.residual_network = model.word_vectors
        self.embedded = True


    def embed_edges(self):
        self.x_train = self.x_train.tolist()
        self.x_test = self.x_test.tolist()
        for i in tqdm.tqdm(range(len(self.x_train))):
            edge = self.x_train[i]
            self.x_train[i] = np.asarray(self.residual_network[str(
                edge[0])]) * np.asarray(self.residual_network[str(edge[1])])
        for i in tqdm.tqdm(range(len(self.x_test))):
            edge = self.x_test[i]
            self.x_test[i] = np.asarray(self.residual_network[str(
                edge[0])]) * np.asarray(self.residual_network[str(edge[1])])



if __name__ == '__main__':
    config_path = 'configs/node2vec.json'
    with open(config_path, 'r') as f: 
        config = json.load(f)
    dataset = Dataset(**config['dataset'])
