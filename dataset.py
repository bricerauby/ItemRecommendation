import urllib.request
import gzip
import shutil
import os
from collections import deque
import time

import tqdm
import numpy as np
import snap

DEBUG = False


class Dataset(object):
    '''
    Dataset contains the network, the residual network, the train set and the test set

    Parameters
    ----------
    path : str
        The path to the dataset (defaulted to the path set up by the main execution).
    residual_ratio : float
        The proportion of edges to keep in the residual network 
    seed : int
        Seed for reproductibility of the dataset generation

    Attributes
    ----------
    network : snap graph representation of the whole network
    residual_network : snap graph representation of the residual network
    removed_edges : list of the edges that have been removed to obtain the residual network
    kept_edges : list of the edges that have been kept in the residual network
    x_train : the edges in the train set 
    y_train : the labels for the train set 0 is fictive edge 1 is existing edge
    x_test : the edges in the test set 
    y_test : the labels for the test set 0 is fictive edge 1 is existing edge
    '''

    def __init__(self, path='data/amazon-meta.txt', residual_ratio=0.5, seed=0):
        self.network = snap.LoadEdgeList(snap.PNGraph, path)
        self.residual_network = snap.LoadEdgeList(snap.PNGraph, path)

        self.removed_edges = set()
        self.kept_edges = set()
        # get the number of the edges to remove
        n_edges_to_keep = int(
            (residual_ratio) * self.residual_network.GetEdges())
        n_edges_to_remove = self.residual_network.GetEdges() - n_edges_to_keep

        # set the seed
        np.random.seed(seed)
        Rnd = snap.TRnd(seed)
        Rnd.Randomize()

        print('removing edges randomly')
        start = time.time()

        # we get a connected graph with less edges than what we need to keep
        # then we randomly re-add edges to reach the correct residual ratio

        # Make sure that the tree has not too many edges already, it is very
        # likely to do only one iteration (it is very expensive otherwise)
        min_n_edges, min_node_id = self.network.GetEdges(), None
        for i in range(self.network.GetNodes()):
            node_id = self.residual_network.GetRndNId()
            BfsTree = snap.GetBfsTree(self.network, node_id, True, True)
            n_edges = BfsTree.GetEdges()
            is_acceptable = n_edges < self.network.GetEdges() * residual_ratio
            if is_acceptable:
                break
            elif min_n_edges > n_edges:
                min_node_id = node_id
                min_n_edges = n_edges

            # we remove the node in order to sample without replacement
            self.residual_network.DelNode(node_id)

        if is_acceptable:
            print('small enough spanning tree was found in {} iteration'.format(i+1))
        else:
            BfsTree = snap.GetBfsTree(self.network, min_node_id, True, True)
            print('small enough spanning tree was not found, the smallest had {} edges'.format(
                min_n_edges))

        # Keep only the node from the network we will add the edges
        self.residual_network.Clr()
        for node in self.network.Nodes():
            self.residual_network.AddNode(node.GetId())

        # we add the edges from the spanning tree
        self.kept_edges = set()
        for edge in BfsTree.Edges():
            self.kept_edges.add(edge.GetId())
            self.residual_network.AddEdge(*edge.GetId())

        # we shuffle the index to have access to a random permutation
        # of the edges while iterating over the edges
        shuffled_indexes = np.random.permutation(
            np.arange(self.network.GetEdges()))

        # we use a set for O(1) addition and check operation
        indexes_to_keep = set(
            shuffled_indexes[:n_edges_to_keep-len(self.kept_edges)])

        for i, edge in enumerate(self.network.Edges()):
            edge_id = edge.GetId()
            if i in indexes_to_keep:
                # if the edge is already added we will add another edge from the permutation
                if edge_id in self.kept_edges:
                    # it is very unlikely that there is no index higher than the current index 
                    # left in the permutation
                    for new_index_to_keep in shuffled_indexes[n_edges_to_keep-len(self.kept_edges):]:
                        if new_index_to_keep > i: #need to be higher than i in order to be added later
                            indexes_to_keep.add(new_index_to_keep)
                            break
                else:
                    self.kept_edges.add(edge_id)
                    self.residual_network.AddEdge(*edge_id)
            else:
                self.removed_edges.add(edge_id)

        self.residual_network.Defrag()
        print('keep ratio : {}'.format(self.residual_network.GetEdges()/self.network.GetEdges()))
        print('the network is connected : {}'.format(
            snap.IsWeaklyConn(self.residual_network)))

        print('time taken {} to generate the residual network'.format(
            time.time()-start))

        n_train = 2 * len(self.kept_edges)
        n_test = 2 * len(self.removed_edges)

        self.fictive_edges = []

        print('generating random fictive edges')
        for i in tqdm.tqdm(range(n_train//2 + n_test//2)):
            Id_src = self.network.GetRndNId(Rnd)
            Id_dst = self.network.GetRndNId(Rnd)
            while self.network.IsEdge(Id_src, Id_dst) or Id_dst == Id_src:
                Id_src = self.network.GetRndNId(Rnd)
                Id_dst = self.network.GetRndNId(Rnd)
            self.fictive_edges.append((Id_src, Id_dst))

        self.x_train, self.y_train = [], []
        self.x_test, self.y_test = [], []

        self.x_test += list(self.removed_edges)
        self.y_test += len(self.x_test) * [1]
        self.x_test += self.fictive_edges[:n_test//2]
        self.y_test += n_test * [0]

        self.x_train += list(self.kept_edges)
        self.y_train += len(self.x_train) * [1]
        self.x_train += self.fictive_edges[n_test//2:]
        self.y_train += n_train//2 * [0]

        self.x_train, self.y_train = np.asarray(
            self.x_train), np.asarray(self.y_train)
        shuffled_indexes = np.random.permutation(n_train)
        self.x_train = self.x_train[shuffled_indexes]
        self.y_train = self.y_train[shuffled_indexes]

    def get_split(self):
        return self.x_train, self.y_train, self.x_test, self.y_test

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