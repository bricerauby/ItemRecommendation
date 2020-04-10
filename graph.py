import snap
from embedding import Node2vec
DEBUG = True

def perform_random_walks(graph, N, L):
    '''
    :param graph: networkx graph
    :param N: the number of walks for each node
    :param L: the walk length
    :return walks: the list of walks
    '''
    nodelist = list(graph.Nodes())
    walks = []
    
    for _ in range(N):
        np.random.shuffle(nodelist)
    
        for node in nodelist:
            # Set the initial node
            walk = [node]
            
            while len(walk) < L:
                
                current_node = walk[-1]
                # Uniformly choose the next node at random
                nb_list = list(nx.neighbors(graph, current_node))
                nb = np.random.choice(a=nb_list, size=1)[0]
                # Append the chosen node
                walk.append(nb)

            walks.append(walk)
        
    return walks

class Graph(object):
    '''
    Graph contains the network and its embedding

    Parameters
    ----------
    network : 
    embedding: the algorithm used for the graph embedding
    seed : int
        Seed for reproductibility of the dataset generation

    Attributes
    ----------
    
    '''
    def __init__(self, network, embedding='node2vec', seed=0):
        self.network = network
        
    def get_embeddings(self, edges):
        raise NotImplementedError

if __name__ == '__main__': 
    if DEBUG : 
        network = snap.LoadEdgeList(snap.PNGraph, 'data/amazon-meta.txt')
        graph = Graph(network)
