import tempfile
import networkx
import snap
import node2vec
import time 
def snap2nx(snap_graph):
    fo = tempfile.NamedTemporaryFile()		
    snap.SaveEdgeList(snap_graph, fo.name, "Save as tab-separated list of edges")
    nx_graph = networkx.read_edgelist(fo.name)
    fo.close()
    return nx_graph

class Embedding():
    def __init__(self, graph):
        self.graph = graph
        self.embedding = None

    def get_embedding(self):
        return self.embedding


class Node2vec(Embedding):
    def __init__(self, graph):
        Embedding.__init__(self,graph)
        self.nx_graph = snap2nx(self.graph)
        start = time.time()
        self.node2vec = node2vec.Node2Vec(self.nx_graph, walk_length=10, num_walks=1, p=1, q=1)
        print('time taken for the random walk generation {}'.format(time.time()-start))
        self.walks = self.node2vec.walks
    def get_walks(self):
        return self.node2vec.walks


if __name__ == "__main__":
    from dataset import Dataset
    dataset = Dataset()
    start = time.time()
    graph_nx = snap2nx(dataset.network)
    print('time taken for conversion {}'.format(time.time()-start))
    embedding = Node2vec(dataset.network)
    
