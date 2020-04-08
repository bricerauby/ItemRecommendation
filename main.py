from dataset import Dataset
from embedding import Graph
# we load the dataset and perform edge extraction to get the train test split
dataset = Dataset()
x_train, y_train, x_test, y_test = dataset.get_split()

# we perform the embedding of the nodes in the network using the residual network 
graph = Graph(dataset.residual_network)

# we get the embeddings for both set
x_train_edges = graph.get_embeddings(x_train)
x_test_edges = graph.get_embeddings(x_test)

# we do the classification