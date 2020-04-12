import torch
import torch.nn as nn
import torch.nn.functional as F
from tqdm import trange
import torch.optim as optim
### https://github.com/DMPierre/LINE/
import random
from decimal import *
import numpy as np
import collections
from tqdm import tqdm
import json
from dataset import Dataset

args = {}
args["graph_path"] = 'data/amazon-meta.txt'
# Hyperparams.
args["order"] = 2
args["negsamplesize"] = 5
args["dimension"] = 128
args["batchsize"] = 1024
args["epochs"] = 10
args["learning_rate"] = 0.025  # As starting value in paper
args["negativepower"] = 0.75 

class VoseAlias(object):
    """
    Adding a few modifs to https://github.com/asmith26/Vose-Alias-Method
    """
    def __init__(self, dist):
        """
        (VoseAlias, dict) -> NoneType
        """
        self.dist = dist
        self.alias_initialisation()

    def alias_initialisation(self):
        """
        Construct probability and alias tables for the distribution.
        """
        # Initialise variables
        n = len(self.dist)
        self.table_prob = {}   # probability table
        self.table_alias = {}  # alias table
        scaled_prob = {}       # scaled probabilities
        small = []             # stack for probabilities smaller that 1
        large = []             # stack for probabilities greater than or equal to 1
        # Construct and sort the scaled probabilities into their appropriate stacks
#         print("1/2. Building and sorting scaled probabilities for alias table...")
        for o, p in self.dist.items():
            scaled_prob[o] = Decimal(p) * n

            if scaled_prob[o] < 1:
                small.append(o)
            else:
                large.append(o)

#         print("2/2. Building alias table...")
        # Construct the probability and alias tables
        while small and large:
            s = small.pop()
            l = large.pop()
            self.table_prob[s] = scaled_prob[s]
            self.table_alias[s] = l
            scaled_prob[l] = (scaled_prob[l] + scaled_prob[s]) - Decimal(1)
            if scaled_prob[l] < 1:
                small.append(l)
            else:
                large.append(l)
        # The remaining outcomes (of one stack) must have probability 1
        while large:
            self.table_prob[large.pop()] = Decimal(1)

        while small:
            self.table_prob[small.pop()] = Decimal(1)
        self.listprobs = list(self.table_prob)
    def alias_generation(self):
        """
        Yields a random outcome from the distribution.
        """
        # Determine which column of table_prob to inspect
        col = random.choice(self.listprobs)
        # Determine which outcome to pick in that column
        if self.table_prob[col] >= random.uniform(0, 1):
            return col
        else:
            return self.table_alias[col]
    def sample_n(self, size):
        """
        Yields a sample of size n from the distribution, and print the results to stdout.
        """
        for i in range(size):
            yield self.alias_generation()
def makeDist(graph, power=0.75):
    maxindex = len(graph.nodes())
    nb_edges = len(graph.edges())
    nodedistdict = {}
    edgedistdict = {}
    nodedegrees = {}
    for node, outdegree in graph.degree():
        nodedistdict[int(node)] = np.power(outdegree, power) / nb_edges
        nodedegrees[int(node)] = outdegree
    for edge in graph.edges():
        edgedistdict[(int(edge[0]), int(edge[1]))] = 1 / nb_edges
    return edgedistdict, nodedistdict, nodedegrees, maxindex

def negSampleBatch(sourcenode, targetnode, negsamplesize,
                   nodedegrees, nodesaliassampler, t=10e-3):
    """
    For generating negative samples.
    """
    negsamples = 0
    while negsamples < negsamplesize:
        samplednode = nodesaliassampler.sample_n(1)
        if (samplednode == sourcenode) or (samplednode == targetnode):
            continue
        else:
            negsamples += 1
            yield samplednode
            
def makeData(samplededges, negsamplesize, nodedegrees, nodesaliassampler):
    for e in samplededges:
        sourcenode, targetnode = e[0], e[1]
        negnodes = []
        for negsample in negSampleBatch(sourcenode, targetnode, negsamplesize,
                                        nodedegrees, nodesaliassampler):
            for node in negsample:
                negnodes.append(node)
        yield [e[0], e[1]] + negnodes  
        
class Line(nn.Module):
    def __init__(self, size, embed_dim=128, order=1):
        super(Line, self).__init__()
        assert order in [1, 2], print("Order should either be int(1) or int(2)")
        self.embed_dim = embed_dim
        self.order = order
        self.nodes_embeddings = nn.Embedding(size, embed_dim)
        if order == 2:
            self.contextnodes_embeddings = nn.Embedding(size, embed_dim)
            # Initialization
            self.contextnodes_embeddings.weight.data = self.contextnodes_embeddings.weight.data.uniform_(
                -.5, .5) / embed_dim
        # Initialization
        self.nodes_embeddings.weight.data = self.nodes_embeddings.weight.data.uniform_(
            -.5, .5) / embed_dim
        
    def forward(self, v_i, v_j, negsamples, device):
        v_i = self.nodes_embeddings(v_i)
        if self.order == 2:
            v_j = self.contextnodes_embeddings(v_j)
            negativenodes = -self.contextnodes_embeddings(negsamples)
        else:
            v_j = self.nodes_embeddings(v_j)
            negativenodes = -self.nodes_embeddings(negsamples)
        mulpositivebatch = torch.mul(v_i, v_j)
        positivebatch = F.logsigmoid(torch.sum(mulpositivebatch, dim=1))
        mulnegativebatch = torch.mul(v_i.view(len(v_i), 1, self.embed_dim), negativenodes)
        negativebatch = torch.sum(F.logsigmoid(torch.sum(mulnegativebatch, dim=2)), dim=1)
        loss = positivebatch + negativebatch
        return -torch.mean(loss)
    
class Line_model :  
    def __init__(self, graph, args=args, 
                 vectors_path = 'line_model.json') :
        self.args = args
        self.model = None
        self.vectors_path = vectors_path
        self.line = None
        self.graph = graph
    def save_embedding(self):
        with open(self.vectors_path, 'w') as f:
            json.dump(self.word_vectors, f)
            
    def load_embedding(self):
        with open(self.vectors_path, 'r') as f:
            self.word_vectors = json.load(f)
   
    def get_embeddings(self, edges, device):
        if device.type == 'cuda' : edges = edges.cuda()
        emb_1 = self.nodes_embeddings(edges[:, 0]).data.detach().cpu().numpy()  
        emb_2 = self.nodes_embeddings(edges[:, 2]).data.detach().cpu().numpy()
        emb = np.multiply(emb_1, emb_2)
        return emb
    
    def train(self) :
        # Create dict of distribution when opening file
        args = self.args
        print("Reading edgelist file...")
        edgedistdict, nodedistdict, nodedegrees, maxindex = makeDist(
            self.graph, args['negativepower'])

        nb_edges = len(edgedistdict)
        edgesaliassampler = VoseAlias(edgedistdict)
        del edgedistdict
        nodesaliassampler = VoseAlias(nodedistdict)
        del nodedistdict
        import gc; gc.collect()

        batchrange = int(nb_edges / args['batchsize'])
        line = Line(maxindex + 1, embed_dim=args['dimension'], order=args["order"])

        opt = optim.SGD(line.parameters(), lr=args['learning_rate'],
                        momentum=0.9, nesterov=True)

        device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        if device.type == 'cuda' : line.cuda()
        lossdata = {"it": [], "loss": []}
        it = 0

        print("Training on {}...".format(device))
        for epoch in range(args['epochs']):
            print("Epoch {}".format(epoch))
            import time; time.sleep(0.1)
            for b in trange(batchrange):
                samplededges = edgesaliassampler.sample_n(args['batchsize'])
                batch = list(makeData(samplededges, args['negsamplesize'], nodedegrees,
                                      nodesaliassampler))
                batch = torch.LongTensor(batch)
                if device.type == 'cuda' : batch = batch.cuda()
                v_i = batch[:, 0]
                v_j = batch[:, 1]
                negsamples = batch[:, 2:]
                line.zero_grad()
                loss = line(v_i, v_j, negsamples, device)
                loss.backward()
                opt.step()

                lossdata["loss"].append(loss.item())
                lossdata["it"].append(it)
                it += 1
        self.line = line
        self.word_vectors = {str(elt) : list(value.astype('float')) for (elt, value) in\
                             enumerate(line.nodes_embeddings.weight.data.detach().cpu().numpy())}
        
if __name__ == "__main__":
    
    ## For debugging purpose
#     args = {}
#     args["graph_path"] = 'data/amazon-meta.txt'

#     # Hyperparams.
#     args["order"] = 2
#     args["negsamplesize"] = 1
#     args["dimension"] = 16
#     args["batchsize"] = 2048
#     args["epochs"] = 1
#     args["learning_rate"] = 0.025  # As starting value in paper
#     args["negativepower"] = 0.75 
    
    model = Line_model(graph = Dataset().residual_network, args=args)
    model.train()
    print("Saving the embedding...")
    model.save_embedding()