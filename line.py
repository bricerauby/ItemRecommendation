import torch
import torch.nn as nn
import torch.nn.functional as F
import pickle
from tqdm import trange
import torch.optim as optim
### https://github.com/DMPierre/LINE/
import random
from decimal import *
import numpy as np
import collections
from tqdm import tqdm

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
        print("1/2. Building and sorting scaled probabilities for alias table...")
        for o, p in tqdm(self.dist.items()):
            scaled_prob[o] = Decimal(p) * n

            if scaled_prob[o] < 1:
                small.append(o)
            else:
                large.append(o)

        print("2/2. Building alias table...")
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


def makeDist(graphpath, power=0.75):

    edgedistdict = collections.defaultdict(int)
    nodedistdict = collections.defaultdict(int)

#     weightsdict = collections.defaultdict(int)
    nodedegrees = collections.defaultdict(int)

    weightsum = 0
    negprobsum = 0

    nlines = 0

    with open(graphpath, "r") as graphfile:
        for l in graphfile:
            if l[0] == '#' :
                pass
            else :
                nlines += 1

    print("Reading edgelist file...")
    maxindex = 0
    with open(graphpath, "r") as graphfile:
        for l in tqdm(graphfile, total=nlines):
            if l[0] == '#' :
                pass
            else :
#                 import pdb; pdb.set_trace()

                line = [int(i) for i in l.split()]
                node1, node2, weight = line[0], line[1], 1

                edgedistdict[tuple([node1, node2])] = weight
                nodedistdict[node1] += weight
                nodedistdict[node2] += weight

#                 weightsdict[tuple([node1, node2])] = weight
                nodedegrees[node1] += weight
                nodedegrees[node2] += weight

                weightsum += weight
#                 if weightsum%10000 == 0 :
#                     print('total edges', weightsum)
#                     print('degree 0', nodedegrees[0])
                negprobsum += np.power(weight, power)

                if node1 > maxindex:
                    maxindex = node1
                elif node2 > maxindex:
                    maxindex = node2
    for node, outdegree in nodedistdict.items():
        nodedistdict[node] = np.power(outdegree, power) / negprobsum

    for edge, weight in edgedistdict.items():
        edgedistdict[edge] = weight / weightsum

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
        negativebatch = torch.sum(
            F.logsigmoid(
                torch.sum(mulnegativebatch, dim=2)
            ),
            dim=1)
        loss = positivebatch + negativebatch
        return -torch.mean(loss)
    def save_model(self, path_to_save, loss_data) :
        print("\nDone training, saving model to {}".format(path_to_save))
        torch.save(self, "{}".format(path_to_save))
        
        print("Saving loss data at {}".format('line_loss_evolution'))
        with open('line_loss_evolution', "wb") as ldata:
            pickle.dump(lossdata, ldata)
    
    def get_embeddings(self, edges, device):
        if device.type == 'cuda' : edges = edges.cuda()
        emb_1 = self.nodes_embeddings(edges[:, 0]).data().detach().cpu().numpy()  
        emb_2 = self.nodes_embeddings(edges[:, 2]).data().detach().cpu().numpy()
        emb = np.multiply(emb_1, emb_2)
        return emb

if __name__ == "__main__":
    
    args["graph_path"] = 
    args["save_path"] = 'line_model_weights'

    # Hyperparams.
    args["order"] = 2
    args["negsamplesize"] = 5
    args["dimension"] = 64
    args["batchsize"] = 128
    args["epochs"] = 10
    args["learning_rate"] = 0.025  # As starting value in paper
    args["negativepower"] = 0.75 

    # Create dict of distribution when opening file
    edgedistdict, nodedistdict, nodedegrees, maxindex = makeDist(
        args['graph_path'], args['negativepower'])
    
    nb_edges = len(edgedistdict)
    edgesaliassampler = VoseAlias(edgedistdict)
    del edgedistdict
    nodesaliassampler = VoseAlias(nodedistdict)
    del nodedistdict
    import gc; gc.collect()
    
    batchrange = int(nb_edges / args['batchsize'])
#     print(maxindex)
    line = Line(maxindex + 1, embed_dim=args['dimension'], order=args["order"])
    
    opt = optim.SGD(line.parameters(), lr=args['learning_rate'],
                    momentum=0.9, nesterov=True)
    
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    if device.type == 'cuda' : line.cuda()
    lossdata = {"it": [], "loss": []}
    it = 0

    print("\nTraining on {}...\n".format(device))
    for epoch in range(args['epochs']):
        print("Epoch {}".format(epoch))
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
    line.save_model(args['save_path'], loss_data)
## pour charger le modèle
## model = torch.load("line_model_weights")