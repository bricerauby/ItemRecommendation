import os 
import json

from gensim.models.callbacks import CallbackAny2Vec

class callback(CallbackAny2Vec):
    '''Callback to print loss after each epoch.'''

    def __init__(self, all_loss):
        self.epoch = 0
        self.loss_to_be_subed = 0
        self.all_loss = all_loss

    def on_epoch_end(self, model):
        loss = model.get_latest_training_loss()
        loss_now = loss - self.loss_to_be_subed
        self.loss_to_be_subed = loss
        self.all_loss.append(loss_now)
        print('Loss after epoch {}: {}'.format(self.epoch, loss_now))
        self.epoch += 1

class Embedding():
    
    def __init__(self, graph):
        self.graph = graph
        self.num_nodes = graph.number_of_nodes()
        self.word_vectors = None
        self.vectors_path = None
        self.walks = None
        self.walks_path = None

    def set_paths(self, name):
        if not os.path.exists(name):
            os.makedirs(name)
        self.vectors_path = '{}/embedding.json'.format(name)
        self.walks_path = '{}/walks.json'.format(name)

    def get_embedding(self):
        return self.embedding

    def save_embedding(self):
        with open(self.vectors_path, 'w') as f:
            json.dump(self.word_vectors, f)

    def load_embedding(self):
        with open(self.vectors_path, 'r') as f:
            self.word_vectors = json.load(f)

    def save_walks(self):
        with open(self.walks_path, 'w') as f:
            json.dump(self.walks, f)

    def load_walks(self):
        with open(self.walks_path, 'r') as f:
            self.walks = json.load(f)