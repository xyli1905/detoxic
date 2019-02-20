from .basenetwork import BaseNetwork
import torch
import torch.nn as nn
import torch.nn.functional as F
import pickle
import os

'''NOTICE
    All model below inherit class BaseNetwork, 
    which provide common methods: predict, save and load
'''

'''model BoW
    this is the Bag of Word LR model
'''
class BoW(BaseNetwork):
    def __init__(self, trainable_emb=False):
        super(BoW, self).__init__()
        self.name = "BoW"
        self.trained = False
        self.training_times = 0
        self._chkp_dir = "/Users/xyli1905/Projects/NLP/detoxic/checkpoints"
        self._data_dir = "/Users/xyli1905/Projects/NLP/detoxic/data_proc/processed_data"
        self._vocab_name = "vocab.pkl"
        
        # define parameters
        self._load_vocab()
        self.output_size = 2 # two labels 0,1
        
        # define layers
        self.W = nn.Parameter(torch.randn(self.vocab_size+2, self.output_size))
        self.b = nn.Parameter(torch.zeros(self.output_size))

    def _load_vocab(self):
        vocab_path = os.path.join(self._data_dir, self._vocab_name)
        with open(vocab_path, 'rb') as f:
            # vocab.pkl is np.ndarray
            self.vocab = pickle.load(f)

        self.vocab_size = len(self.vocab)
    
    def weighted_words(self, seq):
        '''
        note idx_seq must be a torch tensor of dtype torch.long
        N is batch size, 60 is the limit of sentence len
        below, W[idx_seq]:(N, 60), output:(N, 1, 300)
        '''
        output = torch.zeros((seq.shape[0], self.output_size))
        for i in range(seq.shape[0]):
            idxseq = seq[i]
            output[i] = torch.sum(self.W[idxseq],0) + self.b
        return output
            
    def forward(self, idx_seq):
        '''
        note idx_seq must be a torch tensor of dtype torch.long
        N is batch size, 60 is the limit of sentence len
        '''
        out = self.weighted_words(idx_seq) # idx_seq:(N, 60), out:(N, 2)
        return out
    

'''model EmbBoW

'''
class EmbBoW(BaseNetwork):
    def __init__(self, trainable_emb=False):
        super(EmbBoW, self).__init__()
        self.name = "EmbBoW"
        self.trained = False
        self.training_times = 0
        self._chkp_dir = "/Users/xyli1905/Projects/NLP/detoxic/checkpoints"
        self._data_dir = "/Users/xyli1905/Projects/NLP/detoxic/data_proc/processed_data"
        self._pretrained_weight_name = "pretrained_weight.pkl"
        
        # define parameters
        self._setup_emb()
        self.trainable_emb = trainable_emb
        self.hidden_size = 64
        self.output_size = 2 # two labels 0,1
        ##self-defined model parameter
        self.W = nn.Parameter(torch.randn(self.vocab_size+2))
        
        # define layers
        self.embed   = nn.Embedding(self.vocab_size+2, self.emb_size)
        self.linear1 = nn.Linear(self.emb_size, self.hidden_size)
        self.linear2 = nn.Linear(self.hidden_size, self.output_size)
        
        # initialize weights of layers
        self.init_weight()

    def _setup_emb(self):
        emb_path = os.path.join(self._data_dir, self._pretrained_weight_name)
        with open(emb_path, 'rb') as f:
            # pretrained_weight.pkl is np.ndarray
            self.pretrained_weight = torch.from_numpy(pickle.load(f))

        self.vocab_size = self.pretrained_weight.shape[0] - 2
        self.emb_size = self.pretrained_weight.shape[1]
        
    def init_weight(self):
        self.embed.weight.data.copy_(self.pretrained_weight)
        self.embed.weight.requires_grad = self.trainable_emb
        # with other layers defaultly initialized
    
    def weighted_embed(self, x, seq):
        '''
        note idx_seq must be a torch tensor of dtype torch.long
        N is batch size, 60 is the limit of sentence len
        below, W[idx_seq]:(N, 60), output:(N, 1, 300)
        '''
        output = torch.zeros((seq.shape[0], 1, x.shape[2]))
        for i in range(seq.shape[0]):
            idxseq = seq[i:i+1]
            output[i] = torch.mm(self.W[idxseq], x[i])
        return output
            
    def forward(self, idx_seq):
        '''
        note idx_seq must be a torch tensor of dtype torch.long
        N is batch size, 60 is the limit of sentence len
        '''
        X = self.embed(idx_seq) # idx_seq:(N, 60), X:(N, 60, 300)
        X = self.weighted_embed(X, idx_seq)
        X = F.relu(self.linear1(X))
        out = self.linear2(X).view(-1,self.output_size)
        return out
    

'''model EmbLR
    note in this notebook, vocab & Pweight are global variables
    in this new version, we set W -> (1,300),
    namely use embeddings to identify words & assign them weights
'''
class EmbLR(BaseNetwork):
    def __init__(self, trainable_emb=False):
        super(EmbLR, self).__init__()
        self.name = "EmbLR"
        self.trained = False
        self.training_times = 0
        self._chkp_dir = "/Users/xyli1905/Projects/NLP/detoxic/checkpoints"
        self._data_dir = "/Users/xyli1905/Projects/NLP/detoxic/data_proc/processed_data"
        self._pretrained_weight_name = "pretrained_weight.pkl"
        
        # define parameters
        self._setup_emb()
        self.trainable_emb = trainable_emb
        self.hidden_size = 64
        self.output_size = 2 # two labels 0,1
        ##self-defined model parameter
        self.W = nn.Parameter(torch.randn(self.emb_size, 1))
        
        # define layers
        self.embed   = nn.Embedding(self.vocab_size+2, self.emb_size)
        self.linear1 = nn.Linear(self.emb_size, self.hidden_size)
        self.linear2 = nn.Linear(self.hidden_size, self.output_size)
        
        # initialize weights of layers
        self.init_weight()

    def _setup_emb(self):
        emb_path = os.path.join(self._data_dir, self._pretrained_weight_name)
        with open(emb_path, 'rb') as f:
            # pretrained_weight.pkl is np.ndarray
            self.pretrained_weight = torch.from_numpy(pickle.load(f))

        self.vocab_size = self.pretrained_weight.shape[0] - 2
        self.emb_size = self.pretrained_weight.shape[1]
        
    def init_weight(self):
        self.embed.weight.data.copy_(self.pretrained_weight)
        self.embed.weight.requires_grad = self.trainable_emb
        # with other layers defaultly initialized

    def emb_selfcorr(self, x):
        '''
        basically do matmul(E.T, E), dim=(300, 300), kind of a self-correlation
        '''
        output = torch.zeros((x.shape[0], self.emb_size, self.emb_size))
        for i in range(x.shape[0]):
            output[i] = torch.mm(torch.t(x[i]), x[i])
        return output
        
    def forward(self, idx_seq):
        '''
        note idx_seq must be a torch tensor of dtype torch.long
        N is batch size, 60: sentence_cutoff, 300: embedding_dim
        input idx_seq:(N, 60)
        '''
        X = self.embed(idx_seq)  # emb vec X:(N, 60, 300)
        X = self.emb_selfcorr(X) # give corr mat: (N, 300, 300)
        X = torch.squeeze(torch.matmul(X, self.W))   # weighted sum of embeddings (N, 300)
        X = F.relu(self.linear1(X))
        out = self.linear2(X).view(-1,self.output_size)
        return out   
#






