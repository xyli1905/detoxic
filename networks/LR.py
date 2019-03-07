from .basenetwork import BaseNetwork
from .embedding import EmbeddingLayer
import torch
import torch.nn as nn
import torch.nn.functional as F
import pickle
import os

'''model BoW
    the Bag of Word LR model (word count vectorization)
'''
class BoW(BaseNetwork):
    def __init__(self, opt):
        super(BoW, self).__init__(opt)
        self.name = "BoW"

        # define parameters
        self._load_vocab()
        self._output_size = 2
        
        # define layers
        self.W = nn.Parameter(torch.randn(self._vocab_size+2, self._output_size))
        self.b = nn.Parameter(torch.zeros(self._output_size))

    def _load_vocab(self):
        vocab_path = os.path.join(self._opt.data_dir, self._opt.vocab_name)
        with open(vocab_path, 'rb') as f:
            # vocab.pkl is np.ndarray
            self.vocab = pickle.load(f)
        self._vocab_size = len(self.vocab)
    
    def weighted_words(self, seq):
        '''
        weighted sum of word-count voectors

        idx_seq : input torch tensor of dtype torch.long
        '''
        output = torch.zeros((seq.shape[0], self._output_size))
        for i in range(seq.shape[0]):
            idxseq = seq[i]
            output[i] = torch.sum(self.W[idxseq],0) + self.b
        return output
            
    def forward(self, idx_seq):
        '''
        idx_seq : input torch tensor of dtype torch.long
        
        Relevant dim parameters:
         batchsize, sentencecutoff or B,S in short
        '''
        out = self.weighted_words(idx_seq) # idx_seq:(B, S), out:(B, 2)
        return out
    

'''model EmbBoW
    modified Bag of Word model, with words represented by embedding vector
'''
class EmbBoW(BaseNetwork):
    def __init__(self, opt):
        super(EmbBoW, self).__init__(opt)
        self.name = "EmbBoW"
        
        # define parameters
        self._hidden_size = 64
        self._output_size = 2

        # define layers
        self.embed = EmbeddingLayer(self._opt)
        self.W = nn.Parameter(torch.randn(self.embed.vocab_size + 2))
        self.linear1 = nn.Linear(self._emb_size, self._hidden_size)
        self.linear2 = nn.Linear(self._hidden_size, self._output_size)
        
        # initialize weights of layers
        self.init_weight()
        
    def init_weight(self):
        # fornow leave layers (except emb) defaultly initialized
        pass
    
    def weighted_embed(self, x, seq):
        '''
        Weighted sum of embedding vectors

        x : embedding vector from previous layer
        idx_seq : input torch tensor of dtype torch.long
        '''
        output = torch.zeros((seq.shape[0], 1, x.shape[2]))
        for i in range(seq.shape[0]):
            idxseq = seq[i:i+1]
            output[i] = torch.mm(self.W[idxseq], x[i])
        return output

    def forward(self, idx_seq, use_encoding=False):
        '''
        idx_seq : input torch tensor of dtype torch.long; 

        Relevant dim parameters:
         batchsize, sentencecutoff, embedsize or B,S,E in short
        '''
        X = self.embed(idx_seq) # idx_seq:(B, S), X:(B, S, E)
        X = self.weighted_embed(X, idx_seq) # X: (B, E)
        encoding = F.relu(self.linear1(X))
        if use_encoding:
            return encoding
        out = self.linear2(encoding).view(-1,self._output_size)
        return out
    

'''model EmbLR
    further modification of Bag of Word model, weight are not assigned to
    a word but been calculated based on its embedding vector
    roughly speaking, an no-memory RNN model
'''
class EmbLR(BaseNetwork):
    def __init__(self, opt):
        super(EmbLR, self).__init__(opt)
        self.name = "EmbLR"
        
        # define layer size
        self._hidden_size = 128#64 debug
        self._output_size = 2
        
        # define layers
        self.embed = EmbeddingLayer(self._opt)
        self.W = nn.Parameter(torch.randn(self.embed.emb_size, 1))
        self.linear1 = nn.Linear(self.embed.emb_size, self._hidden_size)
        self.linear2 = nn.Linear(self._hidden_size, self._output_size)
        
        # initialize weights of layers
        self.init_weight()
        
    def init_weight(self):
        # fornow leave layers (except emb) defaultly initialized
        pass

    def emb_selfcorr(self, x):
        '''
        basically do matmul(emb.T, emb), a kind of a self-correlation
        '''
        output = torch.zeros((x.shape[0], self.embed.emb_size, self.embed.emb_size))
        for i in range(x.shape[0]):
            output[i] = torch.mm(torch.t(x[i]), x[i])
        return output
        
    def forward(self, idx_seq, use_encoding=False):
        '''
        idx_seq : input torch tensor of dtype torch.long

        Relevant dim parameters:
         batchsize, sentencecutoff, embedsize or B,S,E in short
        '''
        X = self.embed(idx_seq)  # emb vec X:(B, S, E)
        X = self.emb_selfcorr(X) # give corr mat: (B, E, E)
        X = torch.squeeze(torch.matmul(X, self.W)) # weighted sum of embed: (B, E)
        encoding = F.relu(self.linear1(X))
        if use_encoding:
            return encoding
        out = self.linear2(encoding).view(-1,self._output_size)
        return out
#






