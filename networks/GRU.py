# Gated Recurrent Unit (GRU); Baseline model
from .basenetwork import BaseNetwork
from .embedding import EmbeddingLayer
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
import pickle
import os

'''model GRU
    base GRU layer, can be: 1- or 2-layer; uni- or bi-directional
'''
class GRUnet(BaseNetwork):
    def __init__(self, opt):
        super(GRUnet, self).__init__(opt)
        self.name = "GRU"

        # define parameters
        self._output_size = 2
        self._dropout_rate = opt.dropout_rate
        ## below for rnn layer
        self._cell_hidden_size = 128
        self._num_layers = opt.number_layers
        self._cell_dropout = 0 # no internal dropout for now
        self._bidirectional = opt.is_bidirectional
        self._num_dir = 2 if self._bidirectional else 1
        self._h_state_vsize = self._num_layers*self._num_dir

        # define layers
        self.embed = EmbeddingLayer(self._opt)
        self._input_size = self.embed.emb_size
        self.gru = nn.GRU(input_size = self._input_size, 
                          hidden_size = self._cell_hidden_size,
                          num_layers = self._num_layers,
                          batch_first = True, # (batch, seqlen, embdim)
                          dropout = self._cell_dropout,
                          bidirectional = self._bidirectional
                         )
        self.dropout = nn.Dropout(self._dropout_rate)
        self.linear = nn.Linear(self._cell_hidden_size*self._num_dir, self._output_size)

        # initialize weights of layers
        self.init_weight()
 
    def init_weight(self):
        # fornow leave layers (except emb) defaultly initialized
        pass
  
    def forward(self, idx_seq, use_encoding=False):
        X = self.embed(idx_seq) # X:(b, seqlen, embdim)
        h = Variable(torch.zeros(self._h_state_vsize, idx_seq.shape[0], self._cell_hidden_size))
        X, h = self.gru(X, h)
        encoding = self.dropout(X[:,-1,:])
        if use_encoding:
            return encoding
        out = self.linear(encoding).view(-1,self._output_size)
        return out
#
