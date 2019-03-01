# Long- and Short-Term Memory (LSTM); Baseline model
from .basenetwork import BaseNetwork
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
import pickle
import os

class LSTMmodel(BaseNetwork):
    def __init__(self, opt):
        super(LSTMmodel, self).__init__(opt)
        self.name = "baseLSTM"
        self._opt = opt
        
        # define parameters
        self._setup_emb()
        self._hidden_size = 64
        self._output_size = 2 # two labels 0,1
        self._dropout_rate = 0.2
        ## below for rnn layer
        self._number_layers = 1
        self._input_size = self._emb_size # 300
        self._cell_hidden_size = 128
        self._cell_dropout = 0 # no dropout in GRU for now
        self._bidirectional = False
        self._num_dir = 2 if self._bidirectional else 1
        self._h_state_vsize = self._number_layers*self._num_dir
        
        # define layers
        self.embed = nn.Embedding(self._vocab_size+2, self._emb_size)
        self.gru = nn.GRU(input_size = self._input_size, 
                          hidden_size = self._cell_hidden_size, 
                          batch_first = True, # (batch, seqlen, embdim)
                          dropout = self._cell_dropout,
                          bidirectional = self._bidirectional
                         )
        self.dropout = nn.Dropout(self._dropout_rate)
        self.linear = nn.Linear(self._cell_hidden_size, self._output_size)
        
        # initialize weights of layers
        self.init_weight()

    def _setup_emb(self):
        emb_path = os.path.join(self._opt.data_dir, self._opt.pretrained_weight_name)
        with open(emb_path, 'rb') as f:
            # pretrained_weight.pkl is np.ndarray
            self.pretrained_weight = torch.from_numpy(pickle.load(f))

        self._vocab_size = self.pretrained_weight.shape[0] - 2
        self._emb_size = self.pretrained_weight.shape[1]
        
    def init_weight(self):
        self.embed.weight.data.copy_(self.pretrained_weight)
        self.embed.weight.requires_grad = self._opt.trainable_emb
        # with other layers defaultly initialized
        
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
