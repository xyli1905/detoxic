from .basenetwork import BaseNetwork
import torch
import torch.nn as nn
import torch.nn.functional as F
import pickle
import os

'''NOTE
now only support loading pretrained Embeddings, e.g. Glove, wiki, ... 
Later: Bert

use a seperate class for embeddinglayer is for the purpose of easy 
modification and extension to different ways of using different embeddings
'''
class EmbeddingLayer(BaseNetwork):
    def __init__(self, opt):
        super(EmbeddingLayer, self).__init__(opt)
        self.name = "Embedding"
        self._load_pretrianed_emb()
        self._emb = nn.Embedding(self.vocab_size + 2, self.emb_size)
        self._initialize()

    def _load_pretrianed_emb(self):
        '''
        load from pretrained embedding (Type: np.ndarray); choosing in baseoption
        '''
        emb_path = os.path.join(self._opt.data_dir, self._opt.pretrained_weight_name)
        with open(emb_path, 'rb') as f:
            # pretrained_weight.pkl is np.ndarray
            self.pretrained_weight = torch.from_numpy(pickle.load(f))

        self.vocab_size = self.pretrained_weight.shape[0] - 2
        self.emb_size = self.pretrained_weight.shape[1]

    def _initialize(self):
        '''
        set pretrained Embeddings (e.g. Glove, Bert);
        '''
        self._emb.weight.data.copy_(self.pretrained_weight)
        self._emb.weight.requires_grad = self._opt.is_emb_trainable #normally False

    def forward(self, input):
        return self._emb(input)
