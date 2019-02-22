# define embedding layer, (may) load pretrained embeddings
# import torch
# import torch.nn as nn
# import numpy as np


# class EmbeddingLayer(nn.Module):
#     def __init__(self, vocabulary_size, embedding_size):
#         super(EmbeddingLayer, self).__init__()
#         # ---------------------- #
#         # define local variables #
#         # ---------------------- #
#         self._name = "embedding_layer"
#         self.weight_data_path = "/Users/xyli1905/Projects/NLP/detoxic/data_proc/pretrained_weight.npy"
#         # ------------- #
#         # define layers #
#         # ------------- #
#         self._emb = nn.Embedding(vocabulary_size, embedding_size)

#     # either randomly initialized -> embedding_type = "random"
#     # or load pretrained weights,
#     # choices for pretrained embeddings:
#     #       glove.840B.300d.                ->  embedding_type = "glove"
#     #       GoogleNews-vectors-negative300  ->  embedding_type = "google"
#     #       paragram_300_sl999              ->  embedding_type = "paragram"
#     #       wiki-news-300d-1M               ->  embedding_type = "wiki"
#     def initialize(self):
#         # initial version
#         pretrained_weight = np.load(self.weight_data_path)
#         self._emb.weight.data.copy_(torch.from_numpy(pretrained_weight))

#     def forward(self, inputs):
#         embeds = self._emb(inputs)
#         #embeds = self._emb(inputs).view((1, -1))
#         return embeds

# def loadVocab():
#     # load existed vocabulary
#     vocab_path =  "/Users/xyli1905/Projects/NLP/detoxic/data_proc/vocab.txt"
#     with open(vocab_path, 'r') as f:
#         vocab = [word.split("\n")[0] for word in f.readlines()]
#     return vocab

# def main():

#     vocab = loadVocab()
#     word_to_idx = {word: i for i, word in enumerate(vocab)}

#     vocabulary_size = len(vocab)
#     embedding_size = 300

#     emblayer = EmbeddingLayer(vocabulary_size, embedding_size)
#     emblayer.initialize()

#     test_word = "card"
#     inputs = torch.tensor(word_to_idx[test_word], dtype=torch.long)

#     result = emblayer.forward(inputs)
#     print(result)

# if __name__ == '__main__':
#     main()