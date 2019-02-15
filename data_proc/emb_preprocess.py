# from pretrained embedding data producing pretrained weight for embedding layer
import os
import torch
from torch.nn import Embedding
import numpy as np

np.random.seed()

class EmbeddingPreprocessor:
    def __init__(self, emb_name="glove.840B.300d.txt", vocab_name="vocab.txt"):
        self.embedding_dir = "/Users/xyli1905/Projects/embeddings/"
        self.vocab_dir = "/Users/xyli1905/Projects/NLP/detoxic/data_proc/"
        self._name = "embdding_preprocessor"
        self.emb_name = emb_name
        self.emb_path = ""
        self.emb_dim = -1
        self.vocab_name = vocab_name
        self.vocab_path = ""

    def _find_path(self, fname, fpath):
        for root, dirs, files in os.walk(fpath):
            if fname in files:
                return os.path.join(root, fname)

    def _name_parser(self):
        # return wanted loader for the embedding
        # predefined key words for distinguishing embeddings
        key_glove = "glove"
        key_googlenews = "GoogleNews"
        key_paragram = "paragram"
        key_wiki = "wiki"

        self.emb_path = self._find_path(self.emb_name, self.embedding_dir)

        if key_glove in self.emb_name:
            embloader = self.loadGlove
            # later will add auto detect function for emb_dim
            #emb_param = {"emb_dim": 300}
            self.emb_dim = 300

            return embloader
        else:
            raise NotImplementedError('embeddings other than Glove are not implemented yet')

    def loadEmbedding(self):
        embloader = self._name_parser()
        embedding = embloader()
        return embedding

    def loadGlove(self):
        emb_dim = self.emb_dim
        glove_path = self.emb_path
        glove_file = open(glove_path, "r")
        glove = {}
        for line in glove_file.readlines():
            try:
                splitLine = line.split()
                word = " ".join(splitLine[:-emb_dim])
                embedding = np.array([float(val) for val in splitLine[-emb_dim:]])
                glove[word] = embedding
            except:
                print(line)
        return glove

    def release_mem(self, embedding):
        '''
        assumeing emb is a dictionary
        '''
        print("\nclearing embedding ...")
        embedding.clear()

    def loadVocab(self):
        # load existed vocabulary
        self.vocab_path = self._find_path(self.vocab_name, self.vocab_dir)
        with open(self.vocab_path, 'r') as f:
            vocab = [word.split("\n")[0] for word in f.readlines()]
        return vocab

    def get_embedding_vec(self, embedding, word):
        #emb_dim = emb_param["emb_dim"]
        try:
            embedding_vec = embedding[word]
        except:
            embedding_vec = np.random.normal(scale=0.6, size=(self.emb_dim, ))
            
        return embedding_vec

    def get_pretrained_weight(self, embedding, vocab):
        if self.emb_dim == -1:
            raise ValueError('get_pretrained_weight must be used after loading embedding')
        # based on vocab getting the needed part from pretrained embedding
        pretrained_weight = np.ndarray((len(vocab), self.emb_dim))
        for i, word in enumerate(vocab):
            pretrained_weight[i] = self.get_embedding_vec(embedding, word)

        return pretrained_weight


def main():
    # initialize preprocessor
    Preprocessor = EmbeddingPreprocessor(emb_name="glove.840B.300d.txt", vocab_name="vocab.txt")

    # load data
    print("\nloading pretrained embedding ...")
    embedding = Preprocessor.loadEmbedding()
    print("\nloading pre-built vocabulary ...")
    vocab = Preprocessor.loadVocab()

    # get pretrained_weight
    print("\nextracting pretrained weight ...")
    pretrained_weight = Preprocessor.get_pretrained_weight(embedding, vocab)
    np.save("pretrained_weight", pretrained_weight)

    # test for pretrained weight
    emb_dim = Preprocessor.emb_dim
    embed = Embedding(len(vocab), emb_dim)
    embed.weight.data.copy_(torch.from_numpy(pretrained_weight))

    word_to_idx = {word: i for i, word in enumerate(vocab)}

    test_word = "card"
    torch_word_idx = torch.tensor(word_to_idx[test_word], dtype=torch.long)

    print("\ntest for output of embedding layer")
    print(embed(torch_word_idx))
    print(embedding[test_word])

    # release mem used by embedding (~5.4G)
    Preprocessor.release_mem(embedding)


if __name__ == '__main__':
    main()
#
