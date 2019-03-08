# from pretrained embedding data producing pretrained weight for embedding layer
import os
import pickle
import torch
from torch.nn import Embedding
import numpy as np

np.random.seed()

'''method list:

    load_embedding(file_name)
    load_vocabulary_pkl(file_name)
    dump_weight_mat_pkl(data, file_name)
    release_emb_mem(embedding)
    get_pretrained_weight(vocab, embedding)
'''
class EmbeddingPreprocessor:
    def __init__(self):
        self._name = "embedding_preprocessor"
        self.embedding_dir = "/Users/xyli1905/Projects/embeddings/"
        self.procdata_dir = "/Users/xyli1905/Projects/NLP/detoxic/data_proc/processed_data"
        self.vocab_dir = self.procdata_dir
        self.output_dir = self.procdata_dir
        self.emb_dim = -1

    def _find_file(self, fname, fpath):
        for root, dirs, files in os.walk(fpath):
            if fname in files:
                return os.path.join(root, fname)

        raise RuntimeError("%s not found in %s" % (fname, fpath))

    def _name_parser(self, file_name):
        '''
        return wanted loader for the embedding;
        presently only support Glove
        '''
        # predefined key words for distinguishing embeddings
        key_glove = "glove"
        key_googlenews = "GoogleNews"
        key_paragram = "paragram"
        key_wiki = "wiki"

        emb_path = self._find_file(file_name, self.embedding_dir)

        if key_glove in file_name:
            embloader = self._load_glove
            dim_string = file_name.split(".")[-2]
            self.emb_dim = int(dim_string[:-1])
        else:
            raise NotImplementedError('For now, only support Glove embedding')

        return embloader, emb_path

    def load_embedding(self, file_name="glove.840B.300d.txt"):
        embloader, emb_path = self._name_parser(file_name)
        embedding = embloader(emb_path)
        return embedding

    def _load_glove(self, glove_path):
        if self.emb_dim == -1:
            raise ValueError("self.emb_dim not updated properly")

        emb_dim = self.emb_dim
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

    def load_vocabulary_pkl(self, file_name="vocab.pkl"):
        # load existed vocabulary
        vocab_path = self._find_file(file_name, self.vocab_dir)
        with open(vocab_path, 'rb') as f:
            vocab = pickle.load(f)
        return vocab

    def dump_weight_mat_pkl(self, data, file_name):
        output_file = os.path.join(self.output_dir, file_name)
        with open(output_file, 'wb') as f:
            pickle.dump(data, f)
        print(" file %s saved to %s" % (file_name, output_file))

    def release_emb_mem(self, embedding):
        '''
        assumeing emb is a dictionary
        '''
        assert isinstance(embedding, dict)
        print("\nclear embedding memory")
        embedding.clear()

    def _get_embedding_vec(self, embedding, word):
        try:
            embedding_vec = embedding[word]
        except:
            embedding_vec = np.random.normal(scale=0.6, size=(self.emb_dim, ))
            
        return embedding_vec

    def get_pretrained_weight(self, vocab, embedding):
        if self.emb_dim == -1:
            raise ValueError('get_pretrained_weight must be called after loading embedding')

        # based on vocab getting the needed part from pretrained embedding
        # if a word in vocab but not in embedding, assign it a random vec
        pretrained_weight = np.ndarray((len(vocab), self.emb_dim))
        for i, word in enumerate(vocab):
            pretrained_weight[i] = self._get_embedding_vec(embedding, word)

        # then handle the case, when a word is not in vacab, namley "unknown" word;
        # randomly initialize an "unknown_vec"
        unknown_vec = np.random.normal(scale=0.6, size=(self.emb_dim, ))

        # at last handle the placeholder for nothing,
        # since we require all input seq have the same length;
        # use a zero vector "nothing_vec"
        nothing_vec = np.zeros(self.emb_dim)

        # add unknown_vec and nothing_vec at the end of pretrained_weight rows
        # and return
        return np.vstack([pretrained_weight, unknown_vec, nothing_vec])


# demo of using EmbeddingPreprocessor class
def main():
    # initialize preprocessor
    Eproc = EmbeddingPreprocessor()

    # load data
    print("\nloading pretrained embedding ...")
    embedding = Eproc.load_embedding(file_name="glove.840B.300d.txt")
    print("\nloading pre-built vocabulary ...")
    vocab = Eproc.load_vocabulary_pkl(file_name="vocab.pkl")

    # get pretrained_weight
    print("\nextracting pretrained weight ...")
    pretrained_weight = Eproc.get_pretrained_weight(vocab, embedding)
    # np.save("pretrained_weight", pretrained_weight)
    Eproc.dump_weight_mat_pkl(pretrained_weight, file_name="pretrained_weight.pkl")

    # test for pretrained weight
    # emb_dim = Eproc.emb_dim
    # embed = Embedding(len(vocab), emb_dim)
    # embed.weight.data.copy_(torch.from_numpy(pretrained_weight))

    # word_to_idx = {word: i for i, word in enumerate(vocab)}

    # test_word = "card"
    # torch_word_idx = torch.tensor(word_to_idx[test_word], dtype=torch.long)

    # print("\ntest for output of embedding layer")
    # print(embed(torch_word_idx))
    # print(embedding[test_word])

    # # release mem used by embedding (~5.4G)
    # Eproc.release_emb_mem(embedding)


if __name__ == '__main__':
    main()
#
