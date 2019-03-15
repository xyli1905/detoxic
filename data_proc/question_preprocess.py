# for data pre-processing
import pandas as pd
import numpy as np
import collections
import nltk
import re
import os
import pickle
import time

np.random.seed()

'''methods list:

    load_data_csv(file_name)
    load_data_pkl(file_name)
    dump_data_pkl(data, file_name)
    tokenizer(seqlist)          --seqlist：training/test data--
    build_vocabulary(token, freq_cutoff=1)
    seq_len_dist(token)
    build_data_mat(vocab, *arg) --arg: token, label or token--
'''
class QuestionPreprocessor:
    def __init__(self, sentence_cutoff=60):
        self._name = "question_preprocessor"
        self.sentence_cutoff = sentence_cutoff
        # -------------------- #
        # define in/output dir #
        # -------------------- #
        self.dataset_dir = "/Users/xyli1905/Projects/Datasets/Kaggle/QuoraToxicDetect/"
        self.output_dir = "/Users/xyli1905/Projects/NLP/detoxic/data_proc/processed_data"
        # ---------------------------------------------------- #
        # define regular expression for question preprocessing #
        # ---------------------------------------------------- #
        # terms that are not defined in embedding will be mapped into a single embedding vector, e.g.
        # - unrecognized symbols
        # - characters in chinese & other languages
        #
        # NOTE the following were/will be preprocessed:
        # [] money (maybe not important?)
        # [] dates/time ?how?
        # [] general math expression ?how?
        # [x] numbers --> numbersymb (a preliminary preprocessing)
        # [x] web address / url --> webaddress
        # [x] latex math expressions --> latexmathexpression
        # [x] single/double quotation marks
        # [x] bracketed words
        # [x] split words connected by "/", substitute "/" with " or " (a preliminary preprocessing)
        #
        # for latex math expression
        self._process_latex_expression = re.compile(r"(\[math)((\S|\s)+?)(math\])")
        # for quotation
        self._process_quotation_single = re.compile(r"(\s+|\,\.\?|^)(\')((\s|\S)+?)(\')")
        self._process_quotation_double1 = re.compile(r"(\s+|\,\.\?|^)(\'\')((\s|\S)+?)(\'\')")
        self._process_quotation_double2 = re.compile(r"(\s+|\,\.\?|^)(\")((\s|\S)+?)(\")")
        self._process_quotation_triple = re.compile(r"(\s+|\,\.\?|^)(\'\'\')((\s|\S)+?)(\'\'\')")
        self._process_quotation_special = re.compile(r"(\s+|\,\.\?|^)(\')((\s|\S)+?)(\")")
        # for bracketed words
        self._process_bracketed_words  = re.compile(r"(\s*|^)(\()((\s|\S)+?)(\))")
        # for web address
        self._process_url_http = re.compile(r"(http|https)((\S|\s)+?)(\s+|$)")
        self._process_url_www = re.compile(r"(www)((\S|\s)+?)(\s+|$)")
        # for parallel words, "/" --> "or"
        self._process_double_parallel_words = re.compile(r"(\s+|^)([A-Za-z\-]+)(\/)([A-Za-z\-]+)(\s+|[\.\?\,\!]|$)")
        self._process_triple_parallel_words = re.compile(r"(\s+|^)([A-Za-z\-]+)(\/)([A-Za-z\-]+)(\/)([A-Za-z\-]+)(\s+|[\.\?\,\!]|$)")
        # for numbers 1231 & 2321.231 --> numbsymb
        self._process_number = re.compile(r"(\s+|^)((\d+\.\d+)|(\d+))")

    def load_data_csv(self, file_name="train.csv"):
        '''
        this method is for loading training & test datasets
        '''
        file_path = os.path.join(self.dataset_dir, file_name)
        data = pd.read_csv(file_path)

        data_seq = list(data["question_text"])
        # if training data, then summary on data label
        if file_name == "train.csv":
            data_label = np.array(data["target"])
            dist_dic = collections.Counter(data_label)
            print(" In training data:\n num of Label 0: %s \n num of Label 1: %s \n Toxic percentage: %s %s" \
                  % (dist_dic[0], dist_dic[1], dist_dic[1]/(dist_dic[0] + dist_dic[1])*100., "%"))

            return data, data_seq, data_label

        return data, data_seq, np.array([],dtype=np.int64)

    def load_data_pkl(self, file_name):
        '''
        this method is for loading pre-saved files,
        so we load them from the self.output_dir directory
        those files are: train_token.pkl, train_label.pkl, 
                         test_token.pkl, vocab.pkl
                         train_mat.pkl, test_mat.pkl
        '''
        input_path = os.path.join(self.output_dir, file_name)
        with open(input_path, 'rb') as f:
            data = pickle.load(f)
        return data

    def dump_data_pkl(self, data, file_name):
        output_file = os.path.join(self.output_dir, file_name)
        with open(output_file, 'wb') as f:
            pickle.dump(data, f)
        print(" file %s saved to %s" % (file_name, output_file))

    def tokenizer(self, data_seq):
        t_begin = time.time()
        seq_num = len(data_seq)
        data_token = []
        for i in range(seq_num):
            data_seq[i] = data_seq[i].lower()
            try:
                data_seq[i] = self._RE_preprocessing(data_seq[i])
                data_token.append(nltk.word_tokenize(data_seq[i]))
            except:
                try:
                    data_seq[i] = self._RE_preprocessing_except(data_seq[i])
                    data_token.append(nltk.word_tokenize(data_seq[i]))
                except:
                    print("except raised at idx: %s" % (i))
        t_end = time.time()
        print(" Time for tokenizing all questions is %s (s)" % (t_end - t_begin))
        return data_token

    def build_vocabulary(self, data_token, freq_cutoff=1):
        words = []
        for question in data_token:
            for word in question:
                words.append(word)

        if freq_cutoff > 1:
            fdist = nltk.FreqDist(words)
            words = [word for word, freq in fdist.items() if freq >= freq_cutoff]

        vocab = sorted(set(words))

        print(" built vocabulary with %s words" % (len(vocab)))
        return vocab, words

    def seq_len_dist(self, token):
        seq_num = len(token)
        maxlen, _ = self._max_seq_len(token)
        len_dist = np.zeros(maxlen, dtype = np.int)
        for i in range(seq_num):
            len_dist[len(token[i])-1] += 1
        return len_dist

    def build_data_mat(self, vocab, *arg):
        '''
        arg: either train_token + train_label 
             or test_token alone
        '''
        assert len(arg) >= 1, "not enough input arguments"
        assert len(arg) <= 2, "too many input arguments"

        if isinstance(arg[0], list):
            token_list = arg[0]
        else:
            raise ValueError('second input should be token (list)')

        if len(arg) == 2:
            if isinstance(arg[1], np.ndarray):
                label = arg[1]
            else:
                raise ValueError('third input should be label (np.ndarray)')

        vocab_len = len(vocab)
        idx_unknown = vocab_len
        idx_nothing = vocab_len + 1

        word_to_idx = {word: i for i, word in enumerate(vocab)}

        data_mat = np.full((len(token_list), self.sentence_cutoff), idx_nothing)
        for i, sentence in enumerate(token_list):
            for j, word in enumerate(sentence):
                if j >= self.sentence_cutoff:
                    break
                try:
                    data_mat[i,j] = word_to_idx[word]
                except:
                    data_mat[i,j] = idx_unknown

        if len(arg) == 1:
            return data_mat
        elif len(arg) == 2:
            data_mat = np.concatenate([data_mat, label.reshape(-1,1)], axis=1)
            return data_mat

    def _max_seq_len(self, token):
        '''
        measure in number of words
        '''
        maxlen = 0.
        idx = 0
        seq_num = len(token)
        for i in range(seq_num):
            tmplen = len(token[i])
            if tmplen > maxlen:
                idx = i
                maxlen = tmplen
        return maxlen, idx

    def _RE_preprocessing(self, seq):
        '''
        note we use repr(" "+content[2])[1:-1] to turn the string into raw string
        '''
        # replace latex expression with standard symbol
        seq = self._process_latex_expression.sub("latexmathexpression", seq)

        # replace web address: http(s):... or www. with standard symbol
        seq = self._process_url_http.sub("webaddress ", seq)
        seq = self._process_url_www.sub("webaddress ", seq)

        # check for quotations and delete ''' ''', '' '', ' ' or " " symbols
        match = self._process_quotation_triple.findall(seq)
        for content in match:
            seq = self._process_quotation_triple.sub(repr(" "+content[2])[1:-1], seq, 1)

        match = self._process_quotation_double1.findall(seq)
        for content in match:
            seq = self._process_quotation_double1.sub(repr(" "+content[2])[1:-1], seq, 1)

        match = self._process_quotation_single.findall(seq)
        for content in match:
            seq = self._process_quotation_single.sub(repr(" "+content[2])[1:-1], seq, 1)

        match = self._process_quotation_double2.findall(seq)
        for content in match:
            seq = self._process_quotation_double2.sub(repr(" "+content[2])[1:-1], seq, 1)

        match = self._process_quotation_special.findall(seq)
        for content in match:
            seq = self._process_quotation_special.sub(repr(" "+content[2])[1:-1], seq, 1)

        # check for bracketed content and delete the brackets
        match = self._process_bracketed_words.findall(seq)
        for content in match:
            seq = self._process_bracketed_words.sub(repr(" "+content[2])[1:-1], seq, 1)

        # check parallel words, and replace "/" with " or "
        match = self._process_triple_parallel_words.findall(seq)
        for content in match:
            contentstr = " " + content[1] + " or " + content[3] + " or " + content[5] + " " + content[-1]
            seq = self._process_triple_parallel_words.sub(repr(contentstr)[1:-1], seq, 1)

        match = self._process_double_parallel_words.findall(seq)
        for content in match:
            contentstr = " " + content[1] + " or " + content[3] + " " + content[-1]
            seq = self._process_double_parallel_words.sub(repr(contentstr)[1:-1], seq, 1)

        # replace number e.g. 99 or 99.99 as "numbsymb"
        seq = self._process_number.sub(" numbsymb", seq)

        return seq

    def _RE_preprocessing_except(self, seq):
        '''
        note use this if repr(" "+content[2])[1:-1] raise an exception
        '''
        # replace latex expression with standard symbol
        seq = self._process_latex_expression.sub("latexmathexpression", seq)

        # replace web address: http(s):... or www. with standard symbol
        seq = self._process_url_http.sub("webaddress ", seq)
        seq = self._process_url_www.sub("webaddress ", seq)

        # check for quotations and delete ''' ''', '' '', ' ' or " " symbols
        match = self._process_quotation_triple.findall(seq)
        for content in match:
            seq = self._process_quotation_triple.sub(" "+content[2], seq, 1)

        match = self._process_quotation_double1.findall(seq)
        for content in match:
            seq = self._process_quotation_double1.sub(" "+content[2], seq, 1)

        match = self._process_quotation_single.findall(seq)
        for content in match:
            seq = self._process_quotation_single.sub(" "+content[2], seq, 1)

        match = self._process_quotation_double2.findall(seq)
        for content in match:
            seq = self._process_quotation_double2.sub(" "+content[2], seq, 1)

        match = self._process_quotation_special.findall(seq)
        for content in match:
            seq = self._process_quotation_special.sub(" "+content[2], seq, 1)

        # check for bracketed content and delete the brackets
        match = self._process_bracketed_words.findall(seq)
        for content in match:
            seq = self._process_bracketed_words.sub(" "+content[2], seq, 1)

        # check parallel words, and replace "/" with " or "
        match = self._process_triple_parallel_words.findall(seq)
        for content in match:
            contentstr = " " + content[1] + " or " + content[3] + " or " + content[5] + " " + content[-1]
            seq = self._process_triple_parallel_words.sub(contentstr, seq, 1)

        match = self._process_double_parallel_words.findall(seq)
        for content in match:
            contentstr = " " + content[1] + " or " + content[3] + " " + content[-1]
            seq = self._process_double_parallel_words.sub(contentstr, seq, 1)

        # replace number e.g. 99 or 99.99 as "numbsymb"
        seq = self._process_number.sub("numbsymb", seq)

        return seq


# demo of using QuestionPreprocessor class
def main():
    # initalize question preprocessor
    Qproc = QuestionPreprocessor()

    # load data
    print("\nloading training data ...\n")
    train_data, train_label = Qproc.load_data_csv(file_name="train.csv")
    print("\nloading test data ...")
    test_data, _  = Qproc.load_data_csv(file_name="test.csv")

    # tokenize questions
    print("\ntokenizing trianing dataset ...")
    train_token = Qproc.tokenizer(train_data)
    print("\ntokenizing test dataset ...")
    test_token  = Qproc.tokenizer(test_data)
    Qproc.dump_data_pkl(train_token, file_name="train_token.pkl")
    Qproc.dump_data_pkl(test_token, file_name="test_token.pkl")

    # build vocabulary based on both training & test data
    print("\nbuilding vacabulary from traning+test data ...")
    vocab, _ = Qproc.build_vocabulary(train_token + test_token)
    Qproc.dump_data_pkl(vocab, file_name="vocab.pkl")

    # print seq length distribution (in number of words)
    train_seq_len_dist = Qproc.seq_len_dist(train_token)
    print("\nsequence length distribution for training data (in #words):")
    print(train_seq_len_dist)

    # build train_mat (using train_token & train_label) and test_mat
    print("\nbuilding train_mat from train_token+train_label ...")
    train_mat = Qproc.build_data_mat(vocab, train_token, train_label)
    print("\nbuilding test_mat from test_token ...")
    test_mat = Qproc.build_data_mat(vocab, test_token)
    Qproc.dump_data_pkl(train_mat, file_name="train_mat.pkl")
    Qproc.dump_data_pkl(test_mat, file_name="test_mat.pkl")


if __name__ == '__main__':
    main()
#
