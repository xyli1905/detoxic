# for data pre-processing
import pandas as pd
import numpy as np
import collections
import nltk
import re
import os
import time

class Preprocessor:
    def __init__(self):
        # -------------------- #
        # define in/output dir #
        # -------------------- #
        self.dataset_dir = "/Users/xyli1905/Projects/Datasets/Kaggle/QuoraToxicDetect/"
        self.output_dir = "/Users/xyli1905/Projects/NLP/detoxic/data_proc/"
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
        self.process_latex_expression = re.compile(r"(\[math)((\S|\s)+?)(math\])")
        # for quotation
        self.process_quotation_single = re.compile(r"(\s+|\,\.\?|^)(\')((\s|\S)+?)(\')")
        self.process_quotation_double1 = re.compile(r"(\s+|\,\.\?|^)(\'\')((\s|\S)+?)(\'\')")
        self.process_quotation_double2 = re.compile(r"(\s+|\,\.\?|^)(\")((\s|\S)+?)(\")")
        self.process_quotation_triple = re.compile(r"(\s+|\,\.\?|^)(\'\'\')((\s|\S)+?)(\'\'\')")
        self.process_quotation_special = re.compile(r"(\s+|\,\.\?|^)(\')((\s|\S)+?)(\")")
        # for bracketed words
        self.process_bracketed_words  = re.compile(r"(\s*|^)(\()((\s|\S)+?)(\))")
        # for web address
        self.process_url_http = re.compile(r"(http|https)((\S|\s)+?)(\s+|$)")
        self.process_url_www = re.compile(r"(www)((\S|\s)+?)(\s+|$)")
        # for parallel words, "/" --> "or"
        self.process_double_parallel_words = re.compile(r"(\s+|^)([A-Za-z\-]+)(\/)([A-Za-z\-]+)(\s+|[\.\?\,\!]|$)")
        self.process_triple_parallel_words = re.compile(r"(\s+|^)([A-Za-z\-]+)(\/)([A-Za-z\-]+)(\/)([A-Za-z\-]+)(\s+|[\.\?\,\!]|$)")
        # for numbers 1231 & 2321.231 --> numbsymb
        self.process_number = re.compile(r"(\s+|^)((\d+\.\d+)|(\d+))")

    def load_data(self, file_name="train.csv"):
        file_path = os.path.join(self.dataset_dir, file_name)
        data = pd.read_csv(file_path)

        # if training data, then summary on data label
        if file_name == "train.csv":
            data_label = np.array(data["target"])
            dist_dic = collections.Counter(data_label)
            print(" In training data:\n num of Label 0: %s \n num of Label 1: %s \n Toxic percentage: %s %s" \
                  % (dist_dic[0], dist_dic[1], dist_dic[1]/(dist_dic[0] + dist_dic[1])*100., "%"))

        data_seq = list(data["question_text"])
        return data_seq

    def dump_data(self, data, file_name):
        output_file = os.path.join(self.output_dir, file_name)
        with open(output_file, 'w') as f:
            for item in data:
                f.write("%s\n" % item)

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

    def build_vocabulary(self, data_token):
        words = []
        for question in data_token:
            for word in question:
                words.append(word)
        vocab = sorted(set(words))
        print(" built vocabulary with %s words" % (len(vocab)))
        return vocab, words

    def seqlen_dist(self, token):
        seq_num = len(token)
        maxlen, _ = self._max_seq_len(token)
        len_dist = np.zeros(maxlen, dtype = np.int)
        for i in range(seq_num):
            len_dist[len(token[i])-1] += 1
        return len_dist

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
        seq = self.process_latex_expression.sub("latexmathexpression", seq)

        # replace web address: http(s):... or www. with standard symbol
        seq = self.process_url_http.sub("webaddress ", seq)
        seq = self.process_url_www.sub("webaddress ", seq)
        
        # check for quotations and delete ''' ''', '' '', ' ' or " " symbols
        match = self.process_quotation_triple.findall(seq)
        for content in match:
            seq = self.process_quotation_triple.sub(repr(" "+content[2])[1:-1], seq, 1)
        
        match = self.process_quotation_double1.findall(seq)
        for content in match:
            seq = self.process_quotation_double1.sub(repr(" "+content[2])[1:-1], seq, 1)

        match = self.process_quotation_single.findall(seq)
        for content in match:
            seq = self.process_quotation_single.sub(repr(" "+content[2])[1:-1], seq, 1)
            
        match = self.process_quotation_double2.findall(seq)
        for content in match:
            seq = self.process_quotation_double2.sub(repr(" "+content[2])[1:-1], seq, 1)
            
        match = self.process_quotation_special.findall(seq)
        for content in match:
            seq = self.process_quotation_special.sub(repr(" "+content[2])[1:-1], seq, 1)
            
        # check for bracketed content and delete the brackets
        match = self.process_bracketed_words.findall(seq)
        for content in match:
            seq = self.process_bracketed_words.sub(repr(" "+content[2])[1:-1], seq, 1)

        # check parallel words, and replace "/" with " or "
        match = self.process_triple_parallel_words.findall(seq)
        for content in match:
            contentstr = " " + content[1] + " or " + content[3] + " or " + content[5] + " " + content[-1]
            seq = self.process_triple_parallel_words.sub(repr(contentstr)[1:-1], seq, 1)
        
        match = self.process_double_parallel_words.findall(seq)
        for content in match:
            contentstr = " " + content[1] + " or " + content[3] + " " + content[-1]
            seq = self.process_double_parallel_words.sub(repr(contentstr)[1:-1], seq, 1)
            
        # replace number e.g. 99 or 99.99 as "numbsymb"
        seq = self.process_number.sub(" numbsymb", seq)
        
        return seq

    def _RE_preprocessing_except(self, seq):
        '''
        note use this if repr(" "+content[2])[1:-1] raise an exception
        '''
        # replace latex expression with standard symbol
        seq = self.process_latex_expression.sub("latexmathexpression", seq)

        # replace web address: http(s):... or www. with standard symbol
        seq = self.process_url_http.sub("webaddress ", seq)
        seq = self.process_url_www.sub("webaddress ", seq)
        
        # check for quotations and delete ''' ''', '' '', ' ' or " " symbols
        match = self.process_quotation_triple.findall(seq)
        for content in match:
            seq = self.process_quotation_triple.sub(" "+content[2], seq, 1)
        
        match = self.process_quotation_double1.findall(seq)
        for content in match:
            seq = self.process_quotation_double1.sub(" "+content[2], seq, 1)

        match = self.process_quotation_single.findall(seq)
        for content in match:
            seq = self.process_quotation_single.sub(" "+content[2], seq, 1)
            
        match = self.process_quotation_double2.findall(seq)
        for content in match:
            seq = self.process_quotation_double2.sub(" "+content[2], seq, 1)
            
        match = self.process_quotation_special.findall(seq)
        for content in match:
            seq = self.process_quotation_special.sub(" "+content[2], seq, 1)
            
        # check for bracketed content and delete the brackets
        match = self.process_bracketed_words.findall(seq)
        for content in match:
            seq = self.process_bracketed_words.sub(" "+content[2], seq, 1)
            
        # check parallel words, and replace "/" with " or "
        match = self.process_triple_parallel_words.findall(seq)
        for content in match:
            contentstr = " " + content[1] + " or " + content[3] + " or " + content[5] + " " + content[-1]
            seq = self.process_triple_parallel_words.sub(contentstr, seq, 1)
        
        match = self.process_double_parallel_words.findall(seq)
        for content in match:
            contentstr = " " + content[1] + " or " + content[3] + " " + content[-1]
            seq = self.process_double_parallel_words.sub(contentstr, seq, 1)
            
        # replace number e.g. 99 or 99.99 as "numbsymb"
        seq = self.process_number.sub("numbsymb", seq)
        
        return seq


def main():
    # initalize preprocessor
    QuestionPreprocessor = Preprocessor()

    # load data
    print("\nloading training data ...\n")
    train_data = QuestionPreprocessor.load_data(file_name="train.csv")
    print("\nloading test data ...")
    test_data  = QuestionPreprocessor.load_data(file_name="test.csv")

    # tokenize questions
    print("\ntokenizing trianing dataset ...")
    train_token = QuestionPreprocessor.tokenizer(train_data)
    print("\ntokenizing test dataset ...")
    test_token  = QuestionPreprocessor.tokenizer(test_data)

    # build vocabulary based on both training & test data
    print("\nbuilding vacabulary from traning+test data ...")
    vocab, _ = QuestionPreprocessor.build_vocabulary(train_token + test_token)
    QuestionPreprocessor.dump_data(vocab, file_name="vocab_test.txt")

    # print seq length distribution (in number of words)
    train_seqlan_dist = QuestionPreprocessor.seqlen_dist(train_token)
    print("\nsequence length distribution for training data (in #words):")
    print(train_seqlan_dist)


if __name__ == '__main__':
    main()
#
