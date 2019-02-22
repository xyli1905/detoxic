# pass (basic) parameters to the model
import argparse
import os

class BaseOption:
    def __init__(self):
        self._parser = argparse.ArgumentParser()
        self._initialized = False

    def initialize(self):
        # directory options
        self._parser.add_argument('--chkp_dir', type=str,
                                  default='/Users/xyli1905/Projects/NLP/detoxic/checkpoints',
                                  help='directory storing trained models and optimizers')
        self._parser.add_argument('--data_dir', type=str,
                                  default='/Users/xyli1905/Projects/NLP/detoxic/data_proc/processed_data',
                                  help='directory storing preprocessed data')
        self._parser.add_argument('--model_type', type=str,
                                  default='baseline',
                                  help='type of model: baseline, rnn, encoder-decoder')

        # data options
        self._parser.add_argument('--vocab_name', type=str, default='vocab.pkl', help='file name for processed vocabulary')
        self._parser.add_argument('--pretrained_weight_name', type=str, default='pretrained_weight.pkl',
                                  help='file name for processed pretrained weight')
        self._parser.add_argument('--trainable_emb', type=bool, default=False, help='whether allow update pretrained embedding')

        # general options for models
        self._parser.add_argument('--max_epoch', type=int, default=1, help='number of epochs for training')
        self._parser.add_argument('--batch_size', type=int, default=64, help='number of data points in one batch')
        self._parser.add_argument('--save_freq', type=int, default=1, help='frequency (/epoch) for saving model')
        self._parser.add_argument('--load_epoch_idx', type=int, default=0, help='idx of epoch for loading')
        self._parser.add_argument('--lr', type=float, default=0.01, help='learning rate')

        # options for RNN models


        self._initialized = True

    def parse(self):
        if not self._initialized:
            self.initialize()

        self._opt = self._parser.parse_args(args=[])

        args = vars(self._opt)

        # save args to file
        self._save(args)

        return self._opt

    def _save(self, args):
        expr_dir = os.path.join(self._opt.chkp_dir, self._opt.model_type)

        #prepare saving directory
        if not os.path.exists(expr_dir):
            os.makedirs(expr_dir)

        file_name = os.path.join(expr_dir, 'option_list.txt' )
        with open(file_name, 'wt') as opt_file:
            opt_file.write('------------ Options -------------\n')
            for k, v in sorted(args.items()):
                opt_file.write('%s: %s\n' % (str(k), str(v)))
            opt_file.write('-------------- End ----------------\n')