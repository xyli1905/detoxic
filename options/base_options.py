# pass (basic) parameters to the model
import argparse
import os

class BaseOption:
    def __init__(self):
        self._parser = argparse.ArgumentParser()
        self._initialized = False

    def initialize(self):
        # directory options
        self._parser.add_argument('--chkp_dir', type=str, default='./checkpoints', help='directory storing trained models and optimizers')
        self._parser.add_argument('--data_dir', type=str, default='./data_proc/processed_data', help='directory storing preprocessed data')

        # model options
        self._parser.add_argument('--model_type', type=str, default='baseline', help='type of model: baseline, rnn, encoder-decoder')
        self._parser.add_argument('--network_name', type=str, default='GRU', help='name of the model used')

        # data options
        self._parser.add_argument('--train_data_name', type=str, default='train_mat.pkl', help='name for training data')
        # self._parser.add_argument('--test_data_name', type=str, default='test_mat.pkl', help='name for test data')
        self._parser.add_argument('--vocab_name', type=str, default='vocab.pkl', help='file name for processed vocabulary')
        self._parser.add_argument('--pretrained_weight_name', type=str, default='pretrained_weight.pkl',
                                  help='file name for processed pretrained weight')
        self._parser.add_argument('--trainable_emb', type=self.boolean_string, default=False, help='whether allow update pretrained embedding')

        # general options for training
        self._parser.add_argument('--is_train', type=self.boolean_string, default=True, help='flag showing if the model is in training')
        self._parser.add_argument('--max_epoch', type=int, default=1, help='number of epochs for training')
        self._parser.add_argument('--load_epoch_idx', type=int, default=0, help='idx of epoch for loading')
        self._parser.add_argument('--batch_size', type=int, default=64, help='number of data points in one batch')
        self._parser.add_argument('--lr', type=float, default=0.01, help='learning rate')
        #for test we take the last valid_num many data as the validation set
        self._parser.add_argument('--valid_num', type=int, default=100000, help='size of validation set')

        # options for save and display
        self._parser.add_argument('--save_freq', type=int, default=1, help='frequency (/epoch) for saving model')
        self._parser.add_argument('--loss_check_freq', type=int, default=120000, help='frequency (/iters) for outputing loss')
        self._parser.add_argument('--max_loss_check', type=int, default=10, help='upper bound for number of loss check')

        # options for debug
        self._parser.add_argument('--is_debug', type=self.boolean_string, default=False, help='flag for debug')
        self._parser.add_argument('--debug_dir', type=str, default='debug', help='name dir that stores debug outputs')

        # options for triplet loss
        self._parser.add_argument('--triplet', type=self.boolean_string, default=False, help='whether use triplet loss in training')
        self._parser.add_argument('--margin', type=float, default=0.04, help='margin in triplet loss')
        self._parser.add_argument('--iter_size', type=int, default=100000, help='number of triplets in a epoch')

        self._initialized = True

    def boolean_string(self, s):
        if s not in {'False', 'True'}:
            raise ValueError('Not a valid boolean string')
        return s == 'True'

    def parse(self):
        if not self._initialized:
            self.initialize()

        # self._opt = self._parser.parse_args(args=[]) # for jupyter input
        self._opt = self._parser.parse_args()

        # save args to file
        args = vars(self._opt)
        self._save(args)

        # create debug folder if need
        if self._opt.is_debug:
            print("running debuging mode")
            base_path = os.path.join(self._opt.chkp_dir, self._opt.model_type)
            debug_dir = os.path.join(base_path, self._opt.debug_dir)
            if not os.path.exists(debug_dir):
                os.makedirs(debug_dir)
            self._opt.debug_dir = debug_dir
        else:
            print("running normal mode")

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