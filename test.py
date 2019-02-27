# test models on (for now) validation set
from options.base_options import BaseOption
from utils import util

class Test:
    def __init__(self, **jupyter_input):
        # parse param & options
        self._opt = BaseOption().parse()
        assert self._opt.valid_num > 0, '(temp) no data for validation set'

        # jupyter parser is for easy use in jupyter notebook
        if len(jupyter_input) != 0:
            self._model, self._dataset, max_epoch, load_epoch_idx = util.jupyter_parser(jupyter_input)
            if max_epoch > 0:
                self._opt.max_epoch = max_epoch
            if load_epoch_idx > 0:
                self._opt.load_epoch_idx = load_epoch_idx
        else:
            # prepare dataset
            self._dataset = util.load_training_data(self._opt)
            # specify model, may load trained parameters
            self._model = util.set_model(self._opt)

        self._dataset = self._dataset[-self._opt.valid_num:, :]

        if self._opt.load_epoch_idx > 0 or self._opt.load_epoch_idx == -1:
            self._model.load(self._opt.load_epoch_idx)

        # evaluate model
        print("\nEvaluating model: \n%s\n" % (str(self._model)))
        self._test()

    def _test(self):
        self._model.evaluate(self._dataset)


if __name__ == "__main__":
    Test()
#