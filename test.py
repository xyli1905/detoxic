# test models on (for now) validation set
from options.base_options import BaseOption
from model.baseline import BaselineModel
from utils.dataloader import load_training_data


class Test:
    def __init__(self):
        # parse param & options
        self._opt = BaseOption().parse()
        assert self._opt.valid_num > 0, '(temp) no data for validation set'

        # prepare dataset
        self._dataset = load_training_data(self._opt)
        self._dataset = self._dataset[-self._opt.valid_num:, :]

        # specify model, may load trained parameters within
        self._model = BaselineModel(self._opt)

        # evaluate model
        print("\nEvaluating model for: \n%s\n" % (str(self._model._network)))
        print("Validation set size = %d\n" % self._opt.valid_num)
        self._test()

    def _test(self):
        self._model.evaluate(self._dataset)


if __name__ == "__main__":
    Test()
#