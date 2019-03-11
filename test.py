# test models on (for now) validation set
from options.base_options import BaseOption
from model.baseline import BaselineModel
from model.triplet import TripletModel
from utils.dataloader import load_training_data


class Test:
    def __init__(self):
        # parse param & options
        self._opt = BaseOption().parse()
        self._opt.is_train = False
        assert self._opt.valid_num > 0, '(temp) no data for validation set'

        # prepare dataset
        self._dataset = load_training_data(self._opt)
        self._dataset = self._dataset[-self._opt.valid_num:, :]
        # self._dataset = self._dataset[:self._opt.valid_num, :] #for debug only
        # self._dataset = self._dataset[self._opt.valid_num:2*self._opt.valid_num, :]#for debug only

        # specify model, may load trained parameters within
        if self._opt.is_triplet:
            self._model = TripletModel(self._opt) #not add eval() yet
            print("\nEvaluating model for: \n%s\n%s" % (str(self._model._encoder), str(self._model._classifier)))
        else:
            self._model = BaselineModel(self._opt)
            print("\nEvaluating model for: \n%s" % (str(self._model._classifier)))

        # evaluate model
        print("Validation set size = %d\n" % self._opt.valid_num)
        self._test()

    def _test(self):
        self._model.eval()
        self._model.evaluate(self._dataset)


if __name__ == "__main__":
    Test()
#