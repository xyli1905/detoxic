# collection of utility functions
from networks.LR import BoW, EmbBoW, EmbLR
from networks.GRU import GRUmodel
import os
import torch
import pickle


# parse jupyter input, for easy use in jupyter notebook
def jupyter_parser(arg):
    try:
        model = arg["model"]
    except:
        raise ValueError("no Model input")

    try:
        dataset = arg["train_data"]
    except:
        raise ValueError("no Train data input")

    max_epoch=-1
    try:
        max_epoch = arg["max_epoch"]
    except:
        pass

    load_epoch_idx=0
    try:
        load_epoch_idx = arg["load_epoch_idx"]
    except:
        pass

    return model, dataset, max_epoch, load_epoch_idx

# set model for training and test
def set_model(opt):
    model_dict = {"BoW": BoW, "EmbBoW": EmbBoW, "EmbLR": EmbLR, "GRU": GRUmodel}
    try:
        model = model_dict[opt.model_name](opt)
    except:
        raise ValueError('model name not supported')
    return model

# load test data
def load_test_data(opt):
    pass

# load training data
def load_training_data(opt):
    data_path = os.path.join(opt.data_dir, opt.train_data_name)
    with open(data_path, 'rb') as f:
        dataset = torch.from_numpy(pickle.load(f))
    return dataset

# make directory if not already existed
def mkdir(target_dir):
    #prepare saving directory
    if not os.path.exists(target_dir):
        os.makedirs(target_dir)

# function used to find the path of pretrained models & optimizers
def find_file(file_mark, file_dir, epoch_idx):
    '''
    file_mark --- be either opt_{modelname} or net_{modelname}
    epoch_idx --- which epoch to load
    file_dir  --- be the directory for model & optimizer

    return the opt/net file for self._model.name & given idx
    if idx=-1 load the one with max idx
    '''
    # test the validity of input file_mark
    if not ('opt' in file_mark or 'net' in file_mark):
        raise ValueError('util.find_file only applies to net_ or opt_ file')

    if epoch_idx == -1:
        idx_num = -1
        for file in os.listdir(file_dir):
            if file.startswith(file_mark):
                idx_num = max(idx_num, int(file.split('_')[2]))
        assert idx_num >= 0, 'opt file not found'
    else:
        found = False
        for file in os.listdir(file_dir):
            if file.startswith(file_mark):
                found = int(file.split('_')[2]) == epoch_idx
                if found: break
        assert found, 'opt file for epoch %i not found' % epoch_idx
        idx_num = epoch_idx

    fname = '{}_{}_id.pth'.format(file_mark, str(idx_num))
    file_path = os.path.join(file_dir, fname)

    return file_path, idx_num