# collection of utility functions
import os
import torch

# make directory if not already existed
def mkdir(target_dir):
    '''
    prepare saving directory
    '''
    if not os.path.exists(target_dir):
        os.makedirs(target_dir)

# function used to find the path of pretrained models & optimizers
def find_file(file_mark, file_dir, epoch_idx):
    '''
    file_mark --- be either opt_{networkname} or net_{networkname}
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

def save_param(obj, save_dir, obj_type, name, epoch_idx):
    '''
    save parameters for network or optimizer
    '''
    fname = '{}_{}_{}_id.pth'.format(obj_type[:3], name, str(epoch_idx))
    save_path = os.path.join(save_dir, fname)
    torch.save(obj.state_dict(), save_path)
    print(" %s %s saved at %s" % (obj_type, name, save_path))

def load_param(obj, load_dir, obj_type, name, epoch_idx):
    '''
    load parameters for network or optimizer
    if idx=-1 load the one with max idx
    '''
    mark = '{}_{}'.format(obj_type[:3], name)
    file_path, idx_num = find_file(mark, load_dir, epoch_idx)
    obj.load_state_dict(torch.load(file_path))
    print(" %s for %s loaded from %s" % (obj_type, name, file_path))
    return idx_num