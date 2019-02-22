# collection of utility functions
import os



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