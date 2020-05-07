# Pytorch Deep Learning Package
import torch
from torch.autograd import Variable
import torch.nn.functional as F
import torch.utils.data as Data
from torch.utils.data import Dataset, DataLoader


class AbstractDataset(Dataset):
    def __init__(self, feature_matrix, label_matrix):
        self.data=feature_matrix
        self.label=label_matrix

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):
        # print('\nYee:', self.data[index], ', ' ,self.label[index])
        return self.data[index], self.label[index]
    
    def collate_fn(self, datas):
        # datas = [ batch_size X ( data + label ) ]
        print('\ncollate_fn！')
        print('\ndatas: ', datas)
        # return self.data, self.label