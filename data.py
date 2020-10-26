from torch.utils.data import Dataset, DataLoader
import numpy as np
import torch
import json
from skimage.transform import resize as imresize
import random


class fMRIData(Dataset):
    """Facebook's fMRI data set"""

    def __init__(self, resolution=256, percentile=99,
                 directory='/store/CCIMI/sl767/fMRI/knee_mri_training/individual_npy.json'):
        '''
        :param resolution: The spatial resolution of the data
        :param percentile: Percentile the data should be thresholded at
        :param directory: Path to a list of training data. The data is expected to be in npy format.
        '''
        self.resolution = resolution
        self.percentile = percentile
        with open(directory, 'r') as fp:
            self.data = json.load(fp)
        print('Number of training samples found', self.__len__())

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        sample = np.load(self.data[idx]).astype(np.float32)

        # Data preprocessing
        sample = imresize(sample, (self.resolution, self.resolution), order=1)
        p = np.percentile(sample, self.percentile)
        sample[sample > p] = p
        sample = sample - sample.min()
        sample = sample / (sample.max()+1e-3)
        sample = np.expand_dims(sample, axis=0)
        return torch.from_numpy(sample)


class fMRIDataEval(fMRIData):
    def __init__(self, resolution=256, percentile=99,
                 directory='/store/CCIMI/sl767/fMRI/knee_mri_eval/individual_npy.json'):
        super(fMRIDataEval, self).__init__(resolution=resolution, percentile=percentile, directory=directory)


class LUNA(Dataset):
    """Facebook's fMRI data set"""

    def __init__(self, resolution=256, percentile=99,
                 directory='/store/CCIMI/sl767/LUNA/Training_Data/individual_npy.json'):
        '''
        :param resolution: The spatial resolution of the data
        :param percentile: Percentile the data should be thresholded at
        :param directory: Path to a list of training data. The data is expected to be in npy format.
        '''
        self.resolution = resolution
        self.percentile = percentile
        with open(directory, 'r') as fp:
            self.data = json.load(fp)
        print('Number of training samples found', self.__len__())

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        sample = np.load(self.data[idx]).astype(np.float32)

        # Data preprocessing
        sample = np.maximum(sample, 0)
        if not len(sample.shape) == 2:
            print('Faulty sample', idx)
            return self.__getitem__(random.randint(0, self.__len__()-1))
        sample = imresize(sample, (self.resolution, self.resolution), order=1)
        p = np.percentile(sample, self.percentile)
        sample[sample > p] = p
        sample = sample / (sample.max()+1e-3)
        sample = np.expand_dims(sample, axis=0)
        return torch.from_numpy(sample)


class LUNAEval(LUNA):
    def __init__(self, resolution=256, percentile=99,
                 directory='/store/CCIMI/sl767/LUNA/Evaluation_Data/individual_npy.json'):
        super(LUNAEval, self).__init__(resolution=resolution, percentile=percentile, directory=directory)