import numpy as np
import os
from default import *
import h5py

class DataLoader():
    def __init__(self, ndataset):
        self.ndataset = ndataset
        self.dataset=h5py.File(ndataset,'r')
        self.nb_ech =self.dataset['waveforms'].shape[0]

    def load_batch(self,batch_size):
        ibatch = np.sort(np.random.choice(self.nb_ech, size=batch_size, replace=False))
        X = self.dataset['waveforms'][ibatch]
        X = X.reshape((X.shape[0], X.shape[1], 2))
        X = X[:,:512,:]
        X = X / DEF_NORM
        labels = self.dataset['labels'][ibatch]
        labels = labels.reshape((labels.shape[0], labels.shape[2]))
        labels = labels[:,:512]
        Y = np.zeros((labels.shape[0], labels.shape[1], len(list_out)))
        for i in range(len(list_out)):
            Yt = np.where(labels[:, :] == list_out[i], 1, 0)
            Y[:, :, i] = Yt
        return X,Y

    def load_data(self):
        X = self.dataset['waveforms'][:]
        X = X.reshape((X.shape[0], X.shape[1], 2))
        X = X[:,:512,:]
        X = X / DEF_NORM
        labels = self.dataset['labels'][:]
        labels = labels.reshape((labels.shape[0], labels.shape[2]))
        labels = labels[:,:512]
        Y = np.zeros((labels.shape[0], labels.shape[1], len(list_out)))
        for i in range(len(list_out)):
            Yt = np.where(labels[:, :] == list_out[i], 1, 0)
            Y[:, :, i] = Yt
        return X,Y

    def load_test_data(self):
        X = self.dataset['waveforms'][:]
        X = X.reshape((X.shape[0], X.shape[1], 2))
        X = X[:,:512,:]
        X = X / DEF_NORM
        return X

    def load_test_labels(self):
        labels = self.dataset['labels'][:]
        labels = labels.reshape((labels.shape[0], labels.shape[2]))
        labels = labels[:,:512]
        Y = np.zeros((labels.shape[0], labels.shape[1], len(list_out)))
        for i in range(len(list_out)):
            Yt = np.where(labels[:, :] == list_out[i], 1, 0)
            Y[:, :, i] = Yt
        return Y

    def close(self):
        self.dataset.close()
