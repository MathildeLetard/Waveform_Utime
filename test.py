from __future__ import print_function, division
import os
import sys
import scipy
import datetime
import keras
import matplotlib.pyplot as plt
from dataloader import *
import numpy as np
import tensorflow as tf
from default import *
from dataloader import *
from keras.models import Model

class test():
    def __init__(self,
            model_name,
            dataset_name=''):
        self.model_name=model_name
        self.dataset_name = dataset_name
        print('Charging model...')
        self.model=tf.keras.models.load_model(model_name, custom_objects=DEF_CUSTOM_OBJECTS)
        print('Done')
        self.test_loader = DataLoader(self.dataset_name)
        self.data_test = self.test_loader.load_test_data()
        self.labels_test = self.test_loader.load_test_labels()

    def test_data(self):
        pred = self.model.predict(self.data_test)
        labelstemp = pred.argmax(axis=-1).astype(np.uint8)
        labels=np.zeros(labelstemp.shape)
        for i in range(len(list_out)):
            labels = np.where(labelstemp==i, list_out[i], labels)
        return labels

    def save_data(self,pred,output_folder=None):
        np.save(output_folder, pred)

    def plot_res(self,pred):
        Ytemp = pred.argmax(axis=-1).astype(np.uint8)
        labels_pred = np.zeros(Ytemp.shape)
        for i in range(len(list_out)):
            labels_pred = np.where(Ytemp == i, list_out[i], labels_pred)

        Ytemp = self.labels_test.argmax(axis=-1).astype(np.uint8)
        labels_true = np.zeros(Ytemp.shape)
        for i in range(len(list_out)):
            labels_true = np.where(Ytemp == i, list_out[i], labels_true)

        for i in range(0, labels_true.shape[0], 10):
            indices_pred = np.where(labels_pred[i]!=0)[0]
            indices_true = np.where(labels_true[i]!=0)[0]
            fig = plt.figure()
            fig, axs = plt.subplots(2)
            fig.tight_layout(pad=2)
            axs[0].plot(self.data_test[i,:], color='k')
            axs[1].plot(self.data_test[i,:], color='k')
            c = []
            for j in range(len(indices_true)):
                c.append(colours[str(int(labels_true[i,indices_true[j]]))])
            axs[0].vlines(indices_true, 0, 1, colors=c, linewidth=3)
            c = []
            for j in range(len(indices_pred)):
                c.append(colours[str(int(labels_pred[i,indices_pred[j]]))])
            axs[1].vlines(indices_pred, 0, 1, colors=c, linewidth=3)
            axs[0].set_xticks(indices_true)
            axs[0].set_xticklabels(labels_true[i,indices_true], fontsize=10)
            axs[1].set_xticks(indices_pred)
            axs[1].set_xticklabels(labels_pred[i,indices_pred], fontsize=10)
            axs[0].set_title('True labels', fontsize=10)
            axs[1].set_title('Predictions', fontsize=10)
            name_fig='%s/valid/valid_%.3d_iter_%.5d.png'%(self.filepath_save,i,'test')
            fig.savefig(name_fig)
            plt.close('all')


if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser(
        description="Test waveform segmentation")
    parser.add_argument("--data_test", required=True,
                        help="path to h5 folder ")
    parser.add_argument("--model", required=True,
                        help="path to model")
    parser.add_argument("--gpu_ids", nargs='+', type=str, default=DEF_CUDA,
                        help="priority for GPU (default : %s)"%DEF_CUDA)
    parser.add_argument("--output", default=None,
                        help="path to output (if None : same folder as data")
    parser.add_argument("--note", type=str, default='',
                        help="Personal note for the readme file ")
    args = parser.parse_args()
    dataset_name = args.data_test
    model_name = args.model
    output_path = args.output
    cuda_id=args.gpu_ids

    config = tf.ConfigProto()
    config.gpu_options.allow_growth = True
    os.environ["CUDA_VISIBLE_DEVICES"] = str(cuda_id[0])  # Or 2, 3, etc. other than 0
    physical_devices = tf.config.experimental.list_physical_devices('GPU')
    tf.config.experimental.set_memory_growth(physical_devices[0], True)
    session = tf.Session(config=config)
    if output_path is not None:
        if not os.path.isdir(output_path):
            os.makedirs(output_path)
    model_test = test(model_name,dataset_name)
    print('')
    pred=model_test.test_data()
    model_test.save_data(pred,output_folder=output_path)
