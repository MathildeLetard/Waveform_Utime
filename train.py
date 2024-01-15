from __future__ import print_function, division
import scipy
import datetime
import matplotlib.pyplot as plt
import sys
import numpy as np
import os
import h5py
import keras
import tensorflow as tf
from tensorflow.keras.regularizers import l2
from tensorflow.keras.layers import Input, Concatenate, BatchNormalization, Activation, UpSampling1D, Conv1D,  MaxPool1D
from tensorflow.keras.models import Model, load_model
from tensorflow.keras.optimizers import schedules, SGD
from keras import backend as K
from dataloader import *
from default import *
from sklearn.utils import class_weight


def compute_weights(ndataset,list_out):
    dataset=h5py.File(ndataset,'r')
    Y=dataset['labels'][:]
    dataset.close()
    return class_weight.compute_class_weight('balanced', classes=np.unique(np.ravel(Y,order='C')), y=np.ravel(Y,order='C'))

def read_labels(file):
    data = np.load(file)
    data = data.reshape((data.shape[0], data.shape[2]))
    im=np.zeros((data.shape[0],data.shape[1],len(list_out)))
    for i in range(len(list_out)):
        imt=np.where(data[:,:]==list_out[i],1,0)
        im[:,:,i]=imt
    return im

def weighted_categorical_crossentropy(weights):
    """
    A weighted version of keras.objectives.categorical_crossentropy

    Variables:
        weights: numpy array of shape (C,) where C is the number of classes

    Usage:
        weights = np.array([0.5,2,10]) # Class one at 0.5, class 2 twice the normal weights, class 3 10x.
        loss = weighted_categorical_crossentropy(weights)
    """
    weights = K.variable(weights)
    def loss(y_true, y_pred):
        # scale predictions so that the class probas of each sample sum to 1
        y_pred /= K.sum(y_pred, axis=-1, keepdims=True)
        # clip to prevent NaN's and Inf's
        y_pred = K.clip(y_pred, K.epsilon(), 1 - K.epsilon())
        # calc
        loss = y_true * K.log(y_pred) * weights
        loss = -K.sum(loss, axis=-1)
        return loss
    return loss


class temporal_network():
    def __init__(self,
                 dataset_train,
                 dataset_val,
                 list_out,
                 wf_len=DEF_LEN,
                 channels_out=DEF_CH_OUT,
                 channels_in=DEF_CH_IN,
                 filepath_save=DEF_PATH_SAVE,
                 mode_norm=DEF_MODE_NORM,
                 initial_lr=DEF_INITIAL_LR,
                 decay_rate=DEF_DECAY_RATE,
                 decay_steps=DEF_DECAY_STEPS,
                 gf=DEF_FILTERS):
        self.dataset_train=dataset_train
        self.dataset_val=dataset_val
        self.list_out=list_out
        self.wf_len = wf_len
        self.channels_out = channels_out
        self.channels_in = channels_in
        self.mode_norm=mode_norm
        self.initial_lr = initial_lr
        self.decay_rate = decay_rate
        self.decay_steps = decay_steps
        self.mode_norm = mode_norm
        self.filepath_save = filepath_save
        self.train_loader=DataLoader(self.dataset_train)
        self.val_loader=DataLoader(self.dataset_val)
        self.gf = gf
        self.name_readme='%s/%s'%(self.filepath_save,DEF_README)
        self.cost=[]
        self.valid=[]
        self.epo=[]
        self.alpha = compute_weights(self.dataset_train,self.list_out)

        fid = open(self.name_readme, "a")
        fid.write('--------------------\n')
        fid.write('Network Optimization\n')
        fid.write('--------------------\n')
        fid.write('Initial learning rate : %.6f\n' % self.initial_lr)
        fid.write('Decay rate : %.6f\n' % self.decay_rate)
        fid.write('Decay steps : %d\n' % self.decay_steps)
        fid.close()

        lr_schedule = tf.keras.optimizers.schedules.ExponentialDecay(
            initial_learning_rate=self.initial_lr,
            decay_steps=self.decay_steps,
            decay_rate=self.decay_rate)
        # Build the network
        self.tempcnn = self.multiclass_unet1D()
        self.tempcnn.compile(loss=weighted_categorical_crossentropy(self.alpha),
                               optimizer=tf.keras.optimizers.SGD(lr=self.initial_lr, momentum=0.9),
                               metrics=[weighted_categorical_crossentropy(self.alpha)])

    def multiclass_unet1D(self):
        def conv_block(inputs, filters, pool=True):
            x = Conv1D(filters, 3, padding="same")(inputs)
            x = BatchNormalization()(x)
            x = Activation("relu")(x)

            x = Conv1D(filters, 3, padding="same")(x)
            x = BatchNormalization()(x)
            x = Activation("relu")(x)

            if pool == True:
                p = MaxPool1D(2)(x)
                return x, p
            else:
                return x
        inputs = Input((self.wf_len, self.channels_in))

        """ Encoder """
        x1, p1 = conv_block(inputs, self.gf, pool=True)
        x2, p2 = conv_block(p1, self.gf*2, pool=True)
        x3, p3 = conv_block(p2, self.gf*3, pool=True)
        x4, p4 = conv_block(p3, self.gf*4, pool=True)

        """ Bridge """
        b1 = conv_block(p4, self.gf*8, pool=False)

        """ Decoder """
        u1 = UpSampling1D(2)(b1)
        c1 = Concatenate()([u1, x4])
        x5 = conv_block(c1, 4*self.gf, pool=False)

        u2 = UpSampling1D(2)(x5)
        c2 = Concatenate()([u2, x3])
        x6 = conv_block(c2, 3*self.gf, pool=False)

        u3 = UpSampling1D(2)(x6)
        c3 = Concatenate()([u3, x2])
        x7 = conv_block(c3, self.gf*2, pool=False)

        u4 = UpSampling1D(2)(x7)
        c4 = Concatenate()([u4, x1])
        x8 = conv_block(c4, self.gf, pool=False)

        """ Output layer """
        output = Conv1D(self.channels_out, 1, padding="same", activation="softmax")(x8)

        return Model(inputs, output)


    def train(self,epochs, batch_size):
        start_time = datetime.datetime.now()
        moy_loss = 0
        Niter_epoch=int(self.train_loader.nb_ech/batch_size)+1

        for iter in range(epochs*Niter_epoch):
            X, y = self.train_loader.load_batch(batch_size)
            g_loss = self.tempcnn.train_on_batch(X, y)
            elapsed_time = datetime.datetime.now() - start_time
            eval_g_loss =g_loss[0]
            toprint = "[%.7d] [time: %s] --gen losses : %f " % (iter, elapsed_time, eval_g_loss)
            sys.stdout.write(toprint + chr(13))
            moy_loss = moy_loss + eval_g_loss / Niter_epoch
            # If at save interval => save generated image samples
            if iter % Niter_epoch == 0 and iter > 3:
                print('\n')
                self.chek_training(iter, moy_loss)
                moy_loss = 0
        self.dataset_train.close()
        self.dataset_val.close()

    def chek_training(self,iter,gloss=0):
        self.epo.append(iter)
        lossf=0.
        Xval, Yval = self.val_loader.load_data()
        Ypredict = self.tempcnn.predict(Xval)
        valid_loss=K.mean(weighted_categorical_crossentropy(self.alpha)(Yval,Ypredict))

        eval_valid_loss=K.eval(valid_loss)
        self.cost.append(gloss)
        self.valid.append(eval_valid_loss)

        fig=plt.plot(self.epo,self.cost)
        fig=plt.plot(self.epo,self.valid)
        fig=plt.legend(['loss','valid'])
        name_fig='%s/graph.png'%(self.filepath_save)
        fig.figure.savefig(name_fig)
        fig=plt.close()

        print('Iter : %.8d -- Av generator loss : %f ; Valid  LOSS : %f '%(iter,gloss, eval_valid_loss))
        self.tempcnn.save('%s/models/model_%.8d_loss_%.5f_val_%.5f.h5'%(self.filepath_save, iter,gloss,eval_valid_loss))
        self.plot_res(Ypredict,Yval,Xval,iter)


    def plot_res(self,Ypredict,Yval,Xval,iter):
        Ytemp = Ypredict.argmax(axis=-1).astype(np.uint8)
        labels_pred = np.zeros(Ytemp.shape)
        for i in range(len(list_out)):
            labels_pred = np.where(Ytemp == i, list_out[i], labels_pred)

        Ytemp = Yval.argmax(axis=-1).astype(np.uint8)
        labels_true = np.zeros(Ytemp.shape)
        for i in range(len(list_out)):
            labels_true = np.where(Ytemp == i, list_out[i], labels_true)

        for i in range(0, labels_true.shape[0], 50):
            indices_pred = np.where(labels_pred[i]!=0)[0]
            indices_true = np.where(labels_true[i]!=0)[0]
            fig = plt.figure()
            fig, axs = plt.subplots(2)
            fig.tight_layout(pad=2)
            axs[0].plot(Xval[i,:], color='k')
            axs[1].plot(Xval[i,:], color='k')
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


            name_fig='%s/valid/valid_%.3d_iter_%.5d.png'%(self.filepath_save,i,iter)
            fig.savefig(name_fig)
            plt.close('all')



if __name__ == '__main__':
    import argparse

    parser = argparse.ArgumentParser(
        description="Waveform semantic segmentation")
    parser.add_argument("--path_train", required=True,
                        help="path to h5 train file")
    parser.add_argument("--path_val", required=True,
                        help="path to h5 val file")
    parser.add_argument("--batch_size", type=int, default=DEF_BATCH_SIZE,
                        help="Size of the batch (default : %d)"%DEF_BATCH_SIZE)
    parser.add_argument("--epochs", type=int, default=DEF_EPOCHS,
                        help="Number of epochs (default : %d)"%DEF_EPOCHS)
    parser.add_argument("--wflen", type=int, default=DEF_LEN,
                        help="numbers of columns of low res (default : %d)"%DEF_LEN)
    parser.add_argument("--ch_out", type=int, default=DEF_CH_OUT,
                        help="channels in output (default : %d)"%DEF_CH_OUT)
    parser.add_argument("--ch_in", type=int, default=DEF_CH_IN,
                        help="channels in input (default : %d)"%DEF_CH_IN)
    parser.add_argument("--gpu_ids", type=str, default=DEF_CUDA,
                        help="priority for GPU (default : %s)"%DEF_CUDA)
    parser.add_argument("--path_model",  type=str, default=DEF_PATH_SAVE,
                        help="path to save models ")
    parser.add_argument("--pretrain", type=str, default=DEF_PATH_PRETRAIN,
                        help="pretrained model (default : %s)"%DEF_PATH_PRETRAIN)
    parser.add_argument("--norm", default=DEF_MODE_NORM,
                        help="normalisation method (def : %s)"%DEF_MODE_NORM)
    parser.add_argument("--gf", type=int, default=DEF_FILTERS,
                        help="number of filter for the residual part (default : %d)"%DEF_FILTERS)
    parser.add_argument("--initial_lr", type=float, default=DEF_INITIAL_LR,
                        help="Initial learning rate (default : %f)"%DEF_INITIAL_LR)
    parser.add_argument("--decay_steps", type=int, default=DEF_DECAY_STEPS,
                        help="Decay steps (default : %d)"%DEF_DECAY_STEPS)
    parser.add_argument("--decay_rate", type=float, default=DEF_DECAY_RATE,
                        help="Decay rate (default : %f)"%DEF_DECAY_RATE)
    parser.add_argument("--note", type=str, default='',
                        help="Personal note for the readme file ")

    args = parser.parse_args()
    dataset_train=args.path_train
    dataset_val=args.path_val
    batch_size = args.batch_size
    mode_norm=args.norm
    epochs = int(args.epochs)
    wflen=int(args.wflen)
    ch_out = int(args.ch_out)
    ch_in = int(args.ch_in)
    path_model = args.path_model
    cuda_id=args.gpu_ids
    gf= args.gf
    note=args.note
    decay_steps=args.decay_steps
    decay_rate=args.decay_rate
    initial_lr=args.initial_lr

    # Several GPUs
    pretrain=args.pretrain
    config = tf.ConfigProto()
    config.gpu_options.allow_growth = True
    os.environ["CUDA_VISIBLE_DEVICES"] = str(cuda_id[0])  # Or 2, 3, etc. other than 01
    physical_devices = tf.config.experimental.list_physical_devices('GPU')
    tf.config.experimental.set_memory_growth(physical_devices[0], True)
    session = tf.Session(config=config)

    if not os.path.exists(path_model):
        os.makedirs(path_model)
    if not os.path.exists('%s/valid'%path_model):
        os.makedirs('%s/valid'%path_model)
    if not os.path.exists('%s/models'%path_model):
        os.makedirs('%s/models'%path_model)
    if not os.path.exists('%s/python'%path_model):
        os.makedirs('%s/python'%path_model)
    commande='cp *.py %s/python'%path_model
    os.system(commande)
    name_readme='%s/%s'%(path_model,DEF_README)
    fid = open(name_readme, "w")
    fid.write('--------------------\n')
    fid.write('Input parameters\n')
    fid.write('Dataset train (h5)) : %s\n'%dataset_train)
    fid.write('Dataset val (h5)) : %s\n'%dataset_val)
    fid.write('Batch size : %d\n'%batch_size)
    fid.write('Mode norm : %s\n'%mode_norm)
    fid.write('Epochs : %d\n'%epochs)
    fid.write('Size: %d\n'%wflen)
    fid.write('Channels (in, out)  : %d x %d \n'%(ch_in,ch_out))
    fid.write('Number of filters  : %d \n'%(gf))
    fid.write('--------------------\n')
    fid.write('Losses\n')
    fid.write('--------------------\n')
    fid.write('--------------------\n')
    if pretrain is not '':
        fid.write('Pretrain\n')
        fid.write('--------------------\n')
        fid.write(pretrain)
        fid.write('--------------------\n')
    if note is not '':
        fid.write('Note : %s\n'%note)
        fid.write('--------------------\n')
    fid.write('--------------------\n')
    fid.write('Command\n')
    fid.write(' '.join(sys.argv))
    fid.write('\n')
    fid.write('--------------------\n')
    fid.close()
    net = temporal_network(dataset_train=dataset_train,
                 dataset_val=dataset_val,
                 list_out=list_out,
                 wf_len=wflen,
                 channels_out=ch_out,
                 channels_in=ch_in,
                 filepath_save=path_model,
                 mode_norm=mode_norm,
                 initial_lr=initial_lr,
                 decay_rate=decay_rate,
                 decay_steps=decay_steps,
                 gf=gf)
    if os.path.exists(pretrain):
        print('---------------------------------')
        print('load pretrain model %s'%pretrain)
        net.tempcnn=load_model(pretrain,custom_objects=DEF_CUSTOM_OBJECTS)
        print('Done')
        lr_schedule = tf.keras.optimizers.schedules.ExponentialDecay(
            initial_learning_rate=initial_lr,
            decay_steps=decay_steps,
            decay_rate=decay_rate)
        net.tempcnn.compile(loss=weighted_categorical_crossentropy(alpha),
                              optimizer=tf.keras.optimizers.SGD(lr=lr_schedule, momentum=0.9),
                              metrics=["categorical_accuracy"])
        print('---------------------------------')
    else:
        print('---------------------------------')
        print('Train from scratch')
        print('---------------------------------')


    stringlist = []
    net.tempcnn.summary(print_fn=lambda x: stringlist.append(x))
    short_model_summary = "\n".join(stringlist)
    print(short_model_summary)

    fid = open(name_readme, "a")
    fid.write('--------------------\n')
    fid.write('model\n')
    fid.write(short_model_summary)
    fid.close()
    net.train(epochs=epochs, batch_size=batch_size)
    net.train_loader.close()
    net.val_loader.close()
