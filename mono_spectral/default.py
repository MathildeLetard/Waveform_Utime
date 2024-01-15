import keras
from keras import backend as K

list_out = [0,1,2,3,4,5,6,7,8,9,10]
colours = {'0':'w', '1':'#83acde','2':'#164073', '3':'#69073a', '4':'#577522','5':'#0f7770','6':'#53546b','7':'#01472a','8':'#15472c','9':'#d69122','10':'#76aaab'}

DEF_BATCH_SIZE = 5
DEF_EPOCHS = 500
DEF_INTERVAL = 10
DEF_CUDA = "0"
DEF_LEN = 512
DEF_CH_OUT = len(list_out)
DEF_CH_IN = 1
DEF_PATH_SAVE='./'
DEF_PATH_PRETRAIN=''
DEF_MODE_NORM='01'
DEF_README='readme_model.txt'
DEF_INITIAL_LR=0.001
DEF_DECAY_RATE=0.9
DEF_DECAY_STEPS=600000
DEF_FILTERS=16
DEF_NORM = 1.

#paste weights here
alpha=[9.12132340e-02, 4.18386108e+02, 4.11704266e+02, 3.35798870e+02,
 4.00391007e+02, 3.98013388e+02, 4.17344051e+02, 3.97070228e+02,
 1.06898652e+02, 4.21014162e+02, 1.38367990e+02]

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


DEF_CUSTOM_OBJECTS={'loss':weighted_categorical_crossentropy(alpha),
                    'weighted_categorical_crossentropy':weighted_categorical_crossentropy}
