import tensorflow as tf
import tensorflow.keras.backend as K
from keras.losses import binary_crossentropy

# ref: https://github.com/nauyan/Segmentation/blob/master/Code/utils/lossfunctions.py
# ref: https://github.com/nauyan/Segmentation/blob/master/Code/utils/metricfunctions.py
# ref: dice coefficient: https://www.kaggle.com/yerramvarun/understanding-dice-coefficient

# The smooth variable is used to help with backpropagation by not getting the term close to zero  \
# as well as preventing overfitting with a relatively large value.
def dice_coef(y_true, y_pred, smooth=1.):
    y_true_f = K.flatten(y_true)
    y_pred_f = K.flatten(y_pred)
    intersection = K.sum(y_true_f * y_pred_f)
    return (2. * intersection + smooth) / (K.sum(y_true_f) + K.sum(y_pred_f) + smooth)


def old_dice_loss(y_true, y_pred):
    numerator = 2 * tf.reduce_sum(y_true * y_pred, axis=(1, 2, 3))
    denominator = tf.reduce_sum(y_true + y_pred, axis=(1, 2, 3))
    return 1 - numerator / denominator


def dice_loss(y_true, y_pred):
    return 1 - dice_coef(y_true, y_pred)


def bce_dice_loss(y_true, y_pred):
    return binary_crossentropy(y_true, y_pred) + dice_loss(y_true, y_pred)


def focal_loss(y_true, y_pred, alpha=0.25, gamma=2):
    inputs = K.flatten(y_pred)
    targets = K.flatten(y_true)

    BCE = K.binary_crossentropy(targets, inputs)
    BCE_EXP = K.exp(-BCE)
    focal_loss = K.mean(alpha * K.pow((1 - BCE_EXP), gamma) * BCE)
    return focal_loss


def _jaccard_loss(y_true, y_pred, smooth=100):
    """
    Jaccard = (|X & Y|)/ (|X|+ |Y| - |X & Y|)
            = sum(|A*B|)/(sum(|A|)+sum(|B|)-sum(|A*B|))

    The jaccard distance loss is usefull for unbalanced datasets. This has been
    shifted so it converges on 0 and is smoothed to avoid exploding or disapearing
    gradient.

    Ref: https://en.wikipedia.org/wiki/Jaccard_index

    @url: https://gist.github.com/wassname/f1452b748efcbeb4cb9b1d059dce6f96
    @author: wassname
    """
    intersection = K.sum(K.abs(y_true * y_pred), axis=-1)
    sum_ = K.sum(K.abs(y_true) + K.abs(y_pred), axis=-1)
    jac = (intersection + smooth) / (sum_ - intersection + smooth)
    return (1 - jac) * smooth


# https://www.kaggle.com/code/bigironsphere/loss-function-library-keras-pytorch/notebook
def focal_tversky_loss(targets, inputs, alpha=0.5, beta=3, gamma=1, smooth=1e-6):   # alpha=0.5=beta=0.5 is focal loss

        #flatten label and prediction tensors
        inputs = K.flatten(inputs)
        targets = K.flatten(targets)

        #True Positives, False Positives & False Negatives
        TP = K.sum((inputs * targets))
        FP = K.sum(((1-targets) * inputs))
        FN = K.sum((targets * (1-inputs)))

        Tversky = (TP + smooth) / (TP + alpha*FP + beta*FN + smooth)
        FocalTversky = K.pow((1 - Tversky), gamma)

        return FocalTversky


def jaccard_loss(y_true, y_pred, smooth=5):
    y_true_flat = K.flatten(y_true)
    y_pred_flat = K.flatten(y_pred)

    intersection = K.sum(K.abs(y_true_flat * y_pred_flat))
    union = K.sum(K.abs(y_true_flat) + K.abs(y_pred_flat)) - intersection
    jaccard = (intersection + smooth) / (union + smooth)
    return (1 - jaccard) * smooth


