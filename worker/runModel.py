import pandas as pd
import numpy as np
import os
import tensorflow as tf

from tensorflow import keras
from keras.models import load_model
from keras.losses import binary_crossentropy
from dataGeneratorProcess import DataGenerator
from utils.keras.lossAndMetrics import dice_coef
from utils.keras.callbacks import get_callbacks_list_gc
from utils.bucket import upload_object
from google.cloud import storage
from utils.keras.callbacks import get_callbacks_list_gc
from runModel.params import PATH_OF_SECRET_JSON
from utils.augmentation import MAP_AUG_IMG_PIPELINE
from utils.keras.builder import build_unet

# # check if GPU device are available
gpus = tf.config.list_physical_devices('GPU')
if gpus:
    # configure tensorFlow to use GPU computing
    for gpu in gpus:
        tf.config.experimental.set_memory_growth(gpu, True)
    print("GPU detectada. TensorFlow utilizará la GPU!")
else:
    print("No se ha detectado una GPU. TensorFlow utilizará la CPU!")


# function to process message using Google Cloud Pub/Sub system
def train_mitosis_model(expDict):
    print('reading data from ../data')
    df = pd.read_csv('../data/infoDF.csv', sep=';')
    print(df.head(3))

    if 'pretrained_model' in expDict:
        pretrained = expDict['pretrained_model']
        if pretrained == 'unet-best':
            print(f'using pretrained model: {pretrained}')
            model = load_model("../pretrainedModels/UNET-Best.h5", compile=False)
        elif pretrained == 'monuseg':
            print(f'using pretrained model: {pretrained}')
            model = build_unet(n_filters=expDict.get('n_filters', 16),
                               dropout=expDict.get('dropout', 0.3),
                               batchnorm=True)
            # init weights by name to not consider last layer (is 3 channels and desired 1 for mitosis masking...)
            model.load_weights("../pretrainedModels/pretrained-model-monuseg-unet.h5", by_name=True)
        else:
            raise ValueError("pretrained model is not defined on pretrainedModels")

    else:
        print('NOT using pretrained model')
        # define custom UNET
        model = build_unet(n_filters=expDict.get('n_filters', 16),
                           dropout=expDict.get('dropout', 0.3),
                           batchnorm=True)

        # todo: agafar pretrained weights de google cloud ...

    # DEFINE PARAMETERS
    print('defining parameters of model and compile it...')
    lr = expDict.get('learning_rate', 0.001)  # 0.001 by default ...
    opt_adam = keras.optimizers.Adam(learning_rate=lr)
    loss_f = expDict.get('loss', binary_crossentropy)
    model.compile(optimizer=opt_adam, loss=loss_f, metrics=[dice_coef])

    np.random.seed(1234)
    idx = np.random.choice(len(df), int(len(df) * 0.7), replace=False)
    cond = df.index.isin(idx)
    trainDF = df.loc[cond, :]
    valDF = df.loc[~cond, :]

    # TRAIN WITH GENERATORS
    assert 'experiment_name' in expDict, 'need to specify experiment_name'
    EXP_NAME = expDict['experiment_name']
    BATCH_SIZE = expDict.get('batch_size')

    if 'augmentation' in expDict:
        augPipeline = expDict['augmentation']
        assert augPipeline in MAP_AUG_IMG_PIPELINE, 'not detect pipeline on available augmentation pipelines'
        print(f'use augmentations pipeline: {augPipeline}  on train set...')
        augmentation_attr = MAP_AUG_IMG_PIPELINE[augPipeline]
    else:
        print('NOT use augmentations on train set...')
        augmentation_attr = None

    training_generator = DataGenerator(df=trainDF, batch_size=BATCH_SIZE, img_path='../data/images',
                                       mask_path='../data/masks', augmentation=augmentation_attr,
                                       shuffle=True, preprocess=None)  # preprocess None use /255.
    validation_generator = DataGenerator(df=valDF, batch_size=BATCH_SIZE, img_path='../data/images',
                                         mask_path='../data/masks',
                                         preprocess=None, augmentations=None, shuffle=False)

    EPOCH_NUMBER = expDict.get('epoch_number', 10)

    history = model.fit(training_generator, validation_data=validation_generator,
                        verbose=1, epochs=EPOCH_NUMBER,
                        callbacks=get_callbacks_list_gc(EXP_NAME,
                                                        bucket_name='bucket-mitosis',
                                                        firestore_col_name='logs'))


