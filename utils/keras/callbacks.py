import os
import pandas as pd
import tensorflow as tf
import shutil

from utils.image import create_dir
from utils.loadAndSaveResults import store_data_frame

LOGS_DRIVE_PATH = '/content/drive/MyDrive/mitosis_data/logs/'
WEIGHTS_DRIVE_PATH = '/content/drive/MyDrive/mitosis_data'
LOGS_CSV_PATH = 'logsCSV'

def early_stop(monitor='val_loss', patience=10):
    # early stop if patience epochs not improve monitor metric/loss
    return tf.keras.callbacks.EarlyStopping(monitor=monitor, patience=patience)


def reduce_lr(monitor='val_loss', factor=0.2, patience=4, min_lr=0.0000001, verbose=1):
    # reduce learning rate by factor if not improve monitor in patience epochs
    return tf.keras.callbacks.ReduceLROnPlateau(monitor=monitor, factor=factor,
                                                patience=patience, min_lr=min_lr, verbose=verbose)


def checkpoint(output_name, monitor='val_loss', verbose=1,  mode='min',
               save_best_only=True, save_weights_only=True):
    # output_path: to save information about model, for example weights
    output_path = os.path.join(WEIGHTS_DRIVE_PATH, '%s.h5' % str(output_name))
    return tf.keras.callbacks.ModelCheckpoint(output_path, monitor=monitor, verbose=verbose,
                                              save_best_only=save_best_only, mode=mode,
                                              save_weights_only=save_weights_only)


def csv_logger(output_name, append=True, separator=';'):
    # callback to save logs of epochs model as csv
    output_path = os.path.join(LOGS_CSV_PATH, '%s.csv' % str(output_name))
    store_data_frame(pd.DataFrame(), output_path)
    return tf.keras.callbacks.CSVLogger(output_path, append=append, separator=separator)


class UpdateLoggerToDrive(tf.keras.callbacks.Callback):
    def __init__(self, N, output_name, output_drive=LOGS_DRIVE_PATH):
        self.N = N
        self.epoch = 0
        self.output_name = str(output_name)
        create_dir(LOGS_CSV_PATH)
        self.logCSV = os.path.join(LOGS_CSV_PATH, '%s.csv' % str(output_name))
        self.output_drive = os.path.join(output_drive, '%s.csv' % self.output_name)

    def on_batch_end(self, epoch, logs={}):
        if self.epoch % self.N == 0:
          shutil.copy(self.logCSV, self.output_drive)
        self.epoch += 1


def get_callbacks_list(output_name, N=5):
    return [early_stop(), reduce_lr(), checkpoint(output_name),
            csv_logger(output_name), UpdateLoggerToDrive(N, output_name)]


