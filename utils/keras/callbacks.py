import os
import pandas as pd
import numpy as np
import shutil
import tensorflow as tf

from keras.callbacks import Callback
from google.cloud import firestore, storage
from datetime import datetime
from utils.image import create_dir
from utils.loadAndSaveResults import store_data_frame
from utils.bucket import create_path_on_bucket_if_not_exists

# paths of google drive
LOGS_DRIVE_PATH = '/content/drive/MyDrive/mitosis_data/logs/'
WEIGHTS_DRIVE_PATH = '/content/drive/MyDrive/mitosis_data'
LOGS_CSV_PATH = 'logsCSV'


def early_stop(monitor='val_loss', patience=15):
    # early stop if patience epochs not improve monitor metric/loss
    return tf.keras.callbacks.EarlyStopping(monitor=monitor, patience=patience)


def reduce_lr(monitor='val_loss', factor=0.5, patience=7, min_lr=0.000001, verbose=1, cooldown=1):
    # reduce learning rate by factor if not improve monitor in patience epochs
    return tf.keras.callbacks.ReduceLROnPlateau(monitor=monitor, factor=factor,
                                                patience=patience, min_lr=min_lr, verbose=verbose, cooldown=cooldown)


def checkpoint(output_name, monitor='val_loss', verbose=1, mode='min',
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



def checkpoint_google_storage(output_name, bucket_name=None, monitor='val_loss', verbose=1, mode='min',
                              save_best_only=True):
    assert bucket_name is not None, 'must to inplace bucket_name arg!'
    create_path_on_bucket_if_not_exists(bucket_name, f'weights/{output_name}.h5')
    blob_name = f'weights/{output_name}.h5'
    class BestValLossCheckpoint(Callback):
        """
        Guarda los pesos del modelo con el mejor valor de `val_loss` en Google Cloud Storage.

        Args:
            bucket_name: Nombre del bucket de Google Cloud Storage.
            blob_name (output file): Nombre del archivo .h5 en Google Cloud Storage.
            monitor: Métricas que se monitoriza para determinar el mejor modelo.
            verbose: Verbosidad del callback.
            save_best_only: Si es True, solo se guarda el mejor modelo.
            mode: Modo de comparación para la métrica.
        """

        def __init__(self, bucket_name, blob_name, monitor='val_loss', verbose=1,
                     save_best_only=True, mode='min'):
            super().__init__()
            self.bucket_name = bucket_name
            self.blob_name = blob_name
            self.monitor = monitor
            self.verbose = verbose
            self.save_best_only = save_best_only
            self.mode = mode

            self.best_val_loss = None
            self.best_weights = None

        def on_epoch_end(self, epoch, logs=None):
            current_val_loss = logs.get(self.monitor)
            if self.best_val_loss is None or current_val_loss < self.best_val_loss if self.mode == "min" else current_val_loss > self.best_val_loss:
                self.best_val_loss = current_val_loss
                self.best_weights = self.model.get_weights()

                # update weights to google cloud storage
                client = storage.Client()
                bucket = client.bucket(self.bucket_name)
                blob = bucket.blob(self.blob_name)
                print('first elements of best weights \n')
                print(np.array(self.best_weights)[:1])

                aux_model_path = 'weights_model.h5'
                self.model.save(aux_model_path)
                blob.upload_from_filename(aux_model_path)

                if self.verbose > 0:
                    print(f"weights saved on: gs://{self.bucket_name}/{self.blob_name}")

    return BestValLossCheckpoint(bucket_name, blob_name, monitor=monitor, verbose=verbose,
                                 save_best_only=save_best_only, mode=mode)

def csv_logger_firestore(experiment_id, collection_name=None, separator=';'):
    # callback CSVLogger with Firestore
    assert collection_name is not None, 'must to inplace collection_name arg!'

    class FirestoreCSVLogger(tf.keras.callbacks.CSVLogger):
        def __init__(self, experiment_id, collection_name):
            filename = f"{experiment_id}_logs.csv"
            super().__init__(filename=filename, separator=separator, append=True)
            self.experiment_id = experiment_id
            self.collection_name = collection_name

        def on_epoch_end(self, epoch, logs=None):
            super().on_epoch_end(epoch, logs)
            db = firestore.Client(database='mitologs')
            ts = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            ts_concat = ts.replace(' ', '').replace(':', '').replace('-', '')
            logs['timestamp'] = ts  # add timestamp to logs
            logs['experiment'] = self.experiment_id
            logs['epoch'] = epoch
            logs_str = {key: str(value) for key, value in logs.items()}
            document_id = f"{self.experiment_id}_{epoch}_{ts_concat}"  # unique id
            print(f"path doc: {self.collection_name}/{document_id}")
            # doc_ref = db.collection(self.collection_name).document(document_id)
            doc_ref = db.document(f"{self.collection_name}/{document_id}")
            doc_ref.set(logs_str)

    return FirestoreCSVLogger(experiment_id, collection_name)


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


def get_callbacks_list_gc(output_name, bucket_name=None, firestore_col_name=None):
    return [early_stop(), reduce_lr(),
            checkpoint_google_storage(output_name, bucket_name=bucket_name),
            csv_logger_firestore(output_name, collection_name=firestore_col_name)]
