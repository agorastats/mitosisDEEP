import os
import json

from google.cloud import pubsub_v1
from runModel.params import PATH_OF_SECRET_JSON

# Google Cloud credentials
os.environ["GOOGLE_APPLICATION_CREDENTIALS"] = PATH_OF_SECRET_JSON
credentials_path = PATH_OF_SECRET_JSON
project_id = 'train-mitosis-models'
subscriber = pubsub_v1.SubscriberClient.from_service_account_file(credentials_path)

# specify theme on Google Cloud Pub/Sub
topic_name = 'training'

# function to publish message on Google Cloud
def publish_message(data):
    publisher = pubsub_v1.PublisherClient.from_service_account_file(credentials_path)
    topic_path = publisher.topic_path(project_id, topic_name)

    # convert dict to json
    message_data = json.dumps(data).encode('utf-8')

    # publish message
    future = publisher.publish(topic_path, data=message_data)
    print(f'Message published: {future.result()}')


if __name__ == '__main__':
    # define experiment or read file of experiments ...

    # example of experiments
    experimentDict = {
        'experiment_name': 'experiment_1',
        'lr': 0.01,
        'batch_size': 32,
        'epoch_number': 1,
        'pretrained_model_path': 'unet-best', # unet-best or monuseg --> assume models on path ../pretrainedModels/{MODEL_NAME.h5}
        'loss': 'binary_crossentropy',  # look at utils/keras/lossAndMetrics.py for available loss functions (MAP_LOSS_FUNCTIONS)
        'augmentation': 'v2',  # augmentation for train set, look at utils/augmentation.py (MAP_AUG_IMG_PIPELINE) or False to avoid it
        'n_filters': 16,
        'dropout': 0.8

    }
    # MORE PARAMETERS ...
    # todo: 'load_weights': ... to explore (connect google drive, for example: "drive/MyDrive/mitosis_data/pretrained_weights.h5")

    publish_message(experimentDict)



