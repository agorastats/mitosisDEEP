import os
import time

from google.cloud import pubsub_v1
from worker.runModel import train_mitosis_model

PATH_OF_SECRET_JSON = '../secret.json'
os.environ["GOOGLE_APPLICATION_CREDENTIALS"] = PATH_OF_SECRET_JSON
credentials_path = PATH_OF_SECRET_JSON
project_id = 'train-mitosis-models'  # name of project on Google Cloud
subscription_name = 'training-sub'  # subscription name on Google Cloud Pub/Sub

subscriber = pubsub_v1.SubscriberClient.from_service_account_file(credentials_path)

def callback(message):
    print(f"Received message: {message.data}")
    # todo: aqui en mig ha d'anar la funcio de training keras
    train_mitosis_model(message.data)
    message.ack()  # consume sms, disappears


if __name__ == '__main__':
    # create subscription
    subscription_path = subscriber.subscription_path(project_id, subscription_name)

    # init subscriber to look for any messages
    subscriber.subscribe(subscription_path, callback=callback)  # callback is the process:= training model

    # while True to treat messages
    print("Listening for messages...")
    try:
        while True:
            time.sleep(60)
    except KeyboardInterrupt:
        print("Exiting...")



