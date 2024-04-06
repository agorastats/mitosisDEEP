from google.cloud import storage

def upload_object(bucket_name, folder, local_object, output_object):
    """Upload object to Google Cloud Storage on a specific path.
    Assume that client is enabled with credentials"""

    client = storage.Client()
    bucket = client.get_bucket(bucket_name)

    # create object to update
    blob = bucket.blob(f"{folder}/{output_object}")
    # upload local object to blob
    blob.upload_from_filename(local_object)

def create_path_on_bucket_if_not_exists(bucket_name, path):
    storage_client = storage.Client()
    bucket = storage_client.bucket(bucket_name)
    blob = bucket.blob(path)
    if not blob.exists():
        blob.upload_from_string('')