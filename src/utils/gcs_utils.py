import os

from google.cloud import storage, bigquery


PROJECT_ID = os.environ.get('GCP_PROJECT_ID')

def create_gcs_uri(bucket_name: str, blob_name: str) -> str:
    """Creates a GCS URI for a given bucket and blob name."""
    return f"gs://{bucket_name}/{blob_name}"


def upload_to_bucket(local_file_path: str, destination_blob_name: str):
    """Uploads a file to the GCS bucket."""
    try:
        BUCKET_NAME = os.environ.get('GCS_BUCKET_NAME')

        client = storage.Client(project=PROJECT_ID)
        bucket = client.bucket(BUCKET_NAME)
        blob = bucket.blob(destination_blob_name)
        blob.upload_from_filename(local_file_path)
        print(f"File {local_file_path} uploaded to {destination_blob_name}.")
    except Exception as e:
        print(f"Error uploading file to GCS: {str(e)}")
        raise e


def load_csv_to_bigquery(blob_name: str, dataset_id: str, table_id: str, autodetect: bool = True):
    """Loads a CSV file from GCS into BigQuery."""
    try:
        client = bigquery.Client(project=PROJECT_ID)

        job_config = bigquery.LoadJobConfig(
            source_format=bigquery.SourceFormat.CSV,
            skip_leading_rows=1,
            autodetect=autodetect,
            write_disposition=bigquery.WriteDisposition.WRITE_TRUNCATE,  # overwrite on re-run
        )

        resolved_gcs_uri = create_gcs_uri(os.environ.get('GCS_BUCKET_NAME'), blob_name)

        load_job = client.load_table_from_uri(resolved_gcs_uri, f"{dataset_id}.{table_id}", job_config=job_config)
        load_job.result()  # waits for completion

        table = client.get_table(f"{dataset_id}.{table_id}")
        print(f"Loaded {table.num_rows} rows into {dataset_id}.{table_id}.")
    except Exception as e:
        print(f"Error loading CSV to BigQuery: {str(e)}")
        raise e
