from utils.gcs_utils import load_csv_to_bigquery, upload_to_bucket


def upload_raw_sensor_data():
    """Uploads raw data files to GCS."""
    try:
        upload_to_bucket('data/raw/PdM_errors.csv', 'raw-data/PdM_errors.csv')
        upload_to_bucket('data/raw/PdM_failures.csv', 'raw-data/PdM_failures.csv')
        upload_to_bucket('data/raw/PdM_machines.csv', 'raw-data/PdM_machines.csv')
        upload_to_bucket('data/raw/PdM_maint.csv', 'raw-data/PdM_maint.csv')
        upload_to_bucket('data/raw/PdM_telemetry.csv', 'raw-data/PdM_telemetry.csv')
        print("All raw data files uploaded successfully.")
    except Exception as e:
        print(f"Error uploading raw data files: {str(e)}")
        raise e

def upload_raw_data_to_bigquery():
    """Uploads raw data from GCS to BigQuery."""
    try:
        load_csv_to_bigquery('raw-data/PdM_errors.csv', 'raw_data', 'PdM_errors')
        load_csv_to_bigquery('raw-data/PdM_failures.csv', 'raw_data', 'PdM_failures')
        load_csv_to_bigquery('raw-data/PdM_machines.csv', 'raw_data', 'PdM_machines')
        load_csv_to_bigquery('raw-data/PdM_maint.csv', 'raw_data', 'PdM_maint')
        load_csv_to_bigquery('raw-data/PdM_telemetry.csv', 'raw_data', 'PdM_telemetry')
        print("All raw data files loaded into BigQuery successfully.")
    except Exception as e:
        print(f"Error loading raw data files into BigQuery: {str(e)}")
        raise e

if __name__ == '__main__':
    # upload_raw_sensor_data()
    upload_raw_data_to_bigquery()