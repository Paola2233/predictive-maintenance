import io
import os

import polars as pl
from google.cloud import bigquery

PROJECT_ID = os.environ.get('GCP_PROJECT_ID')

def run_query(query: str,) -> pl.DataFrame:
    """Runs a SQL query against BigQuery and returns the results."""
    try:
        client = bigquery.Client(project=PROJECT_ID)
        query_job = client.query(query)
        arrow_table = query_job.to_arrow()

        # Load into a Polars DataFrame
        df = pl.from_arrow(arrow_table)
        return df
    except Exception as e:
        print(f"Error running query: {str(e)}")
        raise e

def df_to_bigquery(df: pl.DataFrame, dataset_id: str, table_id: str):
    """Uploads a Polars DataFrame to BigQuery."""
    try:
        client = bigquery.Client(project=PROJECT_ID)

        # Write DataFrame to stream as parquet file; does not hit disk
        with io.BytesIO() as stream:
            df.write_parquet(stream)
            stream.seek(0)
            parquet_options = bigquery.ParquetOptions()
            parquet_options.enable_list_inference = True
            job = client.load_table_from_file(
                stream,
                destination=f"{dataset_id}.{table_id}",
                project=PROJECT_ID,
                job_config=bigquery.LoadJobConfig(
                    source_format=bigquery.SourceFormat.PARQUET,
                    parquet_options=parquet_options,
                    write_disposition=bigquery.WriteDisposition.WRITE_TRUNCATE,  # overwrite on re-run
                ),
            )
        job.result()  # waits for completion

        table = client.get_table(f"{dataset_id}.{table_id}")
        print(f"Loaded {table.num_rows} rows into {dataset_id}.{table_id}.")
    except Exception as e:
        print(f"Error uploading DataFrame to BigQuery: {str(e)}")
        raise e
