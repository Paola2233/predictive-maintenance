import warnings

warnings.simplefilter(action='ignore', category=FutureWarning)
import logging
import os

import polars as pl

from src.databases.bigquery import df_to_bigquery, run_query


class DataPreprocessor:
    def __init__(
            self,
            target_window_size: int = 30,
        ):
        self.target_window_size = target_window_size

        # GCP configuration from environment variables
        self.project_id = os.environ.get('GCP_PROJECT_ID')
        self.bucket_name = os.environ.get('GCS_BUCKET_NAME')

        # Validate required environment variables
        if not all([self.project_id, self.bucket_name]):
            raise ValueError("Required environment variables (GCP_PROJECT_ID, GCS_BUCKET_NAME) are not set.")

        # Setup logging
        self.setup_logging()

    def setup_logging(self):
        logging.basicConfig(level=logging.INFO)
        self.logger = logging.getLogger(__name__)

    def _load_data(self):
        """Load dataset from GCS CSV files."""
        try:
            self.logger.info("Loading data from BigQuery...")

            errors_df = run_query(f"""
                SELECT * FROM `{self.project_id}.raw_data.PdM_errors`
            """)
            self.errors = errors_df.sort(['machineID', 'datetime'])

            failures_df = run_query(f"""
                SELECT * FROM `{self.project_id}.raw_data.PdM_failures`
            """)
            self.failures = failures_df.sort(['machineID', 'datetime'])

            machines_df = run_query(f"""
                SELECT * FROM `{self.project_id}.raw_data.PdM_machines`
            """)
            self.machines = machines_df

            maintenance_df = run_query(f"""
                SELECT * FROM `{self.project_id}.raw_data.PdM_maint`
            """)
            self.maintenance = maintenance_df.sort(['machineID', 'datetime'])

            telemetry_df = run_query(f"""
                SELECT * FROM `{self.project_id}.raw_data.PdM_telemetry`
            """)
            self.telemetry = telemetry_df.sort(['machineID', 'datetime'])
        except Exception as e:
            self.logger.error(f"Error loading data from BigQuery: {str(e)}")
            raise e

    def _telemetry_grouping(self):
        """Group telemetry data by date and machineID, aggregating features."""
        try:
            self.telemetry_grouped = (
                self.telemetry
                .with_columns(
                    pl.col('datetime').dt.truncate('1d').alias('date')
                )
                .group_by(['date', 'machineID'])
                .agg([
                    pl.col('volt').mean(),
                    pl.col('rotate').mean(),
                    pl.col('pressure').mean(),
                    pl.col('vibration').mean()
                ])
            )
        except Exception as e:
            self.logger.error(f"Error grouping telemetry data: {str(e)}")
            raise e

    def _add_features(self, df: pl.DataFrame, type_name: str, window_sizes: list = [7, 14, 30]):
        """Add rolling features for errors, failures, and maintenance."""
        try:
            df_with_date = df.with_columns(
                pl.col('datetime').dt.truncate('1d').alias('date')
            )

            for window in window_sizes:
                # For each row in telemetry_grouped, count events within the window
                feature_col = f'{type_name}_last_{window}_days'

                # Cross join telemetry_grouped with the event dataframe
                joined = self.telemetry_grouped.join(
                    df_with_date,
                    left_on='machineID',
                    right_on='machineID',
                    how='left'
                )

                # Filter for events within the window
                filtered = joined.filter(
                    (pl.col('date_right') >= pl.col('date') - pl.duration(days=window)) &
                    (pl.col('date_right') <= pl.col('date'))
                )

                # Count events per telemetry row
                counts = filtered.group_by(
                    ['date', 'machineID']
                ).agg(
                    pl.len().alias(feature_col)
                )

                # Join back to telemetry_grouped
                self.telemetry_grouped = self.telemetry_grouped.join(
                    counts,
                    on=['date', 'machineID'],
                    how='left'
                ).with_columns(
                    pl.col(feature_col).fill_null(0)
                )
        except Exception as e:
            self.logger.error(f"Error adding features: {str(e)}")
            raise e

    def _add_target_feature(self, df: pl.DataFrame, window_size: int = 30):
        """Add target feature indicating if a machine will fail in the next 'window_size' days."""
        try:
            df_with_date = df.with_columns(
                pl.col('datetime').dt.truncate('1d').alias('date')
            )

            # Cross join to match each telemetry row with failures
            joined = self.telemetry_grouped.join(
                df_with_date,
                left_on='machineID',
                right_on='machineID',
                how='left'
            )

            # Filter for failures within the next 'window_size' days
            target_filtered = joined.filter(
                (pl.col('date_right') <= pl.col('date') + pl.duration(days=window_size)) &
                (pl.col('date_right') >= pl.col('date'))
            )

            # Check if there are any failures in the future window
            target_feature = target_filtered.group_by(
                ['date', 'machineID']
            ).agg(
                (pl.len() > 0).cast(pl.Int64).alias(f'will_fail_{window_size}_days')
            )

            # Join back to telemetry_grouped
            self.telemetry_grouped = self.telemetry_grouped.join(
                target_feature,
                on=['date', 'machineID'],
                how='left'
            ).with_columns(
                pl.col(f'will_fail_{window_size}_days').fill_null(0)
            )
        except Exception as e:
            self.logger.error(f"Error adding target feature: {str(e)}")
            raise e

    def save_to_bigquery(self, df: pl.DataFrame, dataset: str, table: str):
        """Save the processed DataFrame to BigQuery."""
        try:
            df_to_bigquery(df, dataset, table)
            self.logger.info(f"Processed features saved to BigQuery table '{dataset}.{table}'")
        except Exception as e:
            self.logger.error(f"Error saving to BigQuery: {str(e)}")
            raise e

    def create_features(self, save_df: bool = True):
        """Complete feature engineering pipeline"""

        self._load_data()
        self._telemetry_grouping()

        # Adding features for errors, failures, and maintenance
        self._add_features(self.errors, 'error')
        self._add_features(self.failures, 'failure')
        self._add_features(self.maintenance, 'maint')

        # Adding target feature for future failures
        self._add_target_feature(self.failures, window_size=self.target_window_size)

        # Join with machines data
        final_df = self.telemetry_grouped.join(
            self.machines,
            left_on='machineID',
            right_on='machineID',
            how='left'
        )

        # Map model values
        model_mapping = {'model1': 0, 'model2': 1, 'model3': 2, 'model4': 3}
        final_df = final_df.with_columns(
            pl.col('model').map_elements(
                lambda x: model_mapping.get(x, x),
                return_dtype=pl.Int64
            )
        )

        if save_df:
            self.save_to_bigquery(final_df, dataset='preprocessed_data', table='features_engineered_v2')
        return final_df

if __name__ == "__main__":
    preprocessor = DataPreprocessor(target_window_size=30)
    processed_data = preprocessor.create_features()
