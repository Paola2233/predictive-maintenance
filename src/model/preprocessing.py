import warnings
warnings.simplefilter(action='ignore', category=FutureWarning)

import logging
import os
import polars as pl
from src.databases.bigquery import df_to_bigquery, run_query


TELEMETRY_COLS = ['volt', 'rotate', 'pressure', 'vibration']
ROLLING_WINDOWS = [3, 7, 24]  # hours (telemetry is hourly)


class DataPreprocessor:

    def __init__(
            self,
            target_window_size: int = 30,
    ):
        self.target_window_size = target_window_size

        # GCP configuration from environment variables
        self.project_id = os.environ.get('GCP_PROJECT_ID')
        self.bucket_name = os.environ.get('GCS_BUCKET_NAME')

        if not all([self.project_id, self.bucket_name]):
            raise ValueError("Required environment variables (GCP_PROJECT_ID, GCS_BUCKET_NAME) are not set.")

        self.setup_logging()

    def setup_logging(self):
        logging.basicConfig(level=logging.INFO)
        self.logger = logging.getLogger(__name__)

    def _load_data(self):
        """Load datasets from BigQuery."""
        try:
            self.logger.info("Loading data from BigQuery...")

            errors_df = run_query(f"SELECT * FROM `{self.project_id}.raw_data.PdM_errors`")
            self.errors = errors_df.sort(['machineID', 'datetime'])

            failures_df = run_query(f"SELECT * FROM `{self.project_id}.raw_data.PdM_failures`")
            self.failures = failures_df.sort(['machineID', 'datetime'])

            machines_df = run_query(f"SELECT * FROM `{self.project_id}.raw_data.PdM_machines`")
            self.machines = machines_df

            maintenance_df = run_query(f"SELECT * FROM `{self.project_id}.raw_data.PdM_maint`")
            self.maintenance = maintenance_df.sort(['machineID', 'datetime'])

            telemetry_df = run_query(f"SELECT * FROM `{self.project_id}.raw_data.PdM_telemetry`")
            self.telemetry = telemetry_df.sort(['machineID', 'datetime'])

        except Exception as e:
            self.logger.error(f"Error loading data from BigQuery: {str(e)}")
            raise e

    # ------------------------------------------------------------------
    # Rolling-window telemetry features (NEW)
    # ------------------------------------------------------------------

    def _add_telemetry_rolling_features(self):
        """
        Compute rolling mean and std for each telemetry signal over
        ROLLING_WINDOWS (in hours). The computation is done per machine
        using Polars' group_by + sort to respect temporal order.

        New columns follow the pattern:
            <signal>_mean_<window>h   e.g. volt_mean_3h
            <signal>_std_<window>h    e.g. volt_std_3h
        """
        try:
            self.logger.info("Computing rolling telemetry features...")

            df = self.telemetry.sort(['machineID', 'datetime'])

            # Build all rolling expressions in a single pass per window
            rolling_exprs = []
            for window in ROLLING_WINDOWS:
                for col in TELEMETRY_COLS:
                    rolling_exprs.append(
                        pl.col(col)
                        .rolling_mean(window_size=window, min_periods=1)
                        .over('machineID')
                        .alias(f'{col}_mean_{window}h')
                    )
                    rolling_exprs.append(
                        pl.col(col)
                        .rolling_std(window_size=window, min_periods=2)
                        .over('machineID')
                        .fill_null(0.0)
                        .alias(f'{col}_std_{window}h')
                    )

            df = df.with_columns(rolling_exprs)

            # Store enriched telemetry (still hourly at this point)
            self.telemetry = df

        except Exception as e:
            self.logger.error(f"Error computing rolling telemetry features: {str(e)}")
            raise e

    # ------------------------------------------------------------------
    # Daily aggregation (unchanged logic, now aggregates rolling cols too)
    # ------------------------------------------------------------------

    def _telemetry_grouping(self):
        """
        Group telemetry (already enriched with rolling features) by day
        and machineID. Raw signals are averaged; rolling stats take their
        daily mean/max to preserve signal variation.
        """
        try:
            # Identify all numeric columns to aggregate (exclude keys)
            numeric_cols = [
                c for c in self.telemetry.columns
                if c not in ('machineID', 'datetime')
            ]

            agg_exprs = [pl.col(c).mean() for c in numeric_cols]

            self.telemetry_grouped = (
                self.telemetry
                .with_columns(
                    pl.col('datetime').dt.truncate('1d').alias('date')
                )
                .group_by(['date', 'machineID'])
                .agg(agg_exprs)
            )

        except Exception as e:
            self.logger.error(f"Error grouping telemetry data: {str(e)}")
            raise e

    # ------------------------------------------------------------------
    # Count-based lag features (errors / failures / maintenance)
    # ------------------------------------------------------------------

    def _add_features(self, df: pl.DataFrame, type_name: str, window_sizes: list = [7, 14, 30]):
        """Add rolling count features for errors, failures, and maintenance."""
        try:
            df_with_date = df.with_columns(
                pl.col('datetime').dt.truncate('1d').alias('date')
            )

            for window in window_sizes:
                feature_col = f'{type_name}_last_{window}_days'

                joined = self.telemetry_grouped.join(
                    df_with_date,
                    left_on='machineID',
                    right_on='machineID',
                    how='left'
                )

                filtered = joined.filter(
                    (pl.col('date_right') >= pl.col('date') - pl.duration(days=window)) &
                    (pl.col('date_right') <= pl.col('date'))
                )

                counts = filtered.group_by(
                    ['date', 'machineID']
                ).agg(
                    pl.len().alias(feature_col)
                )

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

    # ------------------------------------------------------------------
    # Target label
    # ------------------------------------------------------------------

    def _add_target_feature(self, df: pl.DataFrame, window_size: int = 30):
        """Add binary target: will the machine fail in the next `window_size` days?"""
        try:
            df_with_date = df.with_columns(
                pl.col('datetime').dt.truncate('1d').alias('date')
            )

            joined = self.telemetry_grouped.join(
                df_with_date,
                left_on='machineID',
                right_on='machineID',
                how='left'
            )

            target_filtered = joined.filter(
                (pl.col('date_right') <= pl.col('date') + pl.duration(days=window_size)) &
                (pl.col('date_right') >= pl.col('date'))
            )

            target_feature = target_filtered.group_by(
                ['date', 'machineID']
            ).agg(
                (pl.len() > 0).cast(pl.Int64).alias(f'will_fail_{window_size}_days')
            )

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

    # ------------------------------------------------------------------
    # Persistence
    # ------------------------------------------------------------------

    def save_to_bigquery(self, df: pl.DataFrame, dataset: str, table: str):
        try:
            df_to_bigquery(df, dataset, table)
            self.logger.info(f"Saved to BigQuery '{dataset}.{table}'")
        except Exception as e:
            self.logger.error(f"Error saving to BigQuery: {str(e)}")
            raise e

    # ------------------------------------------------------------------
    # Main entry point
    # ------------------------------------------------------------------

    def create_features(self, save_df: bool = True) -> pl.DataFrame:
        """Full feature-engineering pipeline."""
        self._load_data()

        # --- NEW: rolling telemetry features BEFORE daily aggregation ---
        self._add_telemetry_rolling_features()

        self._telemetry_grouping()

        # Count-based lag features
        self._add_features(self.errors, 'error')
        self._add_features(self.failures, 'failure')
        self._add_features(self.maintenance, 'maint')

        # Target
        self._add_target_feature(self.failures, window_size=self.target_window_size)

        # Join machine metadata
        model_mapping = {'model1': 0, 'model2': 1, 'model3': 2, 'model4': 3}
        final_df = (
            self.telemetry_grouped
            .join(self.machines, left_on='machineID', right_on='machineID', how='left')
            .with_columns(
                pl.col('model').map_elements(
                    lambda x: model_mapping.get(x, x),
                    return_dtype=pl.Int64
                )
            )
        )

        if save_df:
            self.save_to_bigquery(
                final_df,
                dataset='preprocessed_data',
                table='features_engineered_v2'
            )

        return final_df


if __name__ == "__main__":
    preprocessor = DataPreprocessor(target_window_size=30)
    processed_data = preprocessor.create_features()