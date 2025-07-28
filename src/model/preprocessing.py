import warnings
warnings.simplefilter(action='ignore', category=FutureWarning)
import pandas as pd
import logging


class DataPreprocessor:
    def __init__(self, target_window_size:int=30):
        self.target_window_size = target_window_size

        # Setup logging
        self.setup_logging()

    def setup_logging(self):
        logging.basicConfig(level=logging.INFO)
        self.logger = logging.getLogger(__name__)

    def _load_data(self):
        """Load dataset from a CSV file."""
        try:
            self.errors = pd.read_csv('data/raw/PdM_errors.csv', parse_dates=['datetime']).sort_values(['machineID', 'datetime'])
            self.failures = pd.read_csv('data/raw/PdM_failures.csv', parse_dates=['datetime']).sort_values(['machineID', 'datetime'])
            self.machines = pd.read_csv('data/raw/PdM_machines.csv')
            self.maintenance = pd.read_csv('data/raw/PdM_maint.csv', parse_dates=['datetime']).sort_values(['machineID', 'datetime'])
            self.telemetry = pd.read_csv('data/raw/PdM_telemetry.csv', parse_dates=['datetime']).sort_values(['machineID', 'datetime'])
        except Exception as e:
            self.logger.error(f"Error loading data: {str(e)}")
            raise e

    def _telemetry_grouping(self):
        """Group telemetry data by date and machineID, aggregating features."""
        try:
            self.telemetry = self.telemetry.sort_values(['machineID', 'datetime'])
            self.telemetry['date'] = self.telemetry['datetime'].dt.normalize()
            self.telemetry_grouped = self.telemetry.groupby(by=['date', 'machineID']) \
                .agg({'volt': 'mean', 'rotate': 'mean', 'pressure': 'mean', 'vibration': 'mean'}).reset_index()
        except Exception as e:
            self.logger.error(f"Error grouping telemetry data: {str(e)}")
            raise e

    def _add_features(self, df: pd.DataFrame, type_name: str, window_sizes: list = [7, 14, 30]):
        """Add rolling features for errors, failures, and maintenance."""
        try:
            df = df.sort_values(['machineID', 'datetime'])
            df['date'] = df['datetime'].dt.normalize()

            for idx, row in self.telemetry_grouped.iterrows():
                current_date = row['date']
                machine = row['machineID']

                # Filter rows for the same machine in a window
                for window in window_sizes:
                    # Create a mask for the last 'window' days
                    mask = (
                        (df['machineID'] == machine) &
                        (df['date'] >= current_date - pd.Timedelta(days=window)) &
                        (df['date'] <= current_date)
                    )

                    # Count how many errors, failures or maintenance occurred in that window
                    self.telemetry_grouped.at[idx, f'{type_name}_last_{window}_days'] = mask.sum()
        except Exception as e:
            self.logger.error(f"Error adding features: {str(e)}")
            raise e

    def _add_target_feature(self, df: pd.DataFrame, window_size: int = 30):
        """Add target feature indicating if a machine will fail in the next 'window_size' days."""
        try:
            df = df.sort_values(['machineID', 'datetime'])
            df['date'] = df['datetime'].dt.normalize()

            for idx, row in self.telemetry_grouped.iterrows():
                current_date = row['date']
                machine = row['machineID']

                mask = (
                    (df['machineID'] == machine) &
                    (df['date'] <= current_date + pd.Timedelta(days=window_size)) &
                    (df['date'] >= current_date)
                )

                # See if there are any failures in the next 'window_size' days
                will_fail = 1 if mask.sum() > 0 else 0
                self.telemetry_grouped.at[idx, f'will_fail_{window_size}_days'] = will_fail
        except Exception as e:
            self.logger.error(f"Error adding target feature: {str(e)}")
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

        final_df = self.telemetry_grouped.merge(self.machines, on='machineID', how='left')
        final_df['model'] = final_df['model'].replace({'model1': 0, 'model2': 1, 'model3': 2, 'model4': 3})

        if save_df:
            final_df.to_csv('data/processed/telemetry.csv', index=False)
            self.logger.info("Processed features saved to 'data/processed/telemetry.csv'")
        return final_df

if __name__ == "__main__":
    preprocessor = DataPreprocessor(target_window_size=30)
    processed_data = preprocessor.create_features()
