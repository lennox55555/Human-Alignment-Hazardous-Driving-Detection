'''
Scripts that implements the extract and load pipelines. It outputs an aggregate csv 
under data/aggregate_gaze_data_by_video.csv
Last Updated: Feb 8, 2025

'''
import os
import sys
import pandas as pd
import numpy as np

from dotenv import load_dotenv

load_dotenv()

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

class DataProcessor():
    def __init__(self, data_dir, normalized_data_filename, aggregate_data_filename):
        self.data_dir = data_dir
        self.normalized_data_filename = normalized_data_filename
        self.normalized_gaze_data = os.path.join(self.data_dir, normalized_data_filename)
        self.aggregate_data_by_video= os.path.join(self.data_dir, aggregate_data_filename)

        # Create directory if it doesn't exist
        os.makedirs(self.data_dir, exist_ok=True)
    
    def process_normalized_data(self):
        df = pd.read_csv(self.normalized_gaze_data)
        # renamed the Unnamed : 0 column to index and duration to gazeDuration so it's more appropriate
        df.rename(columns={'Unnamed: 0': 'index', 'duration': 'gazeDuration'}, inplace=True)
        df["noDetection_no_to_subtle_hazard"] = df["noDetectionReason_nohazards"] + df["noDetectionReason_subtlehazards"]

         # Drop demographic and unnecessary columns
        columns_to_drop = [
            "userId", "index",

            # demographics features
            "ethnicity_asian", "ethnicity_black or african american", 
            "ethnicity_hispanic or latino", "ethnicity_middle eastern or north african", 
            "ethnicity_multiracial", "ethnicity_prefer not to say", "ethnicity_white", 
            "gender_female", "gender_male", "gender_prefer-not-to-say", "visuallyImpaired",


            # features relating to location
            "country_ar", "country_fr", "country_tn", "country_us",
            "state_california", "state_florida", "state_massachusetts", "state_north carolina", 
            "state_oregon", "state_south carolina", "state_washington",
            "city_boca raton", "city_boston", "city_chapel hill", "city_charlotte", 
            "city_coconut creek", "city_delray beach", "city_durham", "city_los angeles", 
            "city_miami", "city_olympia", "city_puyallup", "city_raleigh", "city_san diego", 
            "city_seattle", "city_tega cay", "city_west linn", "city_woodland hills",

            # features relating to raw gaze data (before standardization)
            'original_y', 'original_x', 'original_width', 'original_height', 'normalized_to_width', 'normalized_to_height',
            'display_width', 'display_height', 'width', 'height', 'x_offset', 'y_offset',

            # Survey related stuff
            'noDetectionReason_nohazards', 'noDetectionReason_subtlehazards'


        ]

        df.drop(columns=columns_to_drop, axis=1, inplace=True)

        # Convert boolean columns to integers (1 for True, 0 for False)
        df = df.astype({col: int for col in df.select_dtypes(include=["bool"]).columns})
        
        return df


    def aggregate_by_videoId(self, df):
        # Define aggregation functions
        aggregation_rules = {
            'x': lambda x: list(x),
            'y': lambda y: list(y),
            'time': lambda t: list(t),
            'detectionConfidence': 'mean',
            'gazeDuration': 'min',
            'age': 'mean',
            'hazardDetected': 'max',
            'licenseAge': 'mean',
            'noDetection_no_to_subtle_hazard': 'max',
        }

        # Handle attentionFactors_* columns by taking max
        attention_factor_cols = [col for col in df.columns if col.startswith("attentionFactors_")]
        for col in attention_factor_cols:
            aggregation_rules[col] = 'max'

        # Function to compute weighted hazard severity
        def compute_weighted_hazard_severity(hazard_values):
            value_counts = hazard_values.value_counts(normalize=True)  # Relative frequencies
            return sum(value_counts.index * value_counts.values)  # Weighted sum

        aggregation_rules['hazardSeverity'] = compute_weighted_hazard_severity

        # Perform aggregation
        aggregated_df = df.groupby('videoId').agg(aggregation_rules).reset_index()
        
        # Ensure x and y are sorted based on time
        for idx, row in aggregated_df.iterrows():
            sorted_indices = sorted(range(len(row['time'])), key=lambda i: row['time'][i])
            aggregated_df.at[idx, 'x'] = [row['x'][i] for i in sorted_indices]
            aggregated_df.at[idx, 'y'] = [row['y'][i] for i in sorted_indices]

        aggregated_df.rename(columns={'hazardSeverity': 'weightedHazardSeverity',
                                    'detectionConfidence': 'meanDetectionConfidence', 
                                    'age': 'meanAge',
                                    'licenseAge': 'meanLicenseAge',
                                    'gazeDuration': 'minGazeDuration'}, inplace=True)
        return aggregated_df

    def engineer_aggregate_features(self, df):
        aggregated_df = df.copy()
        # Compute mean x, mean y
        aggregated_df['mean_x'] = aggregated_df['x'].apply(lambda lst: np.mean(lst))
        aggregated_df['mean_y'] = aggregated_df['y'].apply(lambda lst: np.mean(lst))

        # Compute numGazes
        aggregated_df['numGazes'] = aggregated_df['x'].apply(len)

        # Compute variance of x and y
        aggregated_df['variance_x'] = aggregated_df['x'].apply(lambda lst: np.var(lst, ddof=1))
        aggregated_df['variance_y'] = aggregated_df['y'].apply(lambda lst: np.var(lst, ddof=1))

        # Compute gazeVariance
        def compute_gaze_variance(x_list, y_list):
            coords = np.column_stack((x_list, y_list))  # Create (x,y) coordinate pairs
            return np.var(np.linalg.norm(coords - np.mean(coords, axis=0), axis=1), ddof=1)

        def compute_spread_feature(x_list, y_list):
            coords = np.column_stack((x_list, y_list))  # Stack x and y as coordinate pairs
            cov_matrix = np.cov(coords, rowvar=False)  # Compute covariance matrix
            return np.trace(cov_matrix)  # Sum of diagonal elements (Var(X) + Var(Y))

        aggregated_df['gazeVariance'] = aggregated_df.apply(lambda row: compute_gaze_variance(row['x'], row['y']), axis=1)
        aggregated_df['spreadFeature'] = aggregated_df.apply(lambda row: compute_spread_feature(row['x'], row['y']), axis=1)
        return aggregated_df

    def save_csv(self, df):
        df.to_csv(self.aggregate_data_by_video, index=False)


    def process_data(self):
        df = self.process_normalized_data()
        df = self.aggregate_by_videoId(df)
        df = self.engineer_aggregate_features(df)

        self.save_csv(df)

        return df
        

def main():
    dataProcessor = DataProcessor(normalized_data_filename='normalized_gaze_data.csv',
                                  aggregate_data_filename='aggregate_gaze_data_by_video.csv')
    
    df = dataProcessor.process_normalized_data()
    df = dataProcessor.aggregate_by_videoId(df)
    df = dataProcessor.engineer_aggregate_features(df)

    dataProcessor.save_csv(df)

    

if __name__ == "__main__":
    main()