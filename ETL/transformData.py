"""
processes eye-tracking data by normalizing gaze coordinates from various screen sizes 
to a standard video resolution (1280x960) while maintaining aspect ratio and accounting 
for letterboxing/pillarboxing offsets. It also filters out low-quality data by removing 
videos where users spent more than 50% of their time looking at the outer 10% of the screen, 
separating the data into two CSV files: one for good quality gaze data (preprocessedAndNormalized.csv) 
and another for filtered out data (badgazedata.csv)

Last Updated Feb 8th
"""

import pandas as pd
import numpy as np
from pathlib import Path
import re
import ast
import os

class GazeDataTransformer:
    def __init__(self, data_dir, survey_csv_filename="survey_results_raw.csv", users_csv_path="users_data_raw.csv"):
        """
        Initialize the gaze data processor
        
        Args:
            input_csv (str): Path to input CSV file
        """
        self.input_dir = os.path.join(data_dir, 'raw')
        self.output_dir = os.path.join(data_dir, 'processed')
        self.survey_df = pd.read_csv(os.path.join(self.input_dir, survey_csv_filename))
        self.users_df = pd.read_csv(os.path.join(self.input_dir, users_csv_path))
        self.VIDEO_WIDTH = 1280
        self.VIDEO_HEIGHT = 960
        self.EDGE_THRESHOLD = 0.10  # 10% from edges
        self.normalized_data = []
        self.bad_gaze_data = []
        self.merged_df = None

        os.makedirs(os.path.dirname(self.output_dir), exist_ok=True)

    
    def clean_and_convert(self, entry):
        '''
        This function converts the string objects meant to be dicts/arrays in the csv to their appropriate data structure

        Input:
            - entry (str): the string to be converted
        
        Returns:
            - The appropriate data structure
        '''
        # Replace ObjectId with the string version of the ID
        cleaned_entry = re.sub(r"ObjectId\('(.*?)'\)", r"'\1'", entry)
        # Convert the string to Python object
        return ast.literal_eval(cleaned_entry)

    def prep_user_df(self, users_df):
        '''
        Prepares all data from the users dataframe before merging

        Input:
            - users_df (pd.Dataframe): the users dataframe
        
        Returns:
            - final_users_df (pd.Dataframe): the cleaned users dataframe
        '''
        # Convert string objects meant to be data structures to their appropriate data structure
        users_df['form'] = users_df['form'].apply(self.clean_and_convert)

        # Create the final_users_df and assign the column userId to email for parity with the survey df
        final_users_df = pd.DataFrame()
        final_users_df['userId'] = users_df['email']

        # Convert the newly created data structures into individual columns and drop the unnecessary _id column    
        final_users_df = pd.concat([final_users_df[['userId']], users_df['form'].apply(pd.Series)], axis=1)
        final_users_df = final_users_df.drop(columns=['_id'])

        return final_users_df

    def prep_survey_df(self, survey_df):
        '''
        Prepares all data from the survey dataframe before merging

        Input:
            - survey_df (pd.Dataframe): the survey dataframe
        
        Returns:
            - final_survey_df (pd.Dataframe): the cleaned survey dataframe
        '''
        # Convert string objects meant to be data structures to their appropriate data structure
        survey_df['windowDimensions'] = survey_df['windowDimensions'].apply(self.clean_and_convert)
        survey_df['gaze'] = survey_df['gaze'].apply(self.clean_and_convert)
        survey_df['formData'] = survey_df['formData'].apply(self.clean_and_convert)

        # Convert the formData and windowDimensions columns in the survey df from a dict to being their own individual columns
        final_survey_df = pd.concat([survey_df, survey_df['formData'].apply(pd.Series), survey_df['windowDimensions'].apply(pd.Series)], axis=1)
        final_survey_df = final_survey_df.drop(columns=['formData', '__v', '_id'])
        
        # Convert start and end times to a meaningful duration and drop the start/end time columns<br>and the windowDimensions
        final_survey_df = final_survey_df.drop(columns=['startTime', 'endTime', 'windowDimensions'])
        return final_survey_df

    def process_merged_df(self, final_survey_df, final_users_df):
        '''
        Merges the survey and user dataframes and continues to clean the dataframe

        Input:
            - final_survey_df (pd.Dataframe): the cleaned survey dataframe
            - final_users_df (pd.Dataframe): the cleaned users dataframe
        
        Returns:
            - merged_df (pd.Dataframe): the cleaned, merged dataframe
        '''
        # Add a key to the gaze dictionaries for if a hazard was present at that specific timestamp
        merged_df = final_survey_df.merge(final_users_df, on='userId', how='left')
        for i in range(merged_df.shape[0]):
            if len(merged_df['gaze'][i]) == 0:
                continue

            min_time = min([gaze_point['time'] for gaze_point in merged_df['gaze'][i]])

            for j in range(len(merged_df['gaze'][i])):
                if merged_df['hazardDetected'][i] == False or len(merged_df['spacebarTimestamps'][i]) == 0:
                    merged_df['gaze'][i][j]['hazard'] = False
                else:
                    k = 1
                    while k < len(merged_df['spacebarTimestamps'][i]):
                        time = merged_df['gaze'][i][j]['time']
                        time_during_hazard = time > merged_df['spacebarTimestamps'][i][k-1] and time < merged_df['spacebarTimestamps'][i][k]
                        if merged_df['hazardDetected'][i] == True and time_during_hazard:
                            merged_df['gaze'][i][j]['hazard'] = True
                        else:
                            merged_df['gaze'][i][j]['hazard'] = False
                        k += 2
                
                merged_df['gaze'][i][j]['time'] = (merged_df['gaze'][i][j]['time'] - min_time) / 1000
        
        # Split the entire dataframe to have one row per timestamp with a value of if it's hazardous or not which will be the label
        merged_df = merged_df.explode('gaze', ignore_index=True)
        normalized = pd.json_normalize(merged_df['gaze'])
        merged_df = merged_df.drop(columns=['gaze']).join(normalized)

        # One hot encode the necessary columns
        attention_factors_exploded = merged_df.explode('attentionFactors')
        attn_factor_cols = pd.get_dummies(attention_factors_exploded['attentionFactors'], prefix='attentionFactors')
        merged_df = merged_df.drop(columns=['attentionFactors', 'spacebarTimestamps', '_id']).join(attn_factor_cols.groupby(level=0).max())

        # fill empty strings with ignore and make all strings lowercase 
        columns_to_clean = ['noDetectionReason', 'country', 'state', 'city', 'ethnicity', 'gender']
        for col in columns_to_clean:
            merged_df[col] = merged_df[col].replace('', pd.NA).fillna('ignore')
            if merged_df[col].dtype == 'object':
                merged_df[col] = merged_df[col].str.lower().replace('', pd.NA).fillna('ignore')

        # convert the city of boca with its full name
        merged_df['city'] = merged_df['city'].replace({'boca': 'boca raton'})

        # one hot encode columns
        merged_df = pd.get_dummies(merged_df, columns=columns_to_clean, prefix=columns_to_clean)

        # drop all columns with the _ignore one-hot-encoded column as those were empty
        merged_df = merged_df.drop(merged_df.filter(like='_ignore').columns, axis=1)

        ### Drop Rows with Missing Data
        merged_df = merged_df.dropna()

        return merged_df


    def calculate_video_display_size(self, screen_width, screen_height):
        """
        Calculate how the video would be displayed on a given screen size
        while maintaining aspect ratio
        """
        video_aspect = self.VIDEO_WIDTH / self.VIDEO_HEIGHT
        screen_aspect = screen_width / screen_height
        
        if screen_aspect > video_aspect:
            display_height = screen_height
            display_width = display_height * video_aspect
        else:
            display_width = screen_width
            display_height = display_width / video_aspect
            
        x_offset = (screen_width - display_width) / 2
        y_offset = (screen_height - display_height) / 2
        
        return display_width, display_height, x_offset, y_offset

    def is_gaze_at_edge(self, x, y):
        """
        Check if gaze point is in the outer 10% of the video
        """
        edge_x = self.VIDEO_WIDTH * self.EDGE_THRESHOLD
        edge_y = self.VIDEO_HEIGHT * self.EDGE_THRESHOLD
        
        return (x < edge_x or 
                x > self.VIDEO_WIDTH - edge_x or 
                y < edge_y or 
                y > self.VIDEO_HEIGHT - edge_y)

    def check_video_quality(self, video_data):
        """
        Check if more than 50% of gaze points are at the edges
        
        Returns:
            bool: True if video should be kept, False if it should be dropped
        """
        edge_points = video_data.apply(
            lambda row: self.is_gaze_at_edge(row['x'], row['y']), 
            axis=1
        ).sum()
        
        return edge_points / len(video_data) <= 0.5

    def normalize_coordinates(self, row):
        """
        Normalize coordinates for a single row of data
        """
        display_width, display_height, x_offset, y_offset = self.calculate_video_display_size(
            row['width'], row['height']
        )
        
        # Remove the offset from the gaze coordinates
        adjusted_x = row['x'] - x_offset
        adjusted_y = row['y'] - y_offset
        
        # Convert from display coordinates to video coordinates
        video_x = (adjusted_x / display_width) * self.VIDEO_WIDTH
        video_y = (adjusted_y / display_height) * self.VIDEO_HEIGHT
        
        # Clip coordinates to video boundaries
        normalized_x = np.clip(video_x, 0, self.VIDEO_WIDTH)
        normalized_y = np.clip(video_y, 0, self.VIDEO_HEIGHT)
        
        new_row = row.copy()
        
        # Store original values
        new_row['original_x'] = row['x']
        new_row['original_y'] = row['y']
        new_row['original_width'] = row['width']
        new_row['original_height'] = row['height']
        
        # Store display calculations
        new_row['display_width'] = display_width
        new_row['display_height'] = display_height
        new_row['x_offset'] = x_offset
        new_row['y_offset'] = y_offset
        
        # Update coordinates and dimensions
        new_row['x'] = normalized_x
        new_row['y'] = normalized_y
        new_row['width'] = self.VIDEO_WIDTH
        new_row['height'] = self.VIDEO_HEIGHT
        
        # Add normalization metadata
        new_row['normalized_to_width'] = self.VIDEO_WIDTH
        new_row['normalized_to_height'] = self.VIDEO_HEIGHT
        
        return new_row

    def normalize_gaze_data(self, merged_df):
        """
        Process the gaze data: normalize coordinates and filter bad videos
        """
        df = merged_df.copy()
        
        # Group by user and video
        user_video_groups = df.groupby(['userId', 'videoId'])
        
        print("Processing and filtering gaze data...")
        total_groups = len(user_video_groups)
        
        for idx, ((user_id, video_id), group_data) in enumerate(user_video_groups):
            if idx % 100 == 0:
                print(f"Processing group {idx}/{total_groups}")
            
            # Normalize coordinates for this group
            normalized_group = [self.normalize_coordinates(row) for _, row in group_data.iterrows()]
            normalized_df = pd.DataFrame(normalized_group)
            
            # Check if this video should be kept or dropped
            if self.check_video_quality(normalized_df):
                self.normalized_data.extend(normalized_group)
            else:
                self.bad_gaze_data.extend(normalized_group)
                print(f"Dropping video {video_id} for user {user_id} due to excessive edge gazing")

    

    
    def prep_merged_df_for_training(self, time_split=0.28):
        '''
        Strips the final merged dataframe to the necessary columns used for training

        Input:
            - merged_df (pd.Dataframe): the cleaned final merged dataframe
            - time_split (float): the time (in seconds) for which to bin the dataframe
        
        Returns:
            - training_df (pd.Dataframe): the cleaned dataframe prepped for training
        '''

        merged_df = self.merged_df
        # Create bins of time_split seconds for the
        merged_df['time_bin'] = (merged_df['time'] // time_split).astype(int)  

        merged_df['hazard'] = merged_df['hazard'].astype(int)

        # gets the merged dataframe grouped by videoId and time bin
        training_df = (
            merged_df.groupby(['videoId', 'time_bin'])
            .agg({
                'x': 'mean',  # Average x position in the interval
                'y': 'mean',  # Average y position in the interval
                'hazard': lambda x: (x.mean() > 0.5)  # Average vote for 
            })
            .reset_index()
        )

        # Create time column representing the start of the interval
        training_df['time'] = training_df['time_bin'] * time_split

         # Apply trimming and recalculating time per videoId
        training_df = training_df.groupby('videoId', group_keys=False).apply(self.trim_and_recalculate_time)

        # Drop the 'time_bin' column if not needed
        training_df = training_df.drop('time_bin', axis=1)
        return training_df


    def save_merged_csv(self, df):
            """Save the data to a CSV file"""
            try:
                # Create output directory if it doesn't exist
                if not os.path.exists(self.output_dir):
                    os.makedirs(self.output_dir)
                
                
                filename = os.path.join(self.output_dir, f"final_user_survey_data.csv")
                
                # Save to CSV
                df.to_csv(filename, index=False)
                print(f"\nMERGED DATA successfully saved to {filename}")
                print(f"Total rows saved: {len(df)}")
                print(f"Columns saved: {', '.join(df.columns)}\n")

            except Exception as e:
                print(f"Error saving MERGED data: {e}")
    
    def save_normalized_results(self, good_output_csv, bad_output_csv):
        """
        Save the processed data to CSV files
        """
        # Save good data
        good_df = pd.DataFrame(self.normalized_data)
        good_df.to_csv(good_output_csv, index=False)
        
        # Save bad data
        bad_df = pd.DataFrame(self.bad_gaze_data)
        bad_df.to_csv(bad_output_csv, index=False)
        
        # Print summary
        print("\nProcessing Summary:")
        print(f"Total videos processed: {len(self.normalized_data) + len(self.bad_gaze_data)}")
        print(f"Good videos saved to: {good_output_csv}")
        print(f"Videos with good gaze data: {len(self.normalized_data)}")
        print(f"Bad videos saved to: {bad_output_csv}")
        print(f"Videos with excessive edge gazing: {len(self.bad_gaze_data)}")
        
        # Print unique screen sizes
        good_df_screens = len(good_df.groupby(['original_width', 'original_height']))
        print(f"\nUnique screen sizes in good data: {good_df_screens}")
        print(f"All coordinates normalized to: {self.VIDEO_WIDTH}x{self.VIDEO_HEIGHT}")
    
    def save_training_csv(self, df):
        """Save the data to a CSV file"""
        try:
            # Create output directory if it doesn't exist
            if not os.path.exists(self.output_dir):
                os.makedirs(self.output_dir)
            
            
            filename = os.path.join(self.output_dir, 'binned_video_dat_wo_user.csv')
            
            # Save to CSV
            df.to_csv(filename, index=False)
            print(f"\nMERGED DATA successfully saved to {filename}")
            print(f"Total rows saved: {len(df)}")
            print(f"Columns saved: {', '.join(df.columns)}\n")

        except Exception as e:
            print(f"Error saving MERGED data: {e}")
    
    # Trim time for each videoId to ensure time difference <= 15
    def trim_and_recalculate_time(self, group):
        while group['time'].max() - group['time'].min() > 15:
            # Drop the row with the earliest time and recalculate
            group = group[group['time'] > group['time'].min()].reset_index(drop=True)

        # Normalize time based on the new min time
        group['time'] = group['time'] - group['time'].min()
        return group

    def transform_data(self, time_split=0.28):
        '''
        Runs the entire pre-process pipeline and returns the final script to be used for training.

        Input:
            - time_split (float): the time (in seconds) for which to bin the dataframe.
        
        Returns:
            (pd.DataFrame): the cleaned dataframe prepped for training.
        '''
        good_output_csv = "normalized_gaze_data.csv"
        bad_output_csv = "badgazedata.csv"

        # Prepare survey and user dataframes.
        final_survey_df = self.prep_survey_df(self.survey_df)
        final_users_df = self.prep_user_df(self.users_df)
        merged_df = self.process_merged_df(final_survey_df, final_users_df)
        

        # Add a 'duration' column: for each video, the duration is the maximum time value.
        merged_df['duration'] = merged_df.groupby('videoId')['time'].transform('max')
        print("Added duration column to merged data.")
        self.merged_df = merged_df
        if not os.path.exists(os.path.join(self.output_dir, good_output_csv)):
            self.normalize_gaze_data(merged_df=merged_df)
        else:
            print('\nNORMALIZED EYE GAZE Data found!!')
            print('\nSkipping this step...')

        # Save the merged CSV for reference.
        self.save_merged_csv(df=merged_df)

        if not os.path.exists(os.path.join(self.output_dir, good_output_csv)):
            self.save_normalized_results(
                good_output_csv=os.path.join(self.output_dir, good_output_csv),
                bad_output_csv=os.path.join(self.output_dir, bad_output_csv)
            )

        # Print percentage of hazards in the merged data.
        hazard_perc = merged_df[merged_df['hazard'] == True].shape[0] / merged_df.shape[0]
        print(f'Percentage of hazards in data: {hazard_perc}')

        # Create the binned CSV for training if not already done.
        if not os.path.exists(os.path.join(self.output_dir, 'binned_video_dat_wo_user.csv')):
            training_df = self.prep_merged_df_for_training(time_split)
            self.save_training_csv(training_df)
        else:
            print('\nBINNED TRAINING Data found!!')
            print('\nSkipping this step...')



    


def main():
    try:
        processor = GazeDataTransformer()
        processor.transform_data()
    except Exception as e:
        print(f"Error: {e}")


if __name__ == "__main__":
    main()

    

# def main():
#     survey_csv_path = '../data/survey_results_20250205_154644.csv'
#     users_csv_path = '../data/users_data_20250205_154700.csv'

#     gazeDataExtractor = Tal(survey_csv_path=survey_csv_path, users_csv_path=users_csv_path)

#     final_df = gazeDataExtractor.pre_process_pipeline()
    
#     # percent of hazards in videos
#     hazard_perc = final_df[final_df['hazard'] == True].shape[0] / final_df.shape[0]
#     print(f'Percentage of hazards in data: {hazard_perc}')

#     final_df.to_csv('binned_video_dat_wo_user.csv')