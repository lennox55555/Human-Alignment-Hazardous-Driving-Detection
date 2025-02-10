import pandas as pd
import re
import ast

def clean_and_convert(entry):
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

def prep_user_df(users_df):
    '''
    Prepares all data from the users dataframe before merging

    Input:
        - users_df (pd.Dataframe): the users dataframe
    
    Returns:
        - final_users_df (pd.Dataframe): the cleaned users dataframe
    '''
    # Convert string objects meant to be data structures to their appropriate data structure
    users_df['form'] = users_df['form'].apply(clean_and_convert)

    # Create the final_users_df and assign the column userId to email for parity with the survey df
    final_users_df = pd.DataFrame()
    final_users_df['userId'] = users_df['email']

    # Convert the newly created data structures into individual columns and drop the unnecessary _id column    
    final_users_df = pd.concat([final_users_df[['userId']], users_df['form'].apply(pd.Series)], axis=1)
    final_users_df = final_users_df.drop(columns=['_id'])

    return final_users_df

def prep_survey_df(survey_df):
    '''
    Prepares all data from the survey dataframe before merging

    Input:
        - survey_df (pd.Dataframe): the survey dataframe
    
    Returns:
        - final_survey_df (pd.Dataframe): the cleaned survey dataframe
    '''
    # Convert string objects meant to be data structures to their appropriate data structure
    survey_df['windowDimensions'] = survey_df['windowDimensions'].apply(clean_and_convert)
    survey_df['gaze'] = survey_df['gaze'].apply(clean_and_convert)
    survey_df['formData'] = survey_df['formData'].apply(clean_and_convert)

    # Convert the formData and windowDimensions columns in the survey df from a dict to being their own individual columns
    final_survey_df = pd.concat([survey_df, survey_df['formData'].apply(pd.Series), survey_df['windowDimensions'].apply(pd.Series)], axis=1)
    final_survey_df = final_survey_df.drop(columns=['formData', '__v', '_id'])
    
    # Convert start and end times to a meaningful duration and drop the start/end time columns<br>and the windowDimensions
    final_survey_df = final_survey_df.drop(columns=['startTime', 'endTime', 'windowDimensions'])
    return final_survey_df

def process_merged_df(final_survey_df, final_users_df):
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

def prep_merged_df_for_training(merged_df, time_split=0.28):
    '''
    Strips the final merged dataframe to the necessary columns used for training

    Input:
        - merged_df (pd.Dataframe): the cleaned final merged dataframe
        - time_split (float): the time (in seconds) for which to bin the dataframe
    
    Returns:
        - training_df (pd.Dataframe): the cleaned dataframe prepped for training
    '''
    # Create bins of time_split seconds for the
    merged_df['time_bin'] = (merged_df['time'] // time_split).astype(int)  

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

    # Drop the 'time_bin' column if not needed
    training_df = training_df.drop('time_bin', axis=1)
    return training_df

def pre_process_pipeline(survey_df, users_df, time_split=0.28):
    '''
    Runs the entire pre-process pipeline and returns the final script to be used for training

    Input:
        - survey_df (pd.Dataframe): the raw survey_df dataframe
        - users_df (pd.Dataframe): the raw users_df dataframe
        - time_split (float): the time (in seconds) for which to bin the dataframe
    
    Returns:
        (pd.Dataframe): the cleaned dataframe prepped for training
    '''
    final_survey_df = prep_survey_df(survey_df)
    final_users_df = prep_user_df(users_df)
    merged_df = process_merged_df(final_survey_df, final_users_df)
    return prep_merged_df_for_training(merged_df, time_split)

def main():
    survey_df = pd.read_csv('../data/survey_results_20250205_154644.csv')
    users_df = pd.read_csv('../data/users_data_20250205_154700.csv')

    final_df = pre_process_pipeline(survey_df, users_df)
    
    # percent of hazards in videos
    hazard_perc = final_df[final_df['hazard'] == True].shape[0] / final_df.shape[0]
    print(f'Percentage of hazards in data: {hazard_perc}')

    final_df.to_csv('binned_video_dat_wo_user.csv')

if __name__ == '__main__':
    main()