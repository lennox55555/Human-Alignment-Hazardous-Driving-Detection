'''
This script will run the project and output results in command line. 
Think of it as a local version of the web app
'''

from ETL.processData import DataProcessor
from models.naiveModel import NaiveHazardDetector

DATA_DIR = './data'
# TODO: ADD CODE TO DATA PROCESSOR THAT GENERATES THE normalized_gaze_data.csv
# TODO: UPDATE THE DATAPROCESSOR CLASS TO ADD THE NAME OF THE FILES AS THE CLASS IS GENERATING THEM
# TODO: SMTH LIKE: self.normalized_data = the_new_stuff instead of taking it as an argument
NORMALIZED_DATA_FILENAME = 'normalized_gaze_data.csv'
AGGREGATED_DATA_FILENAME= 'aggregate_gaze_data_by_video.csv'

def main():

    # Initialize data processor
    processor = DataProcessor(data_dir=DATA_DIR,
                              normalized_data_filename=NORMALIZED_DATA_FILENAME,
                              aggregate_data_filename=AGGREGATED_DATA_FILENAME)
    
    # Process data
    processed_data = processor.process_data()
    
    # Initialize and run naive model
    naive_model = NaiveHazardDetector()
    naive_model.fit(processed_data)
    predictions = naive_model.predict_all(processed_data)

    
    
    # Evaluate Naive Approach
    naive_model.evaluate(processed_data, predictions)

if __name__ == "__main__":
    main()