'''
This script pulls the data from mongoDB and extracts the users' data as well as the survey
results, along with the eye gaze data. 

This script also downloads the driving videos from AWS

Last Updated: Feb 9th, 2025
'''
from pymongo import MongoClient
import pandas as pd
import certifi
from datetime import datetime
import os
import boto3

from dotenv import load_dotenv

load_dotenv()

class S3VideoDownloader():
    def __init__(self):
        self.s3_client = boto3.client('s3',
            aws_access_key_id=os.getenv('AWS_ACCESS_KEY_ID'),
            aws_secret_access_key=os.getenv('AWS_SECRET_ACCESS_KEY'),
            region_name=os.getenv('AWS_REGION')
        )
        self.bucket_name = os.getenv('S3_BUCKET_NAME')
        
        # Use data/driving_videos as video directory
        self.video_dir = '../data/raw/driving_videos'

        # Create directory if it doesn't exist
        os.makedirs(self.video_dir, exist_ok=True)

        self.paginator = self.s3_client.get_paginator('list_objects_v2')
    
    def pull_data(self):
        paginator = self.s3_client.get_paginator('list_objects_v2')
        print('\nAttempting to download videos from AWS...')
        for page in paginator.paginate(Bucket=self.bucket_name):
            if 'Contents' in page:
                for obj in page['Contents']:
                    s3_key = obj['Key']
                    local_file_path = os.path.join(self.video_dir, s3_key)

                    # Create local directory structure if needed
                    os.makedirs(os.path.dirname(local_file_path), exist_ok=True)

                    # Download file
                    print(f"Downloading {s3_key} to {local_file_path}")
                    try:
                        self.s3_client.download_file(self.bucket_name, s3_key, local_file_path)
                    
                    except Exception as e:
                        print(f"Error downloading {s3_key}: {str(e)}")
                        

        print("Download complete!")

class GazeDataExtractor():

    def __init__(self):
        self.data_dir = '../data/raw'
        
        os.makedirs(os.path.dirname(self.data_dir), exist_ok=True)
    
    def connect_to_mongodb(self):
        """Establish connection to MongoDB Atlas"""
        connection_string = os.getenv('MONGODB_URI')
        
        if not connection_string:
            raise ValueError("MongoDB connection string not found in environment variables")
            
        print("Attempting to connect to MongoDB...")
        try:
            client = MongoClient(
                connection_string,
                tlsCAFile=certifi.where(),
                tls=True
            )
            # Test connection
            client.admin.command('ping')
            print("Successfully connected to MongoDB!")
            return client
        except Exception as e:
            print(f"Error connecting to MongoDB: {e}")
            if hasattr(e, 'details'):
                print(f"Additional error details: {e.details}")
            return None
    

    def fetch_users_data(self, client):
        """Fetch data from users collection"""
        try:
            db = client["survey"]
            collection = db["users"]
            
            # Add count before fetching
            doc_count = collection.count_documents({})
            print(f"Found {doc_count} documents in collection")
            
            # Fetch all documents
            documents = list(collection.find())
            print(f"Successfully fetched {len(documents)} documents")
            return documents
        except Exception as e:
            print(f"Error fetching data: {e}")
            return None
    
    def fetch_survey_results_data(self, client, batch_size=1000):
        """
        Fetch data from results collection using batch processing
        to handle large datasets efficiently
        """
        try:
            db = client["survey"]
            collection = db["results"]
            
            # Add count before fetching
            doc_count = collection.count_documents({})
            print(f"Found {doc_count} documents in collection")
            
            # Initialize an empty list to store all documents
            all_documents = []
            
            # Use cursor with batch processing
            cursor = collection.find({}, batch_size=batch_size)
            
            # Track progress
            processed = 0
            for doc in cursor:
                all_documents.append(doc)
                processed += 1
                if processed % batch_size == 0:
                    print(f"Processed {processed} of {doc_count} documents...")
            
            print(f"Successfully fetched {len(all_documents)} documents")
            return all_documents
        
        except Exception as e:
            print(f"Error fetching data: {e}")
            return None

    def save_users_csv(self, data):
        """Save the data to a CSV file"""
        try:
            # Create output directory if it doesn't exist
            if not os.path.exists(self.data_dir):
                os.makedirs(self.data_dir)
                
            # Convert to DataFrame
            df = pd.DataFrame(data)
            
            # Generate filename with timestamp
            # timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = os.path.join(self.data_dir, f"users_data_raw.csv")
            
            # Save to CSV
            df.to_csv(filename, index=False)
            print(f"Data successfully saved to {filename}")
        except Exception as e:
            print(f"Error saving USERS data: {e}")
    
    def save_survey_csv(self, data):
        """Save the data to a CSV file"""
        try:
            # Create output directory if it doesn't exist
            if not os.path.exists(self.data_dir):
                os.makedirs(self.data_dir)
            
            # Convert to DataFrame
            df = pd.DataFrame(data)
            
            # # Generate filename with timestamp
            # timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = os.path.join(self.data_dir, f"survey_results_raw.csv")
            
            # Handle potential MongoDB ObjectId
            if '_id' in df.columns:
                df['_id'] = df['_id'].astype(str)
            
            # Save to CSV
            df.to_csv(filename, index=False)
            print(f"Data successfully saved to {filename}")
            print(f"Total rows saved: {len(df)}")
            print(f"Columns saved: {', '.join(df.columns)}")
            
        except Exception as e:
            print(f"Error saving SURVEY data: {e}")
    
    def extract_data(self):
        print('\nExtracting users and survey result data from MongoDB...')
        client = self.connect_to_mongodb()
        if not client:
            return
        
        try:
            # Fetch data
            users_data = self.fetch_users_data(client)
            survey_data = self.fetch_survey_results_data(client)
            if users_data:
                # Save to CSV
                self.save_users_csv(users_data)
            else:
                print("Error fetching and saving USERS DATA")
            
            if survey_data:
                # Save to CSV
                self.save_survey_csv(survey_data)
            else:
                print("Error fetching and saving SURVEY DATA")

            print("Data extraction from MongoDB completed successfully!")
        finally:
            # Close the connection
            if client:
                client.close()
                print("MongoDB connection closed")
        
        # Download videos from AWS
        video_downloader = S3VideoDownloader()
        video_downloader.pull_data()
        
def main():
    gazeDataExtractor = GazeDataExtractor()
    gazeDataExtractor.extract_data()


if __name__ == '__main__':
    main()