'''
Scripts that pulls the videos from S3 Bucket and saves them locally in a specific folder
Last Updated: Feb 8, 2025

'''
import os
import sys
import boto3
from dotenv import load_dotenv

load_dotenv()

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

class S3VideoDownloader():
    def __init__(self):
        self.s3_client = boto3.client('s3',
            aws_access_key_id=os.getenv('AWS_ACCESS_KEY_ID'),
            aws_secret_access_key=os.getenv('AWS_SECRET_ACCESS_KEY'),
            region_name=os.getenv('AWS_REGION')
        )
        self.bucket_name = os.getenv('S3_BUCKET_NAME')
        
        # Use data/driving_videos as video directory
        self.video_dir = '../data/driving_videos'

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

def main():
    video_downloader = S3VideoDownloader()
    video_downloader.pull_data()

if __name__ == "__main__":
    main()