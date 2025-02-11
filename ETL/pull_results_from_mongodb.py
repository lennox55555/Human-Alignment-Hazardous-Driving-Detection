"""
Pulls the gaze results from mongoDB
Last Updated: Feb 8, 2025
"""
from pymongo import MongoClient
import pandas as pd
import certifi
from datetime import datetime
import os
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

def connect_to_mongodb():
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

def fetch_results_data(client, batch_size=1000):
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

def save_to_csv(data, output_dir="data"):
    """Save the data to a CSV file"""
    try:
        # Create output directory if it doesn't exist
        if not os.path.exists(output_dir):
            os.makedirs(output_dir)
        
        # Convert to DataFrame
        df = pd.DataFrame(data)
        
        # Generate filename with timestamp
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = os.path.join(output_dir, f"survey_results_{timestamp}.csv")
        
        # Handle potential MongoDB ObjectId
        if '_id' in df.columns:
            df['_id'] = df['_id'].astype(str)
        
        # Save to CSV
        df.to_csv(filename, index=False)
        print(f"Data successfully saved to {filename}")
        print(f"Total rows saved: {len(df)}")
        print(f"Columns saved: {', '.join(df.columns)}")
        
    except Exception as e:
        print(f"Error saving data: {e}")

def main():
    # Connect to MongoDB
    client = connect_to_mongodb()
    if not client:
        return
    
    try:
        # Fetch data
        data = fetch_results_data(client)
        if data:
            # Save to CSV
            save_to_csv(data)
            print("Data extraction completed successfully!")
    except Exception as e:
        print(f"Error in main execution: {e}")
    finally:
        # Close the connection
        if client:
            client.close()
            print("MongoDB connection closed")

if __name__ == "__main__":
    main()