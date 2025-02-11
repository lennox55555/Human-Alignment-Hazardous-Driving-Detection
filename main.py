'''
This script will run the project and output results in command line. 
Think of it as a local version of the web app
'''

from ETL.processData import DataProcessor
from ETL.extractData import GazeDataExtractor
from ETL.transformData import GazeDataTransformer
from models.naiveModel import NaiveHazardDetector
from models.deep_learning_train import DeepLearningModel
from models.traditionalCVModel import GazeCVClassifier
import os
import joblib

DATA_DIR = './data'
PROCESSED_VIDEOS_DIR = os.path.join(DATA_DIR, 'processed/driving_videos')
MODEL_CHECKPOINT_DIR = "./models/checkpoints"
DEEP_LEARNING_CHECKPOINT = "deepLearningModel.pth"

def main():
    # TODO: Uncomment this line of code to download the data
    # TODO: Currently commented since that takes a while!!!!
    # Initialize extraction process
    extractor = GazeDataExtractor(DATA_DIR)
    extractor.extract_data()


    # Initialize transformation process
    data_transformer = GazeDataTransformer(data_dir=DATA_DIR)
    data_transformer.transform_data()

    # Initialize processing
    processor = DataProcessor(data_dir=DATA_DIR)
    processed_data = processor.process_data()
    
# ====================================NAIVE APPROACH==============================
    print('\nRunning Naive Approach...')
    # Initialize and run naive model
    naive_model = NaiveHazardDetector()
    naive_model.fit(processed_data)
    predictions = naive_model.predict_all(processed_data)
    
    # Evaluate Naive Approach
    naive_model.evaluate(processed_data, predictions)

# ====================================TRADITIONAL CV APPROACH==============================

    print('\nRunning Traditional CV Approach...')
    csv_path_traditional = os.path.join(DATA_DIR, "processed", "normalized_gaze_data.csv")
    video_folder_traditional = os.path.join(DATA_DIR, "raw", "driving_videos")
    trad_classifier = GazeCVClassifier(
        csv_path=csv_path_traditional,
        video_folder=video_folder_traditional,
        patch_size=600,
        hist_bins=8,
        rolling_window=7,
        cap_time=15,
        model_type="svm"
    )
    X_traditional, y_traditional = trad_classifier.process_data()
    model_traditional, scaler_traditional = trad_classifier.train_classifier(X_traditional, y_traditional)

    checkpoint_dir = os.path.join("models", "checkpoints")
    os.makedirs(checkpoint_dir, exist_ok=True)
    checkpoint_path = os.path.join(checkpoint_dir, "traditionalCVModel.pkl")
    joblib.dump(model_traditional, checkpoint_path)
    print(f"Traditional CV model saved to {checkpoint_path}")


# ====================================DEEP LEARNING CV APPROACH==============================
    if not os.path.exists(os.path.join(MODEL_CHECKPOINT_DIR, DEEP_LEARNING_CHECKPOINT)):
        print('\nStarting Deep Learning Model Training...')
        os.makedirs(MODEL_CHECKPOINT_DIR, exist_ok=True)
        deepLearningModel = DeepLearningModel(data_dir=DATA_DIR, model_checkpoint_dir=MODEL_CHECKPOINT_DIR)
        deepLearningModel.run_training()
        accuracy, recall = deepLearningModel.evaluate()
        print(f"\nTest Accuracy: {accuracy * 100:.2f}%")
        print(f"\nTest Recall: {recall * 100:.2f}%") 
    else:
        deepLearningModel = DeepLearningModel(data_dir=DATA_DIR, model_checkpoint_dir=MODEL_CHECKPOINT_DIR)
        deepLearningModel.predict_one('video238')

        

if __name__ == "__main__":
    main()