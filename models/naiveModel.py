import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import precision_score, recall_score, accuracy_score

class NaiveHazardDetector:
    def __init__(self):
        self.scaler = MinMaxScaler()
        # Thresholds can be adjusted based on data distribution
        self.hazard_severity_threshold = 0.6  # After normalization (0-1 scale)
        self.confidence_threshold = 0.7       # After normalization (0-1 scale)
        self.variance_threshold = 0.3         # After normalization (0-1 scale)
        self.is_fitted = False
        
    def fit(self, data):
        """
        Fit the scaler on variance and spread features
        """
        variance_features = ['variance_x', 'variance_y', 'gazeVariance', 'spreadFeature']
        self.scaler.fit(data[variance_features])
        self.is_fitted = True
        return self
        
    def normalize_features(self, data):
        """
        Normalize the variance-based features
        """
        if not self.is_fitted:
            raise ValueError("Model must be fitted before making predictions!")
            
        variance_features = ['variance_x', 'variance_y', 'gazeVariance', 'spreadFeature']
        data_copy = data.copy()
        data_copy[variance_features] = self.scaler.transform(data[variance_features])
        return data_copy
        
    def predict_single_observation(self, row):
        """
        Apply the heuristic rules to a single observation
        """
        # Get attention factors columns
        attention_cols = [col for col in row.index if 'attentionFactors_' in col]
        
        # Rule 1: Attention factor present with high hazard severity
        if (any(row[attention_cols] == 1) and 
            row['weightedHazardSeverity'] > self.hazard_severity_threshold):
            return 1
            
        # Rule 2: High detection confidence with attention factor
        if (row['meanDetectionConfidence'] > self.confidence_threshold and 
            any(row[attention_cols] == 1)):
            return 1
            
        # Rule 3: Low variance and spread (focused attention)
        if (row['gazeVariance'] < self.variance_threshold and 
            row['spreadFeature'] < self.variance_threshold):
            return 1
            
        # Rule 4: Clear hazard detection with attention factor
        if (row['noDetection_no_to_subtle_hazard'] == 0 and 
            any(row[attention_cols] == 1)):
            return 1
            
        # Default: No hazard
        return 0
        
    def predict_all(self, data):
        """
        Run predictions on entire dataset
        """
        if not self.is_fitted:
            raise ValueError("Model must be fitted before making predictions!")
            
        # Normalize the data first
        normalized_data = self.normalize_features(data)
        
        # Apply prediction to each row
        predictions = normalized_data.apply(self.predict_single_observation, axis=1)
        return predictions
        
    def evaluate(self, data, predictions):
        """
        Evaluate model performance using multiple metrics
        """
        print(f'Evaluating the performance of the heuristic approach...\n')
        true_labels = data['hazardDetected']
        
        results = {
            'accuracy': accuracy_score(true_labels, predictions),
            'precision': precision_score(true_labels, predictions),
            'recall': recall_score(true_labels, predictions)
        }
        
        # Add confusion matrix visualization
        confusion = pd.crosstab(true_labels, predictions, 
                              margins=True, margins_name='Total')
        
        print('Results:\n')
        print(results)

        print('Confusion Matrix\n')
        print(confusion)
        
        
        return results, confusion

def main():
    csv_path = '../data/aggregate_gaze_data_by_video.csv'
    data = pd.read_csv(csv_path)
    naive_model = NaiveHazardDetector()
    naive_model.fit(data)
    predictions = naive_model.predict_all(data)
    results, confusion = naive_model.evaluate(data, predictions)
    

if __name__ == "__main__":
    main()

