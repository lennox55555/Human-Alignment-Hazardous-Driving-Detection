import pandas as pd
import numpy as np
import cv2
from pathlib import Path
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from sklearn.metrics import classification_report, confusion_matrix

class GazeCVClassifier:
    def __init__(self, csv_path, video_folder, patch_size=600, hist_bins=8, rolling_window=3, cap_time=15, model_type="svm"):
        """
        Initialize the classifier with file paths and parameters.

        Parameters:
            csv_path (str): Path to the CSV file containing gaze data.
            video_folder (str): Path to the folder containing video files.
            patch_size (int): Size of the square patch to extract around the gaze point.
            hist_bins (int): Number of bins for each HSV channel histogram.
            rolling_window (int): Window size for computing rolling averages.
            cap_time (int or float): Maximum effective video duration (in seconds).
            model_type (str): Type of model to train; either "svm" or "gradient_boost".
        """
        self.csv_path = csv_path
        self.video_folder = video_folder
        self.patch_size = patch_size
        self.hist_bins = hist_bins
        self.rolling_window = rolling_window
        self.cap_time = cap_time
        self.model_type = model_type

    def calculate_danger_score(self, detection_confidence, hazard_severity):
        """
        Maps the detection confidence to a multiplier and then multiplies it by hazard severity.

        Mapping (example):
            5 -> 1.0
            4 -> 0.8
            3 -> 0.6
            2 -> 0.4
            1 -> 0.2

        Parameters:
            detection_confidence (numeric): The detection confidence score.
            hazard_severity (numeric): The hazard severity score.

        Returns:
            float: The danger score.
        """
        mapping = {5: 1.0, 4: 0.8, 3: 0.6, 2: 0.4, 1: 0.2}
        multiplier = mapping.get(detection_confidence, 0.0)
        return multiplier * hazard_severity

    def adjust_csv_for_duration(self, df):
        """
        For each video (grouped by videoId), if the video’s duration exceeds cap_time seconds,
        drop rows with time < (duration - cap_time) (representing pre-play loading) and subtract
        that shift from the remaining timestamps so that the effective time starts at 0 and ends at cap_time.

        Parameters:
            df (DataFrame): The input gaze data CSV (which must have a 'duration' column).
        
        Returns:
            DataFrame: The adjusted data.
        """
        def adjust_group(group):
            vid = group["videoId"].iloc[0]
            duration = group["duration"].max()
            if duration > self.cap_time:
                shift = duration - self.cap_time
                group_filtered = group[group["time"] >= shift].copy()
                if group_filtered.empty:
                    print(f"Warning: Video {vid} (duration {duration:.2f}) has no data after shifting by {shift:.2f}s. Skipping this video.")
                    return group_filtered
                group_filtered["time"] = group_filtered["time"] - shift
                group_filtered["duration"] = self.cap_time
                print(f"Video {vid}: duration {duration:.2f} > {self.cap_time}s; shifted by {shift:.2f}s.")
                return group_filtered
            else:
                return group
        return df.groupby("videoId", group_keys=False).apply(adjust_group)

    def add_rolling_features(self, df):
        """
        Sorts the DataFrame by videoId and time, and computes a rolling mean for the gaze 
        coordinates 'x' and 'y' over a given window.

        Parameters:
            df (DataFrame): The input gaze CSV.
        
        Returns:
            DataFrame: The data with added columns 'x_roll' and 'y_roll'.
        """
        df = df.sort_values(["videoId", "time"]).copy()
        df["x_roll"] = df.groupby("videoId")["x"].transform(lambda s: s.rolling(self.rolling_window, min_periods=1).mean())
        df["y_roll"] = df.groupby("videoId")["y"].transform(lambda s: s.rolling(self.rolling_window, min_periods=1).mean())
        return df

    def extract_patch_hsv_hist(self, video_path, timestamp, x, y):
        """
        Given a video file and a gaze sample, extracts a square patch centered at (x, y),
        converts it to HSV, and computes a normalized histogram for each HSV channel.

        Parameters:
            video_path (Path): The path to the video file.
            timestamp (float): The (adjusted) time in seconds at which to extract the frame.
            x (float): The x-coordinate of the gaze.
            y (float): The y-coordinate of the gaze.
        
        Returns:
            list: The concatenated HSV histogram features, or None if the frame cannot be read.
        """
        cap = cv2.VideoCapture(str(video_path))
        if not cap.isOpened():
            print(f"Could not open video: {video_path}")
            return None

        fps = cap.get(cv2.CAP_PROP_FPS)
        frame_index = int(round(timestamp * fps))
        cap.set(cv2.CAP_PROP_POS_FRAMES, frame_index)
        ret, frame = cap.read()
        cap.release()
        if not ret:
            print(f"Could not read frame at time {timestamp} in video {video_path}")
            return None

        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        half = self.patch_size // 2
        x = int(round(x))
        y = int(round(y))
        h, w, _ = frame.shape
        y1 = max(0, y - half)
        y2 = min(h, y + half + 1)
        x1 = max(0, x - half)
        x2 = min(w, x + half + 1)
        patch = frame[y1:y2, x1:x2]
        patch_hsv = cv2.cvtColor(patch, cv2.COLOR_RGB2HSV)
        hist_features = []
        for channel in range(3):
            hist = cv2.calcHist([patch_hsv], [channel], None, [self.hist_bins], [0, 256])
            hist = cv2.normalize(hist, hist).flatten()
            hist_features.extend(hist.tolist())
        return hist_features

    def process_data(self):
        """
        Reads the CSV, adjusts timestamps, adds rolling features, and extracts features
        (including HSV histograms and danger score) for each gaze sample.

        Returns:
            X (ndarray): The feature matrix.
            y (ndarray): The label vector.
        """
        df = pd.read_csv(self.csv_path)
        print(f"Loaded {len(df)} rows from CSV.")
        df = self.adjust_csv_for_duration(df)
        print("Adjusted time stamps for videos longer than 15 seconds.")
        df = self.add_rolling_features(df)
        print("Added rolling average features for x and y.")
        feature_list = []
        label_list = []
        for video_id, group in df.groupby("videoId"):
            video_file = next(Path(self.video_folder).glob(f"{video_id}.*"), None)
            if video_file is None:
                print(f"Video file not found for video id {video_id}")
                continue
            print(f"Processing video {video_id} from {video_file} with {len(group)} samples.")
            for idx, row in group.iterrows():
                t = row["time"]
                x = row["x"]
                y = row["y"]
                hsv_hist = self.extract_patch_hsv_hist(video_file, t, x, y)
                if hsv_hist is None:
                    continue
                # Build feature vector: raw x, raw y, rolling averages, time
                feat = [row["x"], row["y"], row["x_roll"], row["y_roll"], row["time"]]
                if "detectionConfidence" in row and "hazardSeverity" in row:
                    det_conf = row["detectionConfidence"]
                    haz_sev = row["hazardSeverity"]
                    feat.extend([det_conf, haz_sev])
                    danger_score = self.calculate_danger_score(det_conf, haz_sev)
                    feat.append(danger_score)
                feat.extend(hsv_hist)
                feature_list.append(feat)
                label_list.append(row["hazard"])
        X = np.array(feature_list)
        y = np.array(label_list)
        print("Final feature matrix shape:", X.shape)
        print("Final labels shape:", y.shape)
        return X, y

    def train_classifier(self, X, y):
        """
        Splits the dataset, scales the features, and trains a classifier.

        Returns:
            model: The trained model.
            scaler: The scaler used for feature normalization.
        """
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
        scaler = StandardScaler()
        X_train_scaled = scaler.fit_transform(X_train)
        X_test_scaled = scaler.transform(X_test)
        if self.model_type.lower() == "svm":
            model = SVC(kernel="rbf", C=1.0, gamma="scale", class_weight="balanced", random_state=42)
        elif self.model_type.lower() == "gradient_boost":
            from sklearn.ensemble import GradientBoostingClassifier
            model = GradientBoostingClassifier(n_estimators=100, learning_rate=0.1, random_state=42)
        else:
            raise ValueError("Unknown model type. Choose 'svm' or 'gradient_boost'.")
        model.fit(X_train_scaled, y_train)
        y_pred = model.predict(X_test_scaled)
        print("=== Classification Report ===")
        print(classification_report(y_test, y_pred))
        print("=== Confusion Matrix ===")
        print(confusion_matrix(y_test, y_pred))
        return model, scaler

# When running this module directly, you can test the class:
if __name__ == "__main__":
    csv_path = "./data/processed/normalized_gaze_data.csv"
    video_folder = "./data/raw/driving_videos"
    classifier = GazeCVClassifier(csv_path, video_folder, patch_size=600, hist_bins=8, rolling_window=7, cap_time=15, model_type="svm")
    X, y = classifier.process_data()
    model, scaler = classifier.train_classifier(X, y)
