import pandas as pd
import numpy as np
import cv2
from pathlib import Path
from skimage.feature import hog  # (Optional: available if you want to experiment with HOG features.)
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from sklearn.metrics import classification_report, confusion_matrix

# =========================
# HELPER FUNCTION: Compute Danger Score
# =========================
def calculate_danger_score(detection_confidence, hazard_severity):
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
        The danger score (float).
    """
    # You can adjust these mappings as needed.
    mapping = {5: 1.0, 4: 0.8, 3: 0.6, 2: 0.4, 1: 0.2}
    multiplier = mapping.get(detection_confidence, 0.0)
    return multiplier * hazard_severity

# =========================
# STEP 1: Adjust CSV timestamps if video duration > 15 seconds.
# =========================
def adjust_csv_for_duration(df, cap_time=15):
    """
    For each video (grouped by videoId), if the video’s duration exceeds cap_time seconds,
    drop rows with time < (duration - cap_time) (these represent pre-play loading),
    and subtract that shift from the remaining timestamps so that the effective time
    starts at 0 and ends at cap_time.
    
    Parameters:
        df (DataFrame): The input gaze data CSV.
        cap_time (float): The maximum (or effective) duration (in seconds). Default is 15.
        
    Returns:
        A DataFrame with adjusted time and duration values.
    """
    def adjust_group(group):
        vid = group["videoId"].iloc[0]
        duration = group["duration"].max()
        if duration > cap_time:
            shift = duration - cap_time
            group_filtered = group[group["time"] >= shift].copy()
            if group_filtered.empty:
                print(f"Warning: Video {vid} (duration {duration:.2f}) has no data after shifting by {shift:.2f}s. Skipping this video.")
                return group_filtered
            group_filtered["time"] = group_filtered["time"] - shift
            group_filtered["duration"] = cap_time
            print(f"Video {vid}: duration {duration:.2f} > {cap_time}s; shifted by {shift:.2f}s.")
            return group_filtered
        else:
            return group
    return df.groupby("videoId", group_keys=False).apply(adjust_group)

# =========================
# STEP 2: Add rolling average features for gaze coordinates.
# =========================
def add_rolling_features(df, window=3):
    """
    Sorts the DataFrame by videoId and time, and computes a rolling mean for the gaze 
    coordinates 'x' and 'y' over a given window. These rolling averages can help capture
    the trend of the gaze over time.
    
    Parameters:
        df (DataFrame): The input gaze CSV.
        window (int): The rolling window size. Default is 3.
        
    Returns:
        A DataFrame with two new columns: 'x_roll' and 'y_roll'.
    """
    df = df.sort_values(["videoId", "time"]).copy()
    df["x_roll"] = df.groupby("videoId")["x"].transform(lambda s: s.rolling(window, min_periods=1).mean())
    df["y_roll"] = df.groupby("videoId")["y"].transform(lambda s: s.rolling(window, min_periods=1).mean())
    return df

# =========================
# STEP 3: Extract HSV histogram features from a patch around the gaze.
# =========================
def extract_patch_hsv_hist(video_path, timestamp, x, y, patch_size=15, bins=8):
    """
    Given a video file and a gaze sample (timestamp, x, y), this function:
      - Opens the video.
      - Computes the frame index corresponding to the timestamp (using the video's FPS).
      - Seeks to that frame and reads it.
      - Extracts a square patch of size (patch_size x patch_size) centered at (x, y).
        (If the patch would extend outside the image boundaries, the patch is clipped.)
      - Converts the patch from RGB (after converting from OpenCV’s default BGR) to HSV.
      - Computes a normalized histogram for each HSV channel with a specified number of bins.
      - Concatenates the three histograms into one feature vector.
    
    Parameters:
        video_path (Path): The path to the video file.
        timestamp (float): The (adjusted) time in seconds.
        x (float): The x-coordinate of the gaze.
        y (float): The y-coordinate of the gaze.
        patch_size (int): The size of the square patch. Default is 15.
        bins (int): The number of bins per HSV channel. Default is 8.
    
    Returns:
        A list of histogram values (length = 3 * bins) or None if the frame cannot be read.
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

    # Convert frame from BGR (OpenCV default) to RGB.
    frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    half = patch_size // 2
    x = int(round(x))
    y = int(round(y))
    h, w, _ = frame.shape
    # Calculate patch boundaries with clipping.
    y1 = max(0, y - half)
    y2 = min(h, y + half + 1)
    x1 = max(0, x - half)
    x2 = min(w, x + half + 1)
    patch = frame[y1:y2, x1:x2]

    # Convert patch to HSV.
    patch_hsv = cv2.cvtColor(patch, cv2.COLOR_RGB2HSV)
    
    # Compute and normalize the histogram for each HSV channel.
    hist_features = []
    for channel in range(3):  # For H, S, V channels.
        hist = cv2.calcHist([patch_hsv], [channel], None, [bins], [0, 256])
        hist = cv2.normalize(hist, hist).flatten()
        hist_features.extend(hist.tolist())
    return hist_features

# =========================
# STEP 4: Process the CSV and extract features for every gaze sample.
# =========================
def process_data(csv_path, video_folder, patch_size=15, hist_bins=8, rolling_window=3, cap_time=15):
    """
    Reads the original gaze CSV file and performs these steps:
      a) Adjust timestamps for each video if its duration > cap_time.
      b) Computes rolling averages for the gaze coordinates.
      c) For each row, loads the corresponding video file, retrieves the frame at the 
         (adjusted) timestamp, and extracts an HSV histogram from a patch around the gaze.
      d) Builds a feature vector for each sample by concatenating:
         - Raw gaze coordinates: x, y
         - Rolling averages: x_roll, y_roll
         - Adjusted timestamp: time
         - detectionConfidence, hazardSeverity, and the computed danger_score
         - HSV histogram features
         
      It also collects the label (assumed to be in a column called "hazard", with values 0/1).
    
    Parameters:
        csv_path (str): Path to the CSV file.
        video_folder (str): Path to the folder containing the video files.
        patch_size, hist_bins, rolling_window, cap_time: Parameters for patch extraction and time adjustment.
    
    Returns:
        X: Feature matrix (NumPy array of shape [num_samples, num_features]).
        y: Label vector (NumPy array).
    """
    df = pd.read_csv(csv_path)
    print(f"Loaded {len(df)} rows from CSV.")
    
    # Adjust timestamps for videos that exceed cap_time.
    df = adjust_csv_for_duration(df, cap_time=cap_time)
    print("Adjusted time stamps for videos longer than 15 seconds.")
    
    # Add rolling average features for gaze coordinates.
    df = add_rolling_features(df, window=rolling_window)
    print("Added rolling average features for x and y.")
    
    feature_list = []
    label_list = []
    
    # Process each video so that we open each video file only once.
    for video_id, group in df.groupby("videoId"):
        video_file = next(Path(video_folder).glob(f"{video_id}.*"), None)
        if video_file is None:
            print(f"Video file not found for video id {video_id}")
            continue
        
        print(f"Processing video {video_id} from {video_file} with {len(group)} samples.")
        for idx, row in group.iterrows():
            t = row["time"]
            x = row["x"]
            y = row["y"]
            hsv_hist = extract_patch_hsv_hist(video_file, t, x, y, patch_size=patch_size, bins=hist_bins)
            if hsv_hist is None:
                continue
            
            # Build the feature vector.
            # Start with raw gaze values, rolling averages, and time.
            feat = [row["x"], row["y"], row["x_roll"], row["y_roll"], row["time"]]
            
            # If detectionConfidence and hazardSeverity exist, add them and compute the danger score.
            if "detectionConfidence" in row and "hazardSeverity" in row:
                det_conf = row["detectionConfidence"]
                haz_sev = row["hazardSeverity"]
                feat.extend([det_conf, haz_sev])
                danger_score = calculate_danger_score(det_conf, haz_sev)
                feat.append(danger_score)
            
            # Append the HSV histogram features.
            feat.extend(hsv_hist)
            feature_list.append(feat)
            
            # Append the corresponding label from the "hazard" column.
            label_list.append(row["hazard"])
    
    X = np.array(feature_list)
    y = np.array(label_list)
    print("Final feature matrix shape:", X.shape)
    print("Final labels shape:", y.shape)
    return X, y

# =========================
# STEP 5: Train and evaluate a classifier.
# =========================
def train_classifier(X, y, model_type="svm"):
    """
    Splits the dataset, scales the features, and trains a classifier.
    
    Parameters:
        X (np.array): Feature matrix.
        y (np.array): Label vector.
        model_type (str): Either "svm" or "gradient_boost". Default is "svm".
    
    For the SVM, class_weight="balanced" is used to help address class imbalance.
    
    Returns:
        The trained model and the scaler.
    """
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    if model_type.lower() == "svm":
        model = SVC(kernel="rbf", C=1.0, gamma="scale", class_weight="balanced", random_state=42)
    elif model_type.lower() == "gradient_boost":
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

# =========================
# MAIN
# =========================
if __name__ == "__main__":
    # Update these paths to point to your CSV file and video folder.
    csv_path = "../data/processed/normalized_gaze_data.csv"
    video_folder = "../data/raw/driving_videos"
    
    # Process the CSV: adjust timestamps, add rolling averages, extract HSV histogram features,
    # and compute the danger score using detectionConfidence and hazardSeverity.
    X, y = process_data(
        csv_path,
        video_folder,
        patch_size=600,      # size of the patch around the gaze point 150 is best rn
        hist_bins=8,        # number of bins for each HSV channel histogram
        rolling_window=3,   # rolling average window for gaze coordinates, try
        cap_time=15         # cap effective video duration at 15 seconds
    )
    
    # Train the classifier. Choose "svm" (default) or "gradient_boost" based on your needs.
    model, scaler = train_classifier(X, y, model_type="svm")
    
    # Optionally, save the trained model and scaler:
    # import joblib
    # joblib.dump(model, "trained_model.pkl")
    # joblib.dump(scaler, "scaler.pkl")
