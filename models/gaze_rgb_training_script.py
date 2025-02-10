import os
import cv2
import time
import pickle
import numpy as np
import pandas as pd
from tqdm import tqdm
from sklearn.svm import SVC
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report

# ----------------
# CONFIGURATIONS
# ----------------
DATA_PATH = os.path.join(os.path.dirname(__file__), "../data/")
VIDEO_PATH = os.path.join(DATA_PATH, "user_gaze_videos/")  
GAZE_DATA_PATH = os.path.join(DATA_PATH, "normalized_gaze_data.csv")

NUM_FRAMES = 10        # Number of frames per video to sample
FRAME_RESIZE = (64, 64)  # Resize frames to 64x64

# ----------------
# PATH CHECKS
# ----------------
if not os.path.exists(VIDEO_PATH):
    raise FileNotFoundError(f"Video path not found: {VIDEO_PATH}")
if not os.path.exists(GAZE_DATA_PATH):
    raise FileNotFoundError(f"Gaze data path not found: {GAZE_DATA_PATH}")

# ----------------
# LOAD GAZE DATA
# ----------------
gaze_data = pd.read_csv(GAZE_DATA_PATH)

# Expect columns: ["userId", "videoId", "time", "hazardDetected"]
required_cols = {"userId", "videoId", "time", "hazardDetected"}
if not required_cols.issubset(gaze_data.columns):
    raise ValueError(f"The gaze data must have columns: {required_cols}")

# Convert hazardDetected column to boolean if it's stored as a string
if gaze_data["hazardDetected"].dtype == object:
    gaze_data["hazardDetected"] = gaze_data["hazardDetected"].str.strip().str.lower() == "true"


gaze_data["videoFilename"] = gaze_data["userId"] + "_" + gaze_data["videoId"].astype(str) + ".mp4"

# ----------------
# FRAME SAMPLING
# ----------------
def extract_frames(video_path, num_frames=10):
    """
    Extract evenly spaced frames (in color) from a video, 
    then optionally resize to reduce dimensionality.
    """
    cap = cv2.VideoCapture(video_path)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    if total_frames == 0:
        cap.release()
        return []

    frame_indices = np.linspace(0, total_frames - 1, num_frames, dtype=int)
    frames = []
    for idx in frame_indices:
        cap.set(cv2.CAP_PROP_POS_FRAMES, idx)
        ret, frame = cap.read()
        if not ret:
            continue
       
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        if FRAME_RESIZE is not None:
            frame_rgb = cv2.resize(frame_rgb, FRAME_RESIZE, interpolation=cv2.INTER_AREA)
        frames.append(frame_rgb)
    
    cap.release()
    return frames

# ----------------------------
# MAIN PIPELINE: RGB FEATURES
# ----------------------------
if __name__ == "__main__":
    X = []
    y = []

    
    video_files = [f for f in os.listdir(VIDEO_PATH) if f.endswith('.mp4')]
    video_files.sort()
    available_gaze_videos = set(gaze_data["videoFilename"].unique())

    for video in tqdm(video_files, desc="Processing videos"):
        video_path = os.path.join(VIDEO_PATH, video)

        # If the video file does not exist in the gaze data, skip it
        if video not in available_gaze_videos:
            print(f"Skipping {video} (No matching gaze data found)")
            continue

        e
        user_id, video_id = video.rsplit("_", 1)
        video_id = video_id.replace(".mp4", "")

        # Get the subset of gaze data for this video
        video_hazard_data = gaze_data[(gaze_data["userId"] == user_id) & (gaze_data["videoId"] == video_id)]
        video_hazard_data.sort_values("time", inplace=True)

        if video_hazard_data.empty:
            print(f"Warning: No hazard data for {video}. All frames will be labeled as 0.")

        # Extract frames
        frames = extract_frames(video_path, NUM_FRAMES)
        if not frames:
            print(f"Warning: No frames extracted for {video}. Skipping.")
            continue

        
        frame_duration = 15.0 / NUM_FRAMES

        # 3. Assign label for each frame
        for frame_idx, frame_rgb in enumerate(frames):
            frame_timestamp = frame_idx * frame_duration
            next_frame_timestamp = (frame_idx + 1) * frame_duration  
            
            # Select all rows within this frame's time band
            hazard_rows = video_hazard_data[
                (video_hazard_data["time"] >= frame_timestamp) &
                (video_hazard_data["time"] < next_frame_timestamp)
            ]
            
          
            hazard_label = 1 if (hazard_rows["hazardDetected"] == True).any() else 0
            feature_vec = frame_rgb.flatten().astype(np.float32)

            X.append(feature_vec)
            y.append(hazard_label)

    # --------------
    # Debugging Label Distribution
    # --------------
    X = np.array(X)
    y = np.array(y)
    print(f"Total frames (X): {len(X)}; Total labels (y): {len(y)}")
    unique_labels, counts = np.unique(y, return_counts=True)
    print(f"Unique labels: {dict(zip(unique_labels, counts))}")

    if len(unique_labels) < 2:
        print("🚨 ERROR: Only one class found. Check hazard labels in your CSV.")
        print(gaze_data["hazardDetected"].value_counts())  #
        raise ValueError("Less than 2 unique classes found. Cannot train a binary classifier.")

    # ---------------------------------
    # TRAIN/VAL/TEST SPLIT
    # ---------------------------------
    # 70% for training, then 15%/15% for validation/test
    X_train, X_temp, y_train, y_temp = train_test_split(
        X, y, test_size=0.3, random_state=42, stratify=y
    )
    X_val, X_test, y_val, y_test = train_test_split(
        X_temp, y_temp, test_size=0.5, random_state=42, stratify=y_temp
    )

    print(f"Train size: {len(X_train)}, Val size: {len(X_val)}, Test size: {len(X_test)}")

    # -------------------------
    # TRAIN THE SVM 
    # -------------------------
    print("Training SVM model on RGB features...")
    clf = SVC(kernel='linear')

    start_time = time.time()
    clf.fit(X_train, y_train)
    train_time = time.time() - start_time
    print(f"SVM training completed in {train_time:.2f} seconds.")

    # -------------------------
    # EVALUATE MODEL
    # -------------------------
    print("\n--- Validation Set Performance ---")
    y_val_pred = clf.predict(X_val)
    print(classification_report(y_val, y_val_pred))

    print("\n--- Test Set Performance ---")
    y_test_pred = clf.predict(X_test)
    print(classification_report(y_test, y_test_pred))

    print("✅ Done. RGB-based gaze model pipeline complete.")

# -------------------------
# SAVE MODEL 
# -------------------------
model_checkpoint = "gaze_svm_model.pkl"

with open(model_checkpoint, "wb") as f:
    pickle.dump(clf, f)

print(f"✅ Model saved successfully: {model_checkpoint}")
