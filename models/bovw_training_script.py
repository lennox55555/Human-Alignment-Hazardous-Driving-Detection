import os
import cv2
import numpy as np
import pickle
import pandas as pd
from tqdm import tqdm
from sklearn.cluster import KMeans
from sklearn.svm import SVC
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report

# ---------------
# CONFIGURATIONS
# ---------------
DATA_PATH = os.path.join(os.path.dirname(__file__), "../data/")
VIDEO_PATH = os.path.join(DATA_PATH, "driving_videos/")
NUM_FRAMES = 10         # Number of frames sampled per video
VOCAB_SIZE = 500        # BoVW Vocabulary size
BATCH_SIZE = 10         # Process videos in batches

# If True, skip re-extracting SIFT features if checkpoint files already exist.
SKIP_SIFT_EXTRACTION = True

# ---------------
# PATH CHECKING
# ---------------
if not os.path.exists(VIDEO_PATH):
    raise FileNotFoundError(f"Video path not found: {VIDEO_PATH}")

# ---------------
# FRAME SAMPLING
# ---------------
def extract_frames(video_path, num_frames=10):
    """Extract evenly spaced frames (in grayscale) from a video."""
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
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        frames.append(gray)
    
    cap.release()
    return frames

# ---------------
# SIFT EXTRACTION
# ---------------
def extract_sift_features(frames, video_id, checkpoint_dir="sift_checkpoints/"):
    """
    Extract SIFT descriptors for each frame *and return them aligned* 
    so we know which frame actually produced descriptors.
    """
    os.makedirs(checkpoint_dir, exist_ok=True)
    checkpoint_file = os.path.join(checkpoint_dir, f"{video_id}_sift.pkl")
    
    # If skipping is allowed and file exists, just load
    if SKIP_SIFT_EXTRACTION and os.path.exists(checkpoint_file):
        with open(checkpoint_file, "rb") as f:
            return pickle.load(f)
    
    # Otherwise, compute fresh
    sift = cv2.SIFT_create()
    descriptors_per_frame = []
    for frame in tqdm(frames, desc=f"Extracting SIFT for {video_id}"):
        keypoints, desc = sift.detectAndCompute(frame, None)
        descriptors_per_frame.append(desc)  
    
    with open(checkpoint_file, "wb") as f:
        pickle.dump(descriptors_per_frame, f)
    
    return descriptors_per_frame

# ---------------
# BOVW VOCAB
# ---------------
def create_bovw_vocab(list_of_all_descriptors, vocab_size=500):
    """
    Clusters descriptors into a 'visual vocabulary' with K-Means.
    Expects list_of_all_descriptors to be a single big list (each element
    is an Nx128 SIFT descriptor array, possibly different N per frame).
    """
    # Flatten all descriptors
    if not list_of_all_descriptors:
        print("Warning: No descriptors found. Cannot create vocabulary.")
        return None

    all_descriptors = np.vstack([d for d in list_of_all_descriptors if d is not None])
    if all_descriptors.size == 0:
        print("Warning: No descriptors found. Cannot create vocabulary.")
        return None
    
    sample_size = min(len(all_descriptors), 50000)
    sample_idxs = np.random.choice(len(all_descriptors), sample_size, replace=False)
    sample_descriptors = all_descriptors[sample_idxs]

    kmeans = KMeans(n_clusters=vocab_size, random_state=42, n_init=10)
    with tqdm(total=1, desc="Fitting KMeans model") as pbar:
        kmeans.fit(sample_descriptors)
        pbar.update(1)
    return kmeans

def compute_bovw_histogram(descriptors_per_frame, kmeans):
    """
    Given a list of descriptor arrays (one per frame), 
    compute a BoVW histogram for each frame. Return stacked histograms.
    """
    histograms = []
    for desc in tqdm(descriptors_per_frame, desc="Computing BoVW histograms"):
        if desc is None or len(desc) == 0:
            continue
        words = kmeans.predict(desc)
        hist, _ = np.histogram(words, bins=np.arange(kmeans.n_clusters + 1))
        histograms.append(hist)
    return np.array(histograms)

# ---------------
# MAIN PIPELINE
# ---------------
if __name__ == "__main__":
    # Load hazard labels once
    HAZARD_DATA_PATH = os.path.join(DATA_PATH, "normalized_gaze_data.csv")
    hazard_data = pd.read_csv(HAZARD_DATA_PATH)

    # List of videos
    video_files = [f for f in os.listdir(VIDEO_PATH) if f.endswith('.mp4')]
    video_files.sort()

 
    all_descriptors = []
    all_labels = []

    for i in range(0, len(video_files), BATCH_SIZE):
        batch_videos = video_files[i : i + BATCH_SIZE]
        for video in tqdm(batch_videos, desc=f"Processing batch {i // BATCH_SIZE + 1}"):
            video_id = os.path.splitext(video)[0]
            
            # Extract frames
            frames = extract_frames(os.path.join(VIDEO_PATH, video), NUM_FRAMES)
            if not frames:
                continue

            #SIFT descriptors for each frame
            descriptors_per_frame = extract_sift_features(frames, video_id)

            #Get hazard data subset
            video_hazard_labels = hazard_data[hazard_data["videoId"] == video_id].copy()
            video_hazard_labels.sort_values("time", inplace=True)

            for frame_idx, desc in enumerate(descriptors_per_frame):
                # If desc is None, skip so that X and y remain in sync
                if desc is None or len(desc) == 0:
                    continue
                
                # Example: each frame covers 15/NUM_FRAMES seconds
                frame_timestamp = frame_idx * (15.0 / NUM_FRAMES)

                label_subset = video_hazard_labels[
                    (video_hazard_labels["time"] <= frame_timestamp)
                    & (video_hazard_labels["time"].shift(-1, fill_value=99999) > frame_timestamp)
                ]["hazardDetected"].values

                # So we unify the check here:
                hazard_label = 0
                if len(label_subset) > 0:
                    val = label_subset[0]
                    if isinstance(val, (bool, np.bool_)):
                        hazard_label = 1 if val else 0
                    else:
                        val_str = str(val).strip().lower()
                        hazard_label = 1 if val_str == "true" else 0

                all_descriptors.append(desc)
                all_labels.append(hazard_label)

    # If we have no descriptors, can't proceed
    if len(all_descriptors) == 0:
        print("No descriptors found in any video. Cannot train.")
        exit(0)

    # ---------------
    # Build/Load BoVW Vocabulary
    # ---------------
    kmeans_checkpoint = "bovw_vocab.pkl"
    if os.path.exists(kmeans_checkpoint):
        print("Loading existing BoVW vocabulary from checkpoint...")
        with open(kmeans_checkpoint, "rb") as f:
            kmeans = pickle.load(f)
    else:
        print("Creating BoVW vocabulary...")
        kmeans = create_bovw_vocab(all_descriptors, VOCAB_SIZE)
        if kmeans is None:
            print("No descriptors to cluster; cannot train.")
            exit(0)
        with open(kmeans_checkpoint, "wb") as f:
            pickle.dump(kmeans, f)

    # ---------------
    # Compute/Load Histograms
    # ---------------
    hist_checkpoint = "bovw_histograms.pkl"
    labels_checkpoint = "bovw_labels.pkl"
    if os.path.exists(hist_checkpoint) and os.path.exists(labels_checkpoint):
        print("Loading histograms and labels from checkpoint...")
        with open(hist_checkpoint, "rb") as f:
            X = pickle.load(f)
        with open(labels_checkpoint, "rb") as f:
            y = pickle.load(f)
    else:
        print("Computing BoVW histograms...")
        X = compute_bovw_histogram(all_descriptors, kmeans)
        y = np.array(all_labels)

        min_len = min(len(X), len(y))
        X, y = X[:min_len], y[:min_len]

        with open(hist_checkpoint, "wb") as f:
            pickle.dump(X, f)
        with open(labels_checkpoint, "wb") as f:
            pickle.dump(y, f)

    print(f"Feature vector count (X): {len(X)}")
    print(f"Label count (y): {len(y)}")
    unique_labels = set(y)
    print(f"Unique labels = {unique_labels}")
    if len(unique_labels) < 2:
        raise ValueError(
            "The number of classes has to be greater than one; "
            f"got {len(unique_labels)} class(es). Check your labeling logic or hazard data!"
        )

    # ---------------
    # Train/Test Split
    # ---------------
    X_train, X_temp, y_train, y_temp = train_test_split(
        X, y, test_size=0.3, random_state=42, stratify=y
    )
    X_val, X_test, y_val, y_test = train_test_split(
        X_temp, y_temp, test_size=0.5, random_state=42, stratify=y_temp
    )

    # ---------------
    # Train Classifier
    # ---------------
    clf = SVC(kernel='linear')
    clf_checkpoint = "bovw_svm_model.pkl"

    if os.path.exists(clf_checkpoint):
        print("Loading existing SVM model from checkpoint...")
        with open(clf_checkpoint, "rb") as f:
            clf = pickle.load(f)
    else:
        print("Training SVM model...")
        clf.fit(X_train, y_train)
        with open(clf_checkpoint, "wb") as f:
            pickle.dump(clf, f)

    # ---------------
    # Evaluate Model
    # ---------------
    pred_checkpoint = "bovw_predictions.pkl"
    if os.path.exists(pred_checkpoint):
        print("Loading existing predictions from checkpoint...")
        with open(pred_checkpoint, "rb") as f:
            y_pred = pickle.load(f)
    else:
        print("Predicting on test set...")
        y_pred = clf.predict(X_test)
        with open(pred_checkpoint, "wb") as f:
            pickle.dump(y_pred, f)

    print(classification_report(y_test, y_pred))
    print("Model pipeline complete.")
