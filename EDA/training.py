import os
import cv2
import pandas as pd
import numpy as np
import torch
from torchvision.models.video import r3d_18, R3D_18_Weights
from torch.utils.data import Dataset, DataLoader
import torch.nn as nn
import torch.optim as optim
from torchvision import transforms
from sklearn.model_selection import train_test_split
from PIL import Image

class VideoDataset(Dataset):
    def __init__(self, df, labels, time_splits=0.5, frames_per_clip=540, clip_size=18):
        self.df = df
        self.frames_per_clip = frames_per_clip
        self.labels = labels
        self.time_splits=time_splits
        self.clip_size=int(clip_size)
        self.transform = transforms.Compose([
            transforms.Resize((112, 112)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])

    def __len__(self):
        return self.df.shape[0]

    def load_video_frames(self, userId, videoId):
        video_path = f'./data/training_videos/{userId}_{videoId}.mp4'
        frames, fps = self.get_video_if_exist(video_path)
        return frames, fps

    def __getitem__(self, idx):
        userId = self.df.iloc[idx]['userId']
        videoId = self.df.iloc[idx]['videoId']
        time = self.df.iloc[idx]['time']

        video_tensor, fps = self.load_video_frames(userId, videoId)

        start_time_in_seconds = max(0, time - self.time_splits)
        start_frame = int(start_time_in_seconds * fps)

        clip_frames = video_tensor[:, start_frame:start_frame + self.clip_size]

        label = self.labels.iloc[idx]

        return clip_frames, label

    def get_video_if_exist(self, output_path):
        # Load existing video and grab frames
        cap = cv2.VideoCapture(output_path)
        fps = cap.get(cv2.CAP_PROP_FPS)
        frames_with_positions = []
        
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break
            frames_with_positions.append(frame)

        # If not reducing the video resolution you can use the line below
        #frames = [torch.from_numpy(frame).permute(2, 0, 1).float() / 255.0 for frame in frames]

        frames = [self.transform(Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))) for frame in frames_with_positions]

        # Pad if fewer frames
        while len(frames) < self.frames_per_clip:
            frames.append(frames[-1])  # Duplicate last frame
        
        cap.release()
        cv2.destroyAllWindows()
        return torch.stack(frames, dim=1), int(fps)

# Interpolation function between (x, y) points
def interpolate_positions(times, xs, ys, fps):
    new_times = np.arange(times[0], times[-1], 1 / fps)
    new_xs = np.interp(new_times, times, xs)
    new_ys = np.interp(new_times, times, ys)
    return new_times, new_xs, new_ys
'''
def overlay_video_with_points(df, video_path, userId, videoId):
    output_path = './data/training_videos/' + videoId + '_' + userId + '.mp4' 

    if os.path.exists(output_path):
        return get_video_if_exist(output_path)
    
    # Extract time, x, y columns for interpolation
    times = df['time'].values
    xs = df['x'].values
    ys = df['y'].values

    # Load the video
    cap = cv2.VideoCapture(video_path)

    # Get video properties
    fps = cap.get(cv2.CAP_PROP_FPS)
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    # Initialize the video writer
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')  # Codec for MP4
    out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))

    # Interpolate the positions
    interpolated_times, interpolated_xs, interpolated_ys = interpolate_positions(times, xs, ys, fps)

    # Initialize a list to store frames
    frames_with_positions = []

    # Read and process each frame
    frame_num = 0
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret or frame_num >= len(interpolated_times):
            break

        # Get the interpolated position for the current frame
        x, y = int(interpolated_xs[frame_num]), int(interpolated_ys[frame_num])

        # Draw a circle at the interpolated position
        cv2.circle(frame, (x, y), 10, (255, 0, 0), -1)  # Green circle for normal

        # Write frame to output video
        out.write(frame)

        # Store the frame in the list
        frames_with_positions.append(frame)

        frame_num += 1

    # Release video resources
    cap.release()
    out.release()
    cv2.destroyAllWindows()

    if len(frames_with_positions) == 0:
        print(output_path)

    return frames_with_positions, fps
'''

def training_loop(model, loader, device):
    # Loss function and optimizer
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=1e-4)

    model.to(device)
    model.train()  # Set the model to training mode

    # Training loop
    for epoch in range(10):
        total_loss = 0.0
        j = 1

        for clips, labels in loader:  # Use DataLoader properly for batches
            clips = clips.to(device)  # Move to GPU if available
            labels = labels.to(device)

            # Forward pass
            outputs = model(clips)
            loss = criterion(outputs, labels)

            # Backward pass and optimization
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            total_loss += loss.item()
            print(f'Current Loss: {total_loss/j}')

            j += 1

        # Average loss per epoch
        avg_loss = total_loss / len(loader)
        print(f"Epoch {epoch+1}, Loss: {avg_loss:.4f}")

def evaluate(model, loader, device):
    model.eval()
    correct, total = 0, 0
    with torch.no_grad():
        for inputs, labels in loader:
            inputs, labels = inputs.to(device), labels.to(device)
            outputs = model(inputs)
            _, predicted = torch.max(outputs, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
    return correct / total
            
def main():
    # Load the data
    df = pd.read_csv('./binned_video_data.csv')  # Replace with your actual file
    device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")

    time_split = df.iloc[1]['time']

    df = df.loc[:10, :]

    training_df = df.drop(columns=['hazard'])
    labels = df['hazard']

    X_train, X_test, y_train, y_test = train_test_split(training_df, labels, test_size=0.2, random_state=42)

    train_dataset = VideoDataset(X_train, y_train, time_splits=time_split, frames_per_clip=540, clip_size=36*time_split)
    test_dataset = VideoDataset(X_test, y_test, time_splits=time_split, frames_per_clip=540, clip_size=36*time_split)

    train_loader = DataLoader(train_dataset, batch_size=2, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=2, shuffle=False)

    # Load the pre-trained I3D model
    model = r3d_18(weights=R3D_18_Weights.KINETICS400_V1)
    for param in model.parameters():
        param.requires_grad = False

    # Modify the final layer for binary classification
    model.fc = torch.nn.Linear(model.fc.in_features, 2)

    training_loop(model, train_loader, device)

    accuracy = evaluate(model, test_loader, device)
    print(f"Test Accuracy: {accuracy * 100:.2f}%")  

if __name__ == '__main__':
    main()