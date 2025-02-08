'''
Given the directory of the videos downloaded from s3, this script will use the normalized_gaze_data
to place the user gaze on the videos
Last Updated: Feb 8, 2025
'''
import pandas as pd
import cv2
import numpy as np
import matplotlib as plt
import ast
import os
from pathlib import Path
from scipy.interpolate import interp1d

class UserGazeVideoCreator:
    def __init__(self, video_folder_path, csv_path, output_folder_path, aggregate_csv, output_folder_aggregated):
        """
        Initialize the video creator with paths
        """
        self.video_folder = Path(video_folder_path)
        self.output_folder = Path(output_folder_path)
        self.output_folder_aggregated = Path(output_folder_aggregated)
        self.output_folder.mkdir(parents=True, exist_ok=True)
        self.output_folder_aggregated.mkdir(parents=True, exist_ok=True)
        
        # Hot pink color in BGR format
        self.dot_color = (147, 20, 255) # RGB(255, 20, 147) in BGR

        # Read and normalize the gaze data
        print("Reading and normalizing gaze data...")
        self.gaze_data = self.read_normalized_gaze_data(csv_path)
        self.aggregate_gaze_data = pd.read_csv(aggregate_csv)
        
        # Get unique users
        self.users = self.gaze_data['userId'].unique()
        print(f"Found {len(self.users)} unique users")
        
    def read_normalized_gaze_data(self, csv_path):
        """Read and normalize the gaze data"""
        df = pd.read_csv(csv_path)
        VIDEO_WIDTH = 1280
        VIDEO_HEIGHT = 960
        
        # Normalize coordinates if they haven't been normalized yet
        if 'normalized_to_width' not in df.columns:
            print("Normalizing coordinates...")
            for idx, row in df.iterrows():
                # Calculate display dimensions
                video_aspect = VIDEO_WIDTH / VIDEO_HEIGHT
                screen_aspect = row['width'] / row['height']

                if screen_aspect > video_aspect:
                    display_height = row['height']
                    display_width = display_height * video_aspect
                else:
                    display_width = row['width']
                    display_height = display_width / video_aspect

                # Calculate offsets
                x_offset = (row['width'] - display_width) / 2
                y_offset = (row['height'] - display_height) / 2

                # Normalize coordinates
                df.at[idx, 'x'] = ((row['x'] - x_offset) / display_width) * VIDEO_WIDTH
                df.at[idx, 'y'] = ((row['y'] - y_offset) / display_height) * VIDEO_HEIGHT

        return df

    def handle_duplicate_timestamps(self, video_gaze_data):
        """
        Handle duplicate timestamps by averaging positions
        """
        # Group by time and calculate mean position
        return video_gaze_data.groupby('time', as_index=False).agg({
            'x': 'mean',
            'y': 'mean'
        }).sort_values('time')

    def interpolate_gaze_points(self, video_gaze_data, fps):
        """
        Create smooth interpolation between gaze points
        """
        # Handle any duplicate timestamps first
        unique_data = self.handle_duplicate_timestamps(video_gaze_data)

        # Get original time points and coordinates
        times = unique_data['time'].values
        x_coords = unique_data['x'].values
        y_coords = unique_data['y'].values

        # Add small amount of noise to any remaining duplicate times
        eps = 1e-10
        for i in range(1, len(times)):
            if times[i] <= times[i-1]:
                times[i] = times[i-1] + eps

        # Create interpolation functions
        x_interp = interp1d(times, x_coords, kind='cubic', bounds_error=False, fill_value='extrapolate')
        y_interp = interp1d(times, y_coords, kind='cubic', bounds_error=False, fill_value='extrapolate')

        # Create timestamps for every frame
        frame_times = np.arange(times[0], times[-1], 1/fps)

        # Interpolate positions for every frame
        x_smooth = x_interp(frame_times)
        y_smooth = y_interp(frame_times)

        return pd.DataFrame({
            'time': frame_times,
            'x': x_smooth,
            'y': y_smooth
        })

    def create_video_for_user(self, user_id, sample_mode=False):
        """
        Create videos for a specific user with their gaze overlay
        """
        # Get all videos for this user
        user_data = self.gaze_data[self.gaze_data['userId'] == user_id]
        unique_videos = user_data['videoId'].unique()

        if sample_mode:
            unique_videos = unique_videos[:1]

        for video_id in unique_videos:
            try:
                # Get gaze data for this video
                video_gaze_data = user_data[user_data['videoId'] == video_id].sort_values('time')

                if len(video_gaze_data) < 4:
                    print(f"Not enough gaze points for video {video_id}")
                    continue

                # Find video file
                video_path = next(self.video_folder.glob(f"{video_id}.*"))
                if not video_path.exists():
                    print(f"Video file not found for {video_id}")
                    continue

                print(f"Processing video {video_id} for user {user_id}...")

                # Open video
                cap = cv2.VideoCapture(str(video_path))
                if not cap.isOpened():
                    print(f"Could not open video: {video_path}")
                    continue

                # Get video properties
                fps = cap.get(cv2.CAP_PROP_FPS)
                frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
                frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

                # Create interpolated gaze points
                smooth_gaze_data = self.interpolate_gaze_points(video_gaze_data, fps)

                # Create output video
                output_filename = f"{user_id}_{video_id}.mp4"
                output_path = self.output_folder / output_filename

                fourcc = cv2.VideoWriter_fourcc(*'mp4v')
                out = cv2.VideoWriter(str(output_path), fourcc, fps, (frame_width, frame_height))

                frame_count = 0
                last_gaze_time = smooth_gaze_data['time'].max()

                while cap.isOpened():
                    ret, frame = cap.read()
                    if not ret:
                        break

                    current_time = frame_count / fps

                    # Stop if we've passed the last gaze point
                    if current_time > last_gaze_time:
                        break

                    # Get interpolated gaze point for current time
                    current_point = smooth_gaze_data[
                        (smooth_gaze_data['time'] >= current_time) &
                        (smooth_gaze_data['time'] < current_time + 1/fps)
                    ]

                    if not current_point.empty:
                        x = int(current_point.iloc[0]['x'])
                        y = int(current_point.iloc[0]['y'])

                        # Draw dot with anti-aliasing
                        cv2.circle(frame, (x, y), 8, self.dot_color, -1, cv2.LINE_AA)

                        # Add subtle glow effect
                        cv2.circle(frame, (x, y), 12, self.dot_color, 2, cv2.LINE_AA)

                    out.write(frame)
                    frame_count += 1

                cap.release()
                out.release()
                print(f"Saved video to {output_path}")

            except Exception as e:
                print(f"Error processing video {video_id}: {str(e)}")
                continue

        


def main():
    # File paths
    video_folder = "../data/driving_videos"
    csv_path = "../data/normalized_gaze_data.csv"
    aggregate_csv_path = "../data/aggregate_gaze_data_by_video.csv"
    output_folder = "../data/user_gaze_videos"
    output_folder_aggregated = "../data/videos_with_all_gazes"

    try:
        creator = UserGazeVideoCreator(video_folder, csv_path, output_folder, aggregate_csv=aggregate_csv_path, output_folder_aggregated=output_folder_aggregated)

        # First create a sample video for one user
        sample_user = creator.users[0]
        print(f"\nCreating sample video for user {sample_user}")
        creator.create_video_for_user(sample_user, sample_mode=True)

        # Ask if user wants to process all videos
        response = input("\nDo you want to process all videos for all users? (y/n): ")
        if response.lower() == 'y':
            for user_id in creator.users:
                print(f"\nProcessing videos for user {user_id}")
                creator.create_video_for_user(user_id)

    except Exception as e:
        print(f"Error: {e}")

if __name__ == "__main__":
    main()