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
        csv_path = "../data/processed/binned_video_dat_wo_user.csv"
        video_folder_path = "../data/raw/driving_videos"
        output_folder_path = "../data/processed/driving_videos"

        self.video_folder = Path(video_folder_path)
        self.output_folder = Path(output_folder_path)
        self.output_folder.mkdir(parents=True, exist_ok=True)
        
        # Hot pink color in BGR format
        self.dot_color = (147, 20, 255) # RGB(255, 20, 147) in BGR

        # Read and normalize the gaze data
        print("Reading and normalizing gaze data...")
        self.gaze_data = pd.read_csv(csv_path)
        
        # Get unique users
        self.videoIds = self.gaze_data['videoId'].unique()
        print(f"Found {len(self.videoIds)} unique videoIds")
        

    def interpolate_gaze_points(self, video_gaze_data, fps):
        """
        Create smooth interpolation between gaze points
        """
        # Handle any duplicate timestamps first
        unique_data = video_gaze_data

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

    # def create_video_for_user(self, sample_mode=False):
    #     """
    #     Create videos for a specific user with their gaze overlay
    #     """
    #     # Get all videos for this user
    #     unique_videos = self.gaze_data['videoId'].unique()

    #     if sample_mode:
    #         unique_videos = unique_videos[:1]

    #     for video_id in unique_videos:
    #         try:
    #             # Get gaze data for this video
    #             video_gaze_data = user_data[user_data['videoId'] == video_id].sort_values('time')

    #             if len(video_gaze_data) < 4:
    #                 print(f"Not enough gaze points for video {video_id}")
    #                 continue

    #             # Find video file
    #             video_path = next(self.video_folder.glob(f"{video_id}.*"))
    #             if not video_path.exists():
    #                 print(f"Video file not found for {video_id}")
    #                 continue

    #             print(f"Processing video {video_id} for user {user_id}...")

    #             # Open video
    #             cap = cv2.VideoCapture(str(video_path))
    #             if not cap.isOpened():
    #                 print(f"Could not open video: {video_path}")
    #                 continue

    #             # Get video properties
    #             fps = cap.get(cv2.CAP_PROP_FPS)
    #             frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    #             frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    #             # Create interpolated gaze points
    #             smooth_gaze_data = self.interpolate_gaze_points(video_gaze_data, fps)

    #             # Create output video
    #             output_filename = f"{user_id}_{video_id}.mp4"
    #             output_path = self.output_folder / output_filename

    #             fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    #             out = cv2.VideoWriter(str(output_path), fourcc, fps, (frame_width, frame_height))

    #             frame_count = 0
    #             last_gaze_time = smooth_gaze_data['time'].max()

    #             while cap.isOpened():
    #                 ret, frame = cap.read()
    #                 if not ret:
    #                     break

    #                 current_time = frame_count / fps

    #                 # Stop if we've passed the last gaze point
    #                 if current_time > last_gaze_time:
    #                     break

    #                 # Get interpolated gaze point for current time
    #                 current_point = smooth_gaze_data[
    #                     (smooth_gaze_data['time'] >= current_time) &
    #                     (smooth_gaze_data['time'] < current_time + 1/fps)
    #                 ]

    #                 if not current_point.empty:
    #                     x = int(current_point.iloc[0]['x'])
    #                     y = int(current_point.iloc[0]['y'])

    #                     # Draw dot with anti-aliasing
    #                     cv2.circle(frame, (x, y), 8, self.dot_color, -1, cv2.LINE_AA)

    #                     # Add subtle glow effect
    #                     cv2.circle(frame, (x, y), 12, self.dot_color, 2, cv2.LINE_AA)

    #                 out.write(frame)
    #                 frame_count += 1

    #             cap.release()
    #             out.release()
    #             print(f"Saved video to {output_path}")

    #         except Exception as e:
    #             print(f"Error processing video {video_id}: {str(e)}")
    #             continue
    




    def add_gaze_per_video(self):
        df = self.gaze_data
        
        # Iterate through all unique userId and videoId combinations
        for selected_video in df[['videoId']].drop_duplicates()['videoId']:
            # Filter the data for the current userId and videoId
            print(selected_video)
            filtered_data = df[df['videoId'] == selected_video].reset_index(drop=True)
            
            # Skip if no data for the current combination
            if filtered_data.empty:
                continue
            
            # Interpolation function
            def interpolate_positions(times, xs, ys, fps):
                new_times = np.arange(times[0], times[-1], 1 / fps)
                new_xs = np.interp(new_times, times, xs)
                new_ys = np.interp(new_times, times, ys)
                return new_times, new_xs, new_ys

            # Extract time, x, y columns for interpolation
            times = filtered_data['time'].values
            xs = filtered_data['x'].values
            ys = filtered_data['y'].values

            # Load the video
            video_path = os.path.join(self.video_folder, f'{selected_video}.mp4')
            cap = cv2.VideoCapture(video_path)

            # Check if video is loaded correctly
            if not cap.isOpened():
                print(f"Warning: Could not open video {video_path}. Skipping...")
                continue

            # Get video properties
            fps = cap.get(cv2.CAP_PROP_FPS)
            frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
            frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

            # Set new dimensions for output video to match the normalized gaze data
            new_width, new_height = 1280, 960

            # Output video setup
            output_file = os.path.join(self.output_folder, f'{selected_video}.mp4')
            fourcc = cv2.VideoWriter_fourcc(*'mp4v')
            out = cv2.VideoWriter(output_file, fourcc, fps, (new_width, new_height))

            # Interpolate the positions
            interpolated_times, interpolated_xs, interpolated_ys = interpolate_positions(times, xs, ys, fps)

            # Initialize current frame index
            frame_num = 0

            while cap.isOpened():
                ret, frame = cap.read()
                if not ret or frame_num >= len(interpolated_times):
                    break

                # Resize the frame to (1280, 960)
                frame = cv2.resize(frame, (new_width, new_height))

                # Scale x and y coordinates to match resized frame dimensions
                x = int(interpolated_xs[frame_num] * (new_width / frame_width))
                y = int(interpolated_ys[frame_num] * (new_height / frame_height))

                x = int(np.clip(x, 0, frame_width - 1))
                y  = int(np.clip(y, 0, frame_height - 1))

                # Draw a circle at the interpolated position
                cv2.circle(frame, (x, y), 10, (0, 255, 0), -1)  # Green circle for normal

                # Write the current frame to the output
                out.write(frame)
                frame_num += 1

            # Release resources
            cap.release()
            out.release()
            print(f"Saved video to {self.output_folder}")

        cv2.destroyAllWindows()

        


def main():
    # File paths
    video_folder = "../data/driving_videos"
    csv_path = "../data/normalized_gaze_data.csv"
    aggregate_csv_path = "../data/aggregate_gaze_data_by_video.csv"
    output_folder = "../data/user_gaze_videos"
    output_folder_aggregated = "../data/videos_with_all_gazes"

    try:
        creator = UserGazeVideoCreator(video_folder, csv_path, output_folder, aggregate_csv=aggregate_csv_path, output_folder_aggregated=output_folder_aggregated)
        print(f"\nCreating video with gazes")
        creator.add_gaze_per_video()
       

    except Exception as e:
        print(f"Error: {e}")

if __name__ == "__main__":
    main()