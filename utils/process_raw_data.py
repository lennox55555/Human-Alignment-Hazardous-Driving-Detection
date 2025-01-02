import os
import subprocess

# Path to the folder containing the video files
video_folder = "/Users/lennoxanderson/Documents/Research/RawDrivingData" 

# Output folder for combined videos
output_folder = "/Users/lennoxanderson/Documents/Research/ProcessedDrivingData"  
os.makedirs(output_folder, exist_ok=True)

# Camera types to process
camera_types = ["front", "left", "right", "back"]

def combine_videos(camera_type):
    # Find all video files for the specific camera type
    video_files = [
        f for f in os.listdir(video_folder)
        if camera_type in f and f.endswith(".mp4")
    ]
    
    # Sort files by filename to ensure sequential order
    video_files.sort()
    
    # Create a temporary text file listing all video files for FFmpeg
    list_file_path = os.path.join(output_folder, f"{camera_type}_files.txt")
    with open(list_file_path, "w") as list_file:
        for video in video_files:
            list_file.write(f"file '{os.path.join(video_folder, video)}'\n")
    
    # Output video path
    output_video_path = os.path.join(output_folder, f"{camera_type}_combined.mp4")
    
    # Run FFmpeg to concatenate the videos
    ffmpeg_command = [
        "ffmpeg",
        "-f", "concat",
        "-safe", "0",
        "-i", list_file_path,
        "-c", "copy",
        output_video_path
    ]
    
    print(f"Combining {camera_type} videos...")
    subprocess.run(ffmpeg_command, check=True)
    print(f"{camera_type.capitalize()} video combined: {output_video_path}")

# Process each camera type
for camera in camera_types:

