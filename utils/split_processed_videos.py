import os
import subprocess

# Input folder containing the combined videos
input_folder = "/Users/lennoxanderson/Documents/Research/ProcessedDrivingData"

# Output folder for split videos
output_folder = "/Users/lennoxanderson/Documents/Research/SplitDrivingData"
os.makedirs(output_folder, exist_ok=True)

# Camera type to process
camera_type = "front"

def split_video(camera_type):
    # Locate the combined video for the specified camera type
    input_video_path = os.path.join(input_folder, f"{camera_type}_combined.mp4")

    if not os.path.exists(input_video_path):
        print(f"Combined video for {camera_type} not found: {input_video_path}")
        return

    # Output file template for the split videos
    output_video_template = os.path.join(output_folder, f"{camera_type}_%03d.mp4")

    # Run FFmpeg to split the video into 15-second segments
    ffmpeg_command = [
        "ffmpeg",
        "-i", input_video_path,
        "-c", "copy",
        "-map", "0",
        "-segment_time", "15",
        "-f", "segment",
        "-reset_timestamps", "1",
        output_video_template
    ]

    print(f"Splitting {camera_type} video into 15-second segments...")
    subprocess.run(ffmpeg_command, check=True)
    print(f"{camera_type.capitalize()} video split completed. Files saved to {output_folder}")

# Process the front camera video
split_video(camera_type)