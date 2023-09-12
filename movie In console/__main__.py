import moviepy
from moviepy.editor import VideoFileClip
import numpy as np
from PIL import Image
import os

def video_to_images(video_path, output_folder):
    # Make the output folder if it doesn't exist
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)
    
    # Load video
    clip = VideoFileClip(video_path)
    
    # Initialize an empty list to store frames
    frames = []
    
    # Iterate through video frames
    for frame in clip.iter_frames(fps=24, dtype='uint8'):
        img = Image.fromarray(frame)
        frames.append(img)

    # Save frames as images
    for i, frame in enumerate(frames):
        frame.save(f"{output_folder}/frame_{i}.png")

if __name__ == "__main__":
    video_path = input("Enter the filepath to the movie: ")  # Replace with the path to your MP4 video
      # Folder to save the images
    video_to_images(video_path, output_folder)