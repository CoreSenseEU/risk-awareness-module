"""
riskam.video

The video processing module.
"""

import glob
import os

import cv2

# pylint: disable=no-member


def images_to_video(
    input_folder, output_filename="output.avi", fps=30, image_extension="jpg"
) -> None:
    # Use glob to get list of images in the folder, sorted by name
    image_paths = sorted(glob.glob(os.path.join(input_folder, f"*.{image_extension}")))
    if not image_paths:
        print("No images found in the specified folder.")
        return

    # Read the first frame to get dimensions
    frame = cv2.imread(image_paths[0])
    height, width, _ = frame.shape
    size = (width, height)

    # Define codec and create VideoWriter object
    # For example: XVID, MJPG, or DIVX; you can adjust this if needed.
    fourcc = cv2.VideoWriter_fourcc(*"XVID")
    out = cv2.VideoWriter(output_filename, fourcc, fps, size)

    for path in image_paths:
        img = cv2.imread(path)
        out.write(img)

    out.release()
    print(f"Video saved as {output_filename}")
