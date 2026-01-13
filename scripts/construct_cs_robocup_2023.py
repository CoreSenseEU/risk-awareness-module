"""
construct_cs_robocup.py

Constructs the CS RoboCup dataset from the raw images extracted from the ROS2 bag.
"""

from pathlib import Path
import sys

sys.path.append(str(Path(__file__).parent.parent))

# pylint: disable=wrong-import-position
from riskam.data import cs_robocup


if __name__ == "__main__":
    # Detect humans in frames: this will be used to construct the dataset
    cs_robocup.detect_humans_in_frames()

    # Second, compute the distances between the detected humans and the camera using depth images
    cs_robocup.compute_human_distances()

    # Finally, construct the dataset
    cs_robocup.construct_cs_robocup()

    # Extract the features
    cs_robocup.extract_features()
