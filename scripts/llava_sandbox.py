"""
llava_sandbox.py

A sandbox script for testing the LLaVA model.
"""

from pathlib import Path
import sys

import numpy as np
from PIL import Image

sys.path.append(str(Path(__file__).parent.parent))

# pylint: disable=wrong-import-position
from riskam.ml.llava import Llava

if __name__ == "__main__":

    IMG_PATHS = [
        # "ml_datasets/cs_robocup_2023/raw_dataset/RB_01/rgb/1537223827.468.png", # Robot GOAL
        # "ml_datasets/cs_robocup_2023/raw_dataset/RB_02/rgb/1537390896.430.png", # Woman with bags
        # "ml_datasets/cs_robocup_2023/raw_dataset/RB_08/rgb/1537360243.597.png", # Man in orange sitting on couch
        # "ml_datasets/cs_robocup_2023/raw_dataset/RB_08/rgb/1537360295.585.png"  # Surprised woman
        # "ml_datasets/cs_robocup_2023/raw_dataset/RB_08/rgb/1537360299.612.png"  # A man in red with face close to the camera, with a woman in the background
        # "ml_datasets/cs_robocup_2023/raw_dataset/RB_08/rgb/1537360313.658.png"  # A man in red with face really close to the camera, alone
        "ml_datasets/cs_robocup_2023/raw_dataset/RB_07/rgb/1537387705.642.png"  # Men arguing in front of the robot
    ]

    llava = Llava()
    llava.process_image(IMG_PATHS)

    for img_path in IMG_PATHS:
        image = Image.open(img_path)
        image.show()
