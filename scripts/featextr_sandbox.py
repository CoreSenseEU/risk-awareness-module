"""
featextr_sandbox.py

A sandbox for risk awareness feature extraction.
"""

from pathlib import Path
import sys

from PIL import Image

sys.path.append(str(Path(__file__).parent.parent))

# pylint: disable=wrong-import-position
from riskam.ml import featextr
from riskam import score, visualization as vis

TEST_RESULTS_DIR = Path("test_results")
MODEL_VIS_DIR = TEST_RESULTS_DIR / "model_vis"
RISK_SCORE_DIR = TEST_RESULTS_DIR / "risk_scores"

if __name__ == "__main__":
    IMG_PATHS = [
        "ml_datasets/cs_robocup_2023/raw_dataset/RB_02/rgb/1537390896.430.png",  # Woman with bags
        "ml_datasets/cs_robocup_2023/raw_dataset/RB_08/rgb/1537360243.597.png",  # Man in orange sitting on couch
        "ml_datasets/cs_robocup_2023/raw_dataset/RB_08/rgb/1537360295.585.png",  # Surprised woman
        "ml_datasets/cs_robocup_2023/raw_dataset/RB_08/rgb/1537360299.612.png",  # A man in red with face close to the camera, with a woman in the background
        "ml_datasets/cs_robocup_2023/raw_dataset/RB_08/rgb/1537360313.658.png",  # A man in red with face really close to the camera, alone
        "ml_datasets/cs_robocup_2023/raw_dataset/RB_07/rgb/1537387705.642.png",  # Men arguing in front of the robot
        "ml_datasets/cs_robocup_2023/raw_dataset/RB_01/rgb/1537223839.849.png",  # Several people looking from the doorway
        "ml_datasets/cs_robocup_2023/raw_dataset/RB_01/rgb/1537223827.468.png",  # Robot GOAL
    ]

    for img_path in IMG_PATHS:
        with Image.open(img_path) as image:
            human_bboxes, rel_depth, risk_features = (
                featextr.extract_human_risk_awareness_features(
                    img_path, track_bboxes=False
                )
            )
            risk_score, max_risk_idx = score.risk_awareness_score(risk_features)

            # model_vis_output_path = MODEL_VIS_DIR / Path(img_path).name
            risk_score_output_path = RISK_SCORE_DIR / Path(img_path).name

            # MODEL_VIS_DIR.mkdir(exist_ok=True, parents=True)
            RISK_SCORE_DIR.mkdir(exist_ok=True, parents=True)

            # if visualization is not None:
            #     visualization.save(model_vis_output_path)

            vis.visualize_risk(
                img_path,
                None,
                human_bboxes,
                rel_depth,
                risk_features,
                risk_score,
                max_risk_idx,
            )
