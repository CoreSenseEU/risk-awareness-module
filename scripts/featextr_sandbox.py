"""
featextr_sandbox.py

A sandbox for risk awareness feature extraction.
"""

import sys
from pathlib import Path

import cv2

sys.path.append(str(Path(__file__).parent.parent))

# pylint: disable=wrong-import-position
from riskam.ml import featextr
from riskam import visualization as vis
from riskam.score import RiskScorer

TEST_RESULTS_DIR = Path("test_results")
RISK_SCORE_DIR = TEST_RESULTS_DIR / "risk_scores"

if __name__ == "__main__":
    IMG_PATHS = [
        "ml_datasets/cs_robocup_2023/raw_dataset/RB_02/rgb/1537390896.430.png",
        "ml_datasets/cs_robocup_2023/raw_dataset/RB_08/rgb/1537360243.597.png",
        "ml_datasets/cs_robocup_2023/raw_dataset/RB_08/rgb/1537360295.585.png",
        "ml_datasets/cs_robocup_2023/raw_dataset/RB_08/rgb/1537360299.612.png",
        "ml_datasets/cs_robocup_2023/raw_dataset/RB_08/rgb/1537360313.658.png",
        "ml_datasets/cs_robocup_2023/raw_dataset/RB_07/rgb/1537387705.642.png",
        "ml_datasets/cs_robocup_2023/raw_dataset/RB_01/rgb/1537223839.849.png",
        "ml_datasets/cs_robocup_2023/raw_dataset/RB_01/rgb/1537223827.468.png",
    ]

    scorer = RiskScorer()
    RISK_SCORE_DIR.mkdir(exist_ok=True, parents=True)

    for img_path in IMG_PATHS:
        cv_image = cv2.imread(img_path)
        if cv_image is None:
            print(f"Could not load {img_path}")
            continue

        human_bboxes, depth_viz, risk_features, track_ids = (
            featextr.extract_human_risk_awareness_features(cv_image, track_bboxes=False)
        )
        risk_score, max_risk_idx, _ = scorer.score(risk_features, track_ids=track_ids)

        annotated = vis.visualize_risk(
            cv_image, human_bboxes, depth_viz, risk_features, risk_score, max_risk_idx
        )
        out_path = RISK_SCORE_DIR / Path(img_path).name
        cv2.imwrite(str(out_path), annotated)
        print(f"{img_path}: risk={risk_score:.3f}")
