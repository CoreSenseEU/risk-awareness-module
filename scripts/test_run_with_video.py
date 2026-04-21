"""
test_run_with_video.py

Runs the risk awareness model on a sequence of images, creating
1) a video from the raw images and 2) a video with the risk score overlay.

Note: depth proximity scores are 0 in this offline mode (no depth sensor data).
"""

import argparse
import sys
from pathlib import Path

import cv2
from tqdm import tqdm

sys.path.append(str(Path(__file__).parent.parent))

# pylint: disable=wrong-import-position
from riskam.data.ml_datasets import DATASETS
from riskam.ml import featextr
from riskam import video, visualization as vis
from riskam.score import RiskScorer

DEFAULT_W_PROX = 0.65
DEFAULT_W_GAZE = 0.25
DEFAULT_W_XPOS = 0.1

TEST_RESULTS_DIR = Path("test_results")
RISK_SCORE_MASTER_DIR = TEST_RESULTS_DIR / "risk_scores"
VIDEO_DIR = TEST_RESULTS_DIR / "videos"

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Run the risk awareness model on a sequence of images."
    )
    parser.add_argument(
        "dataset", type=str, choices=DATASETS.keys(), help="The dataset to process."
    )
    parser.add_argument("run", type=str, help="The run to process.")
    args = parser.parse_args()

    IMG_DIR = DATASETS[args.dataset]["img_dir"]
    if args.dataset == "cs_robocup_2023":
        IMG_DIR = IMG_DIR / args.run / "rgb"

    VIDEO_DIR.mkdir(exist_ok=True, parents=True)
    risk_score_dir = RISK_SCORE_MASTER_DIR / args.dataset / args.run
    risk_score_dir.mkdir(exist_ok=True, parents=True)

    scorer = RiskScorer()

    for img_path in tqdm(sorted(IMG_DIR.iterdir())):
        cv_image = cv2.imread(str(img_path))
        if cv_image is None:
            continue

        human_bboxes, depth_viz, risk_features, track_ids = (
            featextr.extract_human_risk_awareness_features(
                cv_image,
                depth_image_m=None,
                track_bboxes=True,
            )
        )
        risk_score, max_risk_idx, _ = scorer.score(
            risk_features,
            track_ids=track_ids,
            w_proximity=DEFAULT_W_PROX,
            w_gaze=DEFAULT_W_GAZE,
            w_position=DEFAULT_W_XPOS,
        )

        annotated = vis.visualize_risk(
            cv_image, human_bboxes, depth_viz, risk_features, risk_score, max_risk_idx
        )
        out_path = risk_score_dir / Path(img_path).name
        cv2.imwrite(str(out_path), annotated)

    video.images_to_video(
        risk_score_dir,
        VIDEO_DIR / f"{args.dataset}_{args.run}_risk.avi",
        image_extension="png",
        fps=10,
    )
    video.images_to_video(
        IMG_DIR,
        VIDEO_DIR / f"{args.dataset}_{args.run}_raw.avi",
        image_extension="png",
    )
