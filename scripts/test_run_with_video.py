"""
test_run_with_video.py

Runs the risk awareness model on a sequence of images, creating 1) a video from the images,
and 2) a video with the risk score overlay.
"""

import argparse
from pathlib import Path
import sys

from tqdm import tqdm

sys.path.append(str(Path(__file__).parent.parent))

# pylint: disable=wrong-import-position
from riskam.data.ml_datasets import DATASETS
from riskam.ml import featextr
from riskam import score, video, visualization as vis

DEFAULT_W_PROX = 0.65
DEFAULT_W_GAZE = 0.25
DEFAULT_W_XPOS = 0.1
DEFAULT_GAMMA = 1
DEFAULT_GAZE_THRESHOLD_LOWER = 0.1
DEFAULT_GAZE_THRESHOLD_UPPER = 0.2

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

    # Establish the dirs
    IMG_DIR = DATASETS[args.dataset]["img_dir"]

    if args.dataset == "cs_robocup_2023":
        IMG_DIR = IMG_DIR / args.run / "rgb"

    VIDEO_DIR.mkdir(exist_ok=True, parents=True)

    risk_score_dir = RISK_SCORE_MASTER_DIR / args.dataset / args.run
    risk_score_dir.mkdir(exist_ok=True, parents=True)

    # Perform the risk awareness analysis
    for img_path in tqdm(sorted(IMG_DIR.iterdir())):
        human_bboxes, rel_depth, risk_features = (
            featextr.extract_human_risk_awareness_features(
                img_path,
                depth_gamma=DEFAULT_GAMMA,
                gaze_face_offset_lower_threshold_ratio=DEFAULT_GAZE_THRESHOLD_LOWER,
                gaze_face_offset_upper_threshold_ratio=DEFAULT_GAZE_THRESHOLD_UPPER,
                track_bboxes=True,
            )  # noqa: E501
        )
        risk_score, max_risk_idx = score.risk_awareness_score(risk_features)

        risk_score_output_path = risk_score_dir / Path(img_path).name

        vis.visualize_risk(
            img_path,
            risk_score_output_path,
            human_bboxes,
            rel_depth,
            risk_features,
            risk_score,
            max_risk_idx,
        )

    # Process the videos
    video.images_to_video(
        risk_score_dir,
        VIDEO_DIR / f"{args.dataset}_{args.run}_risk.avi",
        image_extension="png",
        fps=10,
    )
    video.images_to_video(
        IMG_DIR, VIDEO_DIR / f"{args.dataset}_{args.run}_raw.avi", image_extension="png"
    )
