"""
featextr_sandbox.py

A sandbox for risk awareness feature extraction. Hardcoded paths below point
at the RB_XX runs of the CS RoboCup 2023 dataset; the nearest matching depth
frame is loaded for each RGB frame.

The cs_robocup_2023 dataset was recorded on the TIAGo + PAL Xtion platform
(``riskam.platforms.TIAGO_XTION``) — structured light with a ~0.6 m near-clip
dead zone, heavy zero-fill, and a 2.5 m safety distance. The platform object
carries every parameter the sandbox needs, pulled from
``DATASETS['cs_robocup_2023']['platform']``. See ``docs/improvement_plan.md``
§3.4 for the full rationale.
"""

import sys
from pathlib import Path

import cv2

sys.path.append(str(Path(__file__).parent.parent))

# pylint: disable=wrong-import-position
from riskam.data.cs_robocup_2023 import CSRobocup2023DepthIndex
from riskam.data.ml_datasets import DATASETS
from riskam.ml import featextr
from riskam.ml.subscores import FrameInputs
from riskam import visualization as vis
from riskam.score import RiskScorer

TEST_RESULTS_DIR = Path("test_results")
RISK_SCORE_DIR = TEST_RESULTS_DIR / "risk_scores"

# Recording-platform calibration (see module docstring).
PLATFORM = DATASETS["cs_robocup_2023"]["platform"]

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

    # The sandbox is a per-frame visual sanity check, not a temporal sequence:
    # the 8 frames come from unrelated runs. RiskScorer averages over the last
    # ``n_frames_aggregate`` frames, which would smear a high-risk close-up
    # frame into a "scene" of unrelated low-risk ones. We reset the scorer
    # between frames so each one's score reflects only that frame.

    # One depth index per RB run, built lazily.
    depth_indices: dict[str, CSRobocup2023DepthIndex] = {}

    for img_path_str in IMG_PATHS:
        img_path = Path(img_path_str)
        cv_image = cv2.imread(str(img_path))
        if cv_image is None:
            print(f"Could not load {img_path}")
            continue

        run = img_path.parent.parent.name  # .../RB_XX/rgb/{ts}.png → RB_XX
        if run not in depth_indices:
            depth_indices[run] = CSRobocup2023DepthIndex(run)
        depth_image_m = depth_indices[run].load_for_rgb(img_path)
        if depth_image_m is None:
            print(f"No matching depth frame for {img_path}; skipping")
            continue

        scorer.reset()
        result = featextr.extract(
            FrameInputs(rgb=cv_image, depth_m=depth_image_m, cmd_vel=None),
            d_safe=PLATFORM.d_safe_m,
            depth_near_clip_m=PLATFORM.sensor.near_clip_m,
            near_clip_valid_frac_max=PLATFORM.sensor.valid_frac_max,
            track_bboxes=False,
        )
        risk_score, max_risk_idx, _ = scorer.score(
            result.features, track_ids=result.track_ids
        )

        annotated = vis.visualize_risk(
            cv_image,
            result.human_bboxes,
            result.features,
            risk_score,
            max_risk_idx,
        )
        out_path = RISK_SCORE_DIR / img_path.name
        cv2.imwrite(str(out_path), annotated)
        print(f"{img_path}: risk={risk_score:.3f}")
