"""
riskam.experiments

The experiment module for RiskAM
"""

import json
from pathlib import Path
from time import time

import cv2
from tqdm import tqdm

from riskam.data.cs_robocup_2023 import CSRobocup2023DepthIndex
from riskam.data.ml_datasets import DATASETS
from riskam.ml import featextr
from riskam.ml.humandet import FRONTAL_PITCH_RATIO_DEFAULT, SIGMA_PITCH_DEFAULT, SIGMA_YAW_DEFAULT
from riskam import score, visualization as vis, video
from riskam.score import RiskScorer


# Dir and file names/paths
EXP_ROOT_DIR = Path(__file__).parent.parent / "exp_results"
RISK_IMAGES_DIRNAME = "risk_images"
RAW_VIDEO_FNAME = "raw_video.avi"
RISK_VIDEO_FNAME = "risk_video.avi"
RESULTS_JSON_FNAME = "results.json"
PREDICTIONS_JSON_FNAME = "predictions.json"
RAW_PREDICTIONS_JSON_FNAME = "raw_predictions.json"


# Parameters — updated for the new pipeline.
# Proximity is now computed from RealSense depth when it is available for the
# frame (via CSRobocup2023DepthIndex); otherwise the sub-score falls back to 0.
RISK_SCORE_WEIGHTS = [
    {"proximity": 0.7, "gaze": 0.25, "position": 0.05},
    {"proximity": 0.475, "gaze": 0.475, "position": 0.05},
    {"proximity": 0.25, "gaze": 0.7, "position": 0.05},
    {"proximity": 0.65, "gaze": 0.25, "position": 0.1},
    {"proximity": 0.45, "gaze": 0.45, "position": 0.1},
    {"proximity": 0.25, "gaze": 0.65, "position": 0.1},
]
GAZE_SIGMA_YAW_VALUES = [0.2, 0.3, 0.5]
GAZE_SIGMA_PITCH_VALUES = [0.3, 0.5]

# pylint: disable=no-member


def _load_ground_truth(dataset: str, run: str | None) -> dict | None:
    """
    Loads the ground truth labels for the given dataset (and run if applicable).
    """
    # Load the ground truth annotations
    try:
        ground_truth_path = DATASETS[dataset]["ground_truth_path"]
    except KeyError:
        print(f"Unknown dataset: '{dataset}', skipping.")
        return None

    if not ground_truth_path.exists():
        print(f"Ground truth annotations not found for '{dataset}', skipping.")
        return None

    with open(ground_truth_path, "r", encoding="utf-8") as f:
        ground_truth = json.load(f)

    # If RoboCup 2023, load the ground truth for the specific run
    if dataset == "cs_robocup_2023" and run is not None:
        try:
            ground_truth = ground_truth[run]
        except KeyError:
            print(f"Ground truth annotations not found for '{run}', skipping.")
            return None
    else:
        print("No run specified for RoboCup 2023, skipping.")
        return None

    return ground_truth


def _eval_prediction(gt: int, pred: float) -> str:
    """
    Evaluates the prediction against the ground truth label.

    Returns 'ok' if the risk score matches the annotated class,
    'under' if the score underestimates the risk, and 'over' if it overestimates.
    """
    if gt == 0:
        return "correct" if pred == 0.0 else "overestimate"
    else:
        if gt == 1:
            lower_bound = score.VERY_SMALL_RISK_VALUE
        else:
            lower_bound = score.RISK_SCORE_BREAKPOINTS[gt - 1]

        upper_bound = score.RISK_SCORE_BREAKPOINTS[gt]

        if lower_bound <= pred < upper_bound:
            return "correct"
        elif pred < lower_bound:
            return "underestimate"
        else:
            return "overestimate"


def inspect_predictions(dataset: str, pred_type: str, run: str | None = None) -> None:
    """
    Inspect predictions of the given type ("correct", "underestimate", "overestimate")
    resulting from an experiment.
    """

    # Establish the dataset img dir
    img_dir = DATASETS[dataset]["img_dir"]

    if dataset == "cs_robocup_2023":
        img_dir = img_dir / run / "rgb"

    # Load the predictions
    predictions_path = EXP_ROOT_DIR / dataset / run / PREDICTIONS_JSON_FNAME

    if not predictions_path.exists():
        print("Predictions not found, run the experiment first.")
        return

    with open(predictions_path, "r", encoding="utf-8") as f:
        predictions = json.load(f)

    for img_name in predictions[pred_type]:
        img_path = img_dir / img_name

        # Load the image
        image = cv2.imread(str(img_path))

        if image is None:
            print(f"Failed to load the image: {img_path}")
            continue

        # Display the image
        cv2.imshow(f"{pred_type.capitalize()} predictions", image)

        # Wait indefinitely for a key press
        key = cv2.waitKeyEx(0)

        # ESC key (27) to exit the browser
        if key == 27:
            print("Exiting the browser...")
            break

    cv2.destroyAllWindows()


def run_experiment(
    dataset: str,
    params: dict,
    run: str | None = None,
    output_images: bool = False,
    overwrite_existing: bool = False,
) -> None:
    """
    Run the experiment for the given dataset with the given experimental params.
    """
    # Load the ground truth labels
    ground_truth = _load_ground_truth(dataset, run)

    if ground_truth is None:
        return

    # Establish the images directory and per-dataset depth index
    img_dir = DATASETS[dataset]["img_dir"]
    depth_index = None

    if dataset == "cs_robocup_2023":
        img_dir = img_dir / run / "rgb"
        depth_index = CSRobocup2023DepthIndex(run)
        if not depth_index:
            print(
                f"[warn] no depth frames found for '{run}'; "
                "proximity sub-score will be 0. "
                "Run scripts/extract_ros2_dataset.py to extract depth."
            )

    # Establish the params slug to identify the experiment
    params_slug = _params_slug(params)

    print(
        f"+++ EXPERIMENT {f"{dataset} / {run} / {params_slug}" if run else f"{dataset} / {params_slug}"} STARTED +++"
    )

    # Establish the output directories
    experiment_dir = EXP_ROOT_DIR / dataset / run / params_slug

    # If not overwriting and the directory exists, stop
    if experiment_dir.exists() and not overwrite_existing:
        print("+++ EXPERIMENT ALREADY COMPLETED, SKIPPING +++")
        return

    risk_img_output_dir = experiment_dir / RISK_IMAGES_DIRNAME
    risk_img_output_dir.mkdir(exist_ok=True, parents=True)

    # Establish the experimental metrics dict
    metrics = {
        "correct": 0,
        "underestimate": 0,
        "overestimate": 0,
        "total": 0,
    }

    predictions = {
        "correct": [],
        "underestimate": [],
        "overestimate": [],
    }

    raw_predictions = {}

    times = []

    scorer = RiskScorer()

    # Perform the risk awareness analysis
    for img_path in tqdm(sorted(img_dir.iterdir())):
        # Skip if there is no ground truth for the image
        if img_path.name not in ground_truth:
            continue

        # Load image with OpenCV (featextr expects a BGR ndarray)
        cv_image = cv2.imread(str(img_path))
        if cv_image is None:
            continue

        # Load the nearest-neighbour depth frame (if available).
        depth_image_m = depth_index.load_for_rgb(img_path) if depth_index else None

        # Extract bboxes, depth visualisation, and risk features.
        t_start = time()
        human_bboxes, depth_viz, risk_features, track_ids = (
            featextr.extract_human_risk_awareness_features(
                cv_image,
                depth_image_m=depth_image_m,
                gaze_sigma_yaw=params["gaze_sigma_yaw"],
                gaze_sigma_pitch=params["gaze_sigma_pitch"],
                gaze_frontal_pitch_ratio=FRONTAL_PITCH_RATIO_DEFAULT,
                track_bboxes=False,
            )
        )
        # Compute the risk score and the index of the highest risk bbox
        risk_score, max_risk_idx, _ = scorer.score(
            risk_features,
            w_proximity=params["w_prox"],
            w_gaze=params["w_gaze"],
            w_position=params["w_pos"],
        )
        times.append(time() - t_start)

        # Evaluate the prediction
        eval_result = _eval_prediction(ground_truth[img_path.name], risk_score)
        metrics["total"] += 1
        metrics[eval_result] += 1
        predictions[eval_result].append(img_path.name)
        raw_predictions[img_path.name] = risk_score

        # Visualize & store the risk visualization (what the model sees)
        if output_images:
            risk_img_output_path = risk_img_output_dir / Path(img_path).name
            annotated = vis.visualize_risk(
                cv_image, human_bboxes, depth_viz, risk_features, risk_score, max_risk_idx
            )
            cv2.imwrite(str(risk_img_output_path), annotated)
    # Calculate the average time per image
    avg_time = sum(times) / len(times)
    metrics["avg_time"] = avg_time

    # Save the results
    results_path = experiment_dir / RESULTS_JSON_FNAME
    results_path.parent.mkdir(exist_ok=True, parents=True)

    predictions_path = experiment_dir / PREDICTIONS_JSON_FNAME
    raw_predictions_path = experiment_dir / RAW_PREDICTIONS_JSON_FNAME

    with open(results_path, "w", encoding="utf-8") as f:
        json.dump(metrics, f, indent=4)

    with open(predictions_path, "w", encoding="utf-8") as f:
        json.dump(predictions, f, indent=4)

    with open(raw_predictions_path, "w", encoding="utf-8") as f:
        json.dump(raw_predictions, f, indent=4)

    # Create the videos
    if output_images:
        # raw_video_path = experiment_dir / RAW_VIDEO_FNAME
        risk_video_path = experiment_dir / RISK_VIDEO_FNAME

        # video.images_to_video(
        #     img_dir,
        #     raw_video_path,
        #     image_extension="png",
        #     fps=30,
        # )
        video.images_to_video(
            risk_img_output_dir,
            risk_video_path,
            image_extension="png",
            fps=10,
        )

    print(
        f"+++ EXPERIMENT {f"{dataset} / {run} / {params_slug}" if run else f"{dataset} / {params_slug}"} COMPLETE +++"
    )
    print("Results:")
    print(f"    - Total images: {metrics['total']}")
    print(
        f"    - Correct predictions: {metrics['correct']} ({metrics['correct']/metrics['total']*100:.2f}%)"
    )
    print(
        f"    - Underestimates: {metrics['underestimate']} ({metrics['underestimate']/metrics['total']*100:.2f}%)"
    )
    print(
        f"    - Overestimates: {metrics['overestimate']} ({metrics['overestimate']/metrics['total']*100:.2f}%)"
    )
    print(f"    - Average time per image: {avg_time:.2f}s")


def run_experiments(
    dataset: str,
    run: str | None = None,
    output_images: bool = False,
    overwrite_existing: bool = False,
) -> None:
    """
    Run the experiments for the given dataset across all parameter configs.
    """
    for risk_weights in RISK_SCORE_WEIGHTS:
        for sigma_yaw in GAZE_SIGMA_YAW_VALUES:
            for sigma_pitch in GAZE_SIGMA_PITCH_VALUES:
                run_experiment(
                    dataset,
                    {
                        "w_prox": risk_weights["proximity"],
                        "w_gaze": risk_weights["gaze"],
                        "w_pos": risk_weights["position"],
                        "gaze_sigma_yaw": sigma_yaw,
                        "gaze_sigma_pitch": sigma_pitch,
                    },
                    run,
                    output_images,
                    overwrite_existing,
                )


def _params_slug(params: dict) -> str:
    """
    Returns a slug (string representation) for the given params dict.
    """
    return "-".join([f"{k}_{v}" for k, v in params.items()])
