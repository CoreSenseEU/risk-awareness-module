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
from riskam.data.splits import (
    CS_ROBOCUP_2023_SPLIT_PATH,
    VAL_BUCKETS,
    filter_by_split,
    load_split,
)
from riskam.eval_metrics import classification_report
from riskam.ml import featextr, humandet
from riskam.ml.humandet import FRONTAL_PITCH_RATIO_DEFAULT, SIGMA_PITCH_DEFAULT, SIGMA_YAW_DEFAULT
from riskam.ml.subscores import FrameInputs
from riskam.provenance import reproducibility_metadata
from riskam.sweep_config import SweepConfig, load_sweep_config
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


# Sweep grid (weights + gaze sigmas) now lives in a YAML config — see
# ``configs/sweeps/default.yaml`` and ``riskam.sweep_config``. The offline
# pipeline loads depth from ``CSRobocup2023DepthIndex``; frames with no
# matching depth are skipped (RGB + depth is RiskAM's supported minimum).

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


def inspect_predictions(
    dataset: str,
    pred_type: str,
    run: str | None = None,
    split: str | None = None,
) -> None:
    """
    Inspect predictions of the given type ("correct", "underestimate", "overestimate")
    resulting from an experiment.
    """

    # Establish the dataset img dir
    img_dir = DATASETS[dataset]["img_dir"]

    if dataset == "cs_robocup_2023":
        img_dir = img_dir / run / "rgb"

    # Load the predictions
    bucket_dir = split if split is not None else "all"
    predictions_path = (
        EXP_ROOT_DIR / dataset / bucket_dir / run / PREDICTIONS_JSON_FNAME
    )

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
    split: str | None = None,
) -> None:
    """
    Run the experiment for the given dataset with the given experimental params.

    ``split`` selects which val/test bucket to evaluate against. ``None``
    means "all annotated frames" (no split filter); ``"val"`` or ``"test"``
    restrict to that bucket using the canonical split file. Results land in
    ``exp_results/<dataset>/<bucket>/<run>/<params_slug>/`` where
    ``bucket`` is ``"all"`` when ``split is None``.
    """
    # Load the ground truth labels
    ground_truth = _load_ground_truth(dataset, run)

    if ground_truth is None:
        return

    # Apply split filter if requested.
    split_meta: dict | None = None
    if split is not None:
        if split not in VAL_BUCKETS:
            print(f"[error] split must be one of {VAL_BUCKETS}; got {split!r}.")
            return
        if dataset != "cs_robocup_2023":
            print(f"[warn] split filtering not supported for '{dataset}'; ignoring.")
            split = None
        elif not CS_ROBOCUP_2023_SPLIT_PATH.exists():
            print(
                f"[error] split file not found at {CS_ROBOCUP_2023_SPLIT_PATH}. "
                "Run scripts/generate_split.py first."
            )
            return
        else:
            split_obj = load_split(CS_ROBOCUP_2023_SPLIT_PATH)
            ground_truth = filter_by_split(ground_truth, run, split_obj, split)
            if not ground_truth:
                print(f"[info] split={split} has no frames for run={run}; skipping.")
                return
            split_meta = {
                "scheme": "stratified_within_run",
                "test_fraction": split_obj.test_fraction,
                "seed": split_obj.seed,
                "bucket": split,
            }

    # Establish the images directory and per-dataset depth index
    img_dir = DATASETS[dataset]["img_dir"]
    depth_index = None

    if dataset == "cs_robocup_2023":
        img_dir = img_dir / run / "rgb"
        depth_index = CSRobocup2023DepthIndex(run)
        if not depth_index:
            print(
                f"[error] no depth frames found for '{run}'; RiskAM's "
                "supported minimum is RGB + absolute depth. "
                "Run scripts/extract_ros2_dataset.py to extract depth, "
                "then re-run this experiment."
            )
            return

    # Establish the params slug to identify the experiment
    params_slug = _params_slug(params)

    print(
        f"+++ EXPERIMENT {f"{dataset} / {run} / {params_slug}" if run else f"{dataset} / {params_slug}"} STARTED +++"
    )

    # Output directory layout: exp_results/<dataset>/<bucket>/<run>/<slug>/
    # where bucket is "all" when no split filter is in effect. This keeps
    # val and test runs as siblings under the dataset root for the
    # cross-run summarizer.
    bucket_dir = split if split is not None else "all"
    experiment_dir = EXP_ROOT_DIR / dataset / bucket_dir / run / params_slug

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

    # Parallel arrays for per-class / regression metrics aggregated post-loop.
    y_true: list[int] = []
    y_pred_continuous: list[float] = []

    times = []

    scorer = RiskScorer()

    # Each offline run is a logically separate session; clear any lingering
    # per-track velocity history from prior runs/configs so the approach
    # sub-score starts fresh.
    humandet.reset_velocity_history()

    # Perform the risk awareness analysis
    for img_path in tqdm(sorted(img_dir.iterdir())):
        # Skip if there is no ground truth for the image
        if img_path.name not in ground_truth:
            continue

        # Load image with OpenCV (featextr expects a BGR ndarray)
        cv_image = cv2.imread(str(img_path))
        if cv_image is None:
            continue

        # Load the nearest-neighbour depth frame. RGB + depth is RiskAM's
        # supported minimum, so a missing depth frame means this RGB frame
        # cannot be evaluated and is skipped (with a per-experiment tally
        # via `skipped_no_depth`).
        depth_image_m = depth_index.load_for_rgb(img_path) if depth_index else None
        if depth_image_m is None:
            metrics["skipped_no_depth"] = metrics.get("skipped_no_depth", 0) + 1
            continue

        # Extract risk features via the sub-score contract.
        # track_bboxes=True enables ByteTrack, required for the approach
        # sub-score (per-track velocity estimation). cmd_vel is None offline
        # until T3.3 adds bag-backed velocity replay; x_offset will therefore
        # report FALLBACK status (centre-offset heuristic).
        t_start = time()
        result = featextr.extract(
            FrameInputs(rgb=cv_image, depth_m=depth_image_m, cmd_vel=None),
            gaze_sigma_yaw=params["gaze_sigma_yaw"],
            gaze_sigma_pitch=params["gaze_sigma_pitch"],
            gaze_frontal_pitch_ratio=FRONTAL_PITCH_RATIO_DEFAULT,
            gaze_algorithm=params.get("gaze_algorithm", "head_pose"),
            track_bboxes=True,
        )
        # Compute the risk score and the index of the highest risk bbox
        risk_score, max_risk_idx, _ = scorer.score(
            result.features,
            track_ids=result.track_ids,
            w_proximity=params["w_prox"],
            w_gaze=params["w_gaze"],
            w_position=params["w_pos"],
            w_approach=params["w_approach"],
        )
        times.append(time() - t_start)

        # Evaluate the prediction
        gt_class = ground_truth[img_path.name]
        eval_result = _eval_prediction(gt_class, risk_score)
        metrics["total"] += 1
        metrics[eval_result] += 1
        predictions[eval_result].append(img_path.name)
        raw_predictions[img_path.name] = risk_score
        y_true.append(gt_class)
        y_pred_continuous.append(risk_score)

        # Visualize & store the risk visualization (what the model sees)
        if output_images:
            risk_img_output_path = risk_img_output_dir / Path(img_path).name
            annotated = vis.visualize_risk(
                cv_image,
                result.human_bboxes,
                result.depth_viz,
                result.features,
                risk_score,
                max_risk_idx,
            )
            cv2.imwrite(str(risk_img_output_path), annotated)
    # Calculate the average time per image
    if times:
        avg_time = sum(times) / len(times)
    else:
        print(
            "[warn] no frames were processed for this experiment "
            "(every frame either missing from ground truth or lacking depth)."
        )
        avg_time = 0.0
    metrics["avg_time"] = avg_time

    # T3.3.4: expanded classification + regression metrics.
    metrics["classification"] = classification_report(y_true, y_pred_continuous)

    # T3.3.9: reproducibility metadata — stamp the environment that produced
    # these results so a future run can be traced back to its code state.
    metrics["provenance"] = reproducibility_metadata()
    metrics["params"] = dict(params)

    # T3.3.10: record which split this experiment ran against (None ↔ "all").
    metrics["split"] = split
    if split_meta is not None:
        metrics["split_meta"] = split_meta

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
    if metrics.get("skipped_no_depth"):
        print(f"    - Skipped (no matching depth frame): {metrics['skipped_no_depth']}")
    total = metrics["total"]
    if total > 0:
        print(
            f"    - Correct predictions: {metrics['correct']} "
            f"({metrics['correct']/total*100:.2f}%)"
        )
        print(
            f"    - Underestimates: {metrics['underestimate']} "
            f"({metrics['underestimate']/total*100:.2f}%)"
        )
        print(
            f"    - Overestimates: {metrics['overestimate']} "
            f"({metrics['overestimate']/total*100:.2f}%)"
        )
        cls = metrics["classification"]
        print(
            f"    - Macro F1: {cls['macro']['f1']:.3f} | "
            f"Accuracy: {cls['micro']['f1']:.3f} | "
            f"MAE: {cls['regression']['mae']:.3f} | "
            f"RMSE: {cls['regression']['rmse']:.3f}"
        )
    print(f"    - Average time per image: {avg_time:.2f}s")


def run_experiments(
    dataset: str,
    run: str | None = None,
    output_images: bool = False,
    overwrite_existing: bool = False,
    split: str | None = None,
    sweep_config: SweepConfig | None = None,
) -> None:
    """Run every experiment cell in the given sweep config.

    If ``sweep_config`` is None the default config at
    ``configs/sweeps/default.yaml`` is loaded.
    """
    if sweep_config is None:
        sweep_config = load_sweep_config()

    print(
        f"+++ SWEEP '{sweep_config.name}' ({len(sweep_config)} experiments) +++"
    )
    for params in sweep_config.iter_experiments():
        run_experiment(
            dataset,
            params,
            run,
            output_images,
            overwrite_existing,
            split=split,
        )


def _params_slug(params: dict) -> str:
    """
    Returns a slug (string representation) for the given params dict.
    """
    return "-".join([f"{k}_{v}" for k, v in params.items()])
