"""
riskam.visualization

Visualization tools for the risk awareness module.
"""

import json
from pathlib import Path

import cv2
import matplotlib.pyplot as plt
import numpy as np

from riskam import experiments
from riskam.score import RISK_SCORE_BREAKPOINTS

# pylint: disable=no-member


AGG_RESULTS_FNAME = "agg_results.json"


plt.rcParams.update(
    {
        "font.family": "serif",
        "font.serif": ["DejaVu Serif"],
    }
)


def visualize_risk(
    image_path: str,
    output_path: str | None,
    bboxes: list[tuple[int, int, int, int]],
    rel_depth: np.ndarray,
    risk_features: dict[str, np.ndarray],
    risk_score: float,
    max_risk_idx: int,
):
    # Load the image
    image = cv2.imread(image_path)
    if image is None:
        raise FileNotFoundError(f"Image not found: {image_path}")

    # Add rel_depth overlay: closest pixels remain as-is and further ones fade to dark gray
    # Normalize rel_depth to range [0,1]
    norm_depth = (rel_depth - rel_depth.min()) / (
        rel_depth.max() - rel_depth.min() + 1e-6
    )
    # Create a dark gray overlay image
    overlay = np.full_like(image, (50, 50, 50))
    # Soften overlay: reduce effect of depth on blending (e.g., only half as strong)
    soft_weight = norm_depth * 0.85
    # Blend each pixel with the softer weight
    image = (
        image.astype(np.float32) * (1 - soft_weight[..., None])
        + overlay.astype(np.float32) * soft_weight[..., None]
    )
    image = np.clip(image, 0, 255).astype(np.uint8)

    # Draw bounding boxes with color based on gaze score
    for i, (x1, y1, x2, y2) in enumerate(bboxes):
        # Ensure coordinates are integers
        x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)
        gaze_score = risk_features["gaze"][i]
        if gaze_score == 0:
            color = (0, 0, 255)  # red
        elif gaze_score == 1:
            color = (0, 255, 0)  # green
        else:
            color = (0, 255, 255)  # yellow
        cv2.rectangle(image, (x1, y1), (x2, y2), color, 2)

    # Add risk score overlay in the top right corner
    text = f"{risk_score:.3f}"
    height, width = image.shape[:2]
    font = cv2.FONT_HERSHEY_SIMPLEX
    font_scale = max(0.5, width / 600)
    thickness = int(font_scale * 2)
    text_size = cv2.getTextSize(text, font, font_scale, thickness)[0]
    text_x = width - text_size[0] - 20
    text_y = 40
    # Background rectangle for text
    cv2.rectangle(
        image,
        (text_x - 10, text_y - 30),
        (text_x + text_size[0] + 10, text_y + 10),
        (0, 0, 0),
        -1,
    )
    # Determine text color based on RISK_SCORE_BREAKPOINTS
    if risk_score == RISK_SCORE_BREAKPOINTS[0]:
        score_color = (255, 200, 150)  # light blue
    elif risk_score <= RISK_SCORE_BREAKPOINTS[1]:
        score_color = (0, 255, 0)  # green
    elif risk_score <= RISK_SCORE_BREAKPOINTS[2]:
        score_color = (0, 255, 255)  # yellow
    else:
        score_color = (0, 0, 255)  # red
    cv2.putText(
        image,
        text,
        (text_x, text_y),
        font,
        font_scale,
        score_color,
        thickness,
        cv2.LINE_AA,
    )

    # Mark the bounding box corresponding to max_risk_idx with an asterisk centered in the box
    if 0 <= max_risk_idx < len(bboxes):
        x1, y1, x2, y2 = bboxes[max_risk_idx]
        # Convert coordinates to int and compute center of bounding box
        x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)
        center = ((x1 + x2) // 2, (y1 + y2) // 2)
        # Compute text size for the asterisk
        star = "*"
        star_size, baseline = cv2.getTextSize(star, font, font_scale, thickness)
        # Adjust position so that the center of the asterisk text is at 'center'
        star_x = center[0] - star_size[0] // 2
        star_y = center[1] + star_size[1] // 2
        # Draw asterisk with an outline for visibility
        cv2.putText(
            image,
            star,
            (star_x, star_y),
            font,
            font_scale,
            (0, 0, 0),
            thickness + 2,
            cv2.LINE_AA,
        )
        cv2.putText(
            image,
            star,
            (star_x, star_y),
            font,
            font_scale,
            (255, 255, 255),
            thickness,
            cv2.LINE_AA,
        )

    if output_path is None:
        # Show on screen
        cv2.imshow("Risk Visualization", image)
        cv2.waitKey(0)
        cv2.destroyAllWindows()
    else:
        # Ensure directory exists and save image
        out_path = Path(output_path)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        cv2.imwrite(str(out_path), image)


def visualize_exp_results(dataset: str) -> None:
    """
    Visualizes experimental results.
    """
    # Aggregate the results
    agg_results = _aggregate_overall_results(dataset)

    # Establish the gfx dir
    gfx_dir = experiments.EXP_ROOT_DIR / dataset / "gfx"
    gfx_dir.mkdir(parents=True, exist_ok=True)

    # Visualize the best results
    _visualize_best_results(gfx_dir, agg_results)

    # Visualize the parameter diagnostics
    _visualize_param_diagnostics(gfx_dir, agg_results)


def _aggregate_overall_results(dataset: str) -> None:
    """
    Aggregates the total rsults.
    """

    # Establish the dict
    agg_results = {}

    # Iterate over the runs
    all_exp_dir = experiments.EXP_ROOT_DIR / dataset
    for run_dir in all_exp_dir.iterdir():
        if not run_dir.is_dir():
            continue

        for exp_dir in run_dir.iterdir():

            # Load the results
            results_path = exp_dir / "results.json"

            # Skip if results.json does not exist
            if not results_path.exists():
                continue

            try:
                with open(results_path, "r", encoding="utf-8") as f:
                    results = json.load(f)
            except json.JSONDecodeError:
                print(f"Error loading {results_path}")
                continue

            # Establish the entry in agg_results
            params = exp_dir.name

            if params not in agg_results:
                agg_results[params] = {
                    "correct": 0,
                    "underestimate": 0,
                    "overestimate": 0,
                    "total": 0,
                    "avg_time": [],
                }

            # Update the results
            for key in ["correct", "underestimate", "overestimate", "total"]:
                agg_results[params][key] += results[key]

            agg_results[params]["avg_time"].append(results["avg_time"])

    # Aggregate the results
    for _, params_entry in agg_results.items():
        params_entry["avg_time"] = float(np.mean(params_entry["avg_time"]))
        params_entry["accuracy"] = params_entry["correct"] / params_entry["total"]
        params_entry["perc_underestimate"] = (
            params_entry["underestimate"] / params_entry["total"]
        )
        params_entry["perc_overestimate"] = (
            params_entry["overestimate"] / params_entry["total"]
        )

    # Save the results
    agg_results_path = experiments.EXP_ROOT_DIR / dataset / AGG_RESULTS_FNAME

    with open(agg_results_path, "w", encoding="utf-8") as f:
        json.dump(agg_results, f, indent=4)

    return agg_results


def aggregate_over_parameter(agg_results: dict, param: str) -> dict:
    """
    Aggregates the results over a specific parameter.
    """

    # Establish the dict
    agg_over_param = {}

    # Iterate over the aggregated results
    for params, exp_result in agg_results.items():
        param_val = None
        params_split = params.split("-")

        if param == "gaze":
            th_lower = params_split[-2].split("_")[-1]
            th_upper = params_split[-1].split("_")[-1]
            param_val = f"({th_lower},{th_upper})"
        elif param == "w":
            w_prox = params_split[0].split("_")[-1]
            w_gaze = params_split[1].split("_")[-1]
            w_pos = params_split[2].split("_")[-1]
            param_val = f"({w_prox},{w_gaze},{w_pos})"
        else:
            for ps in params_split:
                if ps.startswith(param):
                    param_val = ps.split("_")[-1]
                    break

        if param_val is None:
            print(f"Could not find {param} in {params}")
            continue

        # Establish the entry in agg_over_param
        if param_val not in agg_over_param:
            agg_over_param[param_val] = {
                "correct": 0,
                "underestimate": 0,
                "overestimate": 0,
                "total": 0,
                "avg_time": [],
            }

        # Update the results
        for key in ["correct", "underestimate", "overestimate", "total"]:
            agg_over_param[param_val][key] += exp_result[key]

        agg_over_param[param_val]["avg_time"].append(exp_result["avg_time"])

    # Aggregate the results
    for _, params_entry in agg_over_param.items():
        params_entry["avg_time"] = float(np.mean(params_entry["avg_time"]))
        params_entry["accuracy"] = params_entry["correct"] / params_entry["total"]
        params_entry["perc_underestimate"] = (
            params_entry["underestimate"] / params_entry["total"]
        )
        params_entry["perc_overestimate"] = (
            params_entry["overestimate"] / params_entry["total"]
        )

    return agg_over_param


def _visualize_best_results(gfx_dir: str, agg_results: dict) -> None:
    """
    Visualizes the best results achieved overall.
    """

    # Select the best results: i) highest accuracy, ii) lowest underestimation, iii) lowest overestimation
    running_best = {
        "accuracy": 0,
        "accuracy_params": None,
        "safe": 0,
        "safe_params": None,
        "perc_underestimate": 1,
        "perc_underestimate_params": None,
        "perc_overestimate": 1,
        "perc_overestimate_params": None,
    }

    for params, exp_result in agg_results.items():
        if exp_result["accuracy"] > running_best["accuracy"]:
            running_best["accuracy"] = exp_result["accuracy"]
            running_best["accuracy_params"] = params

        safe = exp_result["accuracy"] + exp_result["perc_overestimate"]

        if safe > running_best["safe"]:
            running_best["safe"] = safe
            running_best["safe_params"] = params

        if exp_result["perc_underestimate"] < running_best["perc_underestimate"]:
            running_best["perc_underestimate"] = exp_result["perc_underestimate"]
            running_best["perc_underestimate_params"] = params

        if exp_result["perc_overestimate"] < running_best["perc_overestimate"]:
            running_best["perc_overestimate"] = exp_result["perc_overestimate"]
            running_best["perc_overestimate_params"] = params

    print(
        f"Best accuracy: {running_best['accuracy']} with params {running_best['accuracy_params']}"
    )
    print(
        f"Best safe: {running_best['safe']} with params {running_best['safe_params']}"
    )
    print(
        f"Best underestimation: {running_best['perc_underestimate']} with params {running_best['perc_underestimate_params']}"
    )
    print(
        f"Best overestimation: {running_best['perc_overestimate']} with params {running_best['perc_overestimate_params']}"
    )

    best_params = running_best["safe_params"]

    # Data
    labels = ["Correct", "Overestimate", "Underestimate"]
    sizes = [
        agg_results[best_params]["accuracy"],
        agg_results[best_params]["perc_overestimate"],
        agg_results[best_params]["perc_underestimate"],
    ]
    colors = [
        "#66bb6a",
        "#fbc02d",
        "#e53935",
    ]
    explode = (0.0, 0.0, 0.0)  # Optional: slice separation

    # Plot
    fig, ax = plt.subplots(figsize=(6, 6))
    wedges, texts, autotexts = ax.pie(
        sizes,
        autopct="%1.1f%%",
        startangle=90,
        colors=colors,
        explode=explode,
        wedgeprops={"edgecolor": "black"},
        textprops={"fontsize": 11, "color": "black"},
        labeldistance=1.2,  # Push category labels outside
        pctdistance=0.85,  # Keep percentages inside the wedges
    )
    ax.axis("equal")

    # Add legend (place to the right)
    ax.legend(
        wedges,
        labels,
        # title=r"\textbf{Zones}",
        loc="center left",
        bbox_to_anchor=(1, 0.5),
        fontsize=10,
        title_fontsize=14,
    )

    # plt.title(r"\textbf{Risk Distribution Across Zones}", fontsize=14)

    # Save directly to PDF
    plt.savefig(gfx_dir / "best.pdf", format="pdf", bbox_inches="tight")


def _visualize_param_diagnostics(gfx_dir: str, agg_results: dict) -> None:
    # Configuration
    n_rows = 1
    n_cols = 3
    labels = ["Correct", "Overestimate", "Underestimate"]
    colors = ["#66bb6a", "#fbc02d", "#e53935"]

    TITLES = {
        "w": "Component Weight (Proximity, Gaze, Position)",
        "gamma": "Gamma",
        "gaze": "Gaze Thresholds (Lower, Upper)",
    }

    fig, axes = plt.subplots(n_rows, n_cols, figsize=(16, 8), sharey=True)
    axes = axes.flatten()

    for idx, (ax, param) in enumerate(zip(axes, TITLES.keys())):
        dataset = aggregate_over_parameter(agg_results, param)

        # Filter only keys that are convertible to float for sorting
        def _is_float(x: str) -> bool:
            try:
                float(x)
                return True
            except ValueError:
                return False

        # bar_keys = [k for k in dataset if _is_float(k)]
        bar_keys = sorted(dataset.keys())
        x = np.arange(len(bar_keys))

        accuracy = [dataset[k]["accuracy"] for k in bar_keys]
        over = [dataset[k]["perc_overestimate"] for k in bar_keys]
        under = [dataset[k]["perc_underestimate"] for k in bar_keys]

        # Stack bars
        ax.bar(
            x, accuracy, width=0.6, color=colors[0], label=labels[0] if idx == 0 else ""
        )
        ax.bar(
            x,
            over,
            bottom=accuracy,
            width=0.6,
            color=colors[1],
            label=labels[1] if idx == 0 else "",
        )
        ax.bar(
            x,
            under,
            bottom=np.array(accuracy) + np.array(over),
            width=0.6,
            color=colors[2],
            label=labels[2] if idx == 0 else "",
        )

        ax.set_title(TITLES[param], fontsize=16)
        ax.set_xticks(x)
        ax.set_xticklabels(bar_keys, rotation=45, fontsize=13)
        ax.set_ylim(0, 1.05)
        ax.grid(axis="y", linestyle="--", linewidth=0.5)

    # Remove empty 6th plot
    if len(TITLES) < n_rows * n_cols:
        for i in range(len(TITLES), n_rows * n_cols):
            fig.delaxes(axes[i])

    # Adjust layout and spacing
    fig.subplots_adjust(
        left=0.1, right=0.98, top=0.88, bottom=0.12, hspace=0.4, wspace=0.3
    )

    # Shared Y-axis label, outside left edge
    fig.text(
        0.02, 0.5, "Outcome Proportion", va="center", rotation="vertical", fontsize=14
    )

    # Shared legend above
    fig.legend(
        labels, loc="upper center", ncol=3, bbox_to_anchor=(0.5, 0.97), fontsize=14
    )

    # Save as high-quality PDF
    plt.savefig(gfx_dir / "params.pdf", format="pdf", bbox_inches="tight")
