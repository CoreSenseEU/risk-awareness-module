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


_INTERIOR_RED_BGR = np.array([0, 0, 255], dtype=np.float32)
# Cap on the bbox-interior red overlay so the person stays visible even at
# max proximity. 0.7 mirrors the strength of the legacy depth-fog blend.
_INTERIOR_MAX_ALPHA = 0.7


def visualize_risk(
    image,
    bboxes: list[tuple[int, int, int, int]],
    risk_features: dict[str, np.ndarray],
    risk_score: float,
    max_risk_idx: int,
):
    """Overlay bounding boxes, proximity-red bbox interiors, and the risk score.

    Each detected human's bbox interior is tinted red with opacity proportional
    to its proximity sub-score: ``α = proximity · 0.7``. Close persons (high
    proximity, including T2.7 close-fallback cases at proximity 1.0) appear as
    vivid red panels; far persons (proximity 0) leave the underlying image
    untouched. The semantic — *red = close = danger* — is uniform per bbox,
    driven by the same per-bbox value that feeds the proximity sub-score, so
    the visualisation tracks the score rather than the raw depth field. This
    also sidesteps Xtion-class sensors' heavy zero-fill noise: a noisy raw
    depth field inside a bbox does not splotch the overlay because the overlay
    reads from the (cleaner) per-bbox proximity score instead.
    """
    if bboxes:
        proximities = risk_features.get(
            "proximity", np.zeros(len(bboxes), dtype=float)
        )
        image = image.astype(np.float32)
        for i, (x1, y1, x2, y2) in enumerate(bboxes):
            x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)
            x1 = max(0, x1)
            y1 = max(0, y1)
            x2 = min(image.shape[1], x2)
            y2 = min(image.shape[0], y2)
            if x2 <= x1 or y2 <= y1:
                continue
            alpha = float(np.clip(proximities[i], 0.0, 1.0)) * _INTERIOR_MAX_ALPHA
            if alpha <= 0.0:
                continue
            region = image[y1:y2, x1:x2]
            image[y1:y2, x1:x2] = (1.0 - alpha) * region + alpha * _INTERIOR_RED_BGR
        image = np.clip(image, 0, 255).astype(np.uint8)

    # Draw bounding boxes with a continuous red↔white gradient by gaze score.
    # gaze=0 (unaware) → pure red (BGR 0,0,255); gaze=1 (aware) → white
    # (BGR 255,255,255); intermediate → pinks. A thicker black outline is
    # drawn underneath so the bbox stays visible on red walls / shirts and
    # on white / overexposed backgrounds — both extremes of the gradient
    # would otherwise blend into common scene content.
    for i, (x1, y1, x2, y2) in enumerate(bboxes):
        x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)
        gaze_score = float(np.clip(risk_features["gaze"][i], 0.0, 1.0))
        bg = int(round(255 * gaze_score))
        color = (bg, bg, 255)  # BGR
        cv2.rectangle(image, (x1, y1), (x2, y2), (0, 0, 0), 4)
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
    return image


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


# --------------------------------------------------------------------------- #
# kinematic-metric video overlay (shared by scripts/metric_lab.py `video`
# and scripts/layer2.py `video`)
# --------------------------------------------------------------------------- #
def risk_color_bgr(v: float) -> tuple:
    """Continuous green → yellow → red by fused risk (BGR)."""
    v = float(np.clip(v, 0.0, 1.0))
    return (0, int(255 * min(1.0, 2.0 * (1.0 - v))), int(255 * min(1.0, 2.0 * v)))


def put_boxed_text(img, text, org, font_scale, color, thickness=1) -> None:
    font = cv2.FONT_HERSHEY_SIMPLEX
    (tw, th), baseline = cv2.getTextSize(text, font, font_scale, thickness)
    x, y = org
    x = max(3, min(x, img.shape[1] - tw - 6))  # keep the label on-frame
    cv2.rectangle(
        img, (x - 3, y - th - 3), (x + tw + 3, y + baseline + 1), (0, 0, 0), -1
    )
    cv2.putText(img, text, (x, y), font, font_scale, color, thickness, cv2.LINE_AA)


def annotate_kinematic_frame(
    img, bboxes, kins, scene_risk, scene_held, ego, run: str, rel_t: float
):
    """Overlay the Direction-A channels. Deliberately NO ground truth and
    no deployed-metric output — this is the new metric on its own."""
    from riskam.kinematics import KinematicStatus  # noqa: PLC0415

    h, w = img.shape[:2]
    fused = [k.risk if k.risk is not None else k.hazard for k in kins]
    scene_idx = int(np.argmax(fused)) if fused else -1

    for i, bbox in enumerate(bboxes):
        x1, y1, x2, y2 = (int(c) for c in bbox)
        x1, y1 = max(0, x1), max(0, y1)
        x2, y2 = min(w - 1, x2), min(h - 1, y2)
        k = kins[i]
        color = risk_color_bgr(fused[i])
        cv2.rectangle(img, (x1, y1), (x2, y2), (0, 0, 0), 4)
        cv2.rectangle(img, (x1, y1), (x2, y2), color, 2)

        if k.status is KinematicStatus.DEAD_ZONE:
            geo = "TOO CLOSE (depth dead zone)"
        elif k.status is KinematicStatus.STATIC:
            geo = f"d {k.d_m:.1f}m  static"
        elif k.t_cpa_s > 0.1 and k.d_min_m < k.d_m - 0.05:
            # closest-point-of-approach prediction, shown only when converging
            geo = f"d {k.d_m:.1f}m  closest {k.d_min_m:.1f}m in {k.t_cpa_s:.1f}s"
        else:
            geo = f"d {k.d_m:.1f}m  not closing"
        aw_txt = f"{k.awareness:.2f}" if k.awareness is not None else "n/a"
        chan = f"haz {k.hazard:.2f}  aw {aw_txt}  -> {fused[i]:.2f}"
        put_boxed_text(img, geo, (x1, max(28, y1 - 22)), 0.45, (255, 255, 255))
        put_boxed_text(img, chan, (x1, max(46, y1 - 6)), 0.45, color)

        if i == scene_idx:
            cx, cy = (x1 + x2) // 2, (y1 + y2) // 2
            cv2.putText(img, "*", (cx - 8, cy + 8), cv2.FONT_HERSHEY_SIMPLEX,
                        1.0, (0, 0, 0), 4, cv2.LINE_AA)
            cv2.putText(img, "*", (cx - 8, cy + 8), cv2.FONT_HERSHEY_SIMPLEX,
                        1.0, (255, 255, 255), 2, cv2.LINE_AA)

    # "hold" marks a value carried by the release (memory bridging a
    # detector dropout), not a live measurement.
    risk_txt = f"risk {scene_risk:.2f}" + (" hold" if scene_held else "")
    put_boxed_text(
        img, risk_txt, (w - (220 if scene_held else 130), 32), 0.8,
        risk_color_bgr(scene_risk), thickness=2,
    )
    ego_txt = f"ego {ego.speed:.2f} m/s" if ego is not None else "ego n/a"
    put_boxed_text(
        img, f"{run}  t+{rel_t:5.1f}s  {ego_txt}  kinematic metric (A)",
        (10, 22), 0.45, (255, 255, 255),
    )
    return img
