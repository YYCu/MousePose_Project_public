from pathlib import Path

import numpy as np
import pandas as pd


# ============================================================
# Paths
# ============================================================

EVALUATION_RESULTS_DIR = Path(
    "/path/to/your/validation/results"
)

GT_PATH = (
    EVALUATION_RESULTS_DIR
    / "prepared_data"
    / "ground_truth_long.csv"
)

OUTPUT_DIR = Path(
    "/path/to/save/ear_geometry_results"
)

OUTPUT_DIR.mkdir(
    parents=True,
    exist_ok=True,
)

PER_INSTANCE_CSV = (
    OUTPUT_DIR
    / "ear_geometry_gt_per_instance.csv"
)

SUMMARY_CSV = (
    OUTPUT_DIR
    / "ear_geometry_gt_summary.csv"
)


# ============================================================
# Parameters
# ============================================================

REQUIRED_POINTS = [
    "nose",
    "left_ear",
    "right_ear",
]

# The signed perpendicular distance is measured in pixels. Testing several
# margins avoids choosing one arbitrary definition of "uncertain" in advance.
UNCERTAIN_MARGINS_PX = [0.0, 1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0]

# Degenerate geometries cannot define a reliable head axis or ear direction.
MIN_HEAD_AXIS_LENGTH_PX = 1.0
MIN_EAR_DISTANCE_PX = 1.0

REQUIRED_COLUMNS = {
    "image_id",
    "gt_individual",
    "bodypart",
    "gt_x",
    "gt_y",
    "gt_available",
}


# ============================================================
# Helpers
# ============================================================

def cross_2d(a, b):
    """Return the scalar 2D cross product a x b."""
    return float(a[0] * b[1] - a[1] * b[0])


def as_bool(value):
    """Safely convert common CSV boolean representations to bool."""
    if pd.isna(value):
        return False

    if isinstance(value, (bool, np.bool_)):
        return bool(value)

    if isinstance(value, (int, float, np.integer, np.floating)):
        return bool(value)

    return str(value).strip().lower() in {
        "true",
        "1",
        "yes",
        "y",
    }


def get_gt_point(mouse_df, bodypart):
    """Return a valid GT [x, y] point, or None if unavailable."""
    rows = mouse_df[mouse_df["bodypart"] == bodypart]

    if len(rows) == 0:
        return None

    row = rows.iloc[0]

    if not as_bool(row["gt_available"]):
        return None

    x = row["gt_x"]
    y = row["gt_y"]

    if pd.isna(x) or pd.isna(y):
        return None

    return np.array([float(x), float(y)], dtype=float)


def classify_sign(signed_distance, margin_px):
    """Classify the anatomical left ear relative to ear-centre -> nose."""
    if signed_distance > margin_px:
        return "positive"

    if signed_distance < -margin_px:
        return "negative"

    return "uncertain"


def interpret_stability(stability):
    """Give a cautious interpretation of dominant-sign consistency."""
    if pd.isna(stability):
        return "no_decisive_instances"

    if stability >= 0.90:
        return "promising"

    if stability >= 0.80:
        return "limited"

    return "not_stable_enough"


# ============================================================
# Main
# ============================================================

def main():
    print("=" * 72)
    print("Experiment C0: Ear-centre-to-nose geometry validation on GT")
    print("=" * 72)

    if not GT_PATH.exists():
        raise FileNotFoundError(
            f"Ground-truth file not found:\n{GT_PATH}"
        )

    gt = pd.read_csv(GT_PATH)

    missing_columns = sorted(REQUIRED_COLUMNS - set(gt.columns))
    if missing_columns:
        raise ValueError(
            "The GT CSV is missing required columns: "
            + ", ".join(missing_columns)
        )

    gt = gt[gt["bodypart"].isin(REQUIRED_POINTS)].copy()

    image_count = int(gt["image_id"].nunique())
    candidate_instances = 0
    complete_head_instances = 0
    missing_required_points = 0
    degenerate_geometry = 0
    instance_rows = []

    grouped = gt.groupby(
        ["image_id", "gt_individual"],
        sort=True,
        dropna=False,
    )

    for (image_id, gt_individual), mouse_df in grouped:
        # Match the instance-level definition used in Experiment B: a GT
        # instance exists if at least one of its analysed GT points is available.
        if not mouse_df["gt_available"].map(as_bool).any():
            continue

        candidate_instances += 1

        points = {
            bp: get_gt_point(mouse_df, bp)
            for bp in REQUIRED_POINTS
        }

        missing = [bp for bp, point in points.items() if point is None]
        if missing:
            missing_required_points += 1
            instance_rows.append(
                {
                    "image_id": image_id,
                    "gt_individual": gt_individual,
                    "geometry_available": False,
                    "exclusion_reason": "missing:" + ",".join(missing),
                }
            )
            continue

        complete_head_instances += 1

        nose = points["nose"]
        left_ear = points["left_ear"]
        right_ear = points["right_ear"]

        ear_center = (left_ear + right_ear) / 2.0
        head_axis = nose - ear_center
        left_vector = left_ear - ear_center
        right_vector = right_ear - ear_center

        head_axis_length = float(np.linalg.norm(head_axis))
        ear_distance = float(np.linalg.norm(left_ear - right_ear))

        left_cross = cross_2d(head_axis, left_vector)
        right_cross = cross_2d(head_axis, right_vector)

        if (
            head_axis_length < MIN_HEAD_AXIS_LENGTH_PX
            or ear_distance < MIN_EAR_DISTANCE_PX
        ):
            degenerate_geometry += 1
            instance_rows.append(
                {
                    "image_id": image_id,
                    "gt_individual": gt_individual,
                    "geometry_available": False,
                    "exclusion_reason": "degenerate_geometry",
                    "head_axis_length_px": head_axis_length,
                    "ear_distance_px": ear_distance,
                    "left_cross": left_cross,
                    "right_cross": right_cross,
                }
            )
            continue

        # cross / axis length equals the signed perpendicular distance from
        # the left ear to the directed ear-centre -> nose line, in pixels.
        left_signed_distance = left_cross / head_axis_length
        right_signed_distance = right_cross / head_axis_length

        row = {
            "image_id": image_id,
            "gt_individual": gt_individual,
            "geometry_available": True,
            "exclusion_reason": "",
            "nose_x": nose[0],
            "nose_y": nose[1],
            "left_ear_x": left_ear[0],
            "left_ear_y": left_ear[1],
            "right_ear_x": right_ear[0],
            "right_ear_y": right_ear[1],
            "ear_center_x": ear_center[0],
            "ear_center_y": ear_center[1],
            "head_axis_length_px": head_axis_length,
            "ear_distance_px": ear_distance,
            "left_cross": left_cross,
            "right_cross": right_cross,
            "left_signed_distance_px": left_signed_distance,
            "right_signed_distance_px": right_signed_distance,
        }

        for margin in UNCERTAIN_MARGINS_PX:
            column = f"sign_margin_{margin:g}px"
            row[column] = classify_sign(left_signed_distance, margin)

        instance_rows.append(row)

    instance_df = pd.DataFrame(instance_rows)
    instance_df.to_csv(PER_INSTANCE_CSV, index=False)

    geometry_df = instance_df[
        instance_df["geometry_available"] == True  # noqa: E712
    ].copy()

    summary_rows = []

    for margin in UNCERTAIN_MARGINS_PX:
        column = f"sign_margin_{margin:g}px"
        counts = geometry_df[column].value_counts()

        positive = int(counts.get("positive", 0))
        negative = int(counts.get("negative", 0))
        uncertain = int(counts.get("uncertain", 0))
        decisive = positive + negative

        if positive > negative:
            dominant_sign = "positive"
        elif negative > positive:
            dominant_sign = "negative"
        else:
            dominant_sign = "tie"

        stability = (
            max(positive, negative) / decisive
            if decisive > 0
            else np.nan
        )

        coverage = (
            decisive / candidate_instances
            if candidate_instances > 0
            else np.nan
        )

        summary_rows.append(
            {
                "uncertain_margin_px": margin,
                "validation_images": image_count,
                "candidate_GT_instances": candidate_instances,
                "complete_head_instances": complete_head_instances,
                "valid_geometry_instances": len(geometry_df),
                "positive": positive,
                "negative": negative,
                "uncertain": uncertain,
                "decisive_instances": decisive,
                "dominant_sign": dominant_sign,
                "dominant_sign_stability": stability,
                "decisive_coverage_of_GT_instances": coverage,
                "interpretation": interpret_stability(stability),
            }
        )

    summary_df = pd.DataFrame(summary_rows)
    summary_df.to_csv(SUMMARY_CSV, index=False)

    print("\nLoaded")
    print("-" * 72)
    print("Validation images:", image_count)
    print("Candidate GT instances:", candidate_instances)
    print("Complete nose + both-ear instances:", complete_head_instances)
    print("Missing one or more required points:", missing_required_points)
    print("Degenerate geometries:", degenerate_geometry)
    print("Valid geometries:", len(geometry_df))

    print("\nSign-stability summary")
    print("-" * 72)
    display_columns = [
        "uncertain_margin_px",
        "positive",
        "negative",
        "uncertain",
        "dominant_sign",
        "dominant_sign_stability",
        "decisive_coverage_of_GT_instances",
        "interpretation",
    ]
    print(summary_df[display_columns].to_string(index=False))

    print("\nHow to read the result")
    print("-" * 72)
    print(
        "A dominant_sign_stability near 1.0 means the anatomical left ear "
        "usually stays on the same side of the directed head axis."
    )
    print(
        "The coverage value shows how many GT instances remain usable after "
        "requiring all three head points and excluding uncertain cases."
    )
    print(
        "A useful filter needs both high stability and reasonable coverage."
    )

    print("\nSaved")
    print("-" * 72)
    print(PER_INSTANCE_CSV)
    print(SUMMARY_CSV)


if __name__ == "__main__":
    main()
