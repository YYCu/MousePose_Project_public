#!/usr/bin/env python3

from pathlib import Path

import numpy as np
import pandas as pd


# ============================================================
# Paths
# ============================================================

EVALUATION_RESULTS_DIR = Path(
    "/path/to/your/evaluation/results"
)

SNAPSHOT_NAME = (
    "your_snapshot_folder_name"
)

SNAPSHOT_DIR = (
    EVALUATION_RESULTS_DIR
    / "snapshots"
    / SNAPSHOT_NAME
)

PREDICTION_PATH = (
    SNAPSHOT_DIR
    / "predictions_prepared_long.csv"
)

EAR_GEOMETRY_RESULTS_DIR = Path(
    "/path/to/your/ear_geometry_results"
)

C0_SUMMARY_PATH = (
    EAR_GEOMETRY_RESULTS_DIR
    / "ear_geometry_gt_summary.csv"
)

OUTPUT_DIR = Path(
    "/path/to/save/ear_swap_results"
)

OUTPUT_DIR.mkdir(
    parents=True,
    exist_ok=True,
)

CORRECTED_PREDICTION_PATH = (
    OUTPUT_DIR
    / "predictions_ear_swap_corrected.csv"
)

SWAP_LOG_PATH = (
    OUTPUT_DIR
    / "ear_swap_log.csv"
)

SUMMARY_PATH = (
    OUTPUT_DIR
    / "ear_swap_summary.csv"
)


# ============================================================
# Parameters
# ============================================================

# Must be one of the margins tested by the C0 script.
UNCERTAIN_MARGIN_PX = 5.0

# All three head points must satisfy this likelihood threshold.
MIN_LIKELIHOOD = 0.7

NOSE = "nose"
LEFT_EAR = "left_ear"
RIGHT_EAR = "right_ear"

REQUIRED_COLUMNS = {
    "image_id",
    "pred_individual",
    "bodypart",
    "pred_x",
    "pred_y",
    "likelihood",
}


# ============================================================
# Geometry
# ============================================================

def cross_2d(a, b):
    """Return the scalar two-dimensional cross product."""

    return float(
        a[0] * b[1]
        - a[1] * b[0]
    )


def classify_sign(signed_distance, margin_px):
    """Classify which side of the directed head axis a point occupies."""

    if signed_distance > margin_px:
        return "positive"

    if signed_distance < -margin_px:
        return "negative"

    return "uncertain"


def opposite_sign(sign):
    """Return the opposite decisive sign."""

    if sign == "positive":
        return "negative"

    if sign == "negative":
        return "positive"

    raise ValueError(
        f"Cannot obtain the opposite of sign: {sign}"
    )


# ============================================================
# Input validation
# ============================================================

def check_inputs():

    print("=" * 72)
    print("Checking inputs")
    print("=" * 72)

    for path in [
        PREDICTION_PATH,
        C0_SUMMARY_PATH,
    ]:
        if not path.exists():
            raise FileNotFoundError(
                f"Missing input:\n{path}"
            )

        print("[OK]", path)


def load_dominant_sign():

    summary = pd.read_csv(
        C0_SUMMARY_PATH
    )

    required = {
        "uncertain_margin_px",
        "dominant_sign",
        "dominant_sign_stability",
        "decisive_coverage_of_GT_instances",
    }

    missing = required - set(summary.columns)

    if missing:
        raise ValueError(
            "C0 summary is missing columns: "
            + ", ".join(sorted(missing))
        )

    margin_values = pd.to_numeric(
        summary["uncertain_margin_px"],
        errors="coerce",
    )

    selected = summary[
        np.isclose(
            margin_values,
            UNCERTAIN_MARGIN_PX,
        )
    ]

    if len(selected) != 1:
        available = sorted(
            margin_values
            .dropna()
            .unique()
            .tolist()
        )

        raise ValueError(
            f"Could not find exactly one C0 result for margin "
            f"{UNCERTAIN_MARGIN_PX} px.\n"
            f"Available margins: {available}"
        )

    row = selected.iloc[0]

    dominant_sign = str(
        row["dominant_sign"]
    ).strip().lower()

    if dominant_sign not in {
        "positive",
        "negative",
    }:
        raise ValueError(
            "The selected C0 margin does not have a decisive "
            f"dominant sign: {dominant_sign}"
        )

    stability = row[
        "dominant_sign_stability"
    ]

    coverage = row[
        "decisive_coverage_of_GT_instances"
    ]

    return dominant_sign, stability, coverage


# ============================================================
# Prediction helpers
# ============================================================

def get_single_point(instance_df, bodypart):

    rows = instance_df[
        instance_df["bodypart"] == bodypart
    ]

    if len(rows) != 1:
        return None

    index = rows.index[0]
    row = rows.iloc[0]

    x = row["pred_x"]
    y = row["pred_y"]
    likelihood = row["likelihood"]

    if (
        pd.isna(x)
        or pd.isna(y)
        or pd.isna(likelihood)
    ):
        return None

    if float(likelihood) < MIN_LIKELIHOOD:
        return None

    return {
        "index": index,
        "x": float(x),
        "y": float(y),
        "likelihood": float(likelihood),
    }


def inspect_instance(instance_df, dominant_sign):

    nose = get_single_point(
        instance_df,
        NOSE,
    )

    left_ear = get_single_point(
        instance_df,
        LEFT_EAR,
    )

    right_ear = get_single_point(
        instance_df,
        RIGHT_EAR,
    )

    if nose is None:
        return {
            "decision": "not_evaluated",
            "reason": "missing_or_low_confidence_nose",
        }

    if left_ear is None:
        return {
            "decision": "not_evaluated",
            "reason": "missing_or_low_confidence_left_ear",
        }

    if right_ear is None:
        return {
            "decision": "not_evaluated",
            "reason": "missing_or_low_confidence_right_ear",
        }

    nose_xy = np.array(
        [nose["x"], nose["y"]],
        dtype=float,
    )

    left_xy = np.array(
        [left_ear["x"], left_ear["y"]],
        dtype=float,
    )

    right_xy = np.array(
        [right_ear["x"], right_ear["y"]],
        dtype=float,
    )

    ear_center = (
        left_xy + right_xy
    ) / 2.0

    head_axis = (
        nose_xy - ear_center
    )

    head_axis_length = float(
        np.linalg.norm(head_axis)
    )

    ear_distance = float(
        np.linalg.norm(
            left_xy - right_xy
        )
    )

    if head_axis_length < 1.0:
        return {
            "decision": "not_evaluated",
            "reason": "degenerate_head_axis",
        }

    if ear_distance < 1.0:
        return {
            "decision": "not_evaluated",
            "reason": "degenerate_ear_distance",
        }

    left_vector = (
        left_xy - ear_center
    )

    right_vector = (
        right_xy - ear_center
    )

    left_signed_distance = (
        cross_2d(
            head_axis,
            left_vector,
        )
        / head_axis_length
    )

    right_signed_distance = (
        cross_2d(
            head_axis,
            right_vector,
        )
        / head_axis_length
    )

    left_sign = classify_sign(
        left_signed_distance,
        UNCERTAIN_MARGIN_PX,
    )

    right_sign = classify_sign(
        right_signed_distance,
        UNCERTAIN_MARGIN_PX,
    )

    expected_right_sign = opposite_sign(
        dominant_sign
    )

    result = {
        "left_index": left_ear["index"],
        "right_index": right_ear["index"],
        "nose_x": nose["x"],
        "nose_y": nose["y"],
        "left_ear_x_before": left_ear["x"],
        "left_ear_y_before": left_ear["y"],
        "right_ear_x_before": right_ear["x"],
        "right_ear_y_before": right_ear["y"],
        "left_ear_likelihood_before": left_ear["likelihood"],
        "right_ear_likelihood_before": right_ear["likelihood"],
        "head_axis_length_px": head_axis_length,
        "ear_distance_px": ear_distance,
        "left_signed_distance_px": left_signed_distance,
        "right_signed_distance_px": right_signed_distance,
        "left_sign": left_sign,
        "right_sign": right_sign,
    }

    if (
        left_sign == "uncertain"
        or right_sign == "uncertain"
    ):
        result.update(
            {
                "decision": "unchanged",
                "reason": "uncertain_geometry",
            }
        )

        return result

    if (
        left_sign == dominant_sign
        and right_sign == expected_right_sign
    ):
        result.update(
            {
                "decision": "unchanged",
                "reason": "ear_orientation_consistent",
            }
        )

        return result

    if (
        left_sign == expected_right_sign
        and right_sign == dominant_sign
    ):
        result.update(
            {
                "decision": "swapped",
                "reason": "ear_orientation_reversed",
            }
        )

        return result

    result.update(
        {
            "decision": "unchanged",
            "reason": "inconsistent_geometry",
        }
    )

    return result


# ============================================================
# Main
# ============================================================

def main():

    print("=" * 72)
    print("Ear-Swap Correction")
    print("=" * 72)

    check_inputs()

    dominant_sign, stability, coverage = (
        load_dominant_sign()
    )

    print("\nSelected GT geometry rule")
    print("-" * 72)
    print("Margin:", UNCERTAIN_MARGIN_PX, "px")
    print("Dominant left-ear sign:", dominant_sign)
    print("GT sign stability:", stability)
    print("GT decisive coverage:", coverage)
    print("Minimum likelihood:", MIN_LIKELIHOOD)

    predictions = pd.read_csv(
        PREDICTION_PATH
    )

    missing_columns = sorted(
        REQUIRED_COLUMNS
        - set(predictions.columns)
    )

    if missing_columns:
        raise ValueError(
            "Prediction CSV is missing columns: "
            + ", ".join(missing_columns)
        )

    corrected = predictions.copy()

    log_rows = []

    grouped = predictions.groupby(
        [
            "image_id",
            "pred_individual",
        ],
        sort=True,
        dropna=False,
    )

    for (
        image_id,
        pred_individual,
    ), instance_df in grouped:

        result = inspect_instance(
            instance_df,
            dominant_sign,
        )

        decision = result["decision"]

        if decision == "swapped":

            left_index = result[
                "left_index"
            ]

            right_index = result[
                "right_index"
            ]

            # Exchange the complete prediction values belonging to the ears.
            # The bodypart names remain fixed.
            columns_to_swap = [
                "pred_x",
                "pred_y",
                "likelihood",
            ]

            left_values = (
                corrected.loc[
                    left_index,
                    columns_to_swap,
                ]
                .copy()
            )

            right_values = (
                corrected.loc[
                    right_index,
                    columns_to_swap,
                ]
                .copy()
            )

            corrected.loc[
                left_index,
                columns_to_swap,
            ] = right_values.to_numpy()

            corrected.loc[
                right_index,
                columns_to_swap,
            ] = left_values.to_numpy()

        log_row = {
            "image_id": image_id,
            "pred_individual": pred_individual,
            "decision": decision,
            "reason": result["reason"],
            "uncertain_margin_px": UNCERTAIN_MARGIN_PX,
            "minimum_likelihood": MIN_LIKELIHOOD,
            "dominant_left_ear_sign": dominant_sign,
        }

        for key, value in result.items():
            if key not in {
                "decision",
                "reason",
                "left_index",
                "right_index",
            }:
                log_row[key] = value

        log_rows.append(
            log_row
        )

    log_df = pd.DataFrame(
        log_rows
    )

    corrected.to_csv(
        CORRECTED_PREDICTION_PATH,
        index=False,
    )

    log_df.to_csv(
        SWAP_LOG_PATH,
        index=False,
    )

    decision_counts = (
        log_df["decision"]
        .value_counts()
    )

    reason_counts = (
        log_df["reason"]
        .value_counts()
    )

    swapped_instances = int(
        decision_counts.get(
            "swapped",
            0,
        )
    )

    summary_df = pd.DataFrame(
        [
            {
                "input_prediction_file": str(
                    PREDICTION_PATH
                ),
                "output_prediction_file": str(
                    CORRECTED_PREDICTION_PATH
                ),
                "uncertain_margin_px": UNCERTAIN_MARGIN_PX,
                "minimum_likelihood": MIN_LIKELIHOOD,
                "dominant_left_ear_sign": dominant_sign,
                "GT_dominant_sign_stability": stability,
                "GT_decisive_coverage": coverage,
                "prediction_instances": len(log_df),
                "swapped_instances": swapped_instances,
                "unchanged_instances": int(
                    decision_counts.get(
                        "unchanged",
                        0,
                    )
                ),
                "not_evaluated_instances": int(
                    decision_counts.get(
                        "not_evaluated",
                        0,
                    )
                ),
            }
        ]
    )

    summary_df.to_csv(
        SUMMARY_PATH,
        index=False,
    )

    print("\nDecisions")
    print("-" * 72)
    print(decision_counts.to_string())

    print("\nReasons")
    print("-" * 72)
    print(reason_counts.to_string())

    print("\nSaved")
    print("-" * 72)
    print(CORRECTED_PREDICTION_PATH)
    print(SWAP_LOG_PATH)
    print(SUMMARY_PATH)

    print("\nImportant")
    print("-" * 72)
    print(
        "The original prediction CSV was not modified."
    )
    print(
        "Run matching again using predictions_ear_swap_corrected.csv."
    )


if __name__ == "__main__":
    main()