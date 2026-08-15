#!/usr/bin/env python3

from pathlib import Path

import numpy as np
import pandas as pd
from scipy.optimize import linear_sum_assignment


# ============================================================
# Paths
# ============================================================

# Paths

EVALUATION_RESULTS_DIR = Path(
    "/path/to/your/validation/results"
)

SNAPSHOT_NAME = (
    "your_snapshot_folder_name"
)

GT_PATH = (
    EVALUATION_RESULTS_DIR
    / "prepared_data"
    / "ground_truth_long.csv"
)

PRED_PATH = (
    EVALUATION_RESULTS_DIR
    / "snapshots"
    / SNAPSHOT_NAME
    / "predictions_prepared_long.csv"
)

OUTPUT_BASE = Path(
    "/path/to/save/likelihood_threshold_results"
)

OUTPUT_BASE.mkdir(
    parents=True,
    exist_ok=True,
)


# ============================================================
# Parameters
# ============================================================

THRESHOLDS = [
    0.1,
    0.3,
    0.5,
    0.6,
    0.7,
    0.8,
]

KEYPOINTS = [
    "nose",
    "left_ear",
    "right_ear",
    "body_center",
    "tail_base",
]

# Maximum accepted instance-matching distance
MATCH_THRESHOLD = 100

# Step 18 settings
ERROR_THRESHOLD = 30
ASSOCIATION_MARGIN = 30


# ============================================================
# Input checking
# ============================================================

def check_inputs():

    for path in [
        GT_PATH,
        PRED_PATH,
    ]:
        if not path.exists():
            raise FileNotFoundError(
                f"Missing input:\n{path}"
            )

        print("[OK]", path)


# ============================================================
# Distance helpers
# ============================================================

def euclidean_distance(
    x1,
    y1,
    x2,
    y2,
):

    return float(
        np.sqrt(
            (x1 - x2) ** 2
            + (y1 - y2) ** 2
        )
    )


def instance_distance(
    gt_instance,
    pred_instance,
):

    distances = []

    for bodypart in KEYPOINTS:

        gt_point = gt_instance[
            gt_instance["bodypart"]
            == bodypart
        ]

        pred_point = pred_instance[
            pred_instance["bodypart"]
            == bodypart
        ]

        if (
            len(gt_point) == 0
            or len(pred_point) == 0
        ):
            continue

        gx = gt_point.iloc[0]["gt_x"]
        gy = gt_point.iloc[0]["gt_y"]

        px = pred_point.iloc[0]["pred_x"]
        py = pred_point.iloc[0]["pred_y"]

        if (
            pd.isna(gx)
            or pd.isna(gy)
            or pd.isna(px)
            or pd.isna(py)
        ):
            continue

        distances.append(
            euclidean_distance(
                gx,
                gy,
                px,
                py,
            )
        )

    if len(distances) == 0:
        return 9999.0

    return float(
        np.mean(distances)
    )


# ============================================================
# Step 15: threshold-specific matching
# ============================================================

def run_matching(
    gt,
    pred,
    threshold,
):

    matched = []
    unmatched_gt = []
    unmatched_pred = []

    total_gt = 0
    total_pred = 0

    images = sorted(
        gt["image_id"].unique()
    )

    for image_id in images:

        gt_img = gt[
            gt["image_id"]
            == image_id
        ]

        pred_img = pred[
            pred["image_id"]
            == image_id
        ]

        # Only the five evaluated keypoints define a GT instance
        gt_matching = gt_img[
            gt_img["bodypart"].isin(
                KEYPOINTS
            )
        ]

        gt_valid = (
            gt_matching
            .groupby(
                "gt_individual"
            )
            .filter(
                lambda rows:
                rows["gt_available"].any()
            )
        )

        gt_ids = sorted(
            gt_valid[
                "gt_individual"
            ].unique()
        )

        # Apply the current threshold before matching
        pred_valid = pred_img[
            pred_img["bodypart"].isin(
                KEYPOINTS
            )
            &
            (
                pred_img["likelihood"]
                >= threshold
            )
        ].copy()

        pred_ids = sorted(
            pred_valid[
                "pred_individual"
            ].unique()
        )

        total_gt += len(gt_ids)
        total_pred += len(pred_ids)

        if len(pred_ids) == 0:

            for gt_id in gt_ids:

                unmatched_gt.append(
                    {
                        "image_id":
                            image_id,
                        "gt_individual":
                            gt_id,
                    }
                )

            continue

        if len(gt_ids) == 0:

            for pred_id in pred_ids:

                unmatched_pred.append(
                    {
                        "image_id":
                            image_id,
                        "pred_individual":
                            pred_id,
                    }
                )

            continue

        cost = np.zeros(
            (
                len(gt_ids),
                len(pred_ids),
            ),
            dtype=float,
        )

        for i, gt_id in enumerate(
            gt_ids
        ):

            for j, pred_id in enumerate(
                pred_ids
            ):

                cost[i, j] = (
                    instance_distance(
                        gt_valid[
                            gt_valid[
                                "gt_individual"
                            ] == gt_id
                        ],
                        pred_valid[
                            pred_valid[
                                "pred_individual"
                            ] == pred_id
                        ],
                    )
                )

        rows, columns = (
            linear_sum_assignment(
                cost
            )
        )

        used_gt = set()
        used_pred = set()

        for row, column in zip(
            rows,
            columns,
        ):

            matching_distance = (
                cost[row, column]
            )

            if (
                matching_distance
                > MATCH_THRESHOLD
            ):
                continue

            gt_id = gt_ids[row]
            pred_id = pred_ids[column]

            matched.append(
                {
                    "image_id":
                        image_id,
                    "gt_individual":
                        gt_id,
                    "pred_individual":
                        pred_id,
                    "matching_distance":
                        matching_distance,
                }
            )

            used_gt.add(
                gt_id
            )

            used_pred.add(
                pred_id
            )

        for gt_id in gt_ids:

            if gt_id not in used_gt:

                unmatched_gt.append(
                    {
                        "image_id":
                            image_id,
                        "gt_individual":
                            gt_id,
                    }
                )

        for pred_id in pred_ids:

            if pred_id not in used_pred:

                unmatched_pred.append(
                    {
                        "image_id":
                            image_id,
                        "pred_individual":
                            pred_id,
                    }
                )

    return {
        "matched": matched,
        "unmatched_gt": unmatched_gt,
        "unmatched_pred": unmatched_pred,
        "total_gt": total_gt,
        "total_pred": total_pred,
    }


# ============================================================
# Step 16 raw: threshold-specific errors
# ============================================================

def calculate_raw_errors(
    gt,
    pred_filtered,
    matched,
):

    error_rows = []

    for match in matched:

        image_id = match[
            "image_id"
        ]

        gt_id = match[
            "gt_individual"
        ]

        pred_id = match[
            "pred_individual"
        ]

        gt_mouse = gt[
            (gt["image_id"] == image_id)
            &
            (
                gt["gt_individual"]
                == gt_id
            )
        ]

        pred_mouse = pred_filtered[
            (
                pred_filtered["image_id"]
                == image_id
            )
            &
            (
                pred_filtered[
                    "pred_individual"
                ] == pred_id
            )
        ]

        for bodypart in KEYPOINTS:

            gt_point = gt_mouse[
                gt_mouse["bodypart"]
                == bodypart
            ]

            pred_point = pred_mouse[
                pred_mouse["bodypart"]
                == bodypart
            ]

            if (
                len(gt_point) == 0
                or len(pred_point) == 0
            ):
                continue

            gx = gt_point.iloc[0]["gt_x"]
            gy = gt_point.iloc[0]["gt_y"]

            px = pred_point.iloc[0][
                "pred_x"
            ]

            py = pred_point.iloc[0][
                "pred_y"
            ]

            likelihood = (
                pred_point.iloc[0][
                    "likelihood"
                ]
            )

            if (
                pd.isna(gx)
                or pd.isna(gy)
                or pd.isna(px)
                or pd.isna(py)
                or pd.isna(likelihood)
            ):
                continue

            error = euclidean_distance(
                gx,
                gy,
                px,
                py,
            )

            error_rows.append(
                {
                    "image_id":
                        image_id,
                    "gt_individual":
                        gt_id,
                    "pred_individual":
                        pred_id,
                    "bodypart":
                        bodypart,
                    "likelihood":
                        likelihood,
                    "euclidean_error":
                        error,
                }
            )

    return pd.DataFrame(
        error_rows,
        columns=[
            "image_id",
            "gt_individual",
            "pred_individual",
            "bodypart",
            "likelihood",
            "euclidean_error",
        ],
    )


# ============================================================
# Step 18: threshold-specific association classification
# ============================================================

def classify_association_failures(
    gt,
    pred_filtered,
    raw_errors,
):

    classifications = []

    suspicious_errors = raw_errors[
        raw_errors[
            "euclidean_error"
        ] >= ERROR_THRESHOLD
    ]

    for _, row in (
        suspicious_errors.iterrows()
    ):

        image_id = row[
            "image_id"
        ]

        gt_id = row[
            "gt_individual"
        ]

        pred_id = row[
            "pred_individual"
        ]

        bodypart = row[
            "bodypart"
        ]

        pred_point = pred_filtered[
            (
                pred_filtered["image_id"]
                == image_id
            )
            &
            (
                pred_filtered[
                    "pred_individual"
                ] == pred_id
            )
            &
            (
                pred_filtered["bodypart"]
                == bodypart
            )
        ]

        if len(pred_point) == 0:
            continue

        px = pred_point.iloc[0][
            "pred_x"
        ]

        py = pred_point.iloc[0][
            "pred_y"
        ]

        if pd.isna(px) or pd.isna(py):
            continue

        gt_current = gt[
            (gt["image_id"] == image_id)
            &
            (
                gt["gt_individual"]
                == gt_id
            )
            &
            (
                gt["bodypart"]
                == bodypart
            )
        ]

        if len(gt_current) == 0:
            continue

        gx = gt_current.iloc[0][
            "gt_x"
        ]

        gy = gt_current.iloc[0][
            "gt_y"
        ]

        if pd.isna(gx) or pd.isna(gy):
            continue

        matched_distance = (
            euclidean_distance(
                px,
                py,
                gx,
                gy,
            )
        )

        alternative_distance = None
        alternative_mouse = None

        alternative_gt = gt[
            (gt["image_id"] == image_id)
            &
            (
                gt["gt_individual"]
                != gt_id
            )
            &
            (
                gt["bodypart"]
                == bodypart
            )
        ]

        for _, alternative in (
            alternative_gt.iterrows()
        ):

            alternative_x = (
                alternative["gt_x"]
            )

            alternative_y = (
                alternative["gt_y"]
            )

            if (
                pd.isna(alternative_x)
                or pd.isna(alternative_y)
            ):
                continue

            distance = (
                euclidean_distance(
                    px,
                    py,
                    alternative_x,
                    alternative_y,
                )
            )

            if (
                alternative_distance
                is None
                or distance
                < alternative_distance
            ):
                alternative_distance = (
                    distance
                )

                alternative_mouse = (
                    alternative[
                        "gt_individual"
                    ]
                )

        category = (
            "localisation_failure"
        )

        if (
            alternative_distance
            is not None
            and alternative_distance
            < matched_distance
            - ASSOCIATION_MARGIN
        ):
            category = (
                "instance_association_failure"
            )

        classifications.append(
            {
                "image_id":
                    image_id,
                "bodypart":
                    bodypart,
                "gt_individual":
                    gt_id,
                "pred_individual":
                    pred_id,
                "error":
                    row[
                        "euclidean_error"
                    ],
                "matched_distance":
                    matched_distance,
                "alternative_mouse":
                    alternative_mouse,
                "alternative_distance":
                    alternative_distance,
                "category":
                    category,
            }
        )

    return pd.DataFrame(
        classifications,
        columns=[
            "image_id",
            "bodypart",
            "gt_individual",
            "pred_individual",
            "error",
            "matched_distance",
            "alternative_mouse",
            "alternative_distance",
            "category",
        ],
    )


# ============================================================
# Step 16 clean
# ============================================================

def calculate_clean_errors(
    raw_errors,
    classification,
):

    association_rows = classification[
        classification["category"]
        == "instance_association_failure"
    ]

    association_cases = set(
        association_rows[
            [
                "image_id",
                "gt_individual",
                "pred_individual",
            ]
        ].itertuples(
            index=False,
            name=None,
        )
    )

    if len(raw_errors) == 0:
        return raw_errors.copy(), association_cases

    clean_mask = []

    for _, row in raw_errors.iterrows():

        case = (
            row["image_id"],
            row["gt_individual"],
            row["pred_individual"],
        )

        clean_mask.append(
            case not in association_cases
        )

    clean_errors = raw_errors[
        clean_mask
    ].copy()

    return (
        clean_errors,
        association_cases,
    )


# ============================================================
# Summary helpers
# ============================================================

def create_localization_summary(
    error_df,
):

    if len(error_df) == 0:

        mean_error = np.nan
        rmse = np.nan

    else:

        mean_error = float(
            error_df[
                "euclidean_error"
            ].mean()
        )

        rmse = float(
            np.sqrt(
                np.mean(
                    error_df[
                        "euclidean_error"
                    ] ** 2
                )
            )
        )

    return pd.DataFrame(
        {
            "metric": [
                "Mean Euclidean Distance",
                "RMSE",
                "Number of evaluated keypoints",
            ],
            "value": [
                mean_error,
                rmse,
                len(error_df),
            ],
        }
    )


def create_keypoint_summary(
    error_df,
):

    if len(error_df) == 0:

        return pd.DataFrame(
            columns=[
                "bodypart",
                "count",
                "mean",
                "median",
                "std",
                "max",
            ]
        )

    return (
        error_df
        .groupby(
            "bodypart"
        )["euclidean_error"]
        .agg(
            [
                "count",
                "mean",
                "median",
                "std",
                "max",
            ]
        )
        .reset_index()
    )


# ============================================================
# Main
# ============================================================

def main():

    print("=" * 72)
    print("Experiment B: Threshold-Specific Complete Evaluation")
    print("=" * 72)

    check_inputs()

    gt = pd.read_csv(
        GT_PATH
    )

    pred = pd.read_csv(
        PRED_PATH
    )

    comparison_rows = []

    for threshold in THRESHOLDS:

        print("\n" + "=" * 72)
        print("Threshold:", threshold)
        print("=" * 72)

        output_dir = (
            OUTPUT_BASE
            / f"threshold_{threshold}"
        )

        output_dir.mkdir(
            parents=True,
            exist_ok=True,
        )

        # Apply current threshold once
        pred_filtered = pred[
            pred["likelihood"]
            >= threshold
        ].copy()

        # Step 15
        matching = run_matching(
            gt,
            pred,
            threshold,
        )

        matched = matching[
            "matched"
        ]

        unmatched_gt = matching[
            "unmatched_gt"
        ]

        unmatched_pred = matching[
            "unmatched_pred"
        ]

        pd.DataFrame(
            matched
        ).to_csv(
            output_dir
            / "matched_instances.csv",
            index=False,
        )

        pd.DataFrame(
            unmatched_gt
        ).to_csv(
            output_dir
            / "unmatched_gt.csv",
            index=False,
        )

        pd.DataFrame(
            unmatched_pred
        ).to_csv(
            output_dir
            / "unmatched_predictions.csv",
            index=False,
        )

        total_gt = matching[
            "total_gt"
        ]

        total_pred = matching[
            "total_pred"
        ]

        precision = (
            len(matched) / total_pred
            if total_pred > 0
            else 0.0
        )

        recall = (
            len(matched) / total_gt
            if total_gt > 0
            else 0.0
        )

        detection_summary = (
            pd.DataFrame(
                {
                    "metric": [
                        "GT instances",
                        "Prediction instances",
                        "Matched instances",
                        "Missed GT",
                        "False positives",
                        "Precision",
                        "Recall",
                    ],
                    "value": [
                        total_gt,
                        total_pred,
                        len(matched),
                        len(unmatched_gt),
                        len(unmatched_pred),
                        precision,
                        recall,
                    ],
                }
            )
        )

        detection_summary.to_csv(
            output_dir
            / "detection_summary.csv",
            index=False,
        )

        # Step 16 raw
        raw_errors = (
            calculate_raw_errors(
                gt,
                pred_filtered,
                matched,
            )
        )

        raw_errors.to_csv(
            output_dir
            / "raw_keypoint_errors.csv",
            index=False,
        )

        raw_summary = (
            create_localization_summary(
                raw_errors
            )
        )

        raw_summary.to_csv(
            output_dir
            / "raw_localization_summary.csv",
            index=False,
        )

        # Step 18
        classification = (
            classify_association_failures(
                gt,
                pred_filtered,
                raw_errors,
            )
        )

        classification.to_csv(
            output_dir
            / "error_classification.csv",
            index=False,
        )

        # Step 16 clean
        (
            clean_errors,
            association_cases,
        ) = calculate_clean_errors(
            raw_errors,
            classification,
        )

        clean_errors.to_csv(
            output_dir
            / "keypoint_errors.csv",
            index=False,
        )

        clean_summary = (
            create_localization_summary(
                clean_errors
            )
        )

        clean_summary.to_csv(
            output_dir
            / "localization_summary.csv",
            index=False,
        )

        keypoint_summary = (
            create_keypoint_summary(
                clean_errors
            )
        )

        keypoint_summary.to_csv(
            output_dir
            / "per_keypoint_error.csv",
            index=False,
        )

        comparison_rows.append(
            {
                "threshold":
                    threshold,
                "gt_instances":
                    total_gt,
                "prediction_instances":
                    total_pred,
                "matched_instances":
                    len(matched),
                "missed_gt":
                    len(unmatched_gt),
                "false_positives":
                    len(unmatched_pred),
                "precision":
                    precision,
                "recall":
                    recall,
                "association_failures":
                    len(association_cases),
                "raw_keypoints":
                    len(raw_errors),
                "clean_keypoints":
                    len(clean_errors),
                "clean_mean_error":
                    clean_summary.loc[
                        clean_summary[
                            "metric"
                        ]
                        == "Mean Euclidean Distance",
                        "value",
                    ].iloc[0],
                "clean_rmse":
                    clean_summary.loc[
                        clean_summary[
                            "metric"
                        ]
                        == "RMSE",
                        "value",
                    ].iloc[0],
            }
        )

        print("\nDetection")
        print(detection_summary)

        print("\nRaw localization")
        print(raw_summary)

        print("\nAssociation failures:")
        print(len(association_cases))

        print("\nClean localization")
        print(clean_summary)

    comparison = pd.DataFrame(
        comparison_rows
    )

    comparison.to_csv(
        OUTPUT_BASE
        / "confidence_comparison.csv",
        index=False,
    )

    print("\n" + "=" * 72)
    print("Finished")
    print("=" * 72)

    print(comparison.to_string(index=False))

    print("\nSaved:")
    print(OUTPUT_BASE)


if __name__ == "__main__":
    main()