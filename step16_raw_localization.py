from pathlib import Path
import pandas as pd
import numpy as np


# ============================================================
# Paths
# ============================================================

# Paths

EVALUATION_RESULTS_DIR = Path(
    "/path/to/your/evaluation/results"
)

SNAPSHOT_NAME = (
    "your_snapshot_folder_name"
)

MATCHING_RESULTS_DIR = Path(
    "/path/to/your/instance_matching_results"
)

OUTPUT_DIR = Path(
    "/path/to/save/localization_results"
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

MATCH_PATH = (
    MATCHING_RESULTS_DIR
    / "matched_instances.csv"
)

OUTPUT_DIR.mkdir(
    parents=True,
    exist_ok=True
)


# ============================================================
# Parameters
# ============================================================

KEYPOINTS = [
    "nose",
    "left_ear",
    "right_ear",
    "body_center",
    "tail_base"
]

PRED_LIKELIHOOD_THRESHOLD = 0.1


# ============================================================
# Main
# ============================================================

def main():

    print("="*70)
    print("Keypoint Localization Error")
    print("="*70)


    gt = pd.read_csv(
        GT_PATH
    )

    pred = pd.read_csv(
        PRED_PATH
    )

    matched = pd.read_csv(
        MATCH_PATH
    )


    print("Matched instances:")
    print(len(matched))

    print("Prediction likelihood threshold:")
    print(PRED_LIKELIHOOD_THRESHOLD)


    errors = []


    for _, row in matched.iterrows():

        image_id = row["image_id"]

        gt_id = row["gt_individual"]

        pred_id = row["pred_individual"]


        gt_mouse = gt[
            (gt["image_id"] == image_id) &
            (gt["gt_individual"] == gt_id)
        ]


        pred_mouse = pred[
            (pred["image_id"] == image_id) &
            (pred["pred_individual"] == pred_id)
        ]


        for bp in KEYPOINTS:


            gt_point = gt_mouse[
                gt_mouse["bodypart"] == bp
            ]


            pred_point = pred_mouse[
                pred_mouse["bodypart"] == bp
            ]


            if (
                len(gt_point)==0 or
                len(pred_point)==0
            ):
                continue

            likelihood = pred_point.iloc[0]["likelihood"]

            # Only evaluate predictions retained by the likelihood threshold

            if (
                pd.isna(likelihood) or
                likelihood < PRED_LIKELIHOOD_THRESHOLD
            ):
                continue

            gx = gt_point.iloc[0]["gt_x"]
            gy = gt_point.iloc[0]["gt_y"]

            px = pred_point.iloc[0]["pred_x"]
            py = pred_point.iloc[0]["pred_y"]


            if (
                pd.isna(gx) or
                pd.isna(gy) or
                pd.isna(px) or
                pd.isna(py)
            ):
                continue


            error = np.sqrt(
                (gx-px)**2 +
                (gy-py)**2
            )


            errors.append(
                {
                    "image_id": image_id,
                    "gt_individual": gt_id,
                    "pred_individual": pred_id,
                    "bodypart": bp,
                    "likelihood": likelihood,
                    "euclidean_error": error
                }
            )



    error_df = pd.DataFrame(
        errors
    )


    # save detailed error

    error_df.to_csv(
        OUTPUT_DIR /
        "keypoint_errors.csv",
        index=False
    )


    # ========================================================
    # Summary metrics
    # ========================================================

    mean_error = (
        error_df["euclidean_error"]
        .mean()
    )


    rmse = np.sqrt(
        np.mean(
            error_df["euclidean_error"] ** 2
        )
    )


    summary = pd.DataFrame(
        {
            "metric":
            [
                "Mean Euclidean Distance",
                "RMSE",
                "Number of evaluated keypoints"
            ],

            "value":
            [
                mean_error,
                rmse,
                len(error_df)
            ]
        }
    )


    summary.to_csv(
        OUTPUT_DIR /
        "localization_summary.csv",
        index=False
    )


    # per keypoint

    keypoint_summary = (
        error_df
        .groupby("bodypart")
        ["euclidean_error"]
        .agg(
            [
                "count",
                "mean",
                "median",
                "std",
                "max"
            ]
        )
        .reset_index()
    )


    keypoint_summary.to_csv(
        OUTPUT_DIR /
        "per_keypoint_error.csv",
        index=False
    )


    print("\nFinished!")

    print(summary)

    print("\nPer keypoint:")
    print(keypoint_summary)



if __name__ == "__main__":
    main()