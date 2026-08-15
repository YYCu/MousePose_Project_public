#!/usr/bin/env python3

from pathlib import Path
import pandas as pd
import numpy as np
import shutil


# ============================================================
# Paths
# ============================================================

EVALUATION_RESULTS_DIR = Path(
    "/path/to/your/evaluation/results"
)

SNAPSHOT_NAME = (
    "your_snapshot_folder_name"
)

LOCALIZATION_RESULTS_DIR = Path(
    "/path/to/your/localization_results"
)

ERROR_PATH = (
    LOCALIZATION_RESULTS_DIR
    / "keypoint_errors.csv"
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

# Optional folder containing GT–prediction overlay images
OVERLAY_DIR = Path(
    "/path/to/your/overlay_images"
)

OUTPUT_DIR = Path(
    "/path/to/save/error_classification_results"
)

OUTPUT_DIR.mkdir(
    parents=True,
    exist_ok=True,
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


# difference needed to classify association failure
ASSOCIATION_MARGIN = 30


# only analyse suspicious errors
ERROR_THRESHOLD = 30



# ============================================================
# Utils
# ============================================================

def distance(x1, y1, x2, y2):

    return np.sqrt(
        (x1-x2)**2 +
        (y1-y2)**2
    )


def find_overlay(image_id):

    if not OVERLAY_DIR.exists():
        return None

    image_name = Path(image_id).name

    for img in OVERLAY_DIR.iterdir():

        if not img.is_file():
            continue

        if image_name in img.name:
            return img

    return None



# ============================================================
# Main
# ============================================================

def main():

    print("="*70)
    print("Localization Error Classification")
    print("="*70)



    error_df = pd.read_csv(
        ERROR_PATH
    )

    gt = pd.read_csv(
        GT_PATH
    )

    pred = pd.read_csv(
        PRED_PATH
    )



    print(
        "Loaded errors:",
        len(error_df)
    )


    # only large errors

    error_df = error_df[
        error_df["euclidean_error"]
        >= ERROR_THRESHOLD
    ].copy()



    print(
        "High error cases:",
        len(error_df)
    )



    results = []



    for _, row in error_df.iterrows():


        image_id = row["image_id"]

        gt_id = row["gt_individual"]

        pred_id = row["pred_individual"]

        bp = row["bodypart"]



        # prediction point

        pred_point = pred[
            (pred["image_id"] == image_id)
            &
            (pred["pred_individual"] == pred_id)
            &
            (pred["bodypart"] == bp)
        ]


        if len(pred_point)==0:

            continue



        px = pred_point.iloc[0]["pred_x"]
        py = pred_point.iloc[0]["pred_y"]



        # current GT distance

        gt_current = gt[
            (gt["image_id"] == image_id)
            &
            (gt["gt_individual"] == gt_id)
            &
            (gt["bodypart"] == bp)
        ]


        if len(gt_current)==0:

            continue



        gx = gt_current.iloc[0]["gt_x"]
        gy = gt_current.iloc[0]["gt_y"]


        matched_distance = distance(
            px,
            py,
            gx,
            gy
        )



        # search alternative GT mouse

        alternative_distance = None

        alternative_mouse = None



        other_gt = gt[
            (gt["image_id"] == image_id)
            &
            (gt["gt_individual"] != gt_id)
            &
            (gt["bodypart"] == bp)
        ]



        for _, other in other_gt.iterrows():

            other_x = other["gt_x"]
            other_y = other["gt_y"]

            if (
                pd.isna(other_x)
                or pd.isna(other_y)
            ):
                continue

            d = distance(
                px,
                py,
                other_x,
                other_y
            )

            if (
                alternative_distance is None
                or
                d < alternative_distance
            ):

                alternative_distance = d

                alternative_mouse = (
                    other["gt_individual"]
                )



        # classification

        category = "localisation_failure"


        if alternative_distance is not None:

            if (
                alternative_distance
                <
                matched_distance
                -
                ASSOCIATION_MARGIN
            ):

                category = (
                    "instance_association_failure"
                )



        overlay = find_overlay(
            image_id
        )


        save_path = None


        if overlay:


            folder = (
                OUTPUT_DIR
                /
                category
            )


            folder.mkdir(
                parents=True,
                exist_ok=True
            )


            new_name = (
                f"{bp}_"
                f"{row['euclidean_error']:.1f}px_"
                f"{overlay.name}"
            )


            save_path = (
                folder /
                new_name
            )


            shutil.copy2(
                overlay,
                save_path
            )



        results.append(
            {
                "image_id": image_id,
                "bodypart": bp,
                "gt_individual": gt_id,
                "pred_individual": pred_id,
                "error": row["euclidean_error"],
                "matched_distance": matched_distance,
                "alternative_mouse": alternative_mouse,
                "alternative_distance": alternative_distance,
                "category": category,
                "saved_image": str(save_path)
            }
        )



    result_df = pd.DataFrame(
        results
    )


    result_df.to_csv(
        OUTPUT_DIR /
        "error_classification.csv",
        index=False
    )



    print("\nFinished!")

    print(
        result_df["category"]
        .value_counts()
    )


    print(
        "\nSaved:",
        OUTPUT_DIR
    )



if __name__ == "__main__":

    main()