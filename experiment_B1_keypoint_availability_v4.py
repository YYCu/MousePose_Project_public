from pathlib import Path
import pandas as pd


# ============================================================
# Paths
# ============================================================

EVALUATION_RESULTS_DIR = Path(
    "/path/to/your/validation/results"
)

SNAPSHOT_NAME = (
    "your_snapshot_folder_name"
)

MATCHING_RESULTS_DIR = Path(
    "/path/to/your/instance_matching_results"
)

CLASSIFICATION_RESULTS_DIR = Path(
    "/path/to/your/error_classification_results"
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

ASSOCIATION_PATH = (
    CLASSIFICATION_RESULTS_DIR
    / "error_classification.csv"
)

OUTPUT_DIR = Path(
    "/path/to/save/keypoint_availability_results"
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


THRESHOLDS = [
    0.1,
    0.3,
    0.5,
    0.6,
    0.7,
    0.8
]



# ============================================================
# Helper
# ============================================================

def get_mouse_data(gt, pred, image_id, gt_id, pred_id):

    gt_mouse = gt[
        (gt["image_id"] == image_id)
        &
        (gt["gt_individual"] == gt_id)
    ]


    pred_mouse = pred[
        (pred["image_id"] == image_id)
        &
        (pred["pred_individual"] == pred_id)
    ]

    return gt_mouse, pred_mouse



# ============================================================
# Main
# ============================================================

def main():


    print("=" * 70)
    print(
        "Experiment B.1 Keypoint Availability Analysis v6"
    )
    print("=" * 70)



    # ========================================================
    # Load data
    # ========================================================


    gt = pd.read_csv(
        GT_PATH
    )


    pred = pd.read_csv(
        PRED_PATH
    )


    matched = pd.read_csv(
        MATCH_PATH
    )


    association = pd.read_csv(
        ASSOCIATION_PATH
    )


    association = association[
        association["category"]
        ==
        "instance_association_failure"
    ]


    association_cases = set()


    for _, row in association.iterrows():

        association_cases.add(
            (
                row["image_id"],
                row["gt_individual"],
                row["pred_individual"]
            )
        )


    print(
        "Association failures removed:",
        len(association_cases)
    )



    # remove tail_tip

    gt = gt[
        gt["bodypart"].isin(KEYPOINTS)
    ]


    pred = pred[
        pred["bodypart"].isin(KEYPOINTS)
    ]



    print("\nLoaded")
    print("--------------------------------")

    print(
        "GT rows:",
        len(gt)
    )

    print(
        "Prediction rows:",
        len(pred)
    )

    print(
        "Matched instances:",
        len(matched)
    )
    print(
        "Clean matched instances:",
        len(matched)-len(association_cases)
    )



    # ========================================================
    # Global statistics
    # ========================================================


    actual_gt_keypoints = int(
        gt["gt_available"].sum()
    )


    raw_valid_prediction_keypoints = int(
        (pred["likelihood"] >= 0).sum()
    )



    raw_matched_keypoints_global = 0



    for _, row in matched.iterrows():


        if (
            row["image_id"],
            row["gt_individual"],
            row["pred_individual"]
        ) in association_cases:

            continue


        gt_mouse, pred_mouse = get_mouse_data(
            gt,
            pred,
            row["image_id"],
            row["gt_individual"],
            row["pred_individual"]
        )


        for bp in KEYPOINTS:


            gt_point = gt_mouse[
                gt_mouse["bodypart"] == bp
            ]


            pred_point = pred_mouse[
                pred_mouse["bodypart"] == bp
            ]


            if (
                len(gt_point) == 0
                or
                len(pred_point) == 0
            ):
                continue



            gx = gt_point.iloc[0]["gt_x"]
            gy = gt_point.iloc[0]["gt_y"]

            px = pred_point.iloc[0]["pred_x"]
            py = pred_point.iloc[0]["pred_y"]


            if (
                bool(gt_point.iloc[0]["gt_available"])
                and
                float(pred_point.iloc[0]["likelihood"]) >= 0
                and
                pd.notna(gx)
                and
                pd.notna(gy)
                and
                pd.notna(px)
                and
                pd.notna(py)
            ):

                raw_matched_keypoints_global += 1



    print("\nGlobal")
    print("--------------------------------")

    print(
        "Actual GT keypoints:",
        actual_gt_keypoints
    )


    print(
        "Raw valid prediction keypoints (likelihood >= 0):",
        raw_valid_prediction_keypoints
    )


    print(
        "Raw matched keypoints:",
        raw_matched_keypoints_global
    )



    global_summary = []

    matched_summary = []



    # ========================================================
    # Threshold analysis
    # ========================================================


    for threshold in THRESHOLDS:


        print("\n")
        print("=" * 70)

        print(
            f"Threshold {threshold}"
        )

        print("=" * 70)



        threshold_dir = (
            OUTPUT_DIR /
            f"threshold_{threshold}"
        )


        threshold_dir.mkdir(
            parents=True,
            exist_ok=True
        )



        retained_prediction = pred[
            pred["likelihood"] >= threshold
        ]



        retained_prediction_keypoints = len(
            retained_prediction
        )


        removed_prediction_keypoints = (
            raw_valid_prediction_keypoints
            -
            retained_prediction_keypoints
        )



        print(
            "Raw prediction keypoints:",
            raw_valid_prediction_keypoints
        )


        print(
            "Retained prediction keypoints:",
            retained_prediction_keypoints
        )


        print(
            "Removed prediction keypoints:",
            removed_prediction_keypoints
        )



        global_summary.append(
            {

                "threshold":
                    threshold,

                "actual_GT_keypoints":
                    actual_gt_keypoints,

                "raw_prediction_keypoints":
                    raw_valid_prediction_keypoints,

                "raw_matched_keypoints":
                    raw_matched_keypoints_global,

                "retained_prediction_keypoints":
                    retained_prediction_keypoints,

                "removed_prediction_keypoints":
                    removed_prediction_keypoints,

                "prediction_retention":
                    retained_prediction_keypoints /
                    raw_valid_prediction_keypoints

            }
        )



        # ====================================================
        # Matched keypoints
        # ====================================================


        raw_matched_keypoints = 0

        retained_matched_keypoints = 0



        instance_results = []



        keypoint_results = {

            bp:
            {
                "raw": 0,
                "retained": 0
            }

            for bp in KEYPOINTS

        }


        for _, row in matched.iterrows():


            if (
                row["image_id"],
                row["gt_individual"],
                row["pred_individual"]
            ) in association_cases:

                continue


            gt_mouse, pred_mouse = get_mouse_data(

                gt,
                pred,
                row["image_id"],
                row["gt_individual"],
                row["pred_individual"]
            )


            instance_raw = 0

            instance_retained = 0



            for bp in KEYPOINTS:


                gt_point = gt_mouse[
                    gt_mouse["bodypart"] == bp
                ]


                pred_point = pred_mouse[
                    pred_mouse["bodypart"] == bp
                ]



                if (
                    len(gt_point) == 0
                    or
                    len(pred_point) == 0
                ):
                    continue



                gt_available = bool(
                    gt_point.iloc[0]["gt_available"]
                )


                likelihood = float(
                    pred_point.iloc[0]["likelihood"]
                )

                gx = gt_point.iloc[0]["gt_x"]
                gy = gt_point.iloc[0]["gt_y"]

                px = pred_point.iloc[0]["pred_x"]
                py = pred_point.iloc[0]["pred_y"]


                if (
                    gt_available
                    and
                    likelihood >= 0
                    and
                    pd.notna(gx)
                    and
                    pd.notna(gy)
                    and
                    pd.notna(px)
                    and
                    pd.notna(py)
                ):


                    instance_raw += 1

                    raw_matched_keypoints += 1

                    keypoint_results[bp]["raw"] += 1



                    if likelihood >= threshold:

                        instance_retained += 1

                        retained_matched_keypoints += 1

                        keypoint_results[bp]["retained"] += 1



            instance_results.append(
                {

                    "image_id":
                        row["image_id"],

                    "gt_individual":
                        row["gt_individual"],

                    "pred_individual":
                        row["pred_individual"],

                    "raw_matched_keypoints":
                        instance_raw,

                    "retained_matched_keypoints":
                        instance_retained,

                    "removed_keypoints":
                        instance_raw -
                        instance_retained,

                    "retention_rate":
                        instance_retained / instance_raw
                        if instance_raw > 0
                        else 0

                }
            )



        removed_matched_keypoints = (
            raw_matched_keypoints
            -
            retained_matched_keypoints
        )


        matched_retention = (
            retained_matched_keypoints /
            raw_matched_keypoints
            if raw_matched_keypoints > 0
            else 0
        )



        print("\nMatched keypoints")
        print("--------------------------------")

        print(
            "Raw matched keypoints:",
            raw_matched_keypoints
        )

        print(
            "Retained matched keypoints:",
            retained_matched_keypoints
        )

        print(
            "Removed matched keypoints:",
            removed_matched_keypoints
        )

        print(
            "Matched keypoint retention:",
            matched_retention
        )



        matched_summary.append(
            {

                "threshold":
                    threshold,

                "raw_matched_keypoints":
                    raw_matched_keypoints,

                "retained_matched_keypoints":
                    retained_matched_keypoints,

                "removed_matched_keypoints":
                    removed_matched_keypoints,

                "matched_keypoint_retention":
                    matched_retention

            }
        )



        # ====================================================
        # Save per-instance
        # ====================================================


        pd.DataFrame(
            instance_results
        ).to_csv(
            threshold_dir /
            "per_instance_keypoint_filtering.csv",
            index=False
        )



        # ====================================================
        # Save per-keypoint
        # ====================================================


        keypoint_rows = []


        for bp in KEYPOINTS:


            raw = keypoint_results[bp]["raw"]

            retained = keypoint_results[bp]["retained"]



            keypoint_rows.append(
                {

                    "bodypart":
                        bp,

                    "raw_matched_keypoints":
                        raw,

                    "retained_matched_keypoints":
                        retained,

                    "removed_keypoints":
                        raw - retained,

                    "retention_rate":
                        retained / raw
                        if raw > 0
                        else 0

                }
            )



        pd.DataFrame(
            keypoint_rows
        ).to_csv(
            threshold_dir /
            "per_keypoint_retention.csv",
            index=False
        )



    # ========================================================
    # Save summaries
    # ========================================================


    pd.DataFrame(
        global_summary
    ).to_csv(
        OUTPUT_DIR /
        "global_summary.csv",
        index=False
    )


    pd.DataFrame(
        matched_summary
    ).to_csv(
        OUTPUT_DIR /
        "matched_summary.csv",
        index=False
    )



    print("\n")
    print("=" * 70)
    print("Finished")
    print("=" * 70)



    print("\nGlobal summary")
    print(
        pd.DataFrame(global_summary)
    )


    print("\nMatched summary")
    print(
        pd.DataFrame(matched_summary)
    )



if __name__ == "__main__":

    main()