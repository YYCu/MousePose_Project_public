from pathlib import Path
import pandas as pd
import numpy as np
from scipy.optimize import linear_sum_assignment


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

OUTPUT_DIR = Path(
    "/path/to/save/instance_matching_results"
)

OUTPUT_DIR.mkdir(
    parents=True,
    exist_ok=True,
)


# ============================================================
# Parameters
# ============================================================

# For identity matching
# Do not include tail_tip because it is highly unstable
KEYPOINTS = [
    "nose",
    "left_ear",
    "right_ear",
    "body_center",
    "tail_base"
]


MATCH_THRESHOLD = 100

PRED_LIKELIHOOD_THRESHOLD = 0.1



# ============================================================
# Distance
# ============================================================

def instance_distance(gt_instance, pred_instance):

    distances = []


    for bp in KEYPOINTS:

        gt_row = gt_instance[
            gt_instance["bodypart"] == bp
        ]


        pred_row = pred_instance[
            pred_instance["bodypart"] == bp
        ]


        if len(gt_row) == 0 or len(pred_row) == 0:
            continue


        gx = gt_row.iloc[0]["gt_x"]
        gy = gt_row.iloc[0]["gt_y"]

        px = pred_row.iloc[0]["pred_x"]
        py = pred_row.iloc[0]["pred_y"]


        likelihood = pred_row.iloc[0]["likelihood"]


        if likelihood < PRED_LIKELIHOOD_THRESHOLD:
            continue


        if (
            pd.isna(gx)
            or pd.isna(gy)
            or pd.isna(px)
            or pd.isna(py)
        ):
            continue


        distances.append(
            np.sqrt(
                (gx - px) ** 2 +
                (gy - py) ** 2
            )
        )


    if len(distances) == 0:
        return 9999


    return np.mean(distances)



# ============================================================
# Main
# ============================================================

def main():

    print("=" * 70)
    print("Step15 GT-Prediction Matching")
    print("=" * 70)


    print("\nLoading data...")


    gt = pd.read_csv(GT_PATH)

    pred = pd.read_csv(PRED_PATH)


    print("GT:", gt.shape)

    print("Prediction:", pred.shape)



    matched = []

    unmatched_gt = []

    unmatched_pred = []



    total_gt_instances = 0

    total_pred_instances = 0



    images = sorted(
        gt["image_id"].unique()
    )



    for idx, image_id in enumerate(images):

        print(
            f"\n[{idx+1}/{len(images)}] {image_id}"
        )


        gt_img = gt[
            gt["image_id"] == image_id
        ]


        pred_img = pred[
            pred["image_id"] == image_id
        ]



        # ====================================================
        # Filter valid GT instances
        # ====================================================

        gt_matching = gt_img[
                    gt_img["bodypart"].isin(KEYPOINTS)
                ]
        
        gt_valid = (
            gt_matching
            .groupby("gt_individual")
            .filter(
                lambda x:
                x["gt_available"].any()
            )
        )


        gt_ids = sorted(
            gt_valid["gt_individual"].unique()
        )


        # ====================================================
        # Filter valid predictions
        # ====================================================

        pred_valid = pred_img[
            pred_img["bodypart"].isin(KEYPOINTS)
            &
            (
                pred_img["likelihood"]
                >= PRED_LIKELIHOOD_THRESHOLD
            )
        ]


        pred_ids = sorted(
            pred_valid["pred_individual"].unique()
        )


        total_gt_instances += len(gt_ids)

        total_pred_instances += len(pred_ids)



        # No prediction

        if len(pred_ids) == 0:

            for g in gt_ids:

                unmatched_gt.append(
                    {
                        "image_id": image_id,
                        "gt_individual": g
                    }
                )

            continue



        # No GT

        if len(gt_ids) == 0:

            for p in pred_ids:

                unmatched_pred.append(
                    {
                        "image_id": image_id,
                        "pred_individual": p
                    }
                )

            continue



        # ====================================================
        # Cost matrix
        # ====================================================

        cost = np.zeros(
            (
                len(gt_ids),
                len(pred_ids)
            )
        )


        for i, g in enumerate(gt_ids):

            for j, p in enumerate(pred_ids):

                cost[i, j] = instance_distance(
                    gt_valid[
                        gt_valid["gt_individual"] == g
                    ],
                    pred_valid[
                        pred_valid["pred_individual"] == p
                    ]
                )



        rows, cols = linear_sum_assignment(
            cost
        )



        used_gt = set()

        used_pred = set()



        # ====================================================
        # Accept matches
        # ====================================================

        for r, c in zip(rows, cols):

            distance = cost[r, c]


            if distance > MATCH_THRESHOLD:
                continue


            matched.append(
                {
                    "image_id": image_id,
                    "gt_individual": gt_ids[r],
                    "pred_individual": pred_ids[c],
                    "matching_distance": distance
                }
            )


            used_gt.add(
                gt_ids[r]
            )

            used_pred.add(
                pred_ids[c]
            )



        # ====================================================
        # Unmatched GT
        # ====================================================

        for g in gt_ids:

            if g not in used_gt:

                unmatched_gt.append(
                    {
                        "image_id": image_id,
                        "gt_individual": g
                    }
                )



        # ====================================================
        # Unmatched Prediction
        # ====================================================

        for p in pred_ids:

            if p not in used_pred:

                unmatched_pred.append(
                    {
                        "image_id": image_id,
                        "pred_individual": p
                    }
                )



    # ========================================================
    # Save
    # ========================================================

    pd.DataFrame(matched).to_csv(
        OUTPUT_DIR /
        "matched_instances.csv",
        index=False
    )


    pd.DataFrame(unmatched_gt).to_csv(
        OUTPUT_DIR /
        "unmatched_gt.csv",
        index=False
    )


    pd.DataFrame(unmatched_pred).to_csv(
        OUTPUT_DIR /
        "unmatched_predictions.csv",
        index=False
    )



    print("\nFinished!")
    print("--------------------------------")

    print(
        "Total GT instances:",
        total_gt_instances
    )

    print(
        "Total Prediction instances:",
        total_pred_instances
    )

    print(
        "Matched:",
        len(matched)
    )

    print(
        "Unmatched GT:",
        len(unmatched_gt)
    )

    print(
        "Unmatched Prediction:",
        len(unmatched_pred)
    )



if __name__ == "__main__":
    main()
