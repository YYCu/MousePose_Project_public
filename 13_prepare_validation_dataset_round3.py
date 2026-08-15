from pathlib import Path
import json
import re

import numpy as np
import pandas as pd


# ============================================================
# Paths and settings
# ============================================================

# Paths and settings

PROJECT_ROOT = Path(
    "/path/to/your/MousePose_Project"
)

# Replace with "validation" or "test".
EVALUATION_SET = "validation or test"

# Replace with "round1", "round2", or "round3".
ROUND_NAME = "round1, round2 or round3"

RESULTS_DIR = (
    PROJECT_ROOT
    / "external_evaluation_results"
    / EVALUATION_SET
    / ROUND_NAME
)

MANIFEST_CSV = (
    RESULTS_DIR
    / "validation_image_manifest.csv"
)

# The labeled-data folder of the evaluation DLC project
EVALUATION_LABELED_DATA = Path(
    "/path/to/your/evaluation/dlc/project/labeled-data"
)

SNAPSHOTS_DIR = (
    RESULTS_DIR
    / "snapshots"
)

OUTPUT_DIR = (
    RESULTS_DIR
    / "prepared_data"
)

GT_INDIVIDUALS = [
    "mouse1",
    "mouse2",
    "mouse3",
]

BODYPARTS = [
    "nose",
    "left_ear",
    "right_ear",
    "body_center",
    "tail_base",
    "tail_tip",
]


# ============================================================
# Small helpers
# ============================================================

def read_h5(path):
    """Read a DLC H5 file with one data key."""
    try:
        return pd.read_hdf(path)
    except ValueError:
        with pd.HDFStore(path, "r") as store:
            return store[store.keys()[0]]


def filename_from_index(index_value):
    """Get image filename from a DLC dataframe index."""
    if isinstance(index_value, tuple):
        return Path(str(index_value[-1])).name

    text = str(index_value).replace("\\", "/")
    return Path(text).name


def find_column(df, choices):
    """Return the first matching column name."""
    lower_map = {
        str(column).lower(): column
        for column in df.columns
    }

    for choice in choices:
        if choice.lower() in lower_map:
            return lower_map[choice.lower()]

    return None


def snapshot_info(snapshot_dir):
    """Read snapshot name, epoch and type."""
    info = {
        "snapshot_name": snapshot_dir.name,
        "snapshot_epoch": None,
        "snapshot_type": (
            "best"
            if "best" in snapshot_dir.name.lower()
            else "regular"
        ),
    }

    manifest_path = (
        snapshot_dir
        / "prediction_manifest.json"
    )

    if manifest_path.exists():
        with open(manifest_path, "r", encoding="utf-8") as file:
            data = json.load(file)

        info["snapshot_name"] = data.get(
            "snapshot_name",
            data.get("snapshot", info["snapshot_name"]),
        )

        info["snapshot_epoch"] = data.get(
            "snapshot_epoch",
            data.get("epoch"),
        )

        info["snapshot_type"] = data.get(
            "snapshot_type",
            info["snapshot_type"],
        )

    if info["snapshot_epoch"] is None:
        numbers = re.findall(r"\d+", snapshot_dir.name)
        if numbers:
            info["snapshot_epoch"] = int(numbers[-1])

    return info


# ============================================================
# Validation manifest
# ============================================================

def prepare_manifest():
    manifest = pd.read_csv(MANIFEST_CSV)

    prepared = pd.DataFrame({
        "validation_order": manifest["validation_order"].astype(int),
        "image_id": manifest["image_key"].astype(str),
        "video": manifest["video"].astype(str),
        "image_name": manifest["image"].astype(str),
        "frozen_image_path": manifest[
            "frozen_absolute_path"
        ].astype(str),
        "source_image_path": manifest[
            "original_absolute_path"
        ].astype(str),
    })

    prepared = prepared.sort_values(
        "validation_order"
    ).reset_index(drop=True)

    return prepared

# ============================================================
# Ground Truth
# ============================================================

def prepare_ground_truth(manifest):
    gt_files = sorted(
        EVALUATION_LABELED_DATA.rglob(
            "CollectedData*.h5"
        )
    )

    if not gt_files:
        raise FileNotFoundError(
            f"No CollectedData H5 files found under:\n"
            f"{EVALUATION_LABELED_DATA}"
        )

    manifest_by_name = (
        manifest
        .set_index("image_name")
    )

    gt_values = {}

    for gt_file in gt_files:
        gt_df = read_h5(gt_file)

        for index_value, row in gt_df.iterrows():
            image_name = filename_from_index(
                index_value
            )

            if image_name not in manifest_by_name.index:
                continue

            manifest_row = manifest_by_name.loc[
                image_name
            ]

            if isinstance(manifest_row, pd.DataFrame):
                raise RuntimeError(
                    f"Duplicate image filename in validation manifest: "
                    f"{image_name}"
                )

            image_id = manifest_row["image_id"]

            for column, value in row.items():
                scorer, individual, bodypart, coord = column

                if individual not in GT_INDIVIDUALS:
                    continue

                if bodypart not in BODYPARTS:
                    continue

                if coord not in ["x", "y"]:
                    continue

                key = (
                    image_id,
                    individual,
                    bodypart,
                )

                if key not in gt_values:
                    gt_values[key] = {
                        "gt_scorer": scorer,
                        "gt_x": np.nan,
                        "gt_y": np.nan,
                    }

                gt_values[key][f"gt_{coord}"] = value

    rows = []

    for _, image in manifest.iterrows():
        for individual in GT_INDIVIDUALS:
            for bodypart in BODYPARTS:
                values = gt_values.get(
                    (
                        image["image_id"],
                        individual,
                        bodypart,
                    ),
                    {
                        "gt_scorer": "Yuu",
                        "gt_x": np.nan,
                        "gt_y": np.nan,
                    },
                )

                gt_available = (
                    pd.notna(values["gt_x"])
                    and pd.notna(values["gt_y"])
                )

                rows.append({
                    "image_id": image["image_id"],
                    "validation_order": image[
                        "validation_order"
                    ],
                    "video": image["video"],
                    "image_name": image["image_name"],
                    "frozen_image_path": image[
                        "frozen_image_path"
                    ],
                    "gt_scorer": values["gt_scorer"],
                    "gt_individual": individual,
                    "bodypart": bodypart,
                    "gt_x": values["gt_x"],
                    "gt_y": values["gt_y"],
                    "gt_available": gt_available,
                })

    gt_long = pd.DataFrame(rows)

    gt_summary = (
        gt_long
        .groupby(
            [
                "image_id",
                "validation_order",
                "video",
                "image_name",
            ],
            as_index=False,
        )
        .agg(
            gt_available_point_count=(
                "gt_available",
                "sum",
            ),
            gt_individual_count=(
                "gt_individual",
                lambda values: values[
                    gt_long.loc[
                        values.index,
                        "gt_available"
                    ]
                ].nunique(),
            ),
        )
    )

    return gt_long, gt_summary


# ============================================================
# Predictions
# ============================================================

def prepare_snapshot_predictions(
    snapshot_dir,
    manifest,
):
    prediction_h5 = (
        snapshot_dir
        / "predictions_original.h5"
    )

    if not prediction_h5.exists():
        candidates = sorted(
            snapshot_dir.glob("*.h5")
        )
        if len(candidates) != 1:
            raise FileNotFoundError(
                f"Could not identify prediction H5 in:\n"
                f"{snapshot_dir}"
            )
        prediction_h5 = candidates[0]

    prediction_df = read_h5(
        prediction_h5
    )

    row_manifest_path = (
        snapshot_dir
        / "prediction_row_manifest.csv"
    )

    if row_manifest_path.exists():
        row_manifest = pd.read_csv(
            row_manifest_path
        )

        # 保证按照预测 H5 的行顺序排列
        if "prediction_row" in row_manifest.columns:
            row_manifest = (
                row_manifest
                .sort_values("prediction_row")
                .reset_index(drop=True)
            )

        if "image_key" in row_manifest.columns:
            mapping = (
                row_manifest[["image_key"]]
                .rename(
                    columns={
                        "image_key": "image_id"
                    }
                )
                .merge(
                    manifest,
                    on="image_id",
                    how="left",
                )
            )

        elif "validation_order" in row_manifest.columns:
            mapping = (
                row_manifest[
                    ["validation_order"]
                ]
                .merge(
                    manifest,
                    on="validation_order",
                    how="left",
                )
            )

        elif "frozen_filename" in row_manifest.columns:
            mapping = (
                row_manifest[
                    ["frozen_filename"]
                ]
                .merge(
                    manifest.assign(
                        frozen_filename=manifest[
                            "frozen_image_path"
                        ].map(
                            lambda value: Path(
                                str(value)
                            ).name
                        )
                    ),
                    on="frozen_filename",
                    how="left",
                )
            )

        else:
        # Step 12 已确保预测顺序和 validation manifest 一致
            mapping = manifest.copy()

    else:
        mapping = manifest.copy()

    mapping = mapping.reset_index(drop=True)

    if mapping["image_id"].isna().any():
        raise RuntimeError(
            f"Some prediction rows could not be mapped in "
            f"{snapshot_dir.name}"
        )
    

    if len(prediction_df) != len(mapping):
        raise RuntimeError(
            f"Prediction rows and mapping rows differ in "
            f"{snapshot_dir.name}:\n"
            f"{len(prediction_df)} vs {len(mapping)}"
        )

    info = snapshot_info(snapshot_dir)

    pred_individuals = list(dict.fromkeys(
        prediction_df.columns.get_level_values(
            "individuals"
            if "individuals"
            in prediction_df.columns.names
            else 1
        )
    ))

    rows = []

    for row_number in range(
        len(prediction_df)
    ):
        image = mapping.iloc[row_number]
        prediction_row = prediction_df.iloc[
            row_number
        ]

        for individual in pred_individuals:
            for bodypart in BODYPARTS:
                scorer = (
                    prediction_df.columns
                    .get_level_values(0)[0]
                )

                def get_value(coord):
                    try:
                        return prediction_row[
                            (
                                scorer,
                                individual,
                                bodypart,
                                coord,
                            )
                        ]
                    except KeyError:
                        return np.nan

                pred_x = get_value("x")
                pred_y = get_value("y")
                likelihood = get_value(
                    "likelihood"
                )

                coordinate_available = (
                    pd.notna(pred_x)
                    and pd.notna(pred_y)
                )

                rows.append({
                    "image_id": image["image_id"],
                    "validation_order": image[
                        "validation_order"
                    ],
                    "video": image["video"],
                    "image_name": image["image_name"],
                    "frozen_image_path": image[
                        "frozen_image_path"
                    ],
                    "snapshot_name": info[
                        "snapshot_name"
                    ],
                    "snapshot_epoch": info[
                        "snapshot_epoch"
                    ],
                    "snapshot_type": info[
                        "snapshot_type"
                    ],
                    "prediction_scorer": scorer,
                    "pred_individual": individual,
                    "bodypart": bodypart,
                    "pred_x": pred_x,
                    "pred_y": pred_y,
                    "likelihood": likelihood,
                    "coordinate_available": (
                        coordinate_available
                    ),
                })

    prediction_long = pd.DataFrame(rows)

    prediction_summary = (
        prediction_long
        .groupby(
            [
                "image_id",
                "validation_order",
                "video",
                "image_name",
                "snapshot_name",
                "snapshot_epoch",
                "snapshot_type",
            ],
            as_index=False,
        )
        .agg(
            predicted_coordinate_count=(
                "coordinate_available",
                "sum",
            ),
            predicted_individual_count=(
                "pred_individual",
                "nunique",
            ),
            points_likelihood_ge_0_1=(
                "likelihood",
                lambda values: (
                    values >= 0.1
                ).sum(),
            ),
            points_likelihood_ge_0_5=(
                "likelihood",
                lambda values: (
                    values >= 0.5
                ).sum(),
            ),
            points_likelihood_ge_0_9=(
                "likelihood",
                lambda values: (
                    values >= 0.9
                ).sum(),
            ),
        )
    )

    return (
        prediction_long,
        prediction_summary,
        info,
    )


# ============================================================
# Main
# ============================================================

def main():
    print("=" * 72)
    print(
        "Prepare DeepLabCut evaluation data"
    )
    print("=" * 72)

    OUTPUT_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    manifest = prepare_manifest()

    manifest.to_csv(
        OUTPUT_DIR
        / "validation_image_manifest_prepared.csv",
        index=False,
    )

    print(
        f"Validation images: {len(manifest)}"
    )

    gt_long, gt_summary = (
        prepare_ground_truth(manifest)
    )

    gt_long.to_csv(
        OUTPUT_DIR
        / "ground_truth_long.csv",
        index=False,
    )

    gt_summary.to_csv(
        OUTPUT_DIR
        / "ground_truth_image_summary.csv",
        index=False,
    )

    print(
        f"GT rows: {len(gt_long)}"
    )
    print(
        f"Available GT points: "
        f"{gt_long['gt_available'].sum()}"
    )
    print(
        f"Missing GT points retained: "
        f"{(~gt_long['gt_available']).sum()}"
    )

    snapshot_dirs = sorted(
        path
        for path in SNAPSHOTS_DIR.iterdir()
        if path.is_dir()
    )

    check_rows = []

    for snapshot_dir in snapshot_dirs:
        print(
            f"Preparing: {snapshot_dir.name}"
        )

        (
            prediction_long,
            prediction_summary,
            info,
        ) = prepare_snapshot_predictions(
            snapshot_dir,
            manifest,
        )

        prediction_long.to_csv(
            snapshot_dir
            / "predictions_prepared_long.csv",
            index=False,
        )

        prediction_summary.to_csv(
            snapshot_dir
            / "prediction_image_summary.csv",
            index=False,
        )

        check_rows.append({
            "snapshot_name": info[
                "snapshot_name"
            ],
            "snapshot_epoch": info[
                "snapshot_epoch"
            ],
            "snapshot_type": info[
                "snapshot_type"
            ],
            "expected_images": len(manifest),
            "prediction_images": (
                prediction_long[
                    "image_id"
                ].nunique()
            ),
            "prediction_rows": len(
                prediction_long
            ),
            "status": (
                "pass"
                if prediction_long[
                    "image_id"
                ].nunique() == len(manifest)
                else "fail"
            ),
        })

    check_df = pd.DataFrame(
        check_rows
    )

    check_df.to_csv(
        OUTPUT_DIR
        / "evaluation_dataset_check.csv",
        index=False,
    )

    summary = {
        "validation_images": len(manifest),
        "snapshots": len(snapshot_dirs),
        "gt_rows": len(gt_long),
        "available_gt_points": int(
            gt_long["gt_available"].sum()
        ),
        "missing_gt_points_retained": int(
            (~gt_long["gt_available"]).sum()
        ),
        "prediction_points_removed": False,
        "individual_matching_applied": False,
        "evaluation_metrics_applied": False,
    }

    with open(
        OUTPUT_DIR
        / "evaluation_data_summary.json",
        "w",
        encoding="utf-8",
    ) as file:
        json.dump(
            summary,
            file,
            indent=2,
        )

    print("=" * 72)
    print(
        "Evaluation data prepared successfully"
    )
    print("=" * 72)
    print(
        f"Prepared data saved under:\n"
        f"{OUTPUT_DIR}"
    )
    print(
        "No prediction was removed."
    )
    print(
        "No individual matching or evaluation "
        "was performed."
    )


if __name__ == "__main__":
    main()