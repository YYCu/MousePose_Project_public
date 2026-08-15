from __future__ import annotations

import yaml
import hashlib
import inspect
import json
import platform
import re
import shutil
import sys
from datetime import datetime
from pathlib import Path
from typing import Any

import deeplabcut
import pandas as pd
import torch
from PIL import Image

from deeplabcut.pose_estimation_pytorch.apis.utils import (
    get_model_snapshots,
)


# ============================================================
# 1. Project paths
# ============================================================

PROJECT_ROOT = Path(
    "/path/to/your/MousePose_Project"
)

# Name used for the output folder, such as round1, round2, or round3.
ROUND_NAME = "your_round_name"

# Trained model project
TRAIN_CONFIG = Path(
    "/path/to/your/training/dlc/project/config.yaml"
)

# Independent validation project
VALIDATION_CONFIG = Path(
    "/path/to/your/validation/dlc/project/config.yaml"
)

VALIDATION_PROJECT = VALIDATION_CONFIG.parent

VALIDATION_LABELED_DATA = (
    VALIDATION_PROJECT
    / "labeled-data"
)

# Metadata created and completed in Steps 03 and 04
VALIDATION_METADATA = (
    PROJECT_ROOT
    / "data"
    / "annotations"
    / "val_metadata.csv"
)

OUTPUT_ROOT = (
    PROJECT_ROOT
    / "external_validation_results"
    / ROUND_NAME
)
# ============================================================
# 2. DeepLabCut settings
# ============================================================

SHUFFLE = 1
TRAINING_SET_INDEX = 0

# Maximum number of predicted animals
MAX_INDIVIDUALS = 3

# Apple Silicon
DEVICE = "mps"

# Generate DLC prediction images
SAVE_DLC_PLOTS = True

# This only controls which points are shown on prediction images.
# Raw coordinates and likelihoods are not filtered by this value.
PLOT_PCUTOFF = 0.1

# Set to True only when deliberately rerunning every snapshot.
OVERWRITE_EXISTING_RESULTS = True

SUPPORTED_IMAGE_SUFFIXES = {
    ".png",
    ".jpg",
    ".jpeg",
    ".tif",
    ".tiff",
}


# ============================================================
# 3. General helpers
# ============================================================

def print_header(title: str) -> None:
    print()
    print("=" * 72)
    print(title)
    print("=" * 72)


def require_file(path: Path, description: str) -> None:
    if not path.is_file():
        raise FileNotFoundError(
            f"{description} was not found:\n{path}"
        )


def require_directory(path: Path, description: str) -> None:
    if not path.is_dir():
        raise FileNotFoundError(
            f"{description} was not found:\n{path}"
        )


def normalise_text(value: object) -> str:
    if pd.isna(value):
        return ""

    return str(value).strip()


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()

    with path.open("rb") as file:
        for block in iter(
            lambda: file.read(1024 * 1024),
            b"",
        ):
            digest.update(block)

    return digest.hexdigest()


def safe_name(value: str) -> str:
    value = re.sub(
        r"[^A-Za-z0-9._-]+",
        "_",
        value,
    )

    return value.strip("_")


def json_ready(value: Any) -> Any:
    """Convert common Python/Pandas objects into JSON-safe values."""

    if isinstance(value, Path):
        return str(value)

    if isinstance(value, dict):
        return {
            str(key): json_ready(item)
            for key, item in value.items()
        }

    if isinstance(value, (list, tuple, set)):
        return [
            json_ready(item)
            for item in value
        ]

    if pd.isna(value):
        return None

    if hasattr(value, "item"):
        try:
            return value.item()
        except Exception:
            pass

    return value


def write_json(
    path: Path,
    data: dict[str, Any],
) -> None:
    with path.open(
        "w",
        encoding="utf-8",
    ) as file:
        json.dump(
            json_ready(data),
            file,
            indent=2,
            ensure_ascii=False,
        )


# ============================================================
# 5. Check paths and device
# ============================================================

def verify_inputs() -> None:
    print_header("Checking required files")

    require_file(
        TRAIN_CONFIG,
        "Training config",
    )

    require_file(
        VALIDATION_CONFIG,
        "Validation project config",
    )

    require_directory(
        VALIDATION_LABELED_DATA,
        "Validation labeled-data folder",
    )

    require_file(
        VALIDATION_METADATA,
        "Validation metadata CSV",
    )


    print(f"Training config:\n{TRAIN_CONFIG}")
    print(f"Validation config:\n{VALIDATION_CONFIG}")
    print(f"Validation metadata:\n{VALIDATION_METADATA}")
    print(f"Device: {DEVICE}")


# ============================================================
# 6. Locate validation images
# ============================================================

def find_validation_image(
    video: str,
    image_name: str,
) -> Path:
    """
    Locate one validation image.

    Expected structure:
        validation_project/labeled-data/<video>/<image>
    """

    direct_path = (
        VALIDATION_LABELED_DATA
        / video
        / image_name
    )

    if direct_path.is_file():
        return direct_path.resolve()

    matches = [
        path.resolve()
        for path in VALIDATION_LABELED_DATA.rglob(
            image_name
        )
        if path.is_file()
        and path.suffix.lower()
        in SUPPORTED_IMAGE_SUFFIXES
    ]

    if not matches:
        raise FileNotFoundError(
            "Could not find validation image.\n"
            f"Video: {video}\n"
            f"Image: {image_name}\n"
            f"Searched under:\n"
            f"{VALIDATION_LABELED_DATA}"
        )

    if len(matches) > 1:
        match_text = "\n".join(
            str(path)
            for path in matches
        )

        raise RuntimeError(
            "More than one image has the same filename, "
            "so the image cannot be resolved safely.\n"
            f"Video: {video}\n"
            f"Image: {image_name}\n"
            f"Matches:\n{match_text}"
        )

    return matches[0]


# ============================================================
# 7. Freeze validation images and build image manifest
# ============================================================

def prepare_validation_images(
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    Select usable validation images, calculate hashes and dimensions,
    and copy them into a frozen inference folder.

    DLC analyzes the frozen copies. This guarantees that prediction
    row names can be mapped back to the correct original image.
    """

    print_header("Preparing validation image manifest")

    metadata = pd.read_csv(
        VALIDATION_METADATA
    )

    required_columns = {
        "video",
        "image",
        "usable",
    }

    missing_columns = (
        required_columns
        - set(metadata.columns)
    )

    if missing_columns:
        raise ValueError(
            "val_metadata.csv is missing required columns:\n"
            f"{sorted(missing_columns)}"
        )

    metadata = metadata.copy()

    metadata["video"] = (
        metadata["video"]
        .map(normalise_text)
    )

    metadata["image"] = (
        metadata["image"]
        .map(normalise_text)
    )

    metadata["usable"] = (
        metadata["usable"]
        .map(normalise_text)
        .str.lower()
    )

    usable_metadata = metadata.loc[
        metadata["usable"].eq("yes")
    ].copy()

    if usable_metadata.empty:
        raise ValueError(
            "No rows with usable=yes were found in:\n"
            f"{VALIDATION_METADATA}"
        )

    if usable_metadata["video"].eq("").any():
        raise ValueError(
            "At least one usable row has an empty video value."
        )

    if usable_metadata["image"].eq("").any():
        raise ValueError(
            "At least one usable row has an empty image value."
        )

    duplicate_rows = (
        usable_metadata.duplicated(
            subset=["video", "image"],
            keep=False,
        )
    )

    if duplicate_rows.any():
        duplicates = usable_metadata.loc[
            duplicate_rows,
            ["video", "image"],
        ]

        raise ValueError(
            "Duplicate video + image entries were found:\n"
            f"{duplicates.to_string(index=False)}"
        )

    usable_metadata = (
        usable_metadata
        .reset_index(drop=True)
    )

    frozen_folder = (
        OUTPUT_ROOT
        / "frozen_validation_images"
    )

    if frozen_folder.exists():
        if OVERWRITE_EXISTING_RESULTS:
            shutil.rmtree(frozen_folder)

    frozen_folder.mkdir(
        parents=True,
        exist_ok=True,
    )

    manifest_rows: list[dict[str, Any]] = []

    for order, row in usable_metadata.iterrows():
        video = row["video"]
        image_name = row["image"]

        original_path = find_validation_image(
            video=video,
            image_name=image_name,
        )

        image_hash = file_sha256(
            original_path
        )

        with Image.open(original_path) as image:
            image_width, image_height = image.size

        suffix = original_path.suffix.lower()

        # Unique and stable filename used for inference.
        frozen_name = (
            f"{order:04d}__"
            f"{safe_name(video)}__"
            f"{safe_name(original_path.stem)}"
            f"{suffix}"
        )

        frozen_path = (
            frozen_folder
            / frozen_name
        )

        if frozen_path.exists():
            existing_hash = file_sha256(
                frozen_path
            )

            if existing_hash != image_hash:
                raise RuntimeError(
                    "A frozen validation image already exists "
                    "but its content differs from the source.\n"
                    f"Frozen image: {frozen_path}\n"
                    "Use OVERWRITE_EXISTING_RESULTS=True only "
                    "if you intentionally changed the dataset."
                )
        else:
            shutil.copy2(
                original_path,
                frozen_path,
            )


        manifest_rows.append(
            {
                "validation_order": order,
                "image_key": (
                    f"{video}/{image_name}"
                ),
                "video": video,
                "image": image_name,
                "original_absolute_path": str(
                    original_path
                ),
                "original_relative_to_validation_project": str(
                    original_path.relative_to(
                        VALIDATION_PROJECT.resolve()
                    )
                ),
                "frozen_filename": frozen_name,
                "frozen_absolute_path": str(
                    frozen_path.resolve()
                ),
                "image_suffix": suffix,
                "image_width": image_width,
                "image_height": image_height,
                "image_size_bytes": (
                    original_path.stat().st_size
                ),
                "image_sha256": image_hash,
            }
        )

    manifest = pd.DataFrame(
        manifest_rows
    )


    print(
        f"Usable images: {len(manifest)}"
    )

    print(
        f"Validation videos: "
        f"{manifest['video'].nunique()}"
    )

    print(
        f"Frozen images:\n{frozen_folder}"
    )

    return usable_metadata, manifest


# ============================================================
# 9. Locate the PyTorch training folder
# ============================================================

def get_training_paths(
) -> tuple[Path, Path]:
    """
    Locate pytorch_config.yaml and the trained model folder.

    First use DLC's public helper. If its return format differs across
    DLC 3.x minor versions, fall back to a constrained search inside the
    training project directory.
    """

    returned_value: Any = None
    candidate_paths: list[Path] = []

    if hasattr(deeplabcut, "return_train_network_path"):
        helper = deeplabcut.return_train_network_path
        helper_signature = inspect.signature(helper)

        call_attempts = [
            lambda: helper(
                str(TRAIN_CONFIG),
                SHUFFLE,
                TRAINING_SET_INDEX,
            ),
            lambda: helper(
                config=str(TRAIN_CONFIG),
                shuffle=SHUFFLE,
                trainingsetindex=TRAINING_SET_INDEX,
            ),
        ]

        last_error: Exception | None = None
        for attempt in call_attempts:
            try:
                returned_value = attempt()
                break
            except TypeError as error:
                last_error = error

        if returned_value is None and last_error is not None:
            print(
                "Warning: return_train_network_path() could not be "
                "called with known DLC 3.x signatures. Falling back "
                "to project-folder discovery."
            )
            print(f"Installed signature: {helper_signature}")

        def collect_paths(value: Any) -> None:
            if isinstance(value, (str, Path)):
                candidate_paths.append(
                    Path(value).expanduser()
                )
            elif isinstance(value, dict):
                for item in value.values():
                    collect_paths(item)
            elif isinstance(value, (list, tuple, set)):
                for item in value:
                    collect_paths(item)

        collect_paths(returned_value)

    training_project = TRAIN_CONFIG.parent.resolve()

    for path in training_project.rglob("pytorch_config.yaml"):
        candidate_paths.append(path)

    pytorch_configs = []
    seen: set[Path] = set()
    for candidate in candidate_paths:
        resolved = candidate.expanduser().resolve()
        if resolved.is_dir():
            config_candidate = resolved / "pytorch_config.yaml"
            if config_candidate.is_file():
                resolved = config_candidate.resolve()
        if resolved.name == "pytorch_config.yaml" and resolved.is_file():
            if resolved not in seen:
                pytorch_configs.append(resolved)
                seen.add(resolved)

    if not pytorch_configs:
        raise RuntimeError(
            "Could not locate pytorch_config.yaml for the selected "
            "training project.\n"
            f"Training config: {TRAIN_CONFIG}\n"
            f"return_train_network_path() returned: {returned_value}"
        )

    # Prefer a config inside a train subfolder that contains snapshots.
    scored: list[tuple[int, Path]] = []
    for config_path in pytorch_configs:
        folder = config_path.parent
        snapshot_count = len(list(folder.glob("*.pt")))
        name_score = 1 if "train" in folder.name.lower() else 0
        scored.append((snapshot_count * 10 + name_score, config_path))

    scored.sort(key=lambda item: (item[0], str(item[1])), reverse=True)
    pytorch_config_path = scored[0][1]
    train_folder = pytorch_config_path.parent

    require_file(pytorch_config_path, "PyTorch config")
    require_directory(train_folder, "PyTorch training folder")

    print(f"PyTorch config:\n{pytorch_config_path}")
    print(f"Training folder:\n{train_folder}")

    return pytorch_config_path, train_folder


# ============================================================
# 10. Discover snapshots using DLC
# ============================================================

def discover_snapshots(
    train_folder: Path,
) -> tuple[list[Any], pd.DataFrame]:
    print_header("Discovering model snapshots")

    # For this bottom-up multi-animal project the pose task is
    # represented by the PyTorch model method. get_model_snapshots
    # can determine the available pose snapshots from the model
    # directory.
    try:
        snapshots = get_model_snapshots(
            index="all",
            model_folder=train_folder,
        )

    except TypeError:
    # DLC 3.0.0 requires an explicit Task object.
        from deeplabcut.pose_estimation_pytorch.task import Task

        pytorch_config_path = (
            train_folder
            / "pytorch_config.yaml"
        )

        require_file(
            pytorch_config_path,
            "PyTorch config",
        )

        with pytorch_config_path.open(
            "r",
            encoding="utf-8",
        ) as file:
            pose_config = yaml.safe_load(file)

        if not isinstance(pose_config, dict):
            raise RuntimeError(
                "Could not read pytorch_config.yaml as a mapping:\n"
                f"{pytorch_config_path}"
            )

        method = pose_config.get("method")

        if not method:
            raise RuntimeError(
                "The PyTorch config does not contain a model method.\n"
                f"Expected a top-level 'method' field in:\n"
                f"{pytorch_config_path}"
            )

        print(
            f"Snapshot task method from PyTorch config: {method}"
        )

        task = Task(method)

        snapshots = get_model_snapshots(
            index="all",
            model_folder=train_folder,
            task=task,
        )

    if not snapshots:
        raise RuntimeError(
            "No model snapshots were found in:\n"
            f"{train_folder}"
        )

    rows: list[dict[str, Any]] = []

    for index, snapshot in enumerate(snapshots):
        raw_snapshot_path = getattr(
            snapshot,
            "path",
            snapshot,
        )
        snapshot_path = Path(
            raw_snapshot_path
        ).expanduser().resolve()

        require_file(
            snapshot_path,
            "Snapshot",
        )

        epoch = getattr(
            snapshot,
            "epochs",
            getattr(snapshot, "epoch", None),
        )

        best = bool(
            getattr(
                snapshot,
                "best",
                "-best-" in snapshot_path.stem,
            )
        )

        if epoch is None:
            match = re.search(
                r"(\d+)(?:\.pt)?$",
                snapshot_path.name,
            )

            if match:
                epoch = int(
                    match.group(1)
                )

        snapshot_type = (
            "best"
            if best
            else "regular"
        )

        rows.append(
            {
                "snapshot_index": index,
                "snapshot_name": (
                    snapshot_path.stem
                ),
                "snapshot_filename": (
                    snapshot_path.name
                ),
                "snapshot_path": str(
                    snapshot_path
                ),
                "snapshot_epoch": epoch,
                "snapshot_type": snapshot_type,
                "is_best_snapshot": best,
                "snapshot_size_bytes": (
                    snapshot_path.stat().st_size
                ),
                "snapshot_sha256": file_sha256(
                    snapshot_path
                ),
            }
        )

        print(
            f"[{index:02d}] "
            f"epoch={epoch}, "
            f"type={snapshot_type}, "
            f"file={snapshot_path.name}"
        )

    snapshot_manifest = pd.DataFrame(
        rows
    )

    return snapshots, snapshot_manifest


# ============================================================
# 11. Find and read DLC prediction H5
# ============================================================

def find_prediction_h5(
    snapshot_output: Path,
) -> Path:
    candidates = sorted(
        [
            path
            for path in snapshot_output.glob(
                "*.h5"
            )
            if path.name
            != "predictions_original.h5"
        ],
        key=lambda path: path.stat().st_mtime,
    )

    if len(candidates) != 1:
        candidate_text = "\n".join(
            str(path)
            for path in candidates
        )

        raise RuntimeError(
            "Expected exactly one DLC prediction H5 file "
            "for this snapshot.\n"
            f"Found: {len(candidates)}\n"
            f"Files:\n{candidate_text}"
        )

    return candidates[0]


def find_prediction_csv(
    snapshot_output: Path,
) -> Path | None:
    candidates = sorted(
        [
            path
            for path in snapshot_output.glob(
                "*.csv"
            )
            if path.name
            != "predictions_original.csv"
        ],
        key=lambda path: path.stat().st_mtime,
    )

    if not candidates:
        return None

    if len(candidates) > 1:
        candidate_text = "\n".join(
            str(path)
            for path in candidates
        )

        raise RuntimeError(
            "More than one DLC prediction CSV was generated.\n"
            f"{candidate_text}"
        )

    return candidates[0]


def read_prediction_dataframe(
    prediction_h5: Path,
) -> tuple[pd.DataFrame, str]:
    """
    Read a DLC H5 without assuming that the HDF key is always named
    'predictions'.
    """

    with pd.HDFStore(
        prediction_h5,
        mode="r",
    ) as store:
        keys = store.keys()

    if len(keys) != 1:
        raise RuntimeError(
            "Expected exactly one table inside the prediction H5.\n"
            f"H5: {prediction_h5}\n"
            f"Keys: {keys}"
        )

    key = keys[0]

    dataframe = pd.read_hdf(
        prediction_h5,
        key=key,
    )

    return dataframe, key


# ============================================================
# 12. Resolve MultiIndex levels
# ============================================================

def resolve_column_level(
    columns: pd.MultiIndex,
    possible_names: set[str],
) -> str | int:
    """
    Return a named level when possible.

    DLC normally uses scorer/individuals/bodyparts/coords. This
    function also tolerates missing or slightly changed level names.
    """

    for name in columns.names:
        if name is None:
            continue

        if str(name).lower() in possible_names:
            return name

    raise ValueError(
        "Could not resolve a required prediction column level.\n"
        f"Column level names: {columns.names}\n"
        f"Expected one of: {sorted(possible_names)}"
    )


# ============================================================
# 13. Map prediction rows to validation images
# ============================================================

def build_prediction_row_manifest(
    predictions: pd.DataFrame,
    image_manifest: pd.DataFrame,
) -> pd.DataFrame:
    if len(predictions) != len(image_manifest):
        raise ValueError(
            "Prediction row count does not match validation "
            "image count.\n"
            f"Prediction rows: {len(predictions)}\n"
            f"Validation images: {len(image_manifest)}"
        )

    frozen_lookup = {
        filename: row
        for filename, row in (
            image_manifest
            .set_index("frozen_filename")
            .iterrows()
        )
    }

    prediction_indices = [
        str(index)
        for index in predictions.index
    ]

    matched_rows: list[dict[str, Any]] = []

    for prediction_row, raw_index in enumerate(
        prediction_indices
    ):
        index_path = Path(
            raw_index
        )

        basename = index_path.name

        matched_filename: str | None = None
        mapping_method: str | None = None

        if basename in frozen_lookup:
            matched_filename = basename
            mapping_method = "prediction_index_basename"

        else:
            matching_filenames = [
                filename
                for filename in frozen_lookup
                if filename in raw_index
            ]

            if len(matching_filenames) == 1:
                matched_filename = (
                    matching_filenames[0]
                )
                mapping_method = (
                    "filename_found_inside_prediction_index"
                )

        if matched_filename is None:
            raise RuntimeError(
                "Could not safely map a prediction row to a "
                "frozen validation image.\n"
                f"Prediction row: {prediction_row}\n"
                f"Prediction index: {raw_index}"
            )

        manifest_row = frozen_lookup[
            matched_filename
        ]

        matched_rows.append(
            {
                "prediction_row": prediction_row,
                "prediction_index_raw": raw_index,
                "mapping_method": mapping_method,
                "validation_order": int(
                    manifest_row[
                        "validation_order"
                    ]
                ),
                "image_key": (
                    manifest_row[
                        "image_key"
                    ]
                ),
                "video": (
                    manifest_row[
                        "video"
                    ]
                ),
                "image": (
                    manifest_row[
                        "image"
                    ]
                ),
                "frozen_filename": (
                    matched_filename
                ),
                "image_width": int(
                    manifest_row[
                        "image_width"
                    ]
                ),
                "image_height": int(
                    manifest_row[
                        "image_height"
                    ]
                ),
                "image_sha256": (
                    manifest_row[
                        "image_sha256"
                    ]
                ),
            }
        )

    row_manifest = pd.DataFrame(
        matched_rows
    )

    if row_manifest[
        "frozen_filename"
    ].duplicated().any():
        raise RuntimeError(
            "More than one prediction row was mapped to the same "
            "validation image."
        )

    expected_filenames = set(
        image_manifest[
            "frozen_filename"
        ]
    )

    mapped_filenames = set(
        row_manifest[
            "frozen_filename"
        ]
    )

    if expected_filenames != mapped_filenames:
        missing = (
            expected_filenames
            - mapped_filenames
        )

        unexpected = (
            mapped_filenames
            - expected_filenames
        )

        raise RuntimeError(
            "Prediction-image mapping is incomplete.\n"
            f"Missing: {sorted(missing)}\n"
            f"Unexpected: {sorted(unexpected)}"
        )

    return row_manifest


# ============================================================
# 14. Validate prediction structure
# ============================================================

def inspect_prediction_structure(
    predictions: pd.DataFrame,
) -> dict[str, Any]:
    if not isinstance(
        predictions.columns,
        pd.MultiIndex,
    ):
        raise ValueError(
            "Prediction columns are not a MultiIndex. "
            "This is not the expected multi-animal DLC output."
        )

    individual_level = resolve_column_level(
        predictions.columns,
        {"individual", "individuals"},
    )

    bodypart_level = resolve_column_level(
        predictions.columns,
        {"bodypart", "bodyparts"},
    )

    coords_level = resolve_column_level(
        predictions.columns,
        {"coord", "coords", "coordinate", "coordinates"},
    )

    individuals = list(
        dict.fromkeys(
            str(value)
            for value in predictions.columns.get_level_values(
                individual_level
            )
        )
    )

    bodyparts = list(
        dict.fromkeys(
            str(value)
            for value in predictions.columns.get_level_values(
                bodypart_level
            )
        )
    )

    coordinates = list(
        dict.fromkeys(
            str(value)
            for value in predictions.columns.get_level_values(
                coords_level
            )
        )
    )

    required_coordinates = {
        "x",
        "y",
        "likelihood",
    }

    missing_coordinates = (
        required_coordinates
        - set(coordinates)
    )

    if missing_coordinates:
        raise ValueError(
            "Prediction output is missing required coordinate "
            "fields.\n"
            f"Missing: {sorted(missing_coordinates)}\n"
            f"Found: {coordinates}"
        )

    if not individuals:
        raise ValueError(
            "No predicted individual columns were found."
        )

    if len(individuals) > MAX_INDIVIDUALS:
        raise ValueError(
            "DLC returned more individual slots than requested.\n"
            f"Requested maximum: {MAX_INDIVIDUALS}\n"
            f"Returned: {individuals}"
        )

    return {
        "column_level_names": [
            None
            if name is None
            else str(name)
            for name in predictions.columns.names
        ],
        "individual_level": str(
            individual_level
        ),
        "bodypart_level": str(
            bodypart_level
        ),
        "coords_level": str(
            coords_level
        ),
        "prediction_individuals": individuals,
        "prediction_individual_count": len(
            individuals
        ),
        "prediction_bodyparts": bodyparts,
        "prediction_bodypart_count": len(
            bodyparts
        ),
        "coordinate_fields": coordinates,
    }


# ============================================================
# 15. Prediction completeness table
# ============================================================

def create_prediction_completeness(
    predictions: pd.DataFrame,
    row_manifest: pd.DataFrame,
    snapshot_information: dict[str, Any],
) -> pd.DataFrame:
    """
    Save one row per image × predicted individual.

    No prediction is removed. The confidence threshold here is only
    used to report how many points exceed common thresholds.
    """

    individual_level = resolve_column_level(
        predictions.columns,
        {"individual", "individuals"},
    )

    bodypart_level = resolve_column_level(
        predictions.columns,
        {"bodypart", "bodyparts"},
    )

    coords_level = resolve_column_level(
        predictions.columns,
        {"coord", "coords", "coordinate", "coordinates"},
    )

    individuals = list(
        dict.fromkeys(
            predictions.columns.get_level_values(
                individual_level
            )
        )
    )

    rows: list[dict[str, Any]] = []

    for prediction_row in range(
        len(predictions)
    ):
        identity_row = row_manifest.loc[
            row_manifest[
                "prediction_row"
            ].eq(prediction_row)
        ].iloc[0]

        prediction_series = predictions.iloc[
            prediction_row
        ]

        for individual in individuals:
            individual_columns = (
                predictions.columns.get_level_values(
                    individual_level
                )
                == individual
            )

            individual_series = (
                prediction_series.loc[
                    individual_columns
                ]
            )

            individual_column_index = (
                predictions.columns[
                    individual_columns
                ]
            )

            bodyparts = list(
                dict.fromkeys(
                    individual_column_index.get_level_values(
                        bodypart_level
                    )
                )
            )

            valid_xy_count = 0
            valid_likelihood_count = 0
            likelihood_values: list[float] = []
            above_01 = 0
            above_05 = 0
            above_09 = 0

            for bodypart in bodyparts:
                values: dict[str, float] = {}

                for coord in [
                    "x",
                    "y",
                    "likelihood",
                ]:
                    coordinate_mask = (
                        (
                            individual_column_index.get_level_values(
                                bodypart_level
                            )
                            == bodypart
                        )
                        & (
                            individual_column_index.get_level_values(
                                coords_level
                            )
                            == coord
                        )
                    )

                    matching_values = (
                        individual_series.loc[
                            coordinate_mask
                        ]
                    )

                    if len(matching_values) == 1:
                        values[coord] = (
                            matching_values.iloc[0]
                        )

                x = values.get("x")
                y = values.get("y")
                likelihood = values.get(
                    "likelihood"
                )

                if (
                    pd.notna(x)
                    and pd.notna(y)
                ):
                    valid_xy_count += 1

                if pd.notna(likelihood):
                    likelihood_float = float(
                        likelihood
                    )

                    valid_likelihood_count += 1

                    likelihood_values.append(
                        likelihood_float
                    )

                    if likelihood_float >= 0.1:
                        above_01 += 1

                    if likelihood_float >= 0.5:
                        above_05 += 1

                    if likelihood_float >= 0.9:
                        above_09 += 1

            rows.append(
                {
                    "snapshot_index": (
                        snapshot_information[
                            "snapshot_index"
                        ]
                    ),
                    "snapshot_name": (
                        snapshot_information[
                            "snapshot_name"
                        ]
                    ),
                    "snapshot_epoch": (
                        snapshot_information[
                            "snapshot_epoch"
                        ]
                    ),
                    "snapshot_type": (
                        snapshot_information[
                            "snapshot_type"
                        ]
                    ),
                    "prediction_row": prediction_row,
                    "image_key": (
                        identity_row[
                            "image_key"
                        ]
                    ),
                    "video": (
                        identity_row[
                            "video"
                        ]
                    ),
                    "image": (
                        identity_row[
                            "image"
                        ]
                    ),
                    "pred_individual": str(
                        individual
                    ),
                    "expected_bodypart_slots": len(
                        bodyparts
                    ),
                    "valid_xy_count": (
                        valid_xy_count
                    ),
                    "valid_likelihood_count": (
                        valid_likelihood_count
                    ),
                    "likelihood_ge_0_1_count": (
                        above_01
                    ),
                    "likelihood_ge_0_5_count": (
                        above_05
                    ),
                    "likelihood_ge_0_9_count": (
                        above_09
                    ),
                    "mean_likelihood": (
                        sum(likelihood_values)
                        / len(likelihood_values)
                        if likelihood_values
                        else None
                    ),
                    "minimum_likelihood": (
                        min(likelihood_values)
                        if likelihood_values
                        else None
                    ),
                    "maximum_likelihood": (
                        max(likelihood_values)
                        if likelihood_values
                        else None
                    ),
                }
            )

    return pd.DataFrame(
        rows
    )


# ============================================================
# 17. Run one snapshot
# ============================================================

def run_one_snapshot(
    snapshot_row: pd.Series,
    image_manifest: pd.DataFrame,
) -> dict[str, Any]:
    snapshot_index = int(
        snapshot_row[
            "snapshot_index"
        ]
    )

    snapshot_name = str(
        snapshot_row[
            "snapshot_name"
        ]
    )

    snapshot_epoch = (
        snapshot_row[
            "snapshot_epoch"
        ]
    )

    snapshot_type = str(
        snapshot_row[
            "snapshot_type"
        ]
    )

    snapshot_path = Path(
        snapshot_row[
            "snapshot_path"
        ]
    )

    folder_name = (
        f"index_{snapshot_index:03d}"
        f"__epoch_{snapshot_epoch}"
        f"__{safe_name(snapshot_name)}"
    )

    snapshot_output = (
        OUTPUT_ROOT
        / "snapshots"
        / folder_name
    )

    canonical_h5 = (
        snapshot_output
        / "predictions_original.h5"
    )

    canonical_csv = (
        snapshot_output
        / "predictions_original.csv"
    )

    row_manifest_path = (
        snapshot_output
        / "prediction_row_manifest.csv"
    )

    completeness_path = (
        snapshot_output
        / "prediction_completeness.csv"
    )

    prediction_manifest_path = (
        snapshot_output
        / "prediction_manifest.json"
    )

    required_existing_files = {
        canonical_h5,
        row_manifest_path,
        completeness_path,
        prediction_manifest_path,
    }

    if snapshot_output.exists():
        if OVERWRITE_EXISTING_RESULTS:
            shutil.rmtree(
                snapshot_output
            )

        elif all(
            path.is_file()
            for path in required_existing_files
        ):
            predictions, hdf_key = (
                read_prediction_dataframe(
                    canonical_h5
                )
            )

            structure = (
                inspect_prediction_structure(
                    predictions
                )
            )

            row_manifest = (
                build_prediction_row_manifest(
                    predictions=predictions,
                    image_manifest=image_manifest,
                )
            )

            print(
                f"Skipping existing valid result: "
                f"{snapshot_name}"
            )

            return {
                "snapshot_index": snapshot_index,
                "snapshot_name": snapshot_name,
                "snapshot_epoch": snapshot_epoch,
                "snapshot_type": snapshot_type,
                "status": (
                    "skipped_existing_valid_result"
                ),
                "prediction_rows": len(
                    predictions
                ),
                "hdf_key": hdf_key,
                "individual_count": (
                    structure[
                        "prediction_individual_count"
                    ]
                ),
                "mapped_image_count": len(
                    row_manifest
                ),
            }

        else:
            raise RuntimeError(
                "An incomplete output folder already exists:\n"
                f"{snapshot_output}\n"
                "Delete it manually or set "
                "OVERWRITE_EXISTING_RESULTS=True."
            )

    snapshot_output.mkdir(
        parents=True,
        exist_ok=False,
    )

    image_paths = (
        image_manifest
        .sort_values("validation_order")
        ["frozen_absolute_path"]
        .tolist()
    )

    print_header(
        f"Running snapshot: {snapshot_name}"
    )

    print(
        f"Snapshot index: {snapshot_index}"
    )

    print(
        f"Snapshot epoch: {snapshot_epoch}"
    )

    print(
        f"Validation images: {len(image_paths)}"
    )

    started_at = (
        datetime.now()
        .astimezone()
    )

    analyze_signature = inspect.signature(
        deeplabcut.analyze_images
    )
    analyze_parameters = set(
        analyze_signature.parameters
    )

    analyze_kwargs: dict[str, Any] = {
        "config": str(TRAIN_CONFIG),
        "images": image_paths,
        "shuffle": SHUFFLE,
        "trainingsetindex": TRAINING_SET_INDEX,
        "snapshot_index": snapshot_index,
        "device": DEVICE,
        "max_individuals": MAX_INDIVIDUALS,
        "save_as_csv": True,
        "plotting": SAVE_DLC_PLOTS,
        "pcutoff": PLOT_PCUTOFF,
        "plot_skeleton": True,
    }

    if "destfolder" in analyze_parameters:
        analyze_kwargs["destfolder"] = str(snapshot_output)
    elif "output_dir" in analyze_parameters:
        analyze_kwargs["output_dir"] = str(snapshot_output)
    else:
        raise RuntimeError(
            "analyze_images() provides neither destfolder nor "
            "output_dir. This should have been caught by the "
            "startup API check."
        )

    if "progress_bar" in analyze_parameters:
        analyze_kwargs["progress_bar"] = True

    deeplabcut.analyze_images(**analyze_kwargs)

    finished_at = (
        datetime.now()
        .astimezone()
    )

    dlc_h5 = find_prediction_h5(
        snapshot_output
    )

    dlc_csv = find_prediction_csv(
        snapshot_output
    )

    # Keep DLC's scorer-specific files and create stable copies
    # for all later scripts.
    shutil.copy2(
        dlc_h5,
        canonical_h5,
    )

    if dlc_csv is not None:
        shutil.copy2(
            dlc_csv,
            canonical_csv,
        )

    predictions, hdf_key = (
        read_prediction_dataframe(
            canonical_h5
        )
    )

    structure = (
        inspect_prediction_structure(
            predictions
        )
    )

    row_manifest = (
        build_prediction_row_manifest(
            predictions=predictions,
            image_manifest=image_manifest,
        )
    )

    row_manifest.to_csv(
        row_manifest_path,
        index=False,
    )

    completeness = (
        create_prediction_completeness(
            predictions=predictions,
            row_manifest=row_manifest,
            snapshot_information={
                "snapshot_index": snapshot_index,
                "snapshot_name": snapshot_name,
                "snapshot_epoch": snapshot_epoch,
                "snapshot_type": snapshot_type,
            },
        )
    )

    completeness.to_csv(
        completeness_path,
        index=False,
    )

    likelihood_columns = (
        predictions.columns.get_level_values(
            structure[
                "coords_level"
            ]
        )
        == "likelihood"
    )

    likelihood_data = predictions.loc[
        :,
        likelihood_columns,
    ]

    scorer_values: list[str] = []

    for possible_level in [
        "scorer",
        "scorers",
    ]:
        if possible_level in (
            predictions.columns.names
        ):
            scorer_values = list(
                dict.fromkeys(
                    str(value)
                    for value in predictions.columns.get_level_values(
                        possible_level
                    )
                )
            )

    prediction_manifest = {
        "snapshot_index": snapshot_index,
        "snapshot_name": snapshot_name,
        "snapshot_epoch": snapshot_epoch,
        "snapshot_type": snapshot_type,
        "snapshot_path": str(
            snapshot_path.resolve()
        ),
        "snapshot_sha256": (
            snapshot_row[
                "snapshot_sha256"
            ]
        ),
        "shuffle": SHUFFLE,
        "training_set_index": (
            TRAINING_SET_INDEX
        ),
        "max_individuals_requested": (
            MAX_INDIVIDUALS
        ),
        "device": DEVICE,
        "started_at": (
            started_at.isoformat()
        ),
        "finished_at": (
            finished_at.isoformat()
        ),
        "duration_seconds": (
            finished_at
            - started_at
        ).total_seconds(),
        "requested_image_count": (
            len(image_manifest)
        ),
        "prediction_row_count": (
            len(predictions)
        ),
        "prediction_column_count": (
            predictions.shape[1]
        ),
        "hdf_key": hdf_key,
        "dlc_generated_h5": (
            dlc_h5.name
        ),
        "dlc_generated_csv": (
            None
            if dlc_csv is None
            else dlc_csv.name
        ),
        "canonical_h5": (
            canonical_h5.name
        ),
        "canonical_csv": (
            canonical_csv.name
            if canonical_csv.is_file()
            else None
        ),
        "prediction_row_manifest": (
            row_manifest_path.name
        ),
        "prediction_completeness": (
            completeness_path.name
        ),
        "dlc_scorer_values": (
            scorer_values
        ),
        "minimum_saved_likelihood": (
            float(
                likelihood_data.min().min()
            )
            if (
                not likelihood_data.empty
                and likelihood_data.notna().any().any()
            )
            else None
        ),
        "maximum_saved_likelihood": (
            float(
                likelihood_data.max().max()
            )
            if (
                not likelihood_data.empty
                and likelihood_data.notna().any().any()
            )
            else None
        ),
        "plotting_enabled": (
            SAVE_DLC_PLOTS
        ),
        "plot_pcutoff": (
            PLOT_PCUTOFF
        ),
        "raw_predictions_filtered": False,
        **structure,
    }

    write_json(
        prediction_manifest_path,
        prediction_manifest,
    )

    if len(predictions) != len(
        image_manifest
    ):
        raise RuntimeError(
            "Final prediction row count check failed."
        )

    print(
        f"Completed: {snapshot_name}"
    )

    print(
        f"Prediction individuals: "
        f"{structure['prediction_individuals']}"
    )

    print(
        f"Prediction bodyparts: "
        f"{structure['prediction_bodyparts']}"
    )

    return {
        "snapshot_index": snapshot_index,
        "snapshot_name": snapshot_name,
        "snapshot_epoch": snapshot_epoch,
        "snapshot_type": snapshot_type,
        "status": "completed",
        "prediction_rows": len(
            predictions
        ),
        "hdf_key": hdf_key,
        "individual_count": (
            structure[
                "prediction_individual_count"
            ]
        ),
        "individual_names": "|".join(
            structure[
                "prediction_individuals"
            ]
        ),
        "bodypart_count": (
            structure[
                "prediction_bodypart_count"
            ]
        ),
        "mapped_image_count": len(
            row_manifest
        ),
    }


# ============================================================
# 18. Main
# ============================================================

def main() -> None:
    print_header(
        "DeepLabCut External Validation"
    )

    verify_inputs()

    OUTPUT_ROOT.mkdir(
        parents=True,
        exist_ok=True,
    )

    usable_metadata, image_manifest = (
        prepare_validation_images()
    )

    usable_metadata.to_csv(
        OUTPUT_ROOT
        / "usable_validation_metadata.csv",
        index=False,
    )

    image_manifest.to_csv(
        OUTPUT_ROOT
        / "validation_image_manifest.csv",
        index=False,
    )

    (
        pytorch_config_path,
        train_folder,
    ) = get_training_paths()

    (
        snapshots,
        snapshot_manifest,
    ) = discover_snapshots(
        train_folder=train_folder
    )

    # snapshots is deliberately retained because the snapshot order
    # produced by DLC determines the snapshot_index passed into
    # analyze_images.

    snapshot_manifest.to_csv(
        OUTPUT_ROOT
        / "snapshot_manifest.csv",
        index=False,
    )

    summary_rows: list[
        dict[str, Any]
    ] = []

    summary_path = (
        OUTPUT_ROOT
        / "step09_run_summary.csv"
    )

    for _, snapshot_row in (
        snapshot_manifest
        .sort_values("snapshot_index")
        .iterrows()
    ):
        try:
            result = run_one_snapshot(
                snapshot_row=(
                    snapshot_row
                ),
                image_manifest=(
                    image_manifest
                ),
            )

        except Exception as error:
            failed_result = {
                "snapshot_index": (
                    snapshot_row[
                        "snapshot_index"
                    ]
                ),
                "snapshot_name": (
                    snapshot_row[
                        "snapshot_name"
                    ]
                ),
                "snapshot_epoch": (
                    snapshot_row[
                        "snapshot_epoch"
                    ]
                ),
                "snapshot_type": (
                    snapshot_row[
                        "snapshot_type"
                    ]
                ),
                "status": "failed",
                "error_type": (
                    type(error).__name__
                ),
                "error_message": str(
                    error
                ),
            }

            summary_rows.append(
                failed_result
            )

            pd.DataFrame(
                summary_rows
            ).to_csv(
                summary_path,
                index=False,
            )

            print_header(
                "Step 09 stopped because of an error"
            )

            print(
                f"Snapshot: "
                f"{snapshot_row['snapshot_name']}"
            )

            print(
                f"Error: {error}"
            )

            raise

        summary_rows.append(
            result
        )

        pd.DataFrame(
            summary_rows
        ).to_csv(
            summary_path,
            index=False,
        )

    print_header(
        "Step 09 completed successfully"
    )

    print(
        f"Snapshots processed: "
        f"{len(snapshot_manifest)}"
    )

    print(
        f"Validation images per snapshot: "
        f"{len(image_manifest)}"
    )

    print(
        f"Results saved under:\n"
        f"{OUTPUT_ROOT}"
    )

    print()
    print(
        "Step 09 saved raw multi-animal predictions only."
    )

    print(
        "No GT matching or evaluation metric was applied."
    )


if __name__ == "__main__":
    main()
