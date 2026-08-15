from pathlib import Path
import pandas as pd


# Paths

PROJECT_ROOT = Path(
    "/path/to/your/MousePose_Project"
)

DATASET_DIR = PROJECT_ROOT / "data" / "dataset_split"
ANNOTATION_DIR = PROJECT_ROOT / "data" / "annotations"

ANNOTATION_DIR.mkdir(parents=True, exist_ok=True)


# Settings

SPLITS = ["train", "val", "test"]

IMAGE_EXTENSIONS = {
    ".png",
    ".jpg",
    ".jpeg",
    ".bmp",
    ".tif",
    ".tiff",
}


# Metadata columns

BASE_COLUMNS = [
    "split",
    "video",
    "image",
    "mouse_count",
]

MOUSE_FIELDS = [
    "orientation",
    "pose",
    "key_point_visibility",
    "occlusion_type",
    "mouse_occlusion",
]

MOUSE_COLUMNS = []

for mouse_id in range(1, 4):
    for field in MOUSE_FIELDS:
        MOUSE_COLUMNS.append(
            f"mouse_{mouse_id}_{field}"
        )

ALL_COLUMNS = (
    BASE_COLUMNS
    + MOUSE_COLUMNS
    + ["usable"]
)


# Find images

def find_images(split_dir: Path) -> list[dict]:
    """
    Find all images inside:

    dataset_split/
        train/
            video_name/
                image.png

    Returns one dictionary per image.
    """

    rows = []

    if not split_dir.exists():
        print(f"Split directory does not exist: {split_dir}")
        return rows

    for video_dir in sorted(split_dir.iterdir()):

        if not video_dir.is_dir():
            continue

        image_paths = sorted(
            path
            for path in video_dir.iterdir()
            if (
                path.is_file()
                and path.suffix.lower() in IMAGE_EXTENSIONS
            )
        )

        for image_path in image_paths:

            row = {
                "split": split_dir.name,
                "video": video_dir.name,
                "image": image_path.name,
                "mouse_count": "",
                "usable": "",
            }

            for column in MOUSE_COLUMNS:
                row[column] = ""

            rows.append(row)

    return rows


# Preserve existing annotations

def preserve_existing_annotations(
    new_df: pd.DataFrame,
    csv_path: Path,
) -> pd.DataFrame:
    """
    Preserve annotations from an existing metadata CSV.

    Images are matched using:
        video + image
    """

    if not csv_path.exists():
        return new_df

    old_df = pd.read_csv(
        csv_path,
        dtype=str,
        keep_default_na=False,
    )

    if not {"video", "image"}.issubset(old_df.columns):
        print(
            f"Warning: existing CSV has no video/image columns: "
            f"{csv_path}"
        )
        return new_df

    old_df = old_df.drop_duplicates(
        subset=["video", "image"],
        keep="last",
    )

    old_lookup = old_df.set_index(
        ["video", "image"]
    )

    columns_to_preserve = [
        column
        for column in ALL_COLUMNS
        if (
            column in old_df.columns
            and column not in {
                "split",
                "video",
                "image",
            }
        )
    ]

    for index, row in new_df.iterrows():

        key = (
            row["video"],
            row["image"],
        )

        if key not in old_lookup.index:
            continue

        old_row = old_lookup.loc[key]

        for column in columns_to_preserve:
            new_df.at[index, column] = old_row[column]

    return new_df


# Create metadata for one split

def create_metadata(split: str) -> None:

    split_dir = DATASET_DIR / split
    csv_path = ANNOTATION_DIR / f"{split}_metadata.csv"

    rows = find_images(split_dir)

    new_df = pd.DataFrame(
        rows,
        columns=ALL_COLUMNS,
    )

    new_df = preserve_existing_annotations(
        new_df=new_df,
        csv_path=csv_path,
    )

    new_df.to_csv(
        csv_path,
        index=False,
    )

    print(
        f"{split}: {len(new_df)} frames"
    )

    print(
        f"Saved: {csv_path}"
    )


# Run

def main():

    print("=" * 70)
    print("Creating annotation metadata")
    print("=" * 70)

    for split in SPLITS:
        create_metadata(split)

    print("=" * 70)
    print("Finished")
    print("=" * 70)


if __name__ == "__main__":
    main()
