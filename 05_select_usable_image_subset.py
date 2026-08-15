from pathlib import Path
from collections import defaultdict, deque
import random
import shutil

import pandas as pd



# Paths

PROJECT_ROOT = Path(
    "/path/to/your/MousePose_Project"
)

ANNOTATION_DIR = (
    PROJECT_ROOT
    / "data"
    / "annotations"
)

TRAIN_IMAGE_DIR = (
    PROJECT_ROOT
    / "data"
    / "dataset_split"
    / "train, val or test"
)

OUTPUT_DIR = (
    PROJECT_ROOT
    / "data"
    / "your_output_folder_name"
)

OUTPUT_IMAGE_DIR = OUTPUT_DIR / "images"

OUTPUT_CSV = (
    OUTPUT_DIR
    / "your_output_csv_filename.csv"
)


# Settings

N_IMAGES = 100

RANDOM_SEED = 42

IMAGE_EXTENSIONS = {
    ".png",
    ".jpg",
    ".jpeg",
    ".bmp",
    ".tif",
    ".tiff",
}


# Helper functions

def normalise_column_name(column_name):
    return str(column_name).strip().lower().replace(" ", "_")


def find_column(dataframe, candidates):
    normalised_columns = {
        normalise_column_name(column): column
        for column in dataframe.columns
    }

    for candidate in candidates:
        candidate = normalise_column_name(candidate)

        if candidate in normalised_columns:
            return normalised_columns[candidate]

    return None


def is_usable(value):
    if pd.isna(value):
        return False

    value = str(value).strip().lower()

    usable_values = {
        "yes",
        "y",
        "true",
        "1",
        "usable",
        "use",
        "keep",
    }

    return value in usable_values


def extract_video_name(image_path):
    stem = image_path.stem

    if "_img" in stem:
        return stem.split("_img")[0]

    return image_path.parent.name


def load_annotation_csvs():
    if not ANNOTATION_DIR.exists():
        raise FileNotFoundError(
            f"Annotation folder does not exist:\n{ANNOTATION_DIR}"
        )

    csv_files = sorted(ANNOTATION_DIR.rglob("*.csv"))

    if not csv_files:
        raise FileNotFoundError(
            f"No CSV files found in:\n{ANNOTATION_DIR}"
        )

    dataframes = []

    print("=" * 70)
    print("Annotation CSV files")
    print("=" * 70)

    for csv_path in csv_files:
        try:
            dataframe = pd.read_csv(csv_path)

            if dataframe.empty:
                print(f"Skipped empty CSV: {csv_path.name}")
                continue

            dataframe["_source_csv"] = str(csv_path)

            dataframes.append(dataframe)

            print(
                f"Loaded: {csv_path.name} "
                f"({len(dataframe)} rows)"
            )

        except Exception as error:
            print(
                f"Could not read {csv_path.name}: "
                f"{error}"
            )

    if not dataframes:
        raise RuntimeError(
            "No usable annotation CSV files were loaded."
        )

    combined_dataframe = pd.concat(
        dataframes,
        ignore_index=True,
        sort=False,
    )

    return combined_dataframe


def build_train_image_index():
    if not TRAIN_IMAGE_DIR.exists():
        raise FileNotFoundError(
            f"Train image folder does not exist:\n"
            f"{TRAIN_IMAGE_DIR}"
        )

    image_paths = [
        path
        for path in TRAIN_IMAGE_DIR.rglob("*")
        if (
            path.is_file()
            and path.suffix.lower() in IMAGE_EXTENSIONS
        )
    ]

    if not image_paths:
        raise FileNotFoundError(
            f"No train images found in:\n"
            f"{TRAIN_IMAGE_DIR}"
        )

    image_index = {}
    duplicate_names = defaultdict(list)

    for image_path in image_paths:
        filename = image_path.name

        if filename in image_index:
            duplicate_names[filename].append(image_path)
        else:
            image_index[filename] = image_path

    if duplicate_names:
        duplicate_examples = list(duplicate_names.keys())[:10]

        raise RuntimeError(
            "Duplicate image filenames were found in the train folder.\n"
            "Because the CSV normally identifies images by filename, "
            "the script cannot safely choose between them.\n\n"
            f"Examples:\n{duplicate_examples}"
        )

    print()
    print("=" * 70)
    print("Train images")
    print("=" * 70)
    print(f"Found {len(image_paths)} train images")

    return image_index


def select_balanced_images(candidate_dataframe, random_seed):
    random_generator = random.Random(random_seed)

    grouped_rows = defaultdict(list)

    for row_index, row in candidate_dataframe.iterrows():
        grouped_rows[row["_video_name"]].append(row_index)

    video_names = sorted(grouped_rows.keys())

    random_generator.shuffle(video_names)

    video_queues = {}

    for video_name in video_names:
        indices = grouped_rows[video_name]
        random_generator.shuffle(indices)
        video_queues[video_name] = deque(indices)

    selected_indices = []

    while len(selected_indices) < N_IMAGES:
        added_in_this_round = False

        for video_name in video_names:
            queue = video_queues[video_name]

            if queue:
                selected_indices.append(queue.popleft())
                added_in_this_round = True

                if len(selected_indices) == N_IMAGES:
                    break

        if not added_in_this_round:
            break

    return candidate_dataframe.loc[selected_indices].copy()


# ============================================================
# Main
# ============================================================

def main():
    random.seed(RANDOM_SEED)

    # --------------------------------------------------------
    # 1. Load annotation CSV files
    # --------------------------------------------------------

    annotations = load_annotation_csvs()

    # --------------------------------------------------------
    # 2. Detect relevant CSV columns
    # --------------------------------------------------------

    image_column = find_column(
        annotations,
        [
            "image_path",
            "image",
            "filename",
            "file_name",
            "image_name",
            "image_filename",
            "path",
        ],
    )

    usability_column = find_column(
        annotations,
        [
            "usability",
            "usable",
            "is_usable",
            "use_image",
            "keep",
        ],
    )

    if image_column is None:
        raise KeyError(
            "Could not identify the image filename column.\n\n"
            f"CSV columns are:\n"
            f"{list(annotations.columns)}\n\n"
            "Expected a column such as:\n"
            "image_path, image, filename, file_name, or image_name"
        )

    if usability_column is None:
        raise KeyError(
            "Could not identify the usability column.\n\n"
            f"CSV columns are:\n"
            f"{list(annotations.columns)}\n\n"
            "Expected a column such as:\n"
            "usability, usable, is_usable, or keep"
        )

    print()
    print("=" * 70)
    print("Detected columns")
    print("=" * 70)
    print(f"Image column:     {image_column}")
    print(f"Usability column: {usability_column}")

    # --------------------------------------------------------
    # 3. Build train image index
    # --------------------------------------------------------

    train_image_index = build_train_image_index()

    # --------------------------------------------------------
    # 4. Match CSV rows with train images
    # --------------------------------------------------------

    annotations["_image_filename"] = (
        annotations[image_column]
        .astype(str)
        .map(lambda value: Path(value).name)
    )

    annotations["_train_image_path"] = annotations[
        "_image_filename"
    ].map(train_image_index)

    train_annotations = annotations[
        annotations["_train_image_path"].notna()
    ].copy()

    print()
    print("=" * 70)
    print("CSV and train matching")
    print("=" * 70)
    print(f"All annotation rows:       {len(annotations)}")
    print(f"Rows belonging to train:   {len(train_annotations)}")

    if train_annotations.empty:
        raise RuntimeError(
            "No CSV records matched images in the train folder.\n"
            "Please check the image filename column and directory paths."
        )

    # --------------------------------------------------------
    # 5. Remove usability = No
    # --------------------------------------------------------

    usable_mask = train_annotations[
        usability_column
    ].map(is_usable)

    usable_annotations = train_annotations[
        usable_mask
    ].copy()

    number_excluded = (
        len(train_annotations)
        - len(usable_annotations)
    )

    print(f"Usability = Yes:           {len(usable_annotations)}")
    print(f"Excluded usability = No:   {number_excluded}")

    if len(usable_annotations) < N_IMAGES:
        raise RuntimeError(
            f"Only {len(usable_annotations)} usable train images "
            f"were found, but {N_IMAGES} were requested."
        )

    # --------------------------------------------------------
    # 6. Remove duplicate image records
    # --------------------------------------------------------

    before_deduplication = len(usable_annotations)

    usable_annotations = usable_annotations.drop_duplicates(
        subset="_image_filename",
        keep="first",
    ).copy()

    duplicates_removed = (
        before_deduplication
        - len(usable_annotations)
    )

    print(f"Duplicate rows removed:    {duplicates_removed}")
    print(f"Unique usable images:      {len(usable_annotations)}")

    if len(usable_annotations) < N_IMAGES:
        raise RuntimeError(
            f"After removing duplicate records, only "
            f"{len(usable_annotations)} usable train images remain."
        )

    # --------------------------------------------------------
    # 7. Add video name
    # --------------------------------------------------------

    usable_annotations["_video_name"] = usable_annotations[
        "_train_image_path"
    ].map(extract_video_name)

    # --------------------------------------------------------
    # 8. Select 100 images
    # --------------------------------------------------------

    selected = select_balanced_images(
        usable_annotations,
        RANDOM_SEED,
    )

    if len(selected) != N_IMAGES:
        raise RuntimeError(
            f"Expected to select {N_IMAGES} images, "
            f"but selected {len(selected)}."
        )

    # Sort selected records for easier review
    selected = selected.sort_values(
        by=["_video_name", "_image_filename"]
    ).reset_index(drop=True)

    # --------------------------------------------------------
    # 9. Prepare output directory
    # --------------------------------------------------------

    if OUTPUT_DIR.exists():
        raise FileExistsError(
            f"Output folder already exists:\n"
            f"{OUTPUT_DIR}\n\n"
            "Delete or rename it before running the script again. "
            "This prevents accidentally overwriting your Round 1 selection."
        )

    OUTPUT_IMAGE_DIR.mkdir(
        parents=True,
        exist_ok=False,
    )

    # --------------------------------------------------------
    # 10. Copy selected images
    # --------------------------------------------------------

    copied_paths = []

    for _, row in selected.iterrows():
        source_path = Path(row["_train_image_path"])
        destination_path = (
            OUTPUT_IMAGE_DIR
            / source_path.name
        )

        shutil.copy2(
            source_path,
            destination_path,
        )

        copied_paths.append(
            str(
                Path("images")
                / destination_path.name
            )
        )

    selected["round1_image_path"] = copied_paths
    selected["round1_selection_order"] = range(
        1,
        len(selected) + 1,
    )

    # --------------------------------------------------------
    # 11. Save corresponding CSV
    # --------------------------------------------------------
    columns_to_remove = [
        "_train_image_path",
    ]

    selected_for_csv = selected.drop(
        columns=columns_to_remove,
        errors="ignore",
    )

    selected_for_csv.to_csv(
        OUTPUT_CSV,
        index=False,
    )

    # --------------------------------------------------------
    # 12. Print summary
    # --------------------------------------------------------

    video_counts = (
        selected["_video_name"]
        .value_counts()
        .sort_index()
    )

    print()
    print("=" * 70)
    print("Round 1 dataset created successfully")
    print("=" * 70)
    print(f"Selected images: {len(selected)}")
    print(f"Videos covered:  {selected['_video_name'].nunique()}")
    print()
    print(f"Output folder:\n{OUTPUT_DIR}")
    print()
    print(f"Images:\n{OUTPUT_IMAGE_DIR}")
    print()
    print(f"CSV:\n{OUTPUT_CSV}")
    print()
    print("Selected images per video:")
    print(video_counts.to_string())


if __name__ == "__main__":
    main()
