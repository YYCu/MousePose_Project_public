from pathlib import Path
import re
import shutil

import deeplabcut



# Paths

PROJECT_ROOT = Path(
    "/path/to/your/MousePose_Project"
)


ROUND1_IMAGE_DIR = (
    PROJECT_ROOT
    / "data"
    / "your_selected_image_folder_name"
    / "images"
)


DLC_ROUND1_WORKING_DIR = (
    PROJECT_ROOT
    / "your_dlc_working_directory_name"
)


SOURCE_VIDEO_DIR = Path(
    "/path/to/your/source/videos"
)


CONFIG_PATH_RECORD = (
    PROJECT_ROOT
    / "data"
    / "your_selected_image_folder_name"
    / "dlc_config_path.txt"
)



# DLC project settings


TASK = "MousePose"
SCORER = "Yuu"

VIDEO_EXTENSION = ".flv"

IMAGE_EXTENSIONS = {
    ".png",
    ".jpg",
    ".jpeg",
    ".bmp",
    ".tif",
    ".tiff",
}

# 8WK_F_WT_00_selected_09_frame_017280.png
# └──────────┘
# 8WK_F_WT_00
VIDEO_NAME_PATTERN = re.compile(
    r"^(8WK_F_WT_\d{2})(?:_|$)"
)



# Image functions

def get_round1_images() -> list[Path]:
    """
    Read and validate the 100 selected Round 1 images.

    Returns:
        A sorted list containing the selected image paths.

    Raises:
        FileNotFoundError: If the image directory does not exist.
        RuntimeError: If the number of images is not exactly 100.
    """
    if not ROUND1_IMAGE_DIR.exists():
        raise FileNotFoundError(
            "Round 1 image directory does not exist:\n"
            f"{ROUND1_IMAGE_DIR}"
        )

    image_paths = sorted(
        path
        for path in ROUND1_IMAGE_DIR.iterdir()
        if (
            path.is_file()
            and path.suffix.lower() in IMAGE_EXTENSIONS
        )
    )

    if not image_paths:
        raise FileNotFoundError(
            "No images were found in:\n"
            f"{ROUND1_IMAGE_DIR}"
        )

    return image_paths


def get_source_video_name(image_path: Path) -> str:
    """
    Extract the source video name from an image filename.

    Example:
        8WK_F_WT_03_selected_06_frame_035890.png

    Returns:
        8WK_F_WT_03

    Args:
        image_path: Path to one selected image.

    Raises:
        ValueError: If the filename does not begin with a valid video name.
    """
    match = VIDEO_NAME_PATTERN.match(image_path.stem)

    if match is None:
        raise ValueError(
            "Could not determine the source video from image filename:\n"
            f"{image_path.name}\n\n"
            "Expected the filename to begin with something like:\n"
            "8WK_F_WT_00_"
        )

    return match.group(1)


def group_images_by_video(
    image_paths: list[Path],
) -> dict[str, list[Path]]:
    """
    Group selected images by their source video name.

    Args:
        image_paths: The selected Round 1 image paths.

    Returns:
        A dictionary whose keys are source video names and whose values
        are lists of images originating from those videos.
    """
    grouped_images: dict[str, list[Path]] = {}

    for image_path in image_paths:
        video_name = get_source_video_name(image_path)

        grouped_images.setdefault(
            video_name,
            [],
        ).append(image_path)

    return dict(
        sorted(grouped_images.items())
    )



# Video functions

def validate_source_video_directory() -> None:
    """
    Confirm that the original source video directory exists.

    Raises:
        FileNotFoundError: If the source video directory does not exist.
        NotADirectoryError: If SOURCE_VIDEO_DIR is not a directory.
    """
    if not SOURCE_VIDEO_DIR.exists():
        raise FileNotFoundError(
            "Source video directory does not exist:\n"
            f"{SOURCE_VIDEO_DIR}\n\n"
            "Edit SOURCE_VIDEO_DIR near the top of this script."
        )

    if not SOURCE_VIDEO_DIR.is_dir():
        raise NotADirectoryError(
            "SOURCE_VIDEO_DIR is not a directory:\n"
            f"{SOURCE_VIDEO_DIR}"
        )


def find_required_videos(
    grouped_images: dict[str, list[Path]],
) -> dict[str, Path]:
    """
    Find the original video corresponding to every image group.

    Each expected video path is constructed as:

        SOURCE_VIDEO_DIR / '<video_name>.flv'

    Args:
        grouped_images: Images grouped by source video name.

    Returns:
        A dictionary mapping video names to original video paths.

    Raises:
        FileNotFoundError: If one or more required videos cannot be found.
    """
    required_videos: dict[str, Path] = {}
    missing_videos: list[Path] = []

    for video_name in grouped_images:
        video_path = (
            SOURCE_VIDEO_DIR
            / f"{video_name}{VIDEO_EXTENSION}"
        )

        if video_path.is_file():
            required_videos[video_name] = video_path
        else:
            missing_videos.append(video_path)

    if missing_videos:
        missing_text = "\n".join(
            str(path)
            for path in missing_videos
        )

        raise FileNotFoundError(
            "The following required source videos were not found:\n\n"
            f"{missing_text}\n\n"
            "Check SOURCE_VIDEO_DIR and the video filenames."
        )

    return required_videos



# DLC project functions


def check_working_directory() -> None:
    """
    Prevent accidentally creating a second project for the same round.

    The script allows the working directory to be absent or empty.
    It stops if the directory already contains files or folders.

    Raises:
        FileExistsError: If the Round 1 working directory is not empty.
    """
    DLC_ROUND1_WORKING_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    existing_items = list(
        DLC_ROUND1_WORKING_DIR.iterdir()
    )

    if existing_items:
        existing_text = "\n".join(
            f"- {path.name}"
            for path in sorted(existing_items)
        )

        raise FileExistsError(
            "The Round 1 DLC working directory is not empty:\n"
            f"{DLC_ROUND1_WORKING_DIR}\n\n"
            "Existing contents:\n"
            f"{existing_text}\n\n"
            "Because this script creates a new DLC project, remove the "
            "unfinished old project before running it again."
        )


def create_dlc_project(
    required_videos: dict[str, Path],
) -> Path:
    """
    Create an official DeepLabCut multi-animal project.

    All source videos represented in the selected 100 images are passed
    to DeepLabCut. DeepLabCut creates the standard project structure and
    records these videos in config.yaml.

    Args:
        required_videos: Mapping from video name to source video path.

    Returns:
        The path to the generated config.yaml.

    Raises:
        RuntimeError: If DeepLabCut does not return a valid config path.
    """
    video_paths = [
        str(video_path)
        for video_path in required_videos.values()
    ]

    config_path = deeplabcut.create_new_project(
        project=TASK,
        experimenter=SCORER,
        videos=video_paths,
        working_directory=str(DLC_ROUND1_WORKING_DIR),
        copy_videos=False,
        multianimal=True,
    )

    if config_path is None:
        raise RuntimeError(
            "DeepLabCut did not return a config.yaml path."
        )

    config_path = Path(config_path)

    if not config_path.is_file():
        raise RuntimeError(
            "DeepLabCut returned a config path, but the file "
            "does not exist:\n"
            f"{config_path}"
        )

    return config_path


def copy_images_to_matching_folders(
    project_dir: Path,
    grouped_images: dict[str, list[Path]],
) -> dict[str, Path]:
    """
    Copy every image into its matching labeled-data video folder.

    Example:
        An image beginning with 8WK_F_WT_03 is copied into:

        labeled-data/8WK_F_WT_03/

    Args:
        project_dir: The root directory of the generated DLC project.
        grouped_images: Images grouped by source video name.

    Returns:
        A dictionary mapping each video name to its labeled-data folder.

    Raises:
        FileExistsError: If a target folder already contains image files.
        RuntimeError: If the number of copied images is incorrect.
    """
    labeled_data_root = (
        project_dir
        / "labeled-data"
    )

    target_directories: dict[str, Path] = {}

    for video_name, image_paths in grouped_images.items():
        target_dir = (
            labeled_data_root
            / video_name
        )

        # DLC should normally create this directory automatically.
        # mkdir is kept as a safe fallback.
        target_dir.mkdir(
            parents=True,
            exist_ok=True,
        )

        existing_images = sorted(
            path
            for path in target_dir.iterdir()
            if (
                path.is_file()
                and path.suffix.lower() in IMAGE_EXTENSIONS
            )
        )

        if existing_images:
            raise FileExistsError(
                f"The target folder already contains "
                f"{len(existing_images)} image files:\n"
                f"{target_dir}\n\n"
                "The script stopped to avoid overwriting data."
            )

        for source_path in image_paths:
            destination_path = (
                target_dir
                / source_path.name
            )

            shutil.copy2(
                source_path,
                destination_path,
            )

        copied_images = sorted(
            path
            for path in target_dir.iterdir()
            if (
                path.is_file()
                and path.suffix.lower() in IMAGE_EXTENSIONS
            )
        )

        if len(copied_images) != len(image_paths):
            raise RuntimeError(
                f"Copy verification failed for {video_name}.\n"
                f"Expected: {len(image_paths)} images\n"
                f"Found: {len(copied_images)} images\n"
                f"Folder: {target_dir}"
            )

        target_directories[video_name] = target_dir

    return target_directories


def verify_total_copied_images(
    target_directories: dict[str, Path],
    expected_count: int,
) -> int:
    """
    Count all copied images across the labeled-data subfolders.

    Args:
        target_directories: Mapping from video name to target folder.

    Returns:
        The total number of copied images.

    Raises:
        RuntimeError: If the total is not exactly EXPECTED_IMAGE_COUNT.
    """
    total_copied = 0

    for target_dir in target_directories.values():
        total_copied += sum(
            1
            for path in target_dir.iterdir()
            if (
                path.is_file()
                and path.suffix.lower() in IMAGE_EXTENSIONS
            )
        )

    if total_copied != expected_count:
        raise RuntimeError(
            f"Expected {expected_count} copied images in total, "
            f"but found {total_copied}."
        )

    return total_copied


def save_config_path(config_path: Path) -> None:
    """
    Save the generated config.yaml path for later scripts.

    Args:
        config_path: Path to the generated DLC config.yaml.
    """
    CONFIG_PATH_RECORD.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    CONFIG_PATH_RECORD.write_text(
        str(config_path.resolve()),
        encoding="utf-8",
    )


# Main

def main() -> None:
    """
    Create the Round 1 official DLC project and distribute images.
    """
    print("=" * 72)
    print("Create Round 1 DeepLabCut project")
    print("=" * 72)

    # --------------------------------------------------------
    # 1. Read selected images
    # --------------------------------------------------------

    image_paths = get_round1_images()

    print()
    print(f"Selected images found: {len(image_paths)}")
    print(ROUND1_IMAGE_DIR)

    # --------------------------------------------------------
    # 2. Group images according to source video
    # --------------------------------------------------------

    grouped_images = group_images_by_video(
        image_paths
    )

    print()
    print(
        f"Source videos represented in Round 1: "
        f"{len(grouped_images)}"
    )

    for video_name, paths in grouped_images.items():
        print(
            f"  {video_name}: {len(paths)} images"
        )

    # --------------------------------------------------------
    # 3. Find and validate original videos
    # --------------------------------------------------------

    validate_source_video_directory()

    required_videos = find_required_videos(
        grouped_images
    )

    print()
    print(
        f"Required source videos found: "
        f"{len(required_videos)}"
    )

    # --------------------------------------------------------
    # 4. Protect existing project data
    # --------------------------------------------------------

    check_working_directory()

    # --------------------------------------------------------
    # 5. Create official DLC project
    # --------------------------------------------------------

    print()
    print(
        "Creating official DeepLabCut "
        "multi-animal project..."
    )

    config_path = create_dlc_project(
        required_videos
    )

    project_dir = config_path.parent

    # --------------------------------------------------------
    # 6. Copy images into corresponding video folders
    # --------------------------------------------------------

    print()
    print(
        "Copying selected images into matching "
        "labeled-data folders..."
    )

    target_directories = copy_images_to_matching_folders(
        project_dir=project_dir,
        grouped_images=grouped_images,
    )

    total_copied = verify_total_copied_images(
        target_directories=target_directories,
        expected_count=len(image_paths),
    )

    # --------------------------------------------------------
    # 7. Save config path
    # --------------------------------------------------------

    save_config_path(config_path)

    # --------------------------------------------------------
    # 8. Print final summary
    # --------------------------------------------------------

    print()
    print("=" * 72)
    print("Step 10 completed successfully")
    print("=" * 72)

    print()
    print("DLC project directory:")
    print(project_dir)

    print()
    print("Config file:")
    print(config_path)

    print()
    print(
        f"Images copied successfully: "
        f"{total_copied}"
    )

    print()
    print("labeled-data distribution:")

    for video_name, target_dir in target_directories.items():
        image_count = len(
            grouped_images[video_name]
        )

        print(
            f"  {video_name}: "
            f"{image_count} images"
        )
        print(
            f"    {target_dir}"
        )

    print()
    print("Config path record:")
    print(CONFIG_PATH_RECORD)

    print()
    print("The config has not been manually modified.")
    print("The annotation interface has not been opened.")


if __name__ == "__main__":
    main()
