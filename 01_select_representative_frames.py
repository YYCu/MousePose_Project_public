from pathlib import Path
import cv2
import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
from tqdm import tqdm



# Paths

VIDEO_DIR = Path("/path/to/your/videos")


OUTPUT_DIR = Path("PROJECT_ROOT /data/kmeans_frames")


METADATA_PATH = Path("PROJECT_ROOT/data/kmeans_metadata.csv")


OUTPUT_DIR.mkdir(
    parents=True,
    exist_ok=True
)

# Parameters

MAX_CANDIDATES = 4500

N_CLUSTERS = 20


# Feature extraction

def extract_feature(frame):

    small = cv2.resize(
        frame,
        (64, 64)
    )

    gray = cv2.cvtColor(
        small,
        cv2.COLOR_BGR2GRAY
    )

    return gray.flatten()




# Videos

videos = sorted(
    VIDEO_DIR.glob("*.flv")
)


print(
    f"Found {len(videos)} videos"
)


metadata = []


# Main loop

for video_path in videos:

    video_name = video_path.stem

    print(
        "\nProcessing:",
        video_name
    )


    save_dir = (
        OUTPUT_DIR /
        video_name
    )

    save_dir.mkdir(
        exist_ok=True
    )


    cap = cv2.VideoCapture(
        str(video_path)
    )


    total_frames = int(
        cap.get(
            cv2.CAP_PROP_FRAME_COUNT
        )
    )


    print(
        "Total frames:",
        total_frames
    )


    # automatic interval

    interval = max(
        1,
        total_frames // MAX_CANDIDATES
    )


    print(
        "Sampling interval:",
        interval
    )


    frames = []
    frame_ids = []


    # --------------------------
    # Sample frames
    # --------------------------

    for frame_id in tqdm(
        range(
            0,
            total_frames,
            interval
        )
    ):

        cap.set(
            cv2.CAP_PROP_POS_FRAMES,
            frame_id
        )


        ret, frame = cap.read()


        if not ret:
            continue


        frames.append(
            frame
        )


        frame_ids.append(
            frame_id
        )


    cap.release()


    print(
        "Candidate frames:",
        len(frames)
    )


    # --------------------------
    # Features
    # --------------------------

    features = np.array(
        [
            extract_feature(frame)
            for frame in frames
        ]
    )


    # --------------------------
    # K-means
    # --------------------------

    kmeans = KMeans(
        n_clusters=N_CLUSTERS,
        random_state=42,
        n_init=10
    )


    labels = kmeans.fit_predict(
        features
    )


    centers = kmeans.cluster_centers_


    selected_indices = []


    for cluster_id, center in enumerate(centers):

        members = np.where(
            labels == cluster_id
        )[0]


        distances = np.linalg.norm(
            features[members] - center,
            axis=1
        )


        selected = members[
            np.argmin(distances)
        ]


        selected_indices.append(
            selected
        )


    # sort by time order

    selected_indices = sorted(
        selected_indices
    )


    # --------------------------
    # Save frames
    # --------------------------

    for idx, frame_index in enumerate(selected_indices):

        original_frame = frame_ids[
            frame_index
        ]


        filename = (
            f"{video_name}_"
            f"selected_{idx:02d}_"
            f"frame_{original_frame:06d}.png"
        )


        cv2.imwrite(
            str(
                save_dir / filename
            ),
            frames[frame_index]
        )


        metadata.append(
            {
                "video": video_name,
                "selected_index": idx,
                "original_frame": original_frame,
                "total_frames": total_frames
            }
        )


    print(
        "Saved:",
        len(selected_indices)
    )


# ==========================
# Save metadata
# ==========================

pd.DataFrame(
    metadata
).to_csv(
    METADATA_PATH,
    index=False
)


print("\nFinished!")
print(
    "Metadata saved:",
    METADATA_PATH
)
