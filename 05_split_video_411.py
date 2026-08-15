from pathlib import Path
import shutil
import pandas as pd


# Replace this with the path to your project directory
PROJECT_ROOT = Path(
    "/path/to/your/MousePose_Project"
)


SRC = PROJECT_ROOT / "data" / "kmeans_frames"

OUT = PROJECT_ROOT / "data" / "dataset_split"


for folder in ["train", "val", "test"]:
    (OUT / folder).mkdir(
        parents=True,
        exist_ok=True
    )


videos = sorted(
    [
        p for p in SRC.iterdir()
        if p.is_dir()
    ]
)


print("Total videos:", len(videos))


summary = []


for i, video in enumerate(videos):

    # 每6个video一个cycle
    position = i % 6


    if position < 4:
        split = "train"

    elif position == 4:
        split = "val"

    else:
        split = "test"


    destination = OUT / split / video.name


    shutil.copytree(
        video,
        destination,
        dirs_exist_ok=True
    )


    n_frames = len(
        list(video.glob("*.png"))
    )


    summary.append(
        {
            "video": video.name,
            "split": split,
            "frames": n_frames
        }
    )


    print(
        video.name,
        "->",
        split,
        f"({n_frames} frames)"
    )



# save
df = pd.DataFrame(summary)

df.to_csv(
    OUT / "split_summary.csv",
    index=False
)


print("\n===================")
print("Finished")
print("===================")

print(
    df["split"].value_counts()
)