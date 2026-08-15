from pathlib import Path
import deeplabcut


# ============================================================
# Path
# ============================================================

CONFIG_PATH = Path(
    "/path/to/your/dlc/project/config.yaml"
)


# ============================================================
# Main
# ============================================================

def main():

    print("=" * 70)
    print("Create DeepLabCut training dataset")
    print("=" * 70)

    print("\nConfig:")
    print(CONFIG_PATH)


    deeplabcut.create_training_dataset(
        str(CONFIG_PATH),
        Shuffles=[1],
        net_type="resnet_50",
        augmenter_type="albumentations"
    )


    print("\nFinished.")



if __name__ == "__main__":
    main()
