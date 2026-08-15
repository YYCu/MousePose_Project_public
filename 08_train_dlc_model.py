from pathlib import Path
import sys

import deeplabcut
import yaml


# ============================================================
# Paths
# ============================================================


PROJECT_ROOT = Path(
    "/path/to/your/MousePose_Project"
)


CONFIG_PATH = Path(
    "/path/to/your/dlc/project/config.yaml"
)

DLC_PROJECT_DIR = CONFIG_PATH.parent



# ============================================================
# Training settings
# ============================================================

SHUFFLE = 1
TRAINING_SET_INDEX = 0

EPOCHS = 200
SAVE_EPOCHS = 5


BATCH_SIZE = 2

DEVICE = "mps"


EVALUATE_AFTER_TRAINING = True



# ============================================================
# Validation
# ============================================================

def validate_paths():

    if not DLC_PROJECT_DIR.exists():
        raise FileNotFoundError(
            DLC_PROJECT_DIR
        )


    if not CONFIG_PATH.exists():
        raise FileNotFoundError(
            CONFIG_PATH
        )


    training_dataset_dir = (
        DLC_PROJECT_DIR
        / "training-datasets"
        / "iteration-0"
    )


    if not training_dataset_dir.exists():

        raise FileNotFoundError(
            "Training dataset is missing.\n"
            "Run 07_create_dlc_training_dataset.py first."
        )



def find_pytorch_config():

    config_files = sorted(
        DLC_PROJECT_DIR.glob(
            "dlc-models-pytorch/"
            "iteration-*/"
            "**/"
            "train/"
            "pytorch_config.yaml"
        )
    )


    if not config_files:

        raise FileNotFoundError(
            "No pytorch_config.yaml found."
        )


    shuffle_text = f"shuffle{SHUFFLE}"


    matching = [
        p for p in config_files
        if shuffle_text.lower()
        in str(p).lower()
    ]


    if len(matching) == 1:
        return matching[0]


    if len(config_files) == 1:
        return config_files[0]


    print("Multiple pytorch configs found:")

    for p in config_files:
        print(p)


    raise RuntimeError(
        "Cannot identify pytorch config."
    )



def verify_training_settings(config_path):

    with config_path.open(
        "r",
        encoding="utf-8"
    ) as f:

        config = yaml.safe_load(f)



    snapshots = (
        config
        .get("runner", {})
        .get("snapshots", {})
    )


    actual_save_epochs = (
        snapshots.get("save_epochs")
    )


    actual_max_snapshots = (
        snapshots.get("max_snapshots")
    )


    print("\nPyTorch snapshot settings:")
    print(f"Config file: {config_path}")
    print(f"max_snapshots: {actual_max_snapshots}")
    print(f"save_epochs:   {actual_save_epochs}")


    if actual_save_epochs != SAVE_EPOCHS:

        raise RuntimeError(
            "save_epochs mismatch.\n"
            f"Expected {SAVE_EPOCHS}\n"
            f"Got {actual_save_epochs}"
        )




# ============================================================
# Main
# ============================================================

def main():

    validate_paths()


    pytorch_config = find_pytorch_config()


    verify_training_settings(
        pytorch_config
    )



    print("\n" + "=" * 70)
    print("DeepLabCut Model Training")
    print("=" * 70)

    print(f"Config:             {CONFIG_PATH}")
    print(f"Shuffle:            {SHUFFLE}")
    print(f"Training set index: {TRAINING_SET_INDEX}")
    print(f"Epochs:             {EPOCHS}")
    print(f"Save every:         {SAVE_EPOCHS} epochs")
    print(f"Batch size:         {BATCH_SIZE}")
    print(f"Device:             {DEVICE}")

    print("=" * 70)



    print("\n[1/2] Starting network training...")


    deeplabcut.train_network(
        str(CONFIG_PATH),
        shuffle=SHUFFLE,
        trainingsetindex=TRAINING_SET_INDEX,
        epochs=EPOCHS,
        save_epochs=SAVE_EPOCHS,
        batch_size=BATCH_SIZE,
        device=DEVICE,
    )


    print("\nTraining completed.")



    if EVALUATE_AFTER_TRAINING:

        print(
            "\n[2/2] Evaluating trained network..."
        )


        deeplabcut.evaluate_network(
            str(CONFIG_PATH),
            Shuffles=[SHUFFLE],
            trainingsetindex=TRAINING_SET_INDEX,
            plotting=True,
        )


        print("\nEvaluation completed.")



    print("\n" + "=" * 70)
    print("DeepLabCut training completed.")
    print("=" * 70)



if __name__ == "__main__":

    try:

        main()


    except KeyboardInterrupt:

        print(
            "\nTraining stopped."
        )

        sys.exit(130)


    except Exception as exc:

        print(
            "\nTraining failed."
        )

        print(
            f"{type(exc).__name__}: {exc}"
        )

        raise
