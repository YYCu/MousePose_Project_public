# Multi-Mouse Pose Estimation Workflow

This repository provides a general template for preparing image datasets,
creating DeepLabCut projects, training multi-animal pose-estimation models, and
evaluating predictions on independent validation and test sets.

The workflow was developed for side-view videos containing three mice, but the
scripts can be adapted to other datasets. Paths, project names, body parts,
individual names, thresholds, image counts, and hardware settings must be
reviewed before each script is run. The repository does not include the source
videos, extracted images, DeepLabCut projects, annotations, trained models, or
evaluation results.

## Workflow overview

```text
Video data
    |
    v
01 Select representative frames
    |
    v
02 Create video-level train/validation/test split
    |
    v
03 Create annotation metadata
    |
    v
04 Review and annotate frame metadata
    |
    v
05 Select usable image subsets
    |
    v
06 Create DeepLabCut projects
    |
    +--> Configure and label training images
    |        |
    |        v
    |    07 Create DLC training dataset
    |        |
    |        v
    |    08 Train DLC model
    |
    +--> Configure and label validation/test images as Ground Truth
             |
             v
         09 Run evaluation predictions
             |
             v
         10 Prepare evaluation data
             |
             v
         11 Match GT and predicted instances
             |
             v
         12 Calculate raw localisation error
             |
             v
         13 Classify large localisation errors
             |
             v
         14 Calculate refined localisation error
```

Scripts 15--18 provide optional validation diagnostics and post-processing.

## Important usage rules

- This repository is a template. Replace every placeholder path before running
  a script.
- Split data by source video, not by individual frame, to reduce data leakage.
- Use the training set to fit models.
- Use the validation set for model development, snapshot selection, likelihood
  threshold selection, and other tuning decisions.
- Use the test set only after every model-development decision has been fixed.
- Do not select a model, snapshot, likelihood threshold, matching threshold, or
  ear-correction parameter using test results.
- Use the same body-part definitions and annotation rules in training,
  validation, and test projects.
- Keep the likelihood threshold, keypoint list, and matching threshold
  consistent across scripts that form one evaluation run.

## Suggested repository structure

```text
project-root/
|-- scripts/
|   |-- 01_select_representative_frames.py
|   |-- 02_create_video_level_split.py
|   |-- 03_create_annotation_metadata.py
|   |-- 04_annotate_metadata_tool.py
|   |-- 05_select_usable_image_subset.py
|   |-- 06_create_dlc_project.py
|   |-- 07_create_dlc_training_dataset.py
|   |-- 08_train_dlc_model.py
|   |-- 09_run_evaluation_predictions.py
|   |-- 10_prepare_evaluation_data.py
|   |-- 11_match_instances.py
|   |-- 12_calculate_localization_error.py
|   |-- 13_classify_localization_errors.py
|   |-- 14_calculate_refined_localization_error.py
|   |-- 15_analyze_likelihood_thresholds.py
|   |-- 16_analyze_keypoint_availability.py
|   |-- 17_validate_ear_geometry.py
|   `-- 18_correct_ear_swaps.py
|-- data/
|   |-- kmeans_frames/
|   |-- dataset_split/
|   |   |-- train/
|   |   |-- val/
|   |   `-- test/
|   `-- annotations/
|-- dlc_projects/
|-- external_evaluation_results/
`-- results/
```

The exact folder names may be changed. If a folder name is changed, update all
scripts that read from or write to that folder.

## Requirements

The scripts use packages including:

```text
deeplabcut
numpy
pandas
scipy
opencv-python
scikit-learn
Pillow
tqdm
PyYAML
torch
```

The metadata annotation interface also requires Tkinter. DeepLabCut and PyTorch
must be installed in an environment appropriate for the available hardware.
For example, `mps` may be used on a supported Apple Silicon Mac, `cuda` on a
supported NVIDIA GPU, and `cpu` when no supported accelerator is available.

## Path placeholders

Scripts contain placeholders such as:

```python
PROJECT_ROOT = Path(
    "/path/to/your/MousePose_Project"
)
```

Other common placeholders include:

```python
SPLIT = "train, val or test"
ROUND_NAME = "round1, round2 or round3"
SNAPSHOT_NAME = "your_snapshot_folder_name"
OUTPUT_DIR = Path("/path/to/save/results")
```

Replace these values with real paths or names before running the script. A
placeholder is not interpreted automatically.

## Data preparation

### 01. Select representative frames

```text
01_select_representative_frames.py
```

This script samples frames from each source video, extracts low-resolution
image features, applies k-means clustering, and saves one representative frame
from each cluster.

Review and adjust:

- source video directory;
- output image directory;
- video extension;
- maximum number of candidate frames;
- number of clusters.

Typical output:

```text
data/kmeans_frames/<video_name>/<image_name>.png
```

### 02. Create a video-level split

```text
02_create_video_level_split.py
```

This script assigns complete video folders to training, validation, and test
sets. The default logic uses a 4:1:1 cycle, but the split rule should be adapted
to the dataset and study design.

The input directory must match the output directory from Step 01, unless a
manual image-cleaning stage is performed between the two steps.

Typical output:

```text
data/dataset_split/train/<video_name>/
data/dataset_split/val/<video_name>/
data/dataset_split/test/<video_name>/
```

### 03. Create annotation metadata

```text
03_create_annotation_metadata.py
```

This script scans the three split directories and creates metadata CSV files:

```text
data/annotations/train_metadata.csv
data/annotations/val_metadata.csv
data/annotations/test_metadata.csv
```

Existing annotations are preserved when rows can be matched using the video and
image names.

### 04. Annotate frame metadata

```text
04_annotate_metadata_tool.py
```

This graphical tool records frame usability and descriptive metadata such as
mouse count, orientation, pose, keypoint visibility, and occlusion type.

Run the tool separately for train, validation, and test metadata. The tool
updates the selected metadata CSV in place.

This metadata is separate from DeepLabCut keypoint annotation.

### 05. Select a usable image subset

```text
05_select_usable_image_subset.py
```

This script retains rows marked `usable=yes` and selects a requested number of
images while distributing the selection across source videos.

Review and adjust:

```python
SPLIT = "train, val or test"
N_IMAGES = 100
```

Also replace the output placeholders:

```python
OUTPUT_DIR = (
    PROJECT_ROOT
    / "data"
    / "your_output_folder_name"
)

OUTPUT_CSV = (
    OUTPUT_DIR
    / "your_output_csv_filename.csv"
)
```

Different image counts may be used for different training rounds. Validation
and test subsets should remain fixed once model development begins.

## DeepLabCut project creation and annotation

### 06. Create a DeepLabCut project

```text
06_create_dlc_project.py
```

This script creates a multi-animal DeepLabCut project from a selected image
subset and the corresponding source videos. It copies selected images into the
matching `labeled-data/<video_name>/` folders.

Review and adjust:

- selected-image directory;
- DeepLabCut working directory;
- source-video directory;
- project name and scorer;
- video extension and filename pattern.

The script should determine the number of images from the input folder. It
should not require exactly 100 images.

After the project is created:

1. Open the generated `config.yaml`.
2. Define body parts, skeleton, individuals, and identity settings.
3. Confirm that the same definitions will be used for all projects.
4. Label the images using DeepLabCut or napari-DeepLabCut.
5. Confirm that the expected `CollectedData_*.csv` and
   `CollectedData_*.h5` files were created.

Training, validation, and test projects can all be created from this template.
Only training projects proceed to network training. Validation and test
projects provide independent Ground Truth for evaluation.

## Model training

### 07. Create a DeepLabCut training dataset

```text
07_create_dlc_training_dataset.py
```

This script runs `deeplabcut.create_training_dataset()` for an existing,
labelled training project.

Set:

```python
CONFIG_PATH = Path(
    "/path/to/your/training/dlc/project/config.yaml"
)
```

The network type and augmenter must match the intended experiment. For example:

```python
deeplabcut.create_training_dataset(
    str(CONFIG_PATH),
    Shuffles=[1],
    net_type="resnet_50",
    augmenter_type="albumentations",
)
```

This step is for training projects only. Do not create a training dataset from
the independent validation or test project.

### 08. Train the DeepLabCut model

```text
08_train_dlc_model.py
```

This script runs `deeplabcut.train_network()` using the training dataset
created in Step 07.

The Step 07 and Step 08 `CONFIG_PATH` values must refer to the same project.
Review the shuffle, training-set index, epochs, snapshot interval, batch size,
device, and optional internal DLC evaluation setting.

The evaluation produced by `deeplabcut.evaluate_network()` uses DeepLabCut's
internal split. It is not a substitute for the independent validation or test
evaluation described below.

For multiple training rounds, repeat Steps 05--08 with the required training
subset and project. Keep validation and test data independent from all training
rounds.

## Independent validation and test evaluation

### 09. Run evaluation predictions

```text
09_run_evaluation_predictions.py
```

This script applies a trained model to images from an independent validation or
test DLC project and saves raw predictions for later evaluation.

The training config determines which model is used. The evaluation project,
metadata, result directory, device, and snapshot settings must be updated for
the current run.

For validation, predictions may be generated from multiple saved snapshots to
support model and snapshot selection. The test set should only be used after
all model-development decisions have been finalised. When evaluating on the
test set, generate predictions only from the snapshot selected using the
validation set. Test results must not be used to select a model, snapshot,
likelihood threshold, matching threshold, or post-processing parameter.

### 10. Prepare evaluation data

```text
10_prepare_evaluation_data.py
```

This script converts Ground Truth and prediction H5 data into long-format CSV
files used by the numerical evaluation scripts.

The Step 09 output directory and Step 10 input result directory must be the
same. The script can read `CollectedData*.h5` directly from the evaluation
project's `labeled-data` directory; a separate Ground Truth backup is not
required.

Main outputs include:

```text
prepared_data/ground_truth_long.csv
snapshots/<snapshot>/predictions_prepared_long.csv
```

### 11. Match Ground Truth and predicted instances

```text
11_match_instances.py
```

This script calculates a cost matrix for each image and uses the Hungarian
algorithm to obtain one-to-one GT--prediction assignments. Assigned pairs are
accepted only when their mean shared-keypoint distance is within the matching
threshold.

Review and adjust:

```python
MATCH_THRESHOLD = 100
PRED_LIKELIHOOD_THRESHOLD = 0.7
```

The input snapshot and result directories must correspond to the evaluation run
prepared in Step 10.

Outputs:

```text
matched_instances.csv
unmatched_gt.csv
unmatched_predictions.csv
```

### 12. Calculate raw localisation error

```text
12_calculate_localization_error.py
```

This script calculates Euclidean keypoint errors within accepted instance
pairs. A keypoint is evaluated only when both GT and prediction coordinates are
available and the prediction satisfies the likelihood threshold.

The keypoint list and likelihood threshold must match Step 11.

Outputs:

```text
keypoint_errors.csv
localization_summary.csv
per_keypoint_error.csv
```

### 13. Classify large localisation errors

```text
13_classify_localization_errors.py
```

This script examines errors above a selected screening threshold. If a
prediction is substantially closer to the corresponding keypoint on another GT
mouse than to its assigned GT mouse, the case is classified as an
`instance_association_failure`. Other screened cases are classified as
`localisation_failure`.

Example parameters:

```python
ERROR_THRESHOLD = 30
ASSOCIATION_MARGIN = 30
```

These parameters must be determined without using the test set.

Output:

```text
error_classification.csv
```

### 14. Calculate refined localisation error

```text
14_calculate_refined_localization_error.py
```

This script excludes matched instance pairs identified as association failures
and recalculates localisation metrics. If no association failures are found,
the refined and raw localisation results will be identical.

## Optional validation diagnostics

### 15. Analyse likelihood thresholds

```text
15_analyze_likelihood_thresholds.py
```

This script reruns matching, detection, association classification, and
localisation calculations over several likelihood thresholds. It is intended
for validation-based threshold selection.

Do not use this analysis to choose a threshold from test results. After a
threshold is selected on validation data, use that fixed value in Steps 11--14
for the final test evaluation.

### 16. Analyse keypoint availability

```text
16_analyze_keypoint_availability.py
```

This diagnostic holds accepted matched pairs fixed and measures how many raw
and matched prediction keypoints are retained or removed at different
likelihood thresholds. It does not replace the complete threshold evaluation in
Step 15.

This script is optional and can be omitted without affecting the core metrics.

## Optional ear-swap correction

### 17. Validate ear geometry on Ground Truth

```text
17_validate_ear_geometry.py
```

This script tests whether the anatomical left ear occupies a stable side of the
ear-centre-to-nose head axis in Ground Truth. It evaluates dominant-sign
stability and coverage over several uncertainty margins.

Use validation Ground Truth to determine whether the geometry is sufficiently
stable and to choose the uncertainty margin. Do not tune this rule using test
Ground Truth.

### 18. Correct predicted ear swaps

```text
18_correct_ear_swaps.py
```

This script applies the validation-derived geometry rule to prediction data. It
swaps the complete predicted values for `left_ear` and `right_ear` when the
predicted orientation is decisively reversed.

It creates a new corrected prediction CSV and does not modify the original
prediction file. If corrections are applied, rerun Steps 11--14 using the
corrected prediction CSV.

Separate copies of the evaluation scripts are not required for ear-corrected
predictions. Update `PRED_PATH` in the same general evaluation scripts.

## Recommended execution by dataset

| Stage | Train | Validation | Test |
|---|---:|---:|---:|
| Representative frame selection | Yes | Derived by split | Derived by split |
| Metadata creation and review | Yes | Yes | Yes |
| Usable subset selection | Yes | Yes | Yes |
| DLC project and keypoint annotation | Yes | Yes | Yes |
| DLC training dataset creation | Yes | No | No |
| Network training | Yes | No | No |
| Prediction and numerical evaluation | No | Yes | Yes |
| Snapshot/threshold tuning | No | Yes | No |
| Final held-out reporting | No | Optional | Yes |

## Repeating the workflow across training rounds

For Round 1, Round 2, and Round 3:

1. Select the required usable training subset.
2. Create or update the corresponding training DLC project.
3. Label any newly added images using the same annotation definitions.
4. Create the DLC training dataset.
5. Train the model.
6. Generate validation predictions.
7. Compare rounds using the fixed validation set.

After the final training round, fix the model, snapshot, likelihood threshold,
matching threshold, and optional post-processing settings. Then evaluate once
on the independent test set.

## Notes on reproducibility

- Record the DeepLabCut, PyTorch, Python, and package versions used.
- Record the random seed used for frame and subset selection.
- Keep the split summary and selected-image CSV files.
- Keep the final training and evaluation config files.
- Record which snapshot and thresholds were used for every reported result.
- Do not upload private, restricted, or very large source datasets to GitHub.
- Do not upload trained model snapshots unless redistribution is permitted and
  their size is appropriate for the repository.

## Adapting the template

This workflow is not a one-command package. It is a reusable template and must
be adjusted to the dataset and research question. In particular, review:

- video and image naming conventions;
- number of animals;
- body-part names and skeleton;
- annotation rules;
- split proportions;
- training-round design;
- number of selected images;
- network architecture and augmentation;
- likelihood and matching thresholds;
- device and batch size;
- validation/test folder structure;
- which optional diagnostics are scientifically justified.

Every modification should be documented so that the reported experiment can be
reproduced.
