# MousePose_Project_public

### Evaluation predictions

`09_run_evaluation_predictions.py` can be used with either the validation
set or the test set by updating the evaluation paths in the script.

For validation, predictions may be generated from multiple saved snapshots
to support model and snapshot selection. The test set should only be used
after all model-development decisions have been finalised. When evaluating
on the test set, predictions should therefore be generated only from the
snapshot selected using the validation set. Test-set results must not be
used to select a model, snapshot, or confidence threshold.
