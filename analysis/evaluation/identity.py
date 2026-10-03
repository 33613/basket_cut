"""Pairwise clip-local identity F1 against independent tracklet labels."""

from itertools import combinations
from pathlib import Path

from contracts.schema import read_jsonl, write_json


def evaluate_identity(reference: Path, prediction: Path, output: Path) -> dict:
    if output.resolve() in {reference.resolve(), prediction.resolve()}:
        raise ValueError("Evaluation output cannot overwrite reference or prediction")
    truth = {}
    predicted = {}
    video_ids = set()
    for row in read_jsonl(reference):
        tid = int(row["raw_track_id"])
        if (
            tid < 0
            or tid in truth
            or not isinstance(row["identity_label"], str)
            or not row["identity_label"].strip()
            or not isinstance(row["video_id"], str)
            or not row["video_id"].strip()
        ):
            raise ValueError(
                "Reference requires unique nonnegative tracks and nonempty labels"
            )
        truth[tid] = str(row["identity_label"])
        video_ids.add(row["video_id"])
    if len(video_ids) != 1 or len(truth) < 2:
        raise ValueError(
            "Reference requires >=2 labeled tracklets from exactly one clip"
        )
    for row in read_jsonl(prediction):
        tid = int(row["raw_track_id"])
        if tid in predicted or not row.get("person_id"):
            raise ValueError("Invalid or duplicate predicted identity mapping")
        if row.get("video_id") and row["video_id"] not in video_ids:
            raise ValueError("Prediction and reference video_id differ")
        predicted[tid] = str(row["person_id"])
    tp = fp = fn = tn = 0
    for a, b in combinations(sorted(truth), 2):
        expected = truth[a] == truth[b]
        same = a in predicted and b in predicted and predicted[a] == predicted[b]
        tp += expected and same
        fp += not expected and same
        fn += expected and not same
        tn += not expected and not same
    result = {
        "video_id": next(iter(video_ids)),
        "headline": {
            "name": "identity_pair_f1",
            "value": 2 * tp / (2 * tp + fp + fn) if 2 * tp + fp + fn else None,
        },
        "precision": tp / (tp + fp) if tp + fp else None,
        "recall": tp / (tp + fn) if tp + fn else None,
        "tp": tp,
        "fp": fp,
        "fn": fn,
        "tn": tn,
        "reference_tracklets": len(truth),
        "missing_predicted_tracklets": sorted(set(truth) - set(predicted)),
        "unscored_prediction_tracklets": sorted(set(predicted) - set(truth)),
        "warning": "Pair-weighted identity grouping, not image-query retrieval or event F1. Reference labels must be independent of resolution review. Unlabeled tracks are unscored; this cannot establish full-video or cross-game generalization.",
    }
    write_json(output, result)
    return result
