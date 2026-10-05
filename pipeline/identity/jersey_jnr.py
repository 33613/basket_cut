"""Multi-frame number evidence and abstention; not a player identity classifier."""

from collections import Counter, defaultdict
import math
from pathlib import Path

from contracts.schema import read_jsonl, write_json, write_jsonl_line
from pipeline.identity.jersey import group_numbers, number_consensus


def validate_thresholds(min_score, max_uncertainty, min_margin, min_support, min_gap_s):
    for value in (min_score, max_uncertainty, min_margin):
        if not math.isfinite(value) or not 0 <= value <= 1:
            raise ValueError("JNR score, uncertainty and margin thresholds must be finite in [0,1]")
    if min_support < 2 or not math.isfinite(min_gap_s) or min_gap_s <= 0:
        raise ValueError("Require at least two temporal supports and a positive time gap")


def jnr_consensus(readings, *, min_score=.8, max_uncertainty=.2, min_margin=.2,
                  min_support=2, min_gap_s=.25):
    validate_thresholds(min_score, max_uncertainty, min_margin, min_support, min_gap_s)
    qualified = []
    for row in readings:
        if any(not math.isfinite(float(row[key])) for key in ("raw_score", "uncertainty", "margin", "timestamp_s")):
            raise ValueError("Nonfinite JNR reading")
        if (row.get("rejection_reasons") or float(row["raw_score"]) < min_score
                or float(row["uncertainty"]) > max_uncertainty or float(row["margin"]) < min_margin):
            continue
        qualified.append(row)
    # All strong conflicting digits are retained, even if temporally close.
    result = number_consensus(qualified, min_score, 1)
    for candidate in result["candidates"]:
        independent, last = [], -math.inf
        for row in sorted(candidate["evidence"], key=lambda r: (r["timestamp_s"], r["frame_idx"])):
            if float(row["timestamp_s"]) - last >= min_gap_s:
                independent.append(row)
                last = float(row["timestamp_s"])
        candidate["support_frames"] = len(independent)
        candidate["support_frame_indices"] = [r["frame_idx"] for r in independent]
    if result["status"] != "conflict":
        candidate = result["candidates"][0] if result["candidates"] else None
        accepted = candidate is not None and candidate["support_frames"] >= min_support
        result["number"] = candidate["number"] if accepted else None
        result["status"] = "candidate_consensus" if accepted else "insufficient_evidence" if candidate else "unreadable"
    result.update({"readings": readings, "backend": "uncertainty_jnr", "team": None})
    return result


def prediction_reading(sample, probabilities, uncertainty, *, min_score, max_uncertainty, min_margin):
    import numpy as np
    probs = np.asarray(probabilities, dtype=np.float64)
    if probs.shape != (100,) or not np.isfinite(probs).all() or (probs < 0).any() or (probs > 1).any():
        raise ValueError("JNR must output 100 finite class scores in [0,1]")
    uncertainty = float(uncertainty)
    if not math.isfinite(uncertainty) or not 0 <= uncertainty <= 1.00001:
        raise ValueError("Invalid JNR uncertainty")
    order = np.argsort(-probs, kind="stable")[:3]
    number, score = int(order[0]), float(probs[order[0]])
    margin = score - float(probs[order[1]])
    reasons = []
    if score < min_score:
        reasons.append("low_number_score")
    if uncertainty > max_uncertainty:
        reasons.append("high_uncertainty")
    if margin < min_margin:
        reasons.append("small_top2_margin")
    if number == 0:
        reasons.append("ambiguous_0_or_00")
    return {"raw_track_id": int(sample["track_id"]), "frame_idx": int(sample["frame_idx"]),
            "timestamp_s": float(sample["timestamp_s"]), "crop_path": sample["crop_path"],
            "text": str(number), "raw_score": score, "uncertainty": uncertainty, "margin": margin,
            "top3": [{"number": str(i), "raw_score": float(probs[i])} for i in order],
            "rejection_reasons": reasons}


def recognize_jerseys_jnr(archive_dir: Path, identity_map: Path, output_dir: Path,
                         root: Path, config: Path, checkpoint: Path, *, batch_size=16,
                         device="cuda", trust_checkpoint=False, min_score=.8,
                         max_uncertainty=.2, min_margin=.2, min_support=2, min_gap_s=.25,
                         overwrite=False, backend=None):
    import cv2
    import numpy as np
    validate_thresholds(min_score, max_uncertainty, min_margin, min_support, min_gap_s)
    if batch_size < 1:
        raise ValueError("batch_size must be positive")
    archive, output = archive_dir.resolve(), output_dir.resolve()
    targets = [output / f"jersey_{name}" for name in
               ("tracks.jsonl", "people.jsonl", "readings.jsonl", "predictions.npz", "summary.json")]
    if identity_map.resolve() in targets or archive / "kpr_samples.jsonl" in targets:
        raise ValueError("JNR outputs cannot overwrite inputs")
    if any(path.is_symlink() for path in targets):
        raise ValueError("JNR evidence outputs cannot be symlinks")
    if not overwrite and any(path.exists() for path in targets):
        raise FileExistsError("JNR evidence exists; use --overwrite for these derived files only")
    mappings = list(read_jsonl(identity_map))
    ids = {int(row["raw_track_id"]) for row in mappings}
    if len(ids) != len(mappings):
        raise ValueError("Duplicate identity mappings")
    samples_path = archive / "kpr_samples.jsonl"
    if not samples_path.is_file():
        samples_path = archive / "kpr_sampling_manifest.jsonl"
    samples, seen = [], set()
    for sample in read_jsonl(samples_path) if ids else []:
        if int(sample["track_id"]) not in ids:
            continue
        key = (int(sample["track_id"]), int(sample["frame_idx"]))
        if key in seen:
            raise ValueError("Duplicate JNR sample frame")
        seen.add(key)
        if not math.isfinite(float(sample["timestamp_s"])) or float(sample["timestamp_s"]) < 0:
            raise ValueError("Invalid JNR sample timestamp")
        path = (archive / sample["crop_path"]).resolve()
        if not path.is_relative_to(archive) or not path.is_file():
            raise ValueError("JNR crop is missing or escapes the archive")
        samples.append(sample)
    samples.sort(key=lambda s: (int(s["track_id"]), int(s["frame_idx"])))
    if samples and backend is None:
        from pipeline.identity.jnr_backend import JNRBackend
        backend = JNRBackend(root, config, checkpoint, device=device, trust_checkpoint=trust_checkpoint)
    readings, all_probs, all_uncertainty = [], [], []
    for start in range(0, len(samples), batch_size):
        batch = samples[start:start + batch_size]
        images = [cv2.imread(str(archive / s["crop_path"])) for s in batch]
        if any(image is None for image in images):
            raise ValueError("Cannot decode a JNR sample crop")
        probs, uncertainty = backend.predict(images)
        if np.asarray(probs).shape != (len(batch), 100) or np.asarray(uncertainty).shape != (len(batch),):
            raise ValueError("JNR batch output shape does not match input")
        for sample, scores, value in zip(batch, probs, uncertainty):
            readings.append(prediction_reading(sample, scores, value, min_score=min_score,
                            max_uncertainty=max_uncertainty, min_margin=min_margin))
        all_probs.extend(probs)
        all_uncertainty.extend(uncertainty)
    by_track = defaultdict(list)
    for row in readings:
        by_track[row["raw_track_id"]].append(row)
    thresholds = dict(min_score=min_score, max_uncertainty=max_uncertainty,
                      min_margin=min_margin, min_support=min_support, min_gap_s=min_gap_s)
    tracks = {tid: {"raw_track_id": tid, **jnr_consensus(by_track[tid], **thresholds)} for tid in sorted(ids)}
    groups = defaultdict(list)
    for row in mappings:
        groups[row["person_id"]].append(int(row["raw_track_id"]))
    people = [{"person_id": pid, **group_numbers(tids, tracks)} for pid, tids in sorted(groups.items())]
    provenance = backend.provenance if backend is not None else {"backend": "uncertainty_jnr", "empty_input": True}
    summary = {"backend": "uncertainty_jnr", "provenance": provenance, "thresholds": thresholds,
               "track_count": len(tracks), "person_count": len(people), "sample_count": len(samples),
               "candidate_number_tracks": sum(t["number"] is not None for t in tracks.values()),
               "conflicting_tracks": sum(t["status"] == "conflict" for t in tracks.values()),
               "rejections": dict(Counter(reason for r in readings for reason in r["rejection_reasons"])),
               "sampling": "cached full-person KPR temporal samples; not jersey-legibility selection",
               "warning": "Soccer-trained model on basketball; scores/uncertainty are uncalibrated. "
               "0 and 00 are ambiguous and abstained. No team recognition. Different reliable numbers "
               "may veto a cross-clip merge, but matching numbers never force a merge or a player name."}
    output.mkdir(parents=True, exist_ok=True)
    # The summary is a completion marker, written only after all evidence files succeed.
    for path, values in zip(targets[:3], (tracks.values(), people, readings)):
        with path.open("w", encoding="utf-8") as handle:
            for row in values:
                write_jsonl_line(handle, row)
    np.savez_compressed(targets[3], probabilities=np.asarray(all_probs, dtype=np.float32).reshape(-1, 100),
                        uncertainties=np.asarray(all_uncertainty, dtype=np.float32),
                        track_ids=np.asarray([r["raw_track_id"] for r in readings], dtype=np.int64),
                        frame_indices=np.asarray([r["frame_idx"] for r in readings], dtype=np.int64))
    write_json(targets[4], summary)
    return summary
