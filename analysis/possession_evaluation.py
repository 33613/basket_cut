"""One-to-one temporal boundary evaluation with explicit annotation coverage."""

import csv
import json
import math
from pathlib import Path

from pipeline.media import atomic_json, sha256_file
from pipeline.segmentation.possession import BOUNDARY_CONVENTION


def annotation_template(source):
    duration = source["duration_s"]
    return {"schema_version": 1, "source_sha256": source["sha256"],
        "boundary_convention": BOUNDARY_CONVENTION, "human_verified": False,
        "instructions": "Annotate ALL changes and explicitly reviewed negative periods. Set complete=true only after reviewing the entire range. Do not copy model candidates as GT.",
        "ranges": [{"name": name, "start_s": start, "end_s": end, "complete": False,
                    "boundaries_s": [], "ignore_ranges_s": []}
            for name, start, end in (("development", 0, min(300, duration)), ("validation", 300, duration))
            if end > start]}


def finite(value):
    if type(value) not in (int, float) or not math.isfinite(value):
        raise ValueError("Expected a finite numeric timestamp")
    return float(value)


def validate_annotations(data, source):
    if data.get("source_sha256") != source["sha256"] or data.get("boundary_convention") != BOUNDARY_CONVENTION:
        raise ValueError("Annotation source or boundary convention mismatch")
    ranges = data.get("ranges")
    if not isinstance(ranges, list) or not ranges:
        raise ValueError("Annotation ranges are required")
    end, names = 0, set()
    for row in sorted(ranges, key=lambda r: finite(r["start_s"])):
        low, high = finite(row["start_s"]), finite(row["end_s"])
        if not isinstance(row.get("name"), str) or not row["name"] or row["name"] in names:
            raise ValueError("Annotation ranges need unique names")
        names.add(row["name"])
        if not end <= low < high <= source["duration_s"] or type(row.get("complete")) is not bool:
            raise ValueError("Invalid/overlapping annotation coverage")
        points = row.get("boundaries_s")
        if not isinstance(points, list) or len(set(points)) != len(points) or any(
                not low <= finite(t) < high or t == 0 for t in points):
            raise ValueError("Boundary references must be unique and inside their coverage")
        last = low
        for a, b in sorted(row.get("ignore_ranges_s", [])):
            a, b = finite(a), finite(b)
            if not last <= a < b <= high or any(a <= t < b for t in points):
                raise ValueError("Invalid ignore interval or boundary inside ignored interval")
            last = b
        end = high


def match_boundaries(predictions, references, tolerance_s=2, max_width_s=10):
    """Maximum-cardinality matching; a wide candidate cannot cover many GT changes."""
    if finite(tolerance_s) < 0 or finite(max_width_s) <= 0:
        raise ValueError("Invalid temporal tolerance or maximum candidate width")
    references = list(references)
    edges = []
    for low, high in predictions:
        low, high = finite(low), finite(high)
        if high < low:
            raise ValueError("Invalid prediction interval")
        eligible = []
        if high-low <= max_width_s:
            for j, point in enumerate(references):
                distance = max(low-point, point-high, 0)
                if distance <= tolerance_s:
                    eligible.append(j)
        edges.append(sorted(eligible, key=lambda j: abs(sum((low, high))/2-references[j])))
    owners = {}
    def assign(i, seen):
        for j in edges[i]:
            if j in seen:
                continue
            seen.add(j)
            if j not in owners or assign(owners[j], seen):
                owners[j] = i
                return True
        return False
    for i in range(len(predictions)):
        assign(i, set())
    matches = [{"prediction_index": i, "reference_index": j,
        "reference_s": references[j], "prediction_range_s": list(predictions[i]),
        "midpoint_error_s": (sum(predictions[i])/2)-references[j]}
        for j, i in sorted(owners.items())]
    tp = len(matches)
    return {"TP": tp, "FP": len(predictions)-tp, "FN": len(references)-tp,
        "precision": tp/len(predictions) if predictions else None,
        "recall": tp/len(references) if references else None,
        "F1": 2*tp/(len(predictions)+len(references)) if predictions or references else None,
        "matched_midpoint_mae_s": sum(abs(m["midpoint_error_s"]) for m in matches)/tp if tp else None,
        "mean_prediction_width_s": sum(b-a for a,b in predictions)/len(predictions) if predictions else None,
        "unmatched_prediction_indices": [i for i in range(len(predictions)) if i not in owners.values()],
        "unmatched_reference_indices": [j for j in range(len(references)) if j not in owners], "matches": matches}


def evaluate(plan, annotations, *, tolerance_s=2, max_width_s=10):
    if finite(tolerance_s) < 0 or finite(max_width_s) <= 0:
        raise ValueError("Invalid temporal tolerance or maximum candidate width")
    validate_annotations(annotations, plan["source"])
    if plan.get("boundary_convention") != BOUNDARY_CONVENTION:
        raise ValueError("Re-run segmentation with the agreed inbound-preparation boundary convention")
    report = {"status": "awaiting_complete_human_annotations", "accuracy_evaluated": False,
        "source_sha256": plan["source"]["sha256"], "protocol_sha256": plan["protocol_sha256"],
        "boundary_convention": BOUNDARY_CONVENTION, "tolerance_s": tolerance_s,
        "max_candidate_width_s": max_width_s, "ranges": []}
    if annotations.get("human_verified") is not True:
        return report
    candidates = [r["time_range_s"] for r in plan.get("coarse_candidates", []) if r["kind"] in {"possession_change", "uncertain"}]
    cuts = [[r["start_s"], r["start_s"]] for i,r in enumerate(plan["segments"][1:], start=1)
            if r.get("start_boundary_kind") in {"possession_change", "manual"} or
            (r.get("start_boundary_kind") == "camera_cut" and r.get("possession_id")
             and plan["segments"][i-1].get("possession_id")
             and r["possession_id"] != plan["segments"][i-1]["possession_id"])]
    for row in annotations["ranges"]:
        if not row["complete"]:
            continue
        low, high = row["start_s"], row["end_s"]
        ignores = row.get("ignore_ranges_s", [])
        def in_scope(p):
            # Ignore predictions touching excluded/border regions rather than count partial events as FP.
            return low <= p[0] and p[1] < high and not any(p[0] < b and p[1] >= a for a,b in ignores)
        observed = high-low-sum(b-a for a,b in ignores)
        for stage, predictions in (("coarse", candidates), ("cuts", cuts)):
            values = [p for p in predictions if in_scope(p)]
            metrics = match_boundaries(values, row["boundaries_s"], tolerance_s, max_width_s)
            report["ranges"].append({"name": row["name"], "stage": stage, "start_s": low, "end_s": high,
                "observed_duration_s": observed, "FP_per_minute": metrics["FP"]*60/observed if observed else None,
            "predictions": values, "excluded_prediction_count": len(predictions)-len(values),
            "reference_count": len(row["boundaries_s"]), **metrics})
    if report["ranges"]:
        report.update(status="evaluated_on_declared_coverage", accuracy_evaluated=True)
    return report


def write_evaluation(segmentation_dir, annotation_file, *, tolerance_s=2, max_width_s=10):
    root = Path(segmentation_dir)
    plan = json.loads((root / "segmentation.json").read_text())
    annotations = json.loads(Path(annotation_file).read_text())
    report = evaluate(plan, annotations, tolerance_s=tolerance_s, max_width_s=max_width_s)
    report["annotations_sha256"] = sha256_file(annotation_file)
    report["evaluator_sha256"] = sha256_file(Path(__file__))
    atomic_json(root / "evaluation.json", report)
    fields = ["name", "stage", "start_s", "end_s", "observed_duration_s", "reference_count", "TP", "FP", "FN",
              "precision", "recall", "F1", "FP_per_minute", "matched_midpoint_mae_s", "mean_prediction_width_s"]
    with (root / "evaluation.csv").open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fields, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(report["ranges"])
    return report
