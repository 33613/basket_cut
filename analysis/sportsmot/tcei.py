"""Paired TCEI runs on a frozen, previously evaluated basketball manifest."""

from __future__ import annotations

import csv
import json
import os
import random
import shutil
import time
from dataclasses import asdict
from pathlib import Path

from adapters.tcei import TCEI_COMMIT, source_revision
from analysis.sportsmot.dataset import verify_sources
from analysis.sportsmot.runner import CONFIG, ROOT, prediction_rows, source_code_hashes
from analysis.teamtrack.dataset import save_json, sha256
from analysis.teamtrack.metrics import evaluate_files, write_csv

VARIANTS = ("baseline", "off", "object", "id", "dual")


def discover_baseline(roots, checkpoint):
    """Find completed full runs; ambiguous source scopes require an explicit path."""
    candidates = {}
    expected = sha256(checkpoint)
    excluded = {
        "img1",
        "env",
        "envs",
        "node_modules",
        "model",
        "models",
        "MOTIP",
        "MMAction2",
        "third_party",
    }
    for root in roots:
        for directory, folders, files in os.walk(root):
            folders[:] = [
                n for n in folders if not n.startswith(".") and n not in excluded
            ]
            if not {
                "dataset.json",
                "experiment_protocol.json",
                "summary.csv",
                "status.json",
            }.issubset(files):
                continue
            path = Path(directory).resolve()
            try:
                manifest = json.loads((path / "dataset.json").read_text())
                protocol = json.loads((path / "experiment_protocol.json").read_text())
                status = json.loads((path / "status.json").read_text())
                if (
                    manifest.get("dataset") != "SportsMOT official basketball val"
                    or len(manifest["sequences"]) != manifest["downloaded_count"]
                    or status["status"] != "completed"
                    or protocol["checkpoint_sha256"] != expected
                    or protocol["dataset_sha256"] != sha256(path / "dataset.json")
                    or status["summary_sha256"] != sha256(path / "summary.csv")
                ):
                    continue
                candidates[path] = manifest["sequences"]
                folders[:] = []
            except (KeyError, ValueError, OSError):
                continue
    if not candidates:
        raise ValueError(
            "No completed full basketball baseline found; provide --baseline-run"
        )
    if any(scope != next(iter(candidates.values())) for scope in candidates.values()):
        raise ValueError(
            "Different historical scopes; provide --baseline-run: "
            + ", ".join(map(str, candidates))
        )
    selected = max(candidates, key=lambda p: (p / "summary.csv").stat().st_mtime)
    print(f"[baseline] {selected} ({len(candidates[selected])} sequences)", flush=True)
    return selected


def code_hashes():
    result = source_code_hashes()
    for name in (
        "adapters/tcei.py",
        "pipeline/tracking/tcei.py",
        "cli/benchmark_tcei.py",
        "analysis/teamtrack/metrics.py",
        "analysis/teamtrack/dataset.py",
    ):
        result[name] = sha256(ROOT / name)
    for path in (ROOT / "MOTIP/TrackEval/trackeval").rglob("*.py"):
        result[str(path.relative_to(ROOT))] = sha256(path)
    return result


def prepare(output, baseline, source, checkpoint, limit, variants, cache_scope):
    if output == baseline or output.is_relative_to(baseline):
        raise ValueError("Use an output directory outside the historical baseline")
    source_revision(source)
    original = json.loads((baseline / "dataset.json").read_text())
    old_protocol = json.loads((baseline / "experiment_protocol.json").read_text())
    if old_protocol["dataset_sha256"] != sha256(baseline / "dataset.json"):
        raise ValueError("Historical manifest checksum changed")
    checkpoint_hash = sha256(checkpoint)
    if (
        old_protocol["checkpoint_sha256"] != checkpoint_hash
        or old_protocol["config"] != CONFIG
    ):
        raise ValueError("Use the historical SportsMOT checkpoint/config")
    if original["dataset"] != "SportsMOT official basketball val":
        raise ValueError("Expected the validated official basketball val manifest")
    sequences = original["sequences"][:limit] if limit else original["sequences"]
    if not limit and len(sequences) != original["downloaded_count"]:
        raise ValueError(
            "--limit 0 requires the full historical manifest, not an earlier limited parity run"
        )
    if not sequences or any(
        s["view"] != "basketball" or s["split"] != "val" for s in sequences
    ):
        raise ValueError("Empty or non-basketball/val manifest")
    # Fix the sequence order from the historical manifest; never select by metrics.
    manifest = dict(original, sequences=sequences, limit=limit)
    protocol = {
        "implementation": "motip-hooks-corrected-v1",
        "baseline_run": str(baseline),
        "historical_dataset_sha256": sha256(baseline / "dataset.json"),
        "historical_protocol_sha256": sha256(baseline / "experiment_protocol.json"),
        "source_root": str(source),
        "source_commit": TCEI_COMMIT,
        "reference_sha256": sha256(source / "TCEI/tcei.py"),
        "checkpoint": str(checkpoint),
        "checkpoint_sha256": checkpoint_hash,
        "config": CONFIG,
        "cache_scope": cache_scope,
        "variants": list(variants),
        "seed": 0,
        "fp32": True,
        "preprocessing": "upstream-pil",
        "allow_tf32": False,
        "code_sha256": code_hashes(),
        "dataset": manifest,
    }
    path = output / "protocol.json"
    if path.exists() and json.loads(path.read_text()) != protocol:
        raise ValueError(
            "Experiment inputs/code/settings changed; use a new output directory"
        )
    save_json(path, protocol)
    save_json(output / "dataset.json", manifest)
    for seq in sequences:
        verify_sources(seq)
        for variant in variants:
            dest = output / variant / "val/gt" / seq["sequence"]
            for relative in ("gt/gt.txt", "seqinfo.ini"):
                target = dest / relative
                target.parent.mkdir(parents=True, exist_ok=True)
                source_file = Path(seq["source"]) / relative
                if target.exists() and sha256(target) != sha256(source_file):
                    raise ValueError(f"Modified prepared GT: {target}")
                if not target.exists():
                    shutil.copyfile(source_file, target)
    return protocol


def read_protocol(output):
    protocol = json.loads((output / "protocol.json").read_text())
    if protocol["code_sha256"] != code_hashes():
        raise ValueError("Code changed since preparation; use a new output directory")
    if protocol["dataset"] != json.loads((output / "dataset.json").read_text()):
        raise ValueError("Dataset manifest changed")
    if sha256(Path(protocol["checkpoint"])) != protocol["checkpoint_sha256"]:
        raise ValueError("Checkpoint changed")
    source = Path(protocol["source_root"])
    source_revision(source)
    if sha256(source / "TCEI/tcei.py") != protocol["reference_sha256"]:
        raise ValueError("Author calibration helpers changed")
    return protocol


def infer(output, variant):
    import cv2
    import numpy as np
    import torch
    import torchvision

    from adapters.tcei import load_reference
    from pipeline.tracking.motip_backend import MotipBackend, MotipRuntimeConfig
    from pipeline.tracking.tcei import TceiConfig

    protocol = read_protocol(output)
    if variant not in protocol["variants"]:
        raise ValueError(variant)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    torch.backends.cudnn.benchmark = False
    historical_protocol = Path(protocol["baseline_run"]) / "experiment_protocol.json"
    if sha256(historical_protocol) != protocol["historical_protocol_sha256"]:
        raise ValueError("Historical inference protocol changed")
    old_runtime = json.loads(historical_protocol.read_text())
    if (
        old_runtime["runtime"] != MotipRuntimeConfig(fp16=False).to_dict()
        or old_runtime["allow_tf32_matmul"]
        or old_runtime["allow_tf32_cudnn"]
    ):
        raise ValueError(
            "Historical runtime differs from the FP32/object-max/TF32-off control"
        )
    config = (
        TceiConfig(mode=variant, cache_scope=protocol["cache_scope"])
        if variant != "baseline"
        else None
    )
    folder = output / variant / "val"
    worker_protocol = {
        "experiment_sha256": sha256(output / "protocol.json"),
        "variant": variant,
        "calibration": asdict(config) if config else None,
        "runtime": MotipRuntimeConfig(fp16=False).to_dict(),
        "torch": torch.__version__,
        "torchvision": torchvision.__version__,
        "numpy": np.__version__,
        "opencv": cv2.__version__,
        "cuda": torch.version.cuda,
        "gpu": torch.cuda.get_device_name(0),
    }
    worker_path = folder / "inference_protocol.json"
    if worker_path.exists() and json.loads(worker_path.read_text()) != worker_protocol:
        raise ValueError("GPU environment changed; use a new output directory")
    save_json(worker_path, worker_protocol)
    reference = load_reference(Path(protocol["source_root"])) if config else None
    backend = None
    calibrated_rows = 0
    for seq in protocol["dataset"]["sequences"]:
        name = seq["sequence"]
        verify_sources(seq)
        pred = folder / "predictions/motip/data" / f"{name}.txt"
        done = folder / "completed" / f"{name}.json"
        # Stream caches cannot resume in the middle: replay the entire variant.
        if done.exists() and protocol["cache_scope"] == "sequence":
            completed = json.loads(done.read_text())
            if completed["protocol_sha256"] != sha256(worker_path) or completed[
                "prediction_sha256"
            ] != sha256(pred):
                raise ValueError(f"Modified completed output: {name}")
            calibrated_rows += completed["calibrated_rows"]
            print(f"[{variant}] reuse {name}", flush=True)
            continue
        if backend is None:
            backend = MotipBackend(
                motip_root=ROOT / "MOTIP",
                config_path=CONFIG,
                checkpoint_path=protocol["checkpoint"],
                runtime_config=MotipRuntimeConfig(fp16=False),
                device="cuda:0",
                preprocessing="upstream-pil",
                calibration=config,
                tcei_reference=reference,
            )
        random.seed(0)
        np.random.seed(0)
        torch.manual_seed(0)
        tracker = backend.create_tracker(height=seq["height"], width=seq["width"])
        torch.cuda.synchronize()
        torch.cuda.reset_peak_memory_stats()
        started = time.monotonic()
        pred.parent.mkdir(parents=True, exist_ok=True)
        temporary = pred.with_suffix(".txt.part")
        with temporary.open("w") as stream:
            for index, image in enumerate(seq["images"], 1):
                frame = cv2.imread(str(Path(seq["source"]) / "img1" / image["name"]))
                if frame is None or frame.shape[:2] != (seq["height"], seq["width"]):
                    raise ValueError(
                        f"Cannot decode original frame: {name}/{image['name']}"
                    )
                for record in backend.infer_frame(tracker, frame):
                    x, y, x2, y2 = record["bbox_xyxy"]
                    stream.write(
                        f"{index},{record['track_id']},{x},{y},{x2 - x},{y2 - y},{record['det_score']},-1,-1,-1\n"
                    )
                if index == 1 or index % 100 == 0:
                    print(f"[{variant}] {name}: {index}/{seq['frames']}", flush=True)
        torch.cuda.synchronize()
        seconds = time.monotonic() - started
        prediction_rows(temporary, seq["frames"])
        temporary.replace(pred)
        count = tracker.calibrator.calibrated_rows if config else 0
        calibrated_rows += count
        save_json(
            done,
            {
                "frames": seq["frames"],
                "seconds": seconds,
                "fps": seq["frames"] / seconds,
                "peak_memory_mib": torch.cuda.max_memory_allocated() / 1024**2,
                "calibrated_rows": count,
                "prediction_sha256": sha256(pred),
                "protocol_sha256": sha256(worker_path),
            },
        )
        del tracker
    if variant not in {"baseline", "off"} and not calibrated_rows:
        raise RuntimeError(
            "Calibration never ran; inspect the adapter before evaluating"
        )


def parity(output, sequences, split="val"):
    results = []
    for seq in sequences:
        paths = [
            output / name / split / "predictions/motip/data" / f"{seq['sequence']}.txt"
            for name in ("baseline", "off")
        ]
        rows = [sorted(prediction_rows(path, seq["frames"])) for path in paths]
        exact = rows[0] == rows[1]
        results.append({"sequence": seq["sequence"], "exact_ids_boxes_scores": exact})
    passed = all(row["exact_ids_boxes_scores"] for row in results)
    save_json(output / "parity.json", {"passed": passed, "sequences": results})
    if not passed:
        raise ValueError(
            "Disabled TCEI differs from baseline; stop before calibrated inference"
        )


def evaluate(output):
    save_json(output / "status.json", {"status": "evaluating"})
    try:
        return _evaluate(output)
    except Exception as error:
        save_json(
            output / "status.json",
            {"status": "failed", "error": f"{type(error).__name__}: {error}"},
        )
        raise


def _evaluate(output):
    protocol = read_protocol(output)
    sequences = protocol["dataset"]["sequences"]
    parity(output, sequences)
    rows, speed, environments = [], [], []
    for variant in protocol["variants"]:
        folder = output / variant / "val"
        worker_path = folder / "inference_protocol.json"
        worker = json.loads(worker_path.read_text())
        if worker["experiment_sha256"] != sha256(output / "protocol.json"):
            raise ValueError("Worker used a different experiment protocol")
        environments.append(
            {
                k: worker[k]
                for k in (
                    "torch",
                    "torchvision",
                    "numpy",
                    "opencv",
                    "cuda",
                    "gpu",
                    "runtime",
                )
            }
        )
        for seq in sequences:
            verify_sources(seq)
            completed = json.loads(
                (folder / "completed" / f"{seq['sequence']}.json").read_text()
            )
            pred = folder / "predictions/motip/data" / f"{seq['sequence']}.txt"
            if (
                completed["frames"] != seq["frames"]
                or completed["protocol_sha256"] != sha256(worker_path)
                or completed["prediction_sha256"] != sha256(pred)
                or sha256(folder / "gt" / seq["sequence"] / "gt/gt.txt")
                != seq["gt_sha256"]
                or sha256(folder / "gt" / seq["sequence"] / "seqinfo.ini")
                != seq["seqinfo_sha256"]
            ):
                raise ValueError(
                    f"Incomplete/modified results: {variant}/{seq['sequence']}"
                )
            speed.append(dict(variant=variant, sequence=seq["sequence"], **completed))
        rows.extend(
            dict(variant=variant, **row)
            for row in evaluate_files(folder, sequences, do_preproc=True)
        )
    if any(env != environments[0] for env in environments[1:]):
        raise ValueError("Variants used different GPU environments/runtime settings")
    write_csv(output / "summary.csv", rows)
    write_csv(output / "performance.csv", speed)
    overall = {row["variant"]: row for row in rows if row["scope"] == "overall"}
    metrics = ("HOTA", "AssA", "IDF1", "IDSW", "Frag", "DetA", "MOTA", "FP", "FN")
    write_csv(
        output / "deltas.csv",
        [
            dict(variant=key, **{m: row[m] - overall["baseline"][m] for m in metrics})
            for key, row in overall.items()
        ],
    )
    # Historical metrics are context only; current baseline is the paired control.
    historical = Path(protocol["baseline_run"]) / "summary.csv"
    if historical.exists():
        with historical.open() as stream:
            old = [r for r in csv.DictReader(stream) if r["scope"] == "overall"]
        save_json(
            output / "historical_reference.json",
            {
                "rows": old,
                "compared_scope_matches": (
                    sum(s["frames"] for s in sequences)
                    == sum(
                        s["frames"]
                        for s in json.loads(
                            (
                                Path(protocol["baseline_run"]) / "dataset.json"
                            ).read_text()
                        )["sequences"]
                    )
                ),
                "note": "Use deltas.csv for paired comparison; historical numbers may cover more sequences.",
            },
        )
    save_json(
        output / "status.json",
        {
            "status": "completed",
            "variants": protocol["variants"],
            "sequences": len(sequences),
            "frames": sum(s["frames"] for s in sequences),
            "summary_sha256": sha256(output / "summary.csv"),
        },
    )
    for row in overall.values():
        print(json.dumps(row, ensure_ascii=False), flush=True)
