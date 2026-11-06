"""Paired TCEI on Indoor, reusing TrackID's native-video converter and evaluator."""

from __future__ import annotations

import json
import os
import subprocess
from pathlib import Path

from adapters.tcei import TCEI_COMMIT, source_revision
from analysis.sportsmot.runner import ROOT, source_code_hashes
from analysis.sportsmot.tcei import parity
from analysis.teamtrack.dataset import save_json, sha256
from analysis.teamtrack.metrics import write_csv
from analysis.trackid.dataset import SPLIT, verify_sources
from analysis.trackid.dataset import prepare as prepare_dataset


def code_hashes():
    result = source_code_hashes()
    paths = [
        ROOT / "cli/benchmark_tcei.py",
        ROOT / "adapters/tcei.py",
        ROOT / "pipeline/tracking/tcei.py",
        ROOT / "analysis/teamtrack/inference.py",
        ROOT / "analysis/teamtrack/metrics.py",
        ROOT / "analysis/teamtrack/dataset.py",
    ]
    paths.extend((ROOT / "analysis/trackid").glob("*.py"))
    paths.extend((ROOT / "MOTIP/TrackEval/trackeval").rglob("*.py"))
    return dict(result, **{str(p.relative_to(ROOT)): sha256(p) for p in paths})


def prepare(output, raw, source, checkpoint, limit, variants, cache_scope):
    source_revision(source)
    if (
        limit < 0
        or len(set(variants)) != len(variants)
        or not {"baseline", "off"}.issubset(variants)
    ):
        raise ValueError(
            "Require unique variants including baseline/off and a nonnegative limit"
        )
    variants = [
        "baseline",
        "off",
        *[v for v in variants if v not in {"baseline", "off"}],
    ]
    # Every variant uses the same original converter. Never export/resample JPEGs.
    manifest = None
    for variant in variants:
        current = prepare_dataset(
            raw, output / variant, subset="Indoor", limit_per_view=limit
        )
        if manifest is not None and current != manifest:
            raise ValueError("Variants selected different source videos or GT")
        manifest = current
    protocol = {
        "dataset": "TrackID3x3 Indoor",
        "split": SPLIT,
        "source_root": str(source),
        "source_commit": TCEI_COMMIT,
        "reference_sha256": sha256(source / "TCEI/tcei.py"),
        "checkpoint": str(checkpoint),
        "checkpoint_sha256": sha256(checkpoint),
        "variants": variants,
        "cache_scope": cache_scope,
        "code_sha256": code_hashes(),
        "dataset_sha256": sha256(output / variants[0] / SPLIT / "dataset.json"),
    }
    path = output / "protocol.json"
    if path.exists() and json.loads(path.read_text()) != protocol:
        raise ValueError("Experiment changed; use a new output directory")
    save_json(path, protocol)
    save_json(output / "dataset.json", manifest)
    print(
        f"[Indoor] {len(manifest['sequences'])} sequences, "
        f"{sum(s['frames'] for s in manifest['sequences'])} native frames/variant",
        flush=True,
    )
    return protocol


def verified(output):
    protocol = json.loads((output / "protocol.json").read_text())
    if (
        protocol["code_sha256"] != code_hashes()
        or sha256(Path(protocol["checkpoint"])) != protocol["checkpoint_sha256"]
    ):
        raise ValueError("Code/checkpoint changed; use a new output directory")
    source = Path(protocol["source_root"])
    source_revision(source)
    if sha256(source / "TCEI/tcei.py") != protocol["reference_sha256"]:
        raise ValueError("Reference helpers changed")
    if sha256(output / "dataset.json") != protocol["dataset_sha256"]:
        raise ValueError("Experiment manifest changed")
    for variant in protocol["variants"]:
        path = output / variant / SPLIT / "dataset.json"
        if sha256(path) != protocol["dataset_sha256"]:
            raise ValueError("Variant source scope changed")
    verify_sources(output / protocol["variants"][0])
    return protocol


def infer(output, variant):
    from adapters.tcei import load_reference
    from analysis.teamtrack.inference import infer as native_infer
    from pipeline.tracking.tcei import TceiConfig

    protocol = verified(output)
    if variant not in protocol["variants"]:
        raise ValueError(variant)
    config = (
        TceiConfig(mode=variant, cache_scope=protocol["cache_scope"])
        if variant != "baseline"
        else None
    )
    reference = load_reference(Path(protocol["source_root"])) if config else None
    native_infer(
        output / variant,
        SPLIT,
        Path(protocol["checkpoint"]),
        calibration=config,
        tcei_reference=reference,
        audit_performance=True,
        extra_protocol={
            "experiment_sha256": sha256(output / "protocol.json"),
            "variant": variant,
        },
    )
    if config is not None and variant != "off":
        markers = (output / variant / SPLIT / "completed").glob("*.json")
        if not sum(json.loads(p.read_text())["calibrated_rows"] for p in markers):
            raise RuntimeError("Calibration never ran")


def evaluate(output):
    from analysis.trackid.benchmark import evaluate as evaluate_variant

    protocol = verified(output)
    sequences = json.loads((output / "dataset.json").read_text())["sequences"]
    parity(output, sequences, SPLIT)
    rows, performance, environments = [], [], []
    for variant in protocol["variants"]:
        folder = output / variant / SPLIT
        worker = json.loads((folder / "inference_protocol.json").read_text())
        expected = {
            "experiment_sha256": sha256(output / "protocol.json"),
            "variant": variant,
        }
        if worker.get("experiment") != expected:
            raise ValueError("Worker used a different experiment")
        # Evaluation normally runs without torch in the control environment.
        # The config itself is already serialized by the GPU worker; check mode here.
        mode = (
            worker["calibration"]["mode"]
            if worker["calibration"] is not None
            else "baseline"
        )
        if mode != variant:
            raise ValueError("Variant calibration mismatch")
        environments.append(
            {
                key: worker[key]
                for key in (
                    "runtime",
                    "preprocessing",
                    "frame_decoder",
                    "device",
                    "torch",
                    "torchvision",
                    "opencv",
                )
            }
        )
        for seq in sequences:
            completion = json.loads(
                (folder / "completed" / f"{seq['sequence']}.json").read_text()
            )
            performance.append(
                dict(variant=variant, sequence=seq["sequence"], **completion)
            )
        rows.extend(
            dict(variant=variant, **row) for row in evaluate_variant(output / variant)
        )
    if any(env != environments[0] for env in environments[1:]):
        raise ValueError("Variants used different GPU environments/runtime settings")
    write_csv(output / "summary.csv", rows)
    write_csv(output / "performance.csv", performance)
    overall = {row["variant"]: row for row in rows if row["scope"] == "overall"}
    metrics = ("HOTA", "AssA", "IDF1", "IDSW", "Frag", "DetA", "MOTA", "FP", "FN")
    write_csv(
        output / "deltas.csv",
        [
            dict(variant=key, **{k: row[k] - overall["baseline"][k] for k in metrics})
            for key, row in overall.items()
        ],
    )
    save_json(
        output / "status.json",
        {
            "status": "completed",
            "dataset": "TrackID3x3 Indoor",
            "variants": protocol["variants"],
            "sequences": len(sequences),
            "frames": sum(s["frames"] for s in sequences),
            "summary_sha256": sha256(output / "summary.csv"),
        },
    )
    for row in overall.values():
        print(json.dumps(row, ensure_ascii=False), flush=True)


def run(args):
    output = args.output_dir.expanduser().resolve()
    if args.stage == "status":
        path = output / "status.json"
        print(path.read_text() if path.exists() else '{"status":"not_started"}')
        protocol_path = output / "protocol.json"
        if protocol_path.exists():
            for variant in json.loads(protocol_path.read_text())["variants"]:
                print(
                    variant,
                    len(list((output / variant / SPLIT / "completed").glob("*.json"))),
                )
        return
    if args.stage == "infer":
        infer(output, args.variant)
        return
    try:
        if args.stage == "all":
            if args.dataset_root is None or args.source_root is None:
                raise ValueError("TrackID requires --dataset-root and --source-root")
            if args.env_file:
                from cli.serve_web import load_environment

                load_environment(args.env_file)
            from analysis.teamtrack.metrics import trackeval_module
            from contracts.execution import ExecutionSettings

            settings = ExecutionSettings()
            trackeval_module()
            protocol = prepare(
                output,
                args.dataset_root.expanduser().resolve(),
                args.source_root.expanduser().resolve(),
                (args.checkpoint or settings.motip_checkpoint).expanduser().resolve(),
                args.limit,
                args.variants,
                args.cache_scope,
            )
            sequences = json.loads((output / "dataset.json").read_text())["sequences"]
            for variant in protocol["variants"]:
                save_json(
                    output / "status.json", {"status": "running", "variant": variant}
                )
                subprocess.run(
                    [
                        str(settings.motip_python),
                        "-u",
                        "-m",
                        "cli.benchmark_tcei",
                        "--dataset",
                        "trackid-indoor",
                        "--stage",
                        "infer",
                        "--variant",
                        variant,
                        "--output-dir",
                        str(output),
                    ],
                    cwd=ROOT,
                    env=dict(os.environ, WANDB_MODE="disabled"),
                    check=True,
                )
                if variant == "off":
                    parity(output, sequences, SPLIT)
        evaluate(output)
    except Exception as error:
        save_json(
            output / "status.json",
            {"status": "failed", "error": f"{type(error).__name__}: {error}"},
        )
        raise
