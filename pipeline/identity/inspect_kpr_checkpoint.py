"""Verify the downloaded multi-dataset KPR checkpoint before loading it.

PyTorch checkpoints use pickle.  This command therefore validates the exact
SHA-256 published by the Hugging Face repository before calling ``torch.load``.
Run it only for the trusted Tracking Laboratory artifact documented here.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from collections.abc import Mapping
from pathlib import Path
from typing import Any


DEFAULT_CHECKPOINT = Path(
    "/root/autodl-tmp/models/kpr/"
    "kpr_dancetrack_sportsmot_posetrack21_occludedduke_market_split0.pth.tar"
)
PUBLISHED_SHA256 = "c7f3a74d86a0bb56940b2703508a50f1d3dbee4d755049272ef5caa18457db3f"


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Verify and inspect the official multi-dataset KPR checkpoint"
    )
    parser.add_argument("--checkpoint", type=Path, default=DEFAULT_CHECKPOINT)
    parser.add_argument(
        "--expected-sha256",
        default=PUBLISHED_SHA256,
        help="Expected digest published by the Hugging Face file page",
    )
    parser.add_argument(
        "--skip-tensor-scan",
        action="store_true",
        help="Skip the finite-value scan over floating-point model tensors",
    )
    parser.add_argument("--output", type=Path)
    return parser


def sha256_file(path: Path, chunk_size: int = 8 * 1024 * 1024) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while chunk := handle.read(chunk_size):
            digest.update(chunk)
    return digest.hexdigest()


def nested_value(value: Any, *keys: str) -> Any:
    current = value
    for key in keys:
        if isinstance(current, Mapping):
            current = current.get(key)
        else:
            current = getattr(current, key, None)
        if current is None:
            return None
    return current


def json_safe(value: Any) -> Any:
    if value is None or isinstance(value, (str, int, float, bool)):
        return value
    if isinstance(value, Mapping):
        return {str(key): json_safe(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [json_safe(item) for item in value]
    return str(value)


def locate_state_dict(checkpoint: Any, torch) -> tuple[str | None, Mapping | None]:
    if not isinstance(checkpoint, Mapping):
        return None, None
    for key in ("state_dict", "model_state_dict", "model"):
        candidate = checkpoint.get(key)
        if isinstance(candidate, Mapping) and any(
            torch.is_tensor(value) for value in candidate.values()
        ):
            return key, candidate
    if any(torch.is_tensor(value) for value in checkpoint.values()):
        return "<root>", checkpoint
    return None, None


def run(args: argparse.Namespace) -> dict[str, Any]:
    checkpoint_path = args.checkpoint.expanduser().resolve()
    if not checkpoint_path.is_file():
        raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")

    actual_sha256 = sha256_file(checkpoint_path)
    expected_sha256 = args.expected_sha256.lower().strip()
    sha256_matches = actual_sha256 == expected_sha256
    if not sha256_matches:
        raise RuntimeError(
            "Checkpoint SHA-256 mismatch; refusing to unpickle it. "
            f"expected={expected_sha256}, actual={actual_sha256}"
        )

    try:
        import torch
    except ImportError as exc:
        raise RuntimeError("Run this command inside the KPR Conda environment") from exc

    checkpoint = torch.load(checkpoint_path, map_location="cpu")
    state_key, state_dict = locate_state_dict(checkpoint, torch)
    if state_dict is None:
        raise RuntimeError("No tensor state_dict found in the checkpoint")

    tensor_count = 0
    parameter_count = 0
    non_finite_keys = []
    for key, value in state_dict.items():
        if not torch.is_tensor(value):
            continue
        tensor_count += 1
        parameter_count += value.numel()
        if (
            not args.skip_tensor_scan
            and (torch.is_floating_point(value) or torch.is_complex(value))
            and not bool(torch.isfinite(value).all())
        ):
            non_finite_keys.append(str(key))

    embedded_config = checkpoint.get("config") if isinstance(checkpoint, Mapping) else None
    result = {
        "valid": bool(
            sha256_matches
            and state_dict is not None
            and embedded_config is not None
            and not non_finite_keys
        ),
        "path": str(checkpoint_path),
        "size_bytes": checkpoint_path.stat().st_size,
        "sha256": actual_sha256,
        "sha256_matches_published_value": sha256_matches,
        "checkpoint_keys": (
            sorted(str(key) for key in checkpoint.keys())
            if isinstance(checkpoint, Mapping)
            else []
        ),
        "state_dict_key": state_key,
        "tensor_count": tensor_count,
        "parameter_count": parameter_count,
        "non_finite_tensor_keys": non_finite_keys,
        "embedded_config_present": embedded_config is not None,
        "embedded_training_sources": json_safe(
            nested_value(embedded_config, "data", "sources")
        ),
        "embedded_training_targets": json_safe(
            nested_value(embedded_config, "data", "targets")
        ),
        "embedded_model_name": json_safe(
            nested_value(embedded_config, "model", "name")
        ),
        "embedded_backbone": json_safe(
            nested_value(embedded_config, "model", "kpr", "backbone")
        ),
        "tensor_scan_performed": not args.skip_tensor_scan,
    }
    if not result["valid"]:
        raise RuntimeError(
            "Checkpoint container loaded, but required config or finite tensors "
            f"failed validation: {json.dumps(result, ensure_ascii=False)}"
        )

    if args.output:
        output_path = args.output.expanduser().resolve()
        output_path.parent.mkdir(parents=True, exist_ok=True)
        with output_path.open("w", encoding="utf-8") as handle:
            json.dump(result, handle, ensure_ascii=False, indent=2, sort_keys=True)
            handle.write("\n")
    print(json.dumps(result, ensure_ascii=False, indent=2, sort_keys=True))
    return result


def main() -> None:
    run(build_parser().parse_args())


if __name__ == "__main__":
    main()

