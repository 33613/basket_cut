"""Strictly verify JNR architecture/checkpoint compatibility and a synthetic forward."""

import argparse
import json
from pathlib import Path
from contracts.execution import ExecutionSettings


def main():
    settings = ExecutionSettings()
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=settings.jnr_root)
    parser.add_argument("--config", type=Path, default=settings.jnr_config)
    parser.add_argument("--checkpoint", type=Path, default=settings.jnr_checkpoint)
    parser.add_argument("--device", choices=("cpu", "cuda"), default="cpu")
    parser.add_argument("--trust-checkpoint", action="store_true")
    args = parser.parse_args()
    import numpy as np
    from pipeline.identity.jnr_backend import JNRBackend
    backend = JNRBackend(**vars(args))
    scores, uncertainty = backend.predict([np.zeros((256, 128, 3), dtype=np.uint8)])
    if scores.shape != (1, 100) or uncertainty.shape != (1,) or not np.isfinite(scores).all() or not np.isfinite(uncertainty).all():
        raise RuntimeError("Nonfinite or malformed JNR forward output")
    print(json.dumps({**backend.provenance, "forward_shape": list(scores.shape),
                      "warning": "Synthetic forward verifies loading, not basketball number accuracy"}, indent=2))


if __name__ == "__main__":
    main()
