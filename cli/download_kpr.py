"""List or download the project-supported KPR model."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from contracts.paths import runtime_root


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", default="multidataset-sports")
    parser.add_argument(
        "--output-dir", type=Path, default=runtime_root() / "models/kpr"
    )
    parser.add_argument("--list", action="store_true")
    parser.add_argument("--force-download", action="store_true")
    return parser


def main() -> None:
    args = build_parser().parse_args()
    from tools.models.kpr_download import KPRDownloadOptions, download_kpr_model

    result = download_kpr_model(
        KPRDownloadOptions(
            model=args.model,
            output_dir=args.output_dir,
            list_only=args.list,
            force_download=args.force_download,
        )
    )
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
