"""Download one public uncertainty-jnr model from the author's link."""

import argparse
import json
from pathlib import Path
from contracts.paths import runtime_root
from tools.models.jnr_download import MODELS, download_jnr


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", choices=sorted(MODELS), default="vit-small")
    parser.add_argument("--output-dir", type=Path, default=runtime_root() / "models/jnr")
    parser.add_argument("--list", action="store_true")
    args = parser.parse_args()
    print(json.dumps(MODELS[args.model] if args.list else download_jnr(args.model, args.output_dir), indent=2))


if __name__ == "__main__":
    main()
