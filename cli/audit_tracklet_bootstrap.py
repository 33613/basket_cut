"""Export native initialization crops, scene frames and Qwen decisions on CPU."""

import argparse
import json
from pathlib import Path


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--run-dir",
        type=Path,
        required=True,
        help="Results directory containing bootstrap_tracklets/",
    )
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--context-samples", type=int, default=6)
    args = parser.parse_args()
    print(
        "正在核对来源和候选图片，导出原尺寸复核报告；只运行 CPU，不调用模型。",
        flush=True,
    )
    from analysis.tracklet_audit import audit

    report, files = audit(
        args.run_dir, args.output_dir, context_samples=args.context_samples
    )
    print(json.dumps(report, ensure_ascii=False, indent=2))
    print(json.dumps(files, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
