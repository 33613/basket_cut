"""Initialize complete-coverage annotation sheets or evaluate possession cuts."""

import argparse
import json
from pathlib import Path

from analysis.possession_evaluation import annotation_template, write_evaluation
from pipeline.media import atomic_json


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--segmentation-dir", required=True, type=Path)
    p.add_argument("--annotations", required=True, type=Path)
    p.add_argument("--initialize", action="store_true")
    p.add_argument("--tolerance-s", type=float, default=2)
    p.add_argument("--max-candidate-width-s", type=float, default=10)
    args = p.parse_args()
    if args.initialize:
        if args.annotations.exists():
            raise FileExistsError("Annotation file already exists; preserving previous work")
        source = json.loads((args.segmentation_dir / "protocol.json").read_text())["source"]
        result = annotation_template(source)
        atomic_json(args.annotations, result)
    else:
        result = write_evaluation(args.segmentation_dir, args.annotations,
            tolerance_s=args.tolerance_s, max_width_s=args.max_candidate_width_s)
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
