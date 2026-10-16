"""Preview, reprocess, reject observations or undo a match memory update on CPU."""
import argparse
import json
from pathlib import Path

from pipeline.identity.gallery_memory import MemoryPolicy
from workflows.gallery_maintenance import maintain


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--run-dir", type=Path, required=True, help="Results directory containing library/ and bootstrap/")
    p.add_argument("--operation", choices=("preview", "reprocess", "rollback_memory", "reject_observation", "reinstate_observation"), default="preview")
    p.add_argument("--output-dir", type=Path)
    p.add_argument("--expected-revision")
    p.add_argument("--node-id")
    p.add_argument("--strong-distance", type=float)
    p.add_argument("--supported-distance", type=float)
    p.add_argument("--gallery-limit", type=int)
    args = p.parse_args()
    if args.operation != "preview" and not args.expected_revision:
        p.error("Mutations require --expected-revision from the current library manifest")
    changes = {k:v for k,v in vars(args).items() if k in {"strong_distance","supported_distance","gallery_limit"} and v is not None}
    policy = None
    if changes:
        manifest = json.loads((args.run_dir/"library/library_manifest.json").read_text())
        policy = MemoryPolicy(**{**(manifest.get("memory_policy") or {}), **changes}).validate()
    report = maintain(args.run_dir, operation=args.operation, expected_revision=args.expected_revision,
        output_dir=args.output_dir, policy=policy, node_id=args.node_id)
    print(json.dumps(report,ensure_ascii=False,indent=2))


if __name__ == "__main__": main()
