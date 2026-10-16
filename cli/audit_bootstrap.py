"""Inspect existing bootstrap crops, KPR distances and recorded matching gates."""

import argparse
import json
from pathlib import Path


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-dir",required=True,type=Path,help="The match results directory containing bootstrap/ and library/")
    parser.add_argument("--output-dir",required=True,type=Path)
    parser.add_argument("--neighbors",type=int,default=5)
    args=parser.parse_args()
    from analysis.bootstrap_audit import audit
    report,files=audit(args.run_dir,args.output_dir,neighbors=args.neighbors)
    print(json.dumps(report,ensure_ascii=False,indent=2))
    print(json.dumps(files,ensure_ascii=False,indent=2))


if __name__=="__main__":
    main()
