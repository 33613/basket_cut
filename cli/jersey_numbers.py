"""Extract optional jersey number candidates without changing KPR identities."""

import argparse
import json
from pathlib import Path
from contracts.paths import model_path
from pipeline.identity.jersey import recognize_jerseys


def main():
    p = argparse.ArgumentParser(description=__doc__)
    for name in ('archive-dir', 'identity-map', 'output-dir'):
        p.add_argument('--' + name, type=Path, required=True)
    p.add_argument('--model-dir', type=Path, default=model_path('ocr', 'easyocr'))
    p.add_argument('--allow-download', action='store_true')
    p.add_argument('--min-score', type=float, default=.8)
    p.add_argument('--min-support', type=int, default=2)
    p.add_argument('--overwrite', action='store_true')
    print(json.dumps(recognize_jerseys(**vars(p.parse_args())), ensure_ascii=False, indent=2))


if __name__ == '__main__':
    main()
