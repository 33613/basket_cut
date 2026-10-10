"""Register new clip evidence into a persistent match-scoped player library."""
import argparse
import json
from pathlib import Path
from contracts.schema import write_json
from workflows.person_library import PersonLibraryOptions, build_person_library


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--run-dir', required=True, type=Path)
    parser.add_argument('--match-id', required=True)
    parser.add_argument('--clip-name', action='append', help='Register only these completed identity stages')
    parser.add_argument('--max-distance', type=float, default=.2)
    parser.add_argument('--novelty-distance', type=float, default=.5)
    parser.add_argument('--min-margin', type=float, default=.05)
    parser.add_argument('--min-samples', type=int, default=2)
    parser.add_argument('--max-within-distance', type=float, default=.4)
    parser.add_argument('--min-common-parts', type=int, default=2)
    parser.add_argument('--gallery-limit', type=int, default=12)
    parser.add_argument('--use-jersey-evidence', action='store_true')
    parser.add_argument('--registration-output', type=Path)
    args = vars(parser.parse_args())
    registration = args.pop('registration_output')
    names = args.pop('clip_name')
    args['clip_names'] = tuple(names) if names else None
    result = build_person_library(PersonLibraryOptions(**args))
    if registration:
        write_json(registration, {'match_id': result['match_id'], 'revision': result['review_revision']})
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == '__main__':
    main()
