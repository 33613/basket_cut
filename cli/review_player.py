"""Review stable match identities without rerunning models."""
import argparse
import fcntl
import json
from pathlib import Path
from pipeline.identity.player_registry import PlayerRegistry
from workflows.person_library import export_library


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--run-dir', type=Path, required=True)
    parser.add_argument('--operation', choices=('confirm', 'label_group', 'exclude', 'detach', 'restore'), required=True)
    parser.add_argument('--expected-revision', required=True)
    parser.add_argument('--person-id')
    parser.add_argument('--node-id')
    parser.add_argument('--label')
    args = parser.parse_args()
    root = args.run_dir.resolve()
    output = root / 'library'
    if output.is_symlink():
        parser.error('Library cannot be a symlink')
    with (output / '.lock').open('a') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX)
        manifest = json.loads((output / 'library_manifest.json').read_text())
        with PlayerRegistry(output / 'players.sqlite3', manifest['match_id']) as registry:
            registry.review(args.operation, expected_revision=args.expected_revision,
                person_id=args.person_id, node_id=args.node_id, label=args.label)
            result = export_library(registry, root, output)
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == '__main__':
    main()
