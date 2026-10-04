"""Start the dashboard with an explicit private .env file (no shell evaluation)."""

import argparse
import os
import shlex
from pathlib import Path


def load_environment(path: Path):
    for line in path.read_text(encoding='utf-8').splitlines():
        line = line.strip()
        if not line or line.startswith('#'):
            continue
        key, separator, raw = line.removeprefix('export ').partition('=')
        key = key.strip()
        if not separator or not key.startswith('BASKET_') or not key.replace('_', '').isalnum():
            raise ValueError('Configuration only accepts BASKET_* assignments')
        tokens = shlex.split(raw, comments=True)
        if len(tokens) > 1:
            raise ValueError(f'Quote values with spaces: {key}')
        os.environ[key] = tokens[0] if tokens else ''


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--env-file', type=Path)
    p.add_argument('--host', default='127.0.0.1')
    p.add_argument('--port', type=int, default=6006)
    args = p.parse_args()
    if args.env_file:
        load_environment(args.env_file)
    from web.backend.settings import WebSettings
    settings = WebSettings()
    print(f'Experiment index: {settings.output_root}', flush=True)
    print(f'Allowed result imports: {settings.import_root}', flush=True)
    import uvicorn
    uvicorn.run('web.backend.app:app', host=args.host, port=args.port)


if __name__ == '__main__':
    main()
