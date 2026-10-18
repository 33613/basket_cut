"""Retry failed library/tail stages using immutable, completed model evidence."""

import json
import shlex
import subprocess
import time
import uuid
from pathlib import Path

from contracts.schema import write_json
from pipeline.media import sha256_file
from workflows.batch import stages_for
from workflows.commands import CommandBuilder


def repair_player_stages(settings, run_dir, *, execute=None):
    root = Path(run_dir).resolve()
    plan = json.loads((root / 'batch_manifest.json').read_text())
    builder = CommandBuilder(settings)
    options = plan['options']
    stages = stages_for(plan['target'], bool(options.get('jersey_qwen', True)))
    before, tail = stages[:stages.index('players')], stages[stages.index('players'):]
    repaired, skipped = [], []
    for clip in plan['clips']:
        name = clip['name']
        if Path(name).name != name or (root / name).resolve().parent != root:
            raise ValueError('Unsafe clip in recovery manifest')
        directory = root / name
        status_path = directory / 'batch_status.json'
        if not status_path.is_file():
            skipped.append({'clip': name, 'reason': 'not_started'})
            continue
        state = json.loads(status_path.read_text())
        if state.get('status') == 'completed':
            continue
        if (state.get('stages', {}).get('players') not in ('failed', 'running', 'completed')
                or any(state.get('stages', {}).get(s) != 'completed' for s in before)):
            skipped.append({'clip': name, 'reason': 'upstream_model_stage_incomplete'})
            continue
        if state.get('fingerprint') != clip['fingerprint']:
            raise ValueError(f'{name}: result fingerprint differs from registered clip')
        meta = json.loads((directory / 'tracking/video_meta.json').read_text())
        source = Path(meta['input_path']).resolve()
        if clip.get('lineage') and sha256_file(source) != clip['lineage']['sha256']:
            raise ValueError(f'{name}: exported input video changed')
        paths = builder.paths({'source_path': str(source), 'output_dir': str(directory)})
        for stage in before:
            if not builder.artifact_exists(stage, paths):
                raise ValueError(f'{name}: incomplete/stale upstream artifact: {stage}')
        attempt = uuid.uuid4().hex
        backup = directory / 'recovery' / f'{attempt}.json'
        write_json(backup, {'original_status': state,
            'method': 'reuse_completed_model_evidence_retry_player_and_tail_stages',
            'code_sha256': {p: sha256_file(settings.repository_root / p) for p in (
                'workflows/recovery.py', 'workflows/library_inputs.py',
                'pipeline/identity/jersey.py', 'pipeline/identity/player_registry.py')}})
        state.setdefault('recovery_attempts', []).append(str(backup.relative_to(root)))
        state['status'] = 'running'
        state.pop('error', None)
        try:
            for stage in tail:
                command = builder.command(stage, paths, {**options, 'clip_id': clip.get('clip_id', name)})
                if stage == 'render':
                    for flag in ('--actions', '--identity-map'):
                        if flag in command:
                            index = command.index(flag)
                            del command[index:index + 2]
                    command.extend(['--identity-map', str(paths['identity_map'])])
                    if plan['target'] == 'full':
                        command.extend(['--actions', str(paths['linked_actions'])])
                print(f'[recovery {name}] {stage}', flush=True)
                state['stages'][stage] = 'running'
                write_json(status_path, state)
                started = time.monotonic()
                with paths['log'].open('a') as log:
                    log.write(f'\n# Recovery {attempt}\n$ {shlex.join(command)}\n')
                    log.flush()
                    code = execute(command, paths) if execute else subprocess.run(
                        command, cwd=settings.repository_root, stdout=log, stderr=subprocess.STDOUT).returncode
                state.setdefault('stage_metrics', {})[stage] = {
                    'elapsed_s': round(time.monotonic() - started, 3), 'exit_code': code,
                    'reused': False, 'recovery_attempt': attempt}
                if code or not builder.artifact_exists(stage, paths):
                    state['stages'][stage] = 'failed'
                    raise RuntimeError(f'{stage} recovery failed; see {paths["log"]}')
                state['stages'][stage] = 'completed'
                write_json(status_path, state)
            state['status'] = 'completed'
            repaired.append(name)
        except Exception as exc:
            state['status'] = 'failed'
            state['error'] = str(exc)
            print(f'[recovery {name}] FAILED: {exc}', flush=True)
        write_json(status_path, state)
    states = [json.loads(p.read_text()) for row in plan['clips']
              if (p := root / row['name'] / 'batch_status.json').is_file()]
    summary = {'expected_clips': len(plan['clips']), 'processed_clips': len(states),
               'completed_clips': sum(s['status'] == 'completed' for s in states), 'clips': states}
    write_json(root / 'batch_summary.json', summary)
    report = {'repaired_clips': repaired, 'skipped': skipped,
              'failed_clips': [s['name'] for s in states if s['status'] == 'failed'],
              'completed_clips': summary['completed_clips'], 'expected_clips': len(plan['clips']),
              'models_rerun': False}
    write_json(root / 'recovery_summary.json', report)
    return report
