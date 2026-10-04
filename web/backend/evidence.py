"""CPU evidence jobs in a separate cache, including read-only imported runs."""

import subprocess
import threading
from concurrent.futures import ThreadPoolExecutor

from analysis.evaluation.track_review import review_index
from contracts.schema import write_json
from web.backend.reviews import review_paths


def evidence_root(store, project_id, video_id):
    store.get_video(project_id, video_id)
    return store.output_root / project_id / 'evidence' / video_id


class EvidenceJobs:
    def __init__(self, settings, store):
        self.settings, self.store = settings, store
        self.executor = ThreadPoolExecutor(max_workers=1, thread_name_prefix='review-evidence')
        self.lock = threading.RLock()
        self.futures = {}

    def active(self, project_id, video_id):
        future = self.futures.get((project_id, video_id))
        return future is not None and not future.done()

    def submit(self, project_id, video):
        vid = video['video_id']
        paths = review_paths(video)
        # Validate raw scope before doing expensive decoding; never hide legacy lost tracks.
        review_index(paths['tracks'], paths['video_meta'], paths['quality'])
        with self.lock:
            if self.active(project_id, vid):
                raise RuntimeError('Evidence job is already active')
            destination = evidence_root(self.store, project_id, vid)
            destination.mkdir(parents=True, exist_ok=True)
            write_json(destination / 'job.json', {'status': 'queued'})
            self.futures[(project_id, vid)] = self.executor.submit(self._run, video, paths, destination)
            return {'status': 'queued'}

    def _run(self, video, paths, destination):
        write_json(destination / 'job.json', {'status': 'running'})
        command = [str(self.settings.review_python), '-m', 'cli.prepare_track_review',
                   '--input', video['source_path'], '--tracks', str(paths['tracks']),
                   '--video-meta', str(paths['video_meta']), '--output-dir', str(destination), '--overwrite']
        if paths['quality'].is_file():
            command.extend(['--quality', str(paths['quality'])])
        try:
            with (destination / 'job.log').open('w', encoding='utf-8') as log:
                code = subprocess.run(command, cwd=self.settings.repository_root,
                                      stdout=log, stderr=subprocess.STDOUT).returncode
            if code:
                error = (destination / 'job.log').read_text()[-4000:]
                raise RuntimeError(f'Evidence command exited {code}. {error}')
            write_json(destination / 'job.json', {'status': 'completed'})
        except Exception as exc:
            write_json(destination / 'job.json', {'status': 'failed', 'error': str(exc)})
