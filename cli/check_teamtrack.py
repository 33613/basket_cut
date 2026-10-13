"""Exercise TeamTrack conversion and real TrackEval on CPU fixtures."""
from analysis.teamtrack.selfcheck import run_selfcheck

if __name__ == '__main__':
    if not run_selfcheck().wasSuccessful():
        raise SystemExit(1)
