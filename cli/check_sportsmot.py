"""Exercise native SportsMOT validation and TrackEval parity reports on CPU."""
from analysis.sportsmot.selfcheck import run_selfcheck

if __name__ == '__main__':
    if not run_selfcheck().wasSuccessful():
        raise SystemExit(1)
