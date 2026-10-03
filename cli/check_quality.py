"""Run quality and identity resolution regressions without model packages."""

from analysis.quality_selfcheck import run_selfcheck

if __name__ == "__main__":
    if not run_selfcheck().wasSuccessful():
        raise SystemExit(1)
