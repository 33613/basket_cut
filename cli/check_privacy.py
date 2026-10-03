"""Check project-owned tracked/unignored text for deployment and credential leaks."""

import re
import subprocess
from pathlib import Path

SKIP = ("MOTIP/", "MMAction2/", "KPR/")
PATTERNS = {
    "personal_home_path": re.compile(r"/Users" + r"/[^/\s]+/|/home" + r"/[^/\s]+/"),
    "private_deployment": re.compile(r"/root" + r"/autodl|autodl" + r"-container-"),
    "cloud_operation_docs": re.compile(r"Auto" + r"DL|autodl\.com"),
    "credential_literal": re.compile(
        r"(?:gh[pousr]_" + r"[A-Za-z0-9]{20,}|hf_" + r"[A-Za-z0-9]{20,})"
    ),
}


def main():
    root = Path(__file__).resolve().parents[1]
    names = (
        subprocess.check_output(
            ["git", "ls-files", "--cached", "--others", "--exclude-standard", "-z"],
            cwd=root,
        )
        .decode()
        .split("\0")
    )
    violations = []
    for name in sorted(set(names)):
        if not name or not (root / name).is_file():
            continue
        try:
            contents = (root / name).read_text(encoding="utf-8")
        except UnicodeDecodeError:
            continue
        for line_number, line in enumerate(contents.splitlines(), 1):
            for rule, pattern in PATTERNS.items():
                # Public upstream examples may contain their authors' paths;
                # deployment traces and credential literals are checked everywhere.
                if name.startswith(SKIP) and rule == "personal_home_path":
                    continue
                if pattern.search(line):
                    violations.append(f"{name}:{line_number}: {rule}")
    if violations:
        # Never echo a possibly secret value.
        print("\n".join(violations))
        raise SystemExit(1)
    print("Privacy scan passed for project-owned worktree text (not Git history).")


if __name__ == "__main__":
    main()
