"""Bounded, safe selection from TAR/ZIP video archives; no full extraction."""

import json
import os
import shutil
import tarfile
import tempfile
import zipfile
from pathlib import Path, PurePosixPath

from contracts.schema import write_json
from workflows.batch import VIDEO_SUFFIXES


def safe_name(name: str) -> str:
    path = PurePosixPath(name)
    if path.is_absolute() or ".." in path.parts or "\\" in name or ":" in name:
        raise ValueError(f"Unsafe archive member: {name}")
    return path.as_posix()


def extract_videos(archive: Path, output_dir: Path, *, limit: int = 30,
                   max_gb: float = 6, selection: Path | None = None, list_only: bool = False) -> dict:
    if limit < 1 or not __import__('math').isfinite(max_gb) or max_gb <= 0:
        raise ValueError("limit and max-gb must be positive and finite")
    archive = archive.resolve()
    output = output_dir.resolve()
    if archive.is_relative_to(output):
        raise ValueError("Keep archive outside the extraction directory")
    requested = None
    if selection:
        value = json.loads(selection.read_text())
        requested = set(value.get("videos", []) if isinstance(value, dict) else value)
        if not requested:
            raise ValueError("Empty video-key selection")
    zipped = zipfile.is_zipfile(archive)
    reader = zipfile.ZipFile(archive) if zipped else tarfile.open(archive)
    try:
        members = reader.infolist() if zipped else reader.getmembers()
        eligible = []
        seen = set()
        for member in members:
            name = member.filename if zipped else member.name
            if Path(name).suffix.lower() not in VIDEO_SUFFIXES:
                continue
            name = safe_name(name)
            regular = not member.is_dir() and ((member.external_attr >> 16) & 0o170000) != 0o120000 if zipped else member.isfile()
            if not regular:
                raise ValueError(f"Video member is not a regular file: {name}")
            if name in seen:
                raise ValueError(f"Duplicate video member: {name}")
            seen.add(name)
            key = str(PurePosixPath(name).with_suffix(""))
            matches = requested is None or any(key == v or key.endswith('/' + v) for v in requested)
            if matches:
                eligible.append((name, member.file_size if zipped else member.size, member))
        chosen = sorted(eligible, key=lambda row: row[0])[:limit]
        if not chosen:
            raise ValueError("No selected video members. Check archive type/selection; do not extract a raw-frame archive as videos.")
        required = sum(size for _, size, _ in chosen)
        if required > max_gb * 1024**3:
            raise ValueError(f"Selected videos need {required / 1024**3:.2f} GiB, above --max-gb {max_gb}; lower --limit")
        result = {"schema_version": 1, "available_selected_videos": len(eligible),
                  "selected_videos": len(chosen), "selected_bytes": required,
                  "videos": [{"path": n, "size_bytes": s} for n, s, _ in chosen]}
        if list_only:
            return result
        output.mkdir(parents=True, exist_ok=True)
        missing_bytes = 0
        for name, size, _ in chosen:
            target = (output / name).resolve()
            if not target.is_relative_to(output):
                raise ValueError("Extraction path escapes output directory")
            if target.exists():
                if not target.is_file() or target.stat().st_size != size:
                    raise FileExistsError(f"Existing file differs; use a new extraction directory: {target}")
            else:
                missing_bytes += size
        if shutil.disk_usage(output).free < missing_bytes + 1024**3:
            raise ValueError("Insufficient disk: retain at least 1 GiB after extraction")
        for name, size, member in chosen:
            target = output / name
            if target.is_file():
                continue
            target.parent.mkdir(parents=True, exist_ok=True)
            descriptor, temporary = tempfile.mkstemp(prefix=".video-", dir=target.parent)
            try:
                with os.fdopen(descriptor, 'wb') as destination:
                    source = reader.open(member) if zipped else reader.extractfile(member)
                    with source:
                        shutil.copyfileobj(source, destination, length=1024**2)
                if Path(temporary).stat().st_size != size:
                    raise ValueError(f"Truncated member: {name}")
                os.replace(temporary, target)
            finally:
                Path(temporary).unlink(missing_ok=True)
        manifest = output / "video_selection.json"
        if manifest.resolve() == archive:
            raise ValueError("Manifest cannot overwrite archive")
        write_json(manifest, result)
        return {**result, "manifest": str(manifest)}
    finally:
        reader.close()
