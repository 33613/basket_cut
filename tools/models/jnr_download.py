"""Author-published Google Drive models; no unofficial checkpoint mirrors."""

from pathlib import Path
from contracts.schema import write_json
from pipeline.identity.jnr_backend import UPSTREAM_REVISION, sha256_file

MODELS = {
    "vit-small": {"drive_id": "1oc8VdEHHxXQfhNZTvbHbm6gfLRkwGl2o",
                  "filename": "vit_small_soccernet.pt", "config": "small16_reid.yaml",
                  "observed_sha256": "536ec612136ec522f23bcf3ebf88b4d794c6fc02ea0f7e301cd22eeb259df4e6"},
    "vit-base": {"drive_id": "16npJY-gyboRE_HNTQI1dC_fIxh3oxa0S",
                 "filename": "vit_base_soccernet.pt", "config": "base8_reid.yaml"},
}


def download_jnr(model, output_dir: Path):
    import gdown
    spec = MODELS[model]
    output_dir = output_dir.expanduser().resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    path = output_dir / spec["filename"]
    receipt = path.with_suffix(".download.json")
    if path.exists():
        if not receipt.is_file():
            raise FileExistsError("Checkpoint already exists without a receipt. Inspect it, or choose another output directory.")
        import json
        previous = json.loads(receipt.read_text())
        if (previous.get("sha256") != sha256_file(path) or previous.get("drive_id") != spec["drive_id"]
                or spec.get("observed_sha256") and previous.get("sha256") != spec["observed_sha256"]):
            raise ValueError("Existing JNR download does not match its receipt")
        return previous
    partial = path.with_suffix(".download")
    result = gdown.download(id=spec["drive_id"], output=str(partial), quiet=False, resume=True)
    if not result or not partial.is_file() or partial.stat().st_size < 1024 * 1024:
        raise RuntimeError("Google Drive download failed; get the official file in a browser and upload it instead")
    with partial.open("rb") as handle:
        prefix = handle.read(200).lower()
    if b"<html" in prefix or b"<!doctype" in prefix:
        raise RuntimeError("Received an HTML page, not model weights")
    if spec.get("observed_sha256") and sha256_file(partial) != spec["observed_sha256"]:
        raise RuntimeError("JNR weight bytes differ from the author download tested by this adapter; investigate before using")
    partial.rename(path)
    value = {"model": model, **spec, "path": str(path), "size_bytes": path.stat().st_size,
             "sha256": sha256_file(path), "upstream_revision": UPSTREAM_REVISION,
             "source": "https://github.com/lukaszgrad/uncertainty-jnr",
             "warning": "Author does not publish a checksum; observed_sha256 is our tested download, not publisher-signed. Run cli.inspect_jnr."}
    write_json(receipt, value)
    return value
