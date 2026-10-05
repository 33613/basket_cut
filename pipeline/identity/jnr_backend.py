"""Thin inference adapter for Grad's uncertainty-jnr; upstream is not modified."""

from pathlib import Path
import hashlib
import sys


UPSTREAM_REVISION = "f19d9cb90e1a67d5ffe44acbf69fb348fe6525e7"


def sha256_file(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def normalize_state_dict(state):
    cleaned = {}
    for key, value in state.items():
        name = key.replace("_orig_mod.", "")
        if name in cleaned:
            raise ValueError("Duplicate checkpoint key after removing compile wrappers")
        cleaned[name] = value
    return cleaned


class JNRBackend:
    def __init__(self, root: Path, config: Path, checkpoint: Path, *, device="cuda",
                 trust_checkpoint=False):
        import torch
        import yaml
        root, config, checkpoint = root.resolve(), config.resolve(), checkpoint.resolve()
        for path in (root / "src/uncertainty_jnr/model.py",
                     root / "src/uncertainty_jnr/augmentation.py", config, checkpoint):
            if not path.is_file():
                raise FileNotFoundError(path)
        sys.path.insert(0, str(root / "src"))
        from uncertainty_jnr.model import TimmOCRModel
        from uncertainty_jnr.augmentation import get_val_transforms
        import uncertainty_jnr.model as upstream_model
        if not Path(upstream_model.__file__).resolve().is_relative_to(root):
            raise ValueError("Another uncertainty_jnr checkout is already imported")
        value = yaml.safe_load(config.read_text())
        model_config = value["model"]
        self.target_size = tuple(value["data"]["target_size"])
        if len(self.target_size) != 2 or any(int(v) < 1 for v in self.target_size):
            raise ValueError("Invalid JNR target size")
        self.device = torch.device(device)
        if self.device.type == "cuda" and not torch.cuda.is_available():
            raise RuntimeError("JNR requested CUDA but no GPU is available; use --device cpu for inspection")
        # The timm/ namespace is a local model name, not a request for pretrained weights.
        name = model_config["model_name"].removeprefix("timm/")
        self.model = TimmOCRModel(
            model_name=name, pretrained=False,
            classifier_type=model_config.get("classifier_type", "independent"),
            embedding_type=model_config.get("embedding_type", "additive"),
            per_digit_bias=model_config.get("per_digit_bias", True),
            uncertainty_head=model_config.get("uncertainty_head", "dirichlet"),
            size_embedding=model_config.get("size_embedding", False),
            **(model_config.get("model_kwargs") or {}))
        try:
            payload = torch.load(checkpoint, map_location="cpu", weights_only=not trust_checkpoint)
        except Exception as exc:
            raise RuntimeError("Cannot load JNR checkpoint. If weights-only loading rejects an official "
                               "download, inspect its provenance before explicitly using --trust-checkpoint. "
                               "Never trust arbitrary pickle files.") from exc
        if not isinstance(payload, dict) or "model_state_dict" not in payload:
            raise ValueError("Expected the author's checkpoint with model_state_dict")
        # Never silently discard a number classifier or randomly initialized backbone.
        self.model.load_state_dict(normalize_state_dict(payload["model_state_dict"]), strict=True)
        self.model.to(self.device).eval()
        self.transform = get_val_transforms(self.target_size)
        self.provenance = {
            "backend": "uncertainty_jnr", "model_name": model_config["model_name"],
            "uncertainty_head": model_config.get("uncertainty_head", "dirichlet"),
            "checkpoint_sha256": sha256_file(checkpoint), "config_sha256": sha256_file(config),
            "source_sha256": {name: sha256_file(root / "src/uncertainty_jnr" / name)
                              for name in ("model.py", "augmentation.py")},
            "device": str(self.device), "strict_weight_loading": True,
            "raw_score_definition": "maximum of author's number_probs, not calibrated basketball accuracy",
            "trusted_pickle_loading": trust_checkpoint, "class_count": 100,
            "preprocessing": "official JerseyCrop + cubic resize + RGB / 127.5 - 1"}

    def predict(self, images):
        """Input full-person BGR crops; return all 100 class scores and uncertainty."""
        import cv2
        import numpy as np
        import torch
        tensors = []
        for crop in images:
            if crop is None or crop.ndim != 3 or crop.shape[2] != 3 or min(crop.shape[:2]) < 2:
                raise ValueError("Invalid full-person crop")
            image = self.transform(image=cv2.cvtColor(crop, cv2.COLOR_BGR2RGB))["image"]
            tensors.append(torch.from_numpy(np.ascontiguousarray(image.transpose(2, 0, 1))).float() / 127.5 - 1)
        with torch.inference_mode():
            # Float32 avoids exp(logit) overflow in the author's Dirichlet head under FP16.
            result = self.model(torch.stack(tensors).to(self.device))
        probs = result.number_probs.detach().cpu().numpy().astype(np.float32)
        uncertainty = result.uncertainty.detach().cpu().numpy().reshape(-1).astype(np.float32)
        return probs, uncertainty
