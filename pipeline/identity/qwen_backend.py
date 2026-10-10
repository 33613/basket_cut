"""Local, offline Qwen2.5-VL adapter. No model code or weights downloaded at runtime."""

import hashlib
import json
from pathlib import Path

PROMPT = ('Read only the jersey number worn by the central person. Do not infer from identity, '
          'other people, scoreboards or other text. If occluded, blurred, incomplete or absent, '
          'return null. Preserve 0 versus 00. Return exactly JSON: '
          '{"number": "0-99 or 00, or null", "readable": true_or_false}.')


def model_provenance(model_dir):
    root = Path(model_dir).resolve()
    if not (root / "config.json").is_file():
        raise FileNotFoundError("Local Qwen config.json is missing")
    files = sorted(root.glob("*.safetensors"))
    if not files:
        raise FileNotFoundError("Qwen requires local safetensors weights")
    checksum = hashlib.sha256()
    for path in files + sorted(root.glob("*.json")):
        checksum.update(path.name.encode())
        with path.open("rb") as handle:
            for chunk in iter(lambda: handle.read(1024 * 1024), b""):
                checksum.update(chunk)
    return {"backend": "qwen_vl", "model_sha256": checksum.hexdigest(),
            "prompt_sha256": hashlib.sha256(PROMPT.encode()).hexdigest()}


def parse_reading(raw):
    try:
        text = raw.strip()
        if text.startswith("```json") and text.endswith("```"):
            text = text[7:-3].strip()
        value = json.loads(text)
    except (TypeError, ValueError):
        return {"number": None, "rejection_reasons": ["invalid_model_json"]}
    if not isinstance(value, dict) or value.get("readable") is not True:
        return {"number": None, "rejection_reasons": ["unreadable"]}
    number = value.get("number")
    if (not isinstance(number, str) or not number.isascii() or not number.isdigit()
            or not 1 <= len(number) <= 2 or (len(number) == 2 and number.startswith("0") and number != "00")):
        return {"number": None, "rejection_reasons": ["invalid_number"]}
    return {"number": number, "rejection_reasons": []}


class QwenBackend:
    def __init__(self, model_dir, *, device="cuda"):
        import torch
        from transformers import AutoProcessor, Qwen2_5_VLForConditionalGeneration
        if device == "cuda" and not torch.cuda.is_available():
            raise RuntimeError("Qwen CUDA inference requires a GPU")
        self.provenance = model_provenance(model_dir)
        self.processor = AutoProcessor.from_pretrained(str(model_dir), local_files_only=True,
            trust_remote_code=False, min_pixels=256 * 28 * 28, max_pixels=512 * 28 * 28)
        self.model = Qwen2_5_VLForConditionalGeneration.from_pretrained(str(model_dir),
            local_files_only=True, trust_remote_code=False, use_safetensors=True,
            torch_dtype=torch.float16 if device == "cuda" else torch.float32,
            attn_implementation="sdpa").to(device).eval()
        self.device = device

    def read(self, image):
        import torch
        messages = [{"role": "user", "content": [
            {"type": "image", "image": image}, {"type": "text", "text": PROMPT}]}]
        text = self.processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        inputs = self.processor(text=[text], images=[image], padding=True, return_tensors="pt").to(self.device)
        with torch.inference_mode():
            tokens = self.model.generate(**inputs, max_new_tokens=128, do_sample=False)
        return self.processor.batch_decode(tokens[:, inputs.input_ids.shape[1]:],
            skip_special_tokens=True, clean_up_tokenization_spaces=False)[0]
