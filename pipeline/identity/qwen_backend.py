"""Local, offline Qwen3-VL adapter for images and bounded video windows. No model code or weights downloaded at runtime."""

import hashlib
import json
from pathlib import Path

MODEL_TYPE = "qwen3_vl"
IMAGE_BUDGET = {"shortest_edge": 256 * 32 * 32, "longest_edge": 512 * 32 * 32}
VIDEO_BUDGET = {"shortest_edge": 128 * 32 * 32 * 2, "longest_edge": 2048 * 32 * 32 * 2}

PROMPT = ('Read only the jersey number worn by the central person. Do not infer from identity, '
          'other people, scoreboards or other text. If occluded, blurred, incomplete or absent, '
          'return null. Preserve 0 versus 00. Return exactly JSON: '
          '{"number": "0-99 or 00, or null", "readable": true_or_false}.')


def set_video_budget(processor, tokens):
    vp = processor.video_processor
    original = dict(vp.size)
    if tokens is None:
        return original
    if type(tokens) is not int or not 256 <= tokens <= 8192:
        raise ValueError("Video token cap must be an integer between 256 and 8192")
    budget = {**original, "longest_edge": tokens * (vp.patch_size * vp.merge_size)**2 * vp.temporal_patch_size}
    if budget["longest_edge"] < budget["shortest_edge"]:
        raise ValueError("Requested video budget is below processor minimum")
    vp.size = budget
    return original


def model_provenance(model_dir):
    root = Path(model_dir).resolve()
    if not (root / "config.json").is_file():
        raise FileNotFoundError("Local Qwen config.json is missing")
    config = json.loads((root / "config.json").read_text())
    if config.get("model_type") != MODEL_TYPE:
        raise ValueError("Expected Qwen3-VL model_type=qwen3_vl; select Qwen3-VL-8B-Instruct")
    files = sorted(root.glob("*.safetensors"))
    if not files:
        raise FileNotFoundError("Qwen requires local safetensors weights")
    checksum = hashlib.sha256()
    assets = sorted({p for pattern in ("*.json", "*.jinja", "*.txt", "*.model") for p in root.glob(pattern)})
    for path in files + assets:
        checksum.update(path.name.encode())
        with path.open("rb") as handle:
            for chunk in iter(lambda: handle.read(1024 * 1024), b""):
                checksum.update(chunk)
    contract = json.dumps({"architecture": MODEL_TYPE, "image_budget": IMAGE_BUDGET,
                           "video_budget": VIDEO_BUDGET, "attention": "sdpa",
                           "dtype": "bf16_if_supported_else_fp16", "max_image_output": 128}, sort_keys=True)
    return {"backend": "qwen_vl", "architecture": MODEL_TYPE, "model_sha256": checksum.hexdigest(),
            "prompt_sha256": hashlib.sha256(PROMPT.encode()).hexdigest(),
            "inference_contract_sha256": hashlib.sha256(contract.encode()).hexdigest()}


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


def load_processor(model_dir):
    from transformers import AutoProcessor
    processor = AutoProcessor.from_pretrained(str(model_dir), local_files_only=True,
        trust_remote_code=False, use_fast=True)
    if processor.video_processor is None:
        raise ValueError("Qwen3-VL video processor is missing; download the complete snapshot")
    processor.image_processor.size = dict(IMAGE_BUDGET)
    processor.video_processor.size = dict(VIDEO_BUDGET)
    return processor


def ensure_gpu_idle():
    """Deployment smoke checks must not compete with another GPU job."""
    import os
    import subprocess
    result = subprocess.run(
        ["nvidia-smi", "--query-compute-apps=pid,process_name", "--format=csv,noheader"],
        capture_output=True, text=True, check=True)
    busy = []
    for line in result.stdout.splitlines():
        if not line.strip():
            continue
        pid, separator, name = line.partition(",")
        if not separator or not pid.strip().isdigit():
            raise RuntimeError("Cannot interpret GPU process query; inspect nvidia-smi first")
        if int(pid) != os.getpid():
            busy.append(f"PID {pid.strip()} ({name.strip()})")
    if busy:
        raise RuntimeError("GPU is busy; wait for current tests to finish before Qwen smoke: " + "; ".join(busy))


class QwenBackend:
    def __init__(self, model_dir, *, device="cuda"):
        import torch
        from transformers import Qwen3VLForConditionalGeneration
        cuda = torch.device(device).type == "cuda"
        if cuda and not torch.cuda.is_available():
            raise RuntimeError("Qwen CUDA inference requires a GPU")
        self.provenance = model_provenance(model_dir)
        self.processor = load_processor(model_dir)
        dtype = (torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16) if cuda else torch.float32
        self.model = Qwen3VLForConditionalGeneration.from_pretrained(str(model_dir),
            local_files_only=True, trust_remote_code=False, use_safetensors=True,
            dtype=dtype, attn_implementation="sdpa").to(device).eval()
        self.device = device

    def _generate(self, inputs, max_new_tokens=128):
        import torch
        inputs = inputs.to(self.device)
        with torch.inference_mode():
            tokens = self.model.generate(**inputs, max_new_tokens=max_new_tokens, do_sample=False)
        return self.processor.batch_decode(tokens[:, inputs.input_ids.shape[1]:],
            skip_special_tokens=True, clean_up_tokenization_spaces=False)[0]

    def read(self, image):
        messages = [{"role": "user", "content": [
            {"type": "image", "image": image}, {"type": "text", "text": PROMPT}]}]
        text = self.processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        inputs = self.processor(text=[text], images=[image], padding=True, return_tensors="pt")
        return self._generate(inputs)

    def assess_player(self, image):
        from pipeline.identity.player_evidence import player_prompt
        messages = [{"role": "user", "content": [
            {"type": "image", "image": image}, {"type": "text", "text": player_prompt(image.size)}]}]
        text = self.processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        inputs = self.processor(text=[text], images=[image], padding=True, return_tensors="pt")
        return self._generate(inputs, max_new_tokens=256)

    def read_video(self, path, prompt, *, num_frames=8, max_new_tokens=128):
        """Infer one local window. Times in the response refer to this local clip.

        The pinned official processor samples frames and carries decoder FPS/frame
        indices into timestamp tokens. The segmentation service controls windows
        and validates temporal responses separately.
        """
        if isinstance(num_frames, bool) or not isinstance(num_frames, int) or not 4 <= num_frames <= 64 or num_frames % 2:
            raise ValueError("num_frames must be an even integer from 4 to 64")
        path = Path(path).expanduser().resolve()
        if not path.is_file():
            raise FileNotFoundError(path)
        if not isinstance(prompt, str) or not prompt.strip():
            raise ValueError("A nonempty video prompt is required")
        if isinstance(max_new_tokens, bool) or not isinstance(max_new_tokens, int) or not 32 <= max_new_tokens <= 4096:
            raise ValueError("max_new_tokens must be an integer from 32 to 4096")
        messages = [{"role": "user", "content": [
            {"type": "video", "video": str(path)}, {"type": "text", "text": prompt}]}]
        inputs = self.processor.apply_chat_template(messages, tokenize=True, add_generation_prompt=True,
            return_dict=True, return_tensors="pt", num_frames=num_frames, fps=None)
        return self._generate(inputs, max_new_tokens=max_new_tokens)
