"""Independent, bounded review of experimental state-transition candidates."""

import json
from pathlib import Path
import re
import time

from analysis.possession_states import TEAMS
from analysis.segmentation_diagnostic import save_video_inputs, set_video_budget
from pipeline.media import atomic_json, cut_video, sha256_file
from pipeline.segmentation.service import number, parse_json


def review_prompt(duration, candidate):
    return f"""独立检查这段篮球视频是否在指定范围内发生一次真实球权转换。
只根据当前视频观察，不使用先前模型回答或比赛计时钟。局部视频长度 {duration:.3f} 秒，
待核查范围为局部 {candidate[0]:.3f} 至 {candidate[1]:.3f} 秒。范围只是线索，不代表确实有转换。
先查看整个视频，分别说明转换前后的可见控球证据，不推断球员号码、比分或球队名称。
dark 表示深色球衣，light 表示浅色球衣，unknown 表示看不清或无法区分。
球权转换需要对手实际取得控制，例如防守篮板、抢断后控球，或得分后对方发球取得控制。
出手、哨声、犯规、进入罚球站位、裁判持球或进攻篮板本身都不能证明换球权。
不能只因球员跑动方向或镜头移动就确认。片头片尾同队也不排除中间多次转换，应检查全段。
decision 只能为 confirmed、rejected、uncertain。
confirmed：可确认一次由一方到对手的控制转换，before_team 和 after_team 必须为不同的已知方。
此时 time_range_s 为包含转换的局部时间范围 [开始秒,结束秒]，必须在待核查范围内。
给范围而非假装精确到某一帧；依据可见帧及时间标签，最后可见旧控制到首次可见新控制之间。
rejected：可以确定指定范围内没有控制转换，前后都是同一已知方，time_range_s 为 null。
uncertain：控球看不清、多次转换、无法把范围缩小到待核查区间，或证据互相矛盾；time_range_s 为 null。
before_evidence、after_evidence 分别简短说明前后可见证据，reason 简短说明决定。
只返回一个 JSON 对象，字段必须是 decision、before_team、after_team、time_range_s、
before_evidence、after_evidence、reason。before_team 和 after_team 只能为 dark、light、unknown。
"""


def parse_review(raw, duration, candidate):
    value = parse_json(raw)
    decision = value.get("decision")
    before, after = value.get("before_team"), value.get("after_team")
    if (not isinstance(decision, str) or decision not in {"confirmed", "rejected", "uncertain"}
            or not isinstance(before, str) or before not in TEAMS
            or not isinstance(after, str) or after not in TEAMS):
        raise ValueError("Invalid review decision or team labels")
    result = {key: value[key] for key in ("decision", "before_team", "after_team")}
    for key in ("before_evidence", "after_evidence", "reason"):
        text = value.get(key)
        if not isinstance(text, str) or not text.strip() or len(text) > 1000:
            raise ValueError("Review requires concise visible before/after evidence and a reason")
        result[key] = text.strip()
    if "time_range_s" not in value:
        raise ValueError("Review must explicitly include time_range_s")
    bounds = value["time_range_s"]
    if decision == "confirmed":
        if before == after or "unknown" in {before, after}:
            raise ValueError("Confirmed possession change requires two distinct known teams")
        if not isinstance(bounds, list) or len(bounds) != 2:
            raise ValueError("Confirmed change needs a time interval")
        low, high = map(number, bounds)
        if not 0 <= candidate[0] <= low < high <= candidate[1] <= duration:
            raise ValueError("Confirmed interval must stay within the proposed candidate")
        result["time_range_s"] = [low, high]
    else:
        if bounds is not None:
            raise ValueError("Rejected or uncertain reviews must have null time_range_s")
        if decision == "rejected" and (before != after or before == "unknown"):
            raise ValueError("Rejected review needs the same known team before and after")
        result["time_range_s"] = None
    # Model confirmation is never a human-verified boundary or permission to cut.
    return {**result, "needs_human_review": True, "accuracy_evaluated": False}


def review_plan(diagnostic, padding_s=2):
    padding_s = number(padding_s)
    if padding_s < 0:
        raise ValueError("Review padding must be nonnegative")
    duration = number(diagnostic["source"]["duration_s"])
    result = []
    for window in diagnostic["results"]:
        if not re.fullmatch(r"coarse-\d{6}", window["window_id"]):
            raise ValueError("Invalid state window id")
        query = window.get("queries", {}).get("state", {})
        if "parse_error" in query:
            continue
        for index, candidate in enumerate(query.get("derived_candidates", [])):
            low, high = map(number, candidate["source_time_range_s"])
            if not 0 <= window["start_s"] <= low < high <= window["end_s"] <= duration:
                raise ValueError("State candidate outside source video")
            start, end = max(0, low - padding_s), min(duration, high + padding_s)
            result.append({"candidate_id": f"{window['window_id']}-{index:03d}",
                "source_candidate_range_s": [low, high], "source_review_range_s": [start, end],
                "local_candidate_range_s": [low - start, high - start]})
    return result


def review_candidates(diagnostic, candidates, output, backend, *, frames=16, video_tokens=8192,
                      ffmpeg="ffmpeg", ffprobe="ffprobe", capture_inputs=True):
    output = Path(output).resolve()
    source = diagnostic["source"]
    if backend.provenance != diagnostic["model"]:
        raise ValueError("Review model differs from state diagnostic")
    if sha256_file(source["path"]) != source["sha256"]:
        raise ValueError("Original video changed after state diagnostic")
    if Path(source["path"]).resolve().is_relative_to(output):
        raise ValueError("Review output cannot contain source video")
    if output.exists() and any(output.iterdir()):
        raise FileExistsError("Use a new, empty review directory")
    if isinstance(frames, bool) or not isinstance(frames, int) or frames % 2 or not 4 <= frames <= 64:
        raise ValueError("Review frame count must be an even integer from 4 to 64")
    output.mkdir(parents=True, exist_ok=True)
    processor = getattr(backend, "processor", None)
    if video_tokens is not None and processor is None:
        raise ValueError("Setting a review budget requires a processor")
    original_budget = set_video_budget(processor, video_tokens) if processor else None
    result = {"status": "running", "accuracy_evaluated": False, "source": source,
        "model": backend.provenance, "requested_frames": frames, "requested_video_tokens": video_tokens,
        "effective_video_budget": dict(processor.video_processor.size) if processor else None,
        "review_sha256": sha256_file(Path(__file__)), "results": []}
    path = output / "candidate-review.json"
    atomic_json(path, result)
    try:
        for candidate in candidates:
            folder = output / candidate["candidate_id"]
            clip = folder / "window.mp4"
            start, end = candidate["source_review_range_s"]
            cut_video(source["path"], clip, start, end, metadata=source,
                      ffmpeg=ffmpeg, ffprobe=ffprobe, preview=True)
            prompt = review_prompt(end - start, candidate["local_candidate_range_s"])
            record = {**candidate, "prompt": prompt, "num_frames": frames, "max_new_tokens": 1024}
            result["results"].append(record)
            atomic_json(path, result)
            print(f"[review] {candidate['candidate_id']} source {start:.2f}–{end:.2f}s", flush=True)
            original_generate = backend._generate if capture_inputs else None
            if capture_inputs:
                def generate(inputs, max_new_tokens=128):
                    record["video_input"] = save_video_inputs(backend.processor, inputs, folder / "input")
                    return original_generate(inputs, max_new_tokens=max_new_tokens)
                backend._generate = generate
            started = time.monotonic()
            try:
                record["raw_response"] = backend.read_video(clip, prompt, num_frames=frames, max_new_tokens=1024)
            except Exception as exc:
                record["error"] = str(exc)
                raise
            finally:
                record["elapsed_s"] = round(time.monotonic() - started, 3)
                if capture_inputs:
                    backend._generate = original_generate
                atomic_json(path, result)
            try:
                record["review"] = parse_review(record["raw_response"], end - start, candidate["local_candidate_range_s"])
                bounds = record["review"]["time_range_s"]
                record["source_transition_range_s"] = [start + t for t in bounds] if bounds is not None else None
            except (ValueError, TypeError, KeyError) as exc:
                record["parse_error"] = str(exc)
            atomic_json(folder / "response.json", record)
            atomic_json(path, result)
            print(f"[{candidate['candidate_id']}/review] {record['raw_response']}", flush=True)
            print("[review result] " + json.dumps(record.get("review", {"parse_error": record.get("parse_error")})
                | {"source_transition_range_s": record.get("source_transition_range_s")}, ensure_ascii=False), flush=True)
        result["status"] = "completed_not_accuracy_evaluated"
    except Exception as exc:
        result.update(status="failed", error=str(exc))
        raise
    finally:
        if processor:
            processor.video_processor.size = original_budget
        atomic_json(path, result)
    return result
