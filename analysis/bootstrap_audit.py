"""CPU-only audit of immutable photo evidence and recorded identity decisions."""

from collections import Counter
from contextlib import closing
import csv
import html
import json
from pathlib import Path
import sqlite3
import shutil
import tarfile
from urllib.parse import quote

import numpy as np
from PIL import Image, ImageOps

from pipeline.identity.cross_clip import part_distances
from pipeline.media import atomic_json, sha256_file
from workflows.bootstrap import verified_photos


def constrained_groups(distances, frame_ids, threshold):
    """Complete-link hypotheses. Never merge distinct detections from one photo.

    This is a diagnostic alternative, not inferred ground truth or a DB update.
    """
    if not np.isfinite(threshold) or not 0 <= threshold <= 1:
        raise ValueError("Distance threshold must be in [0,1]")
    count = len(frame_ids)
    matrix = np.array(distances, dtype=float, copy=True)
    if matrix.shape != (count, count):
        raise ValueError("Distance/photo shape mismatch")
    for i in range(count):
        for j in range(count):
            if frame_ids[i] == frame_ids[j]:
                matrix[i,j] = np.inf
    groups = {i: [i] for i in range(count)}
    while groups:
        i,j = np.unravel_index(np.argmin(matrix), matrix.shape)
        if not np.isfinite(matrix[i,j]) or matrix[i,j] > threshold:
            break
        groups[i].extend(groups.pop(j))
        merged = np.maximum(matrix[i], matrix[j])
        matrix[i,:], matrix[:,i] = merged, merged
        matrix[j,:] = matrix[:,j] = np.inf
        matrix[i,i] = np.inf
    return sorted((sorted(g) for g in groups.values()), key=lambda g: (-len(g),g))


def quantiles(values):
    values = np.asarray(values, dtype=float)
    values = values[np.isfinite(values)]
    return {name: float(np.quantile(values,q)) if len(values) else None
            for name,q in (("min",0),("p10",.1),("median",.5),("p90",.9),("max",1))}


def decision_reasons(payload):
    if "memory" in payload:
        return list(payload["memory"].get("reasons", []))
    if payload.get("decision") != "pending":
        return []
    evidence = payload.get("match_evidence", {})
    candidates = evidence.get("candidates", [])
    comparable = evidence.get("best_comparable_distance")
    reasons = []
    if comparable is None:
        reasons.append("no_comparable_registered_gallery")
    elif comparable < evidence.get("novelty_distance",.5):
        reasons.append("novelty_threshold_not_reached")
    if not candidates:
        reasons.append("no_nonconflicting_registered_candidate")
    else:
        best = candidates[0]["distance"]
        if best > evidence.get("match_distance",.2):
            reasons.append("match_threshold_exceeded")
        elif len(candidates)>1 and candidates[1]["distance"]-best < evidence.get("min_margin",.05):
            reasons.append("nearest_player_margin_too_small")
    return reasons + [r for r in payload.get("hold_reasons", []) if r != "ambiguous_or_uncomparable_identity"]


def csv_file(path, rows, fields):
    with path.open("w",encoding="utf-8-sig",newline="") as stream:
        writer=csv.DictWriter(stream,fields,extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def audit(run_dir, output_dir, *, neighbors=5):
    run, output = Path(run_dir).resolve(), Path(output_dir).resolve()
    if run == output or run.is_relative_to(output) or output.is_relative_to(run):
        raise ValueError("Use a diagnostic directory separate from the result tree")
    if output.exists() and any(output.iterdir()):
        raise FileExistsError("Use a new empty diagnostic directory")
    if not 1 <= neighbors <= 12:
        raise ValueError("Choose 1–12 nearest neighbors")
    seed = run / "bootstrap"
    photos = verified_photos(run)
    contract = json.loads((seed/"feature_contract.json").read_text())
    numbers = json.loads((seed/"numbers.json").read_text())
    if (contract["photos_sha256"] != sha256_file(seed/"photos.json")
            or contract["features_sha256"] != sha256_file(seed/"features.npz")
            or numbers.get("status") != "completed"
            or numbers["signature"]["photos_sha256"] != contract["photos_sha256"]):
        raise ValueError("Bootstrap evidence hashes/status do not match")
    samples = photos["samples"]
    with np.load(seed/"features.npz",allow_pickle=False) as data:
        vectors, visible = data["embeddings"].astype(np.float32), data["visibility_scores"].astype(np.float32)
    if len(vectors) != len(samples) or len({s["sample_id"] for s in samples}) != len(samples):
        raise ValueError("Photo/feature sample ids or counts do not match")
    matrix = part_distances(vectors,visible,min_common_parts=2)
    database = run/"library/players.sqlite3"
    if database.is_symlink():
        raise ValueError("Database must not be a symlink")
    with closing(sqlite3.connect("file:"+quote(str(database),safe="/")+"?mode=ro",uri=True)) as connection:
        connection.row_factory=sqlite3.Row
        connection.execute("BEGIN")
        metadata=dict(connection.execute("SELECT key,value FROM metadata"))
        feature_contract = json.loads(metadata["feature_contract"]) if metadata.get("feature_contract") else None
        expected_contract = {key:contract[key] for key in ("metadata","shape")} if samples else None
        if feature_contract != expected_contract:
            raise ValueError("Registered KPR contract differs from photo features")
        records={r["id"]:dict(r) for r in connection.execute("SELECT * FROM observations")}
        players={r["id"]:dict(r) for r in connection.execute("SELECT * FROM players")}
        history=[dict(r) for r in connection.execute("SELECT * FROM history ORDER BY created_at,rowid")]
        memory = ({r["id"]:json.loads(r["data"]) for r in connection.execute("SELECT * FROM memory_state")}
                  if connection.execute("SELECT 1 FROM sqlite_master WHERE type='table' AND name='memory_state'").fetchone() else {})
    absent=[s["sample_id"] for s in samples if s["sample_id"] not in records]
    if absent:
        raise ValueError("Some bootstrap observations are not registered; finish initialization first")
    frame_ids=[s["frame_id"] for s in samples]
    cross=matrix.copy()
    for i in range(len(samples)):
        cross[i,[j for j in range(len(samples)) if frame_ids[j]==frame_ids[i]]] = np.inf
    rows, nearest, reasons = [], [], Counter()
    for i,sample in enumerate(samples):
        record=records[sample["sample_id"]]
        payload=json.loads(record["payload"])
        state = memory.get(sample["sample_id"])
        if state:
            payload.update(memory=state,decision=state["decision"],hold_reasons=state["reasons"])
        evidence=payload.get("match_evidence",{})
        ranked=[int(j) for j in np.argsort(cross[i],kind="stable") if np.isfinite(cross[i,j])][:neighbors]
        nearest.append(ranked)
        blocking=decision_reasons(payload)
        reasons.update(blocking)
        candidates=evidence.get("candidates",[])
        reading=numbers["readings"].get(sample["sample_id"],{})
        rows.append({"sample_index":i,"sample_id":sample["sample_id"],"frame_id":sample["frame_id"],
            "timestamp_s":sample["timestamp_s"],"player_id":record["player"],
            "player_status":players[record["player"]]["status"],"decision":payload.get("decision"),
            "gallery_eligible":bool(record["gallery"] and not record["detached"]),
            "memory_status":state.get("status") if state else None,
            "memory_quality":state.get("quality") if state else None,
            "gallery_template_count":len(state.get("templates",[])) if state else None,
            "box_width":sample["bbox_xyxy"][2]-sample["bbox_xyxy"][0],
            "box_height":sample["bbox_xyxy"][3]-sample["bbox_xyxy"][1],
            "det_score":sample["det_score"],"sharpness":sample["sharpness"],
            "visible_parts":int(np.count_nonzero(visible[i]>0)),
            "single_frame_number":reading.get("number"),"raw_number_response":reading.get("raw_response"),
            "recorded_best_distance":candidates[0]["distance"] if candidates else None,
            "recorded_best_comparable_distance":evidence.get("best_comparable_distance"),
            "recorded_min_margin":evidence.get("min_margin"),
            "recorded_match_distance":evidence.get("match_distance"),
            "recorded_novelty_distance":evidence.get("novelty_distance"),
            "recorded_margin":candidates[1]["distance"]-candidates[0]["distance"] if len(candidates)>1 else None,
            "nearest_cross_frame_sample":samples[ranked[0]]["sample_id"] if ranked else None,
            "nearest_cross_frame_distance":float(cross[i,ranked[0]]) if ranked else None,
            "blocking_reasons":";".join(blocking)})
    thresholds=(.1,.15,.2,.25,.3,.35,.4,.5)
    hypotheses={str(t):constrained_groups(matrix,frame_ids,t) for t in thresholds}
    same=[matrix[i,j] for i in range(len(samples)) for j in range(i) if frame_ids[i]==frame_ids[j]]
    different=[matrix[i,j] for i in range(len(samples)) for j in range(i) if frame_ids[i]!=frame_ids[j]]
    close_pending=sum(r["decision"]=="pending" and any(
        rows[j]["decision"]=="pending" and float(cross[i,j])<=.2 for j in range(len(samples)))
        for i,r in enumerate(rows))
    report={"status":"completed_not_accuracy_evaluated","accuracy_evaluated":False,
        "match_id":metadata.get("match_id"),"registry_revision":metadata.get("revision"),
        "memory_policy":json.loads(metadata["memory_policy"]) if metadata.get("memory_policy") else None,
        "source":photos["protocol"]["source"],"feature_contract":contract["metadata"],
        "photo_count":len(photos["frames"]),"crop_count":len(samples),
        "empty_photo_count":sum(not any(s["frame_id"]==f["frame_id"] for s in samples) for f in photos["frames"]),
        "bootstrap_identity_group_count":len({r["player_id"] for r in rows}),
        "status_counts":dict(Counter(players[pid]["status"] for pid in {r["player_id"] for r in rows})),
        "decision_counts":dict(Counter(r["decision"] for r in rows)),
        "eligible_gallery_observations":sum(r["gallery_eligible"] for r in rows),
        "visible_part_counts":dict(Counter(r["visible_parts"] for r in rows)),
        "pending_gate_counts":dict(reasons),
        "pending_with_other_cross_frame_pending_within_0_2":close_pending,
        "distance_distributions":{"same_photo_distinct_detections":quantiles(same),
            "all_cross_photo_pairs":quantiles(different),
            "cross_photo_nearest":quantiles([r["nearest_cross_frame_distance"] for r in rows if r["nearest_cross_frame_distance"] is not None])},
        "single_frame_number_read_count":sum(r["single_frame_number"] is not None for r in rows),
        "threshold_hypotheses":[{"threshold":t,"groups":len(hypotheses[str(t)]),
            "singleton_groups":sum(len(g)==1 for g in hypotheses[str(t)]),
            "groups_with_multiple_photos":sum(len(g)>1 for g in hypotheses[str(t)])} for t in thresholds],
        "warning":"Neighbor similarity and hypothetical group counts are not identity accuracy. Same-photo cannot-links are protocol constraints, not manually verified identity labels. The database is never modified."}
    output.mkdir(parents=True,exist_ok=True)
    evidence_dir=output/"evidence";evidence_dir.mkdir()
    for name in ("photos.json","feature_contract.json","features.npz","numbers.json"):
        shutil.copy2(seed/name,evidence_dir/name)
    media=output/"images";media.mkdir()
    for i,sample in enumerate(samples):
        with Image.open(seed/sample["crop_path"]) as image:
            thumbnail=ImageOps.contain(image.convert("RGB"),(160,220))
            thumbnail.save(media/f"{i:04d}.jpg",quality=85)
    atomic_json(output/"summary.json",report)
    atomic_json(output/"observations.json",rows)
    atomic_json(output/"group-hypotheses.json",{"accuracy_evaluated":False,"groups":hypotheses})
    atomic_json(output/"registry-evidence.json",{"metadata":metadata,"history":history,"memory_state":memory,
        "observations":[{**records[s["sample_id"]],"payload":json.loads(records[s["sample_id"]]["payload"])} for s in samples]})
    np.savez_compressed(output/"distances.npz",distances=matrix)
    csv_file(output/"observations.csv",rows,list(rows[0]) if rows else ["sample_id"])
    pairs=[]
    for i in range(len(samples)):
        for j in range(i):
            pair={"sample_a":samples[i]["sample_id"],"sample_b":samples[j]["sample_id"],
                "same_photo":frame_ids[i]==frame_ids[j],"distance":float(matrix[i,j]) if np.isfinite(matrix[i,j]) else None,
                "common_visible_parts":int(np.count_nonzero((visible[i]>0)&(visible[j]>0)))}
            pair.update({f"part_{p}_distance":float(np.linalg.norm(vectors[i,p]-vectors[j,p])/2)
                if visible[i,p]>0 and visible[j,p]>0 else None for p in range(vectors.shape[1])})
            pairs.append(pair)
    csv_file(output/"pairs.csv",pairs,list(pairs[0]) if pairs else ["sample_a","sample_b","distance"])
    def card(i, caption=""):
        row=rows[i]
        return f'<article><img loading="lazy" src="images/{i:04d}.jpg"><small>{html.escape(row["sample_id"])}<br>{row["timestamp_s"]:.2f}s · {html.escape(row["decision"] or "unknown")}<br>{html.escape(row["player_id"])}<br>原裁剪 {row["box_width"]}×{row["box_height"]} · 可见部位 {row["visible_parts"]}<br>number={html.escape(str(row["single_frame_number"]))}<br>{html.escape(caption)}</small></article>'
    neighbor_sections=[]
    for i in range(len(samples)):
        neighbor_sections.append('<section><h3>查询 '+html.escape(samples[i]["sample_id"])+'</h3><div class="strip">'+
            card(i,"查询图")+''.join(card(j,f'd={cross[i,j]:.4f}; '+('gallery' if rows[j]["gallery_eligible"] else 'held')) for j in nearest[i])+ '</div></section>')
    groups_html=[]
    for threshold,groups in hypotheses.items():
        groups_html.append(f'<div class="hypothesis" data-threshold="{threshold}" '+('' if threshold=='0.2' else 'hidden')+'>'+''.join(
            f'<section><h3>假设组 {k+1} · {len(g)} 张</h3><div class="strip">'+''.join(card(i) for i in g)+'</div></section>' for k,g in enumerate(groups))+'</div>')
    style='body{font:15px system-ui;margin:24px;color:#20343d;background:#f3f6f5}section{background:white;border:1px solid #d6e2dd;border-radius:12px;padding:14px;margin:16px 0}.strip{display:flex;gap:12px;overflow:auto}article{width:170px;flex:none}img{height:220px;max-width:160px;object-fit:contain}small{display:block;overflow-wrap:anywhere}pre{white-space:pre-wrap}button,select{padding:8px;margin:8px}#photos{display:grid;grid-template-columns:repeat(auto-fill,170px);gap:12px}[hidden]{display:none!important}'
    document='<!doctype html><html lang="zh-CN"><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1"><title>初始化图库只读诊断</title><style>'+style+'</style><h1>初始化图库只读诊断</h1><p>不修改球员库，不调用 GPU。相似近邻和以下分组是假设，不是识别准确率；不要为了得到十个组直接提高阈值。</p><nav><a href="#summary">统计</a> · <a href="#photos">全部人物裁剪</a> · <a href="#neighbors">跨帧近邻</a> · <a href="#groups">约束分组对照</a></nav><section id="summary"><h2>统计与拒绝原因</h2><pre>'+html.escape(json.dumps(report,ensure_ascii=False,indent=2))+'</pre></section><h2>按抽帧顺序查看裁剪</h2><div id="photos">'+''.join(card(i,rows[i]["blocking_reasons"]) for i in range(len(samples)))+'</div><h2 id="neighbors">每张图的跨帧近邻</h2><p>重点检查：同人是否排在前列；相似球衣是否混淆；held 相似观察是否没有机会参与原图库匹配。</p>'+''.join(neighbor_sections)+'<h2 id="groups">完整连接分组对照</h2><p>组内任意两图距离均不超过阈值，同一张照片的不同检测禁止同组。没有使用号码强制合并；这是诊断对照，不写入原库。</p><label>距离阈值 <select id="threshold">'+''.join(f'<option value="{t}" '+('selected' if t==.2 else '')+f'>{t}</option>' for t in thresholds)+'</select></label>'+''.join(groups_html)+'<script>document.getElementById("threshold").addEventListener("change",e=>{document.querySelectorAll(".hypothesis").forEach(el=>el.hidden=el.dataset.threshold!==e.target.value)})</script></html>'
    (output/"index.html").write_text(document,encoding="utf-8")
    atomic_json(output/"input-hashes.json",{name:sha256_file(seed/name) for name in
        ("photos.json","feature_contract.json","features.npz","numbers.json")})
    archive=output.parent/(output.name+".tar.gz")
    if archive.exists():
        raise FileExistsError("Audit archive already exists")
    with tarfile.open(archive,"w:gz") as package:
        package.add(output,arcname="bootstrap-audit")
    return report,{"html":str(output/"index.html"),"archive":str(archive),"archive_mib":round(archive.stat().st_size/1024**2,2)}
