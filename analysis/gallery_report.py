"""Portable thumbnail report for CPU memory previews; no ground-truth claims."""
import hashlib
import html
from pathlib import Path

from PIL import Image
from workflows.library_inputs import safe_relative


def write_report(root, snapshot, output):
    output = Path(output)
    images = output / "review_images"
    images.mkdir(exist_ok=True)
    sections = []
    for person in sorted(snapshot["people"], key=lambda p: (p["status"] != "candidate", -len(p["members"]),p["player_id"])):
        cards = []
        for node in person["members"]:
            source = Path(root) / ("bootstrap" if node.get("evidence_kind") == "bootstrap_photo" else node["clip_name"]+"/identity")
            relative = (node.get("cover") or {}).get("crop_path")
            image_html = ""
            if relative:
                image = safe_relative(source, relative)
                # Names derive from a digest rather than external node text.
                name = hashlib.sha256(node["node_id"].encode()).hexdigest()[:24]+".jpg"
                with Image.open(image) as im:
                    im = im.convert("RGB"); im.thumbnail((160,220)); im.save(images/name, quality=85)
                image_html = f'<img loading="lazy" src="review_images/{name}">'
            state = node.get("memory", {})
            text = f'{node["node_id"]} | {node.get("timestamp_s",node["clip_name"])} | {state.get("status","legacy")} | templates={len(state.get("templates",[]))} | {", ".join(state.get("reasons",[]))}'
            cards.append(f'<figure>{image_html}<figcaption>{html.escape(text)}</figcaption></figure>')
        sections.append(f'<section><h2>{html.escape(person["player_id"])} · {html.escape(person["status"])} · {person.get("gallery_template_count",0)} templates</h2><div class="cards">{"".join(cards)}</div></section>')
    (output/"index.html").write_text('<!doctype html><html lang="zh-CN"><meta charset="utf-8"><title>球员图库对照</title><style>body{font:16px sans-serif;margin:24px;background:#f6f7f8}section{background:white;padding:16px;margin:16px 0}.cards{display:flex;flex-wrap:wrap;gap:12px}figure{width:175px;margin:0}img{height:220px;max-width:160px;object-fit:contain}figcaption{overflow-wrap:anywhere;font-size:12px}h2{font-size:16px}</style><h1>球员图库候选与待核验观察</h1><p>模板数量、分组数量不代表身份准确率。这里只使用已有模型结果，不运行 GPU 模型。</p>'+"".join(sections)+'</html>',encoding="utf-8")
