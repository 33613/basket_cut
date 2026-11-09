"""Presentation rendering: one reviewed shooter and observed ball trails only."""

from collections import deque
import html
import json
from pathlib import Path
import subprocess

from contracts.schema import read_jsonl
from pipeline.media import atomic_json, sha256_file
from workflows.player_events import export_player_events, query_player_events


def draw(frame, target, ball, trail, *, frame_idx, fps, label):
    import cv2
    if target:
        x1, y1, x2, y2 = map(round, target['bbox_xyxy'])
        cv2.rectangle(frame, (x1, y1), (x2, y2), (40, 220, 40), 3)
        cv2.putText(frame, label, (max(0, x1), max(22, y1-8)),
                    cv2.FONT_HERSHEY_SIMPLEX, .7, (40, 220, 40), 2)
    point = ball.get('ball_track') if ball else None
    valid = point and point.get('observed') and point.get('confirmed', False)
    if valid:
        tid = point['ball_track_id']
        if trail and (trail[-1][1] != tid or frame_idx-trail[-1][0] > max(1, .12*fps)):
            trail.clear()
        center = tuple(round(x) for x in point['center_xy'])
        trail.append((frame_idx, tid, center))
    while trail and frame_idx-trail[0][0] > .8*fps:
        trail.popleft()
    for a, b in zip(trail, list(trail)[1:]):
        cv2.line(frame, a[2], b[2], (0, 230, 255), 2)
    if valid:
        cv2.circle(frame, center, 6, (0, 230, 255), 2)
    return frame


def render_clip(raw, output, item, target_path, *, label, ffmpeg='ffmpeg', ffprobe='ffprobe', width=1280, height=720):
    import cv2
    targets = {r['frame_idx']: r for r in read_jsonl(target_path)}
    balls = {}
    if item['ball_path']:
        if sha256_file(item['ball_path']) != item['ball_sha256']:
            raise ValueError('Source ball observations changed')
        for row in read_jsonl(item['ball_path']):
            f = row['source_frame_idx']
            if f in balls or abs(row['source_timestamp_s']-f/item['fps']) > 1e-6:
                raise ValueError('Ball source clock mismatch')
            balls[f] = row
    capture = cv2.VideoCapture(str(raw))
    if not capture.isOpened():
        raise ValueError(f'Cannot open raw clip: {raw}')
    w, h = int(capture.get(cv2.CAP_PROP_FRAME_WIDTH)), int(capture.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = float(capture.get(cv2.CAP_PROP_FPS))
    if abs(fps-item['fps']) > 1e-3:
        capture.release()
        raise ValueError('Exported clip FPS differs from source tracking clock')
    audio = subprocess.run([ffprobe, '-v', 'error', '-select_streams', 'a:0',
        '-show_entries', 'stream=index', '-of', 'csv=p=0', str(raw)],
        capture_output=True, text=True, check=True).stdout.strip()
    output = Path(output)
    temporary = output.with_suffix('.partial.mp4')
    command = [ffmpeg, '-nostdin', '-v', 'error', '-y', '-f', 'rawvideo', '-pix_fmt', 'bgr24',
        '-s', f'{w}x{h}', '-r', str(fps), '-i', 'pipe:0']
    command += ['-i', str(raw)] if audio else ['-f', 'lavfi', '-i', 'anullsrc=r=48000:cl=stereo']
    command += ['-map', '0:v:0', '-map', '1:a:0', '-vf',
        f'scale={width}:{height}:force_original_aspect_ratio=decrease,pad={width}:{height}:(ow-iw)/2:(oh-ih)/2,fps=30',
        '-c:v', 'libx264', '-preset', 'fast', '-crf', '20', '-threads', '2', '-pix_fmt', 'yuv420p',
        '-c:a', 'aac', '-ar', '48000', '-ac', '2', '-af', 'apad', '-shortest', '-movflags', '+faststart', str(temporary)]
    trail, count = deque(), 0
    # Diagnostics go to a file rather than a pipe that can deadlock the encoder.
    error_path = output.with_suffix('.ffmpeg.log')
    try:
        with error_path.open('w+') as errors:
            process = subprocess.Popen(command, stdin=subprocess.PIPE, stderr=errors)
            try:
                while True:
                    ok, frame = capture.read()
                    if not ok:
                        break
                    source_frame = round(item['clip_start_s']*fps)+count
                    frame = draw(frame, targets.get(source_frame), balls.get(source_frame), trail,
                                 frame_idx=source_frame, fps=fps, label=label)
                    process.stdin.write(frame.tobytes())
                    count += 1
                process.stdin.close()
                if process.wait() or not count:
                    errors.seek(0)
                    raise RuntimeError('Demo encode failed: '+errors.read()[-2000:])
            except BaseException:
                if process.poll() is None:
                    process.kill()
                    process.wait()
                raise
        temporary.replace(output)
    finally:
        capture.release()
        temporary.unlink(missing_ok=True)
    return count


def render_player_videos(root, *, player_id=None, ffmpeg='ffmpeg', ffprobe='ffprobe', width=1280, height=720):
    root = Path(root).resolve()
    plan = json.loads((root/'plan.json').read_text())
    items = {i['event_id']: i for i in plan['items']}
    people = list(read_jsonl(root/'library/people.jsonl'))
    if player_id and player_id not in {p['player_id'] for p in people}:
        raise ValueError('Unknown player ID')
    out = root/'videos'
    out.mkdir(exist_ok=True)
    results = []
    for person in people:
        pid = person['player_id']
        if player_id and pid != player_id:
            continue
        query = query_player_events(root, pid)
        if not query['items']:
            continue
        label = person.get('identity_label') or pid
        display = next((i['display_name'] for i in items.values() if i['player_key'] == label and i['display_name']), label)
        # OpenCV's built-in font supports ASCII. Full display names remain in HTML.
        overlay = label if label.isascii() else pid
        exported = export_player_events(root, pid, out/pid/'raw', media_source='original',
            before_s=0, after_s=0, ffmpeg=ffmpeg, ffprobe=ffprobe)
        by_id = {r['event_id']: r for r in query['items']}
        rendered = []
        def event_order(raw):
            item = items[by_id[raw['event_id']]['source_event_id']]
            vid = item['video_id']
            return (0, int(vid), item['clip_start_s']) if vid.isdigit() else (1, vid, item['clip_start_s'])
        for raw in sorted(exported['videos'], key=event_order):
            eid = by_id[raw['event_id']]['source_event_id']
            item = items[eid]
            target = out/pid/(eid+'.mp4')
            signature = dict(raw_sha256=raw['sha256'], label=overlay, width=width, height=height,
                targets=sha256_file(root/eid/'target_tracks.jsonl'), ball=item['ball_sha256'])
            marker = target.with_suffix('.json')
            old = json.loads(marker.read_text()) if marker.exists() else {}
            if not (old.get('signature') == signature and target.is_file() and old.get('sha256') == sha256_file(target)):
                render_clip(raw['path'], target, item, root/eid/'target_tracks.jsonl', label=overlay,
                            ffmpeg=ffmpeg, ffprobe=ffprobe, width=width, height=height)
                atomic_json(marker, dict(signature=signature, sha256=sha256_file(target)))
            rendered.append(target)
            print(f'[demo render] {label} video {item["video_id"]}', flush=True)
        # All clips have the same H.264/AAC format and output dimensions/rate.
        listing = out/pid/'concat.txt'
        listing.write_text(''.join("file '"+p.name+"'\n" for p in rendered))
        reel = out/(pid+'.mp4')
        temporary = reel.with_suffix('.partial.mp4')
        try:
            subprocess.run([ffmpeg, '-nostdin', '-v', 'error', '-y', '-f', 'concat', '-safe', '1',
                '-i', str(listing), '-c', 'copy', '-movflags', '+faststart', str(temporary)], check=True)
            temporary.replace(reel)
        finally:
            temporary.unlink(missing_ok=True)
        cover = person['members'][0]['cover'].get('crop_path')
        results.append(dict(player_id=pid, display_name=display, clips=len(rendered),
            video=reel.relative_to(root).as_posix(), cover=cover, sha256=sha256_file(reel)))
    combined = None
    if results:
        listing = out/'all-players-concat.txt'
        listing.write_text(''.join("file '"+Path(r['video']).name+"'\n" for r in results))
        reel = out/'all-players.mp4'
        temporary = out/'all-players.partial.mp4'
        try:
            subprocess.run([ffmpeg, '-nostdin', '-v', 'error', '-y', '-f', 'concat', '-safe', '1',
                '-i', str(listing), '-c', 'copy', '-movflags', '+faststart', str(temporary)], check=True)
            temporary.replace(reel)
        finally:
            temporary.unlink(missing_ok=True)
        combined = reel.relative_to(root).as_posix()
    atomic_json(root/'demo_videos.json', dict(players=results, combined=combined, resolution=[width, height], fps=30))
    cards = []
    for r in results:
        image = f'<img height="120" src="{html.escape(r["cover"], quote=True)}">' if r['cover'] else ''
        cards.append(f'<section>{image}<h2>{html.escape(r["display_name"])}</h2><p>{r["clips"]} 个进球片段</p>'
            f'<video controls preload="none" width="720" src="{r["video"]}"></video>'
            f'<p><a download href="{r["video"]}">下载球员集锦</a></p></section>')
    combined_link = f'<p><a download href="{combined}">下载按球员分组的完整集锦</a></p>' if combined else ''
    (root/'index.html').write_text('<!doctype html><meta charset="utf-8"><title>球员进球集锦</title>'
        '<style>body{font:16px sans-serif;margin:30px}section{padding:20px;border-bottom:1px solid #ddd}'
        'video{max-width:100%}</style><h1>球员进球集锦</h1>'+combined_link+''.join(cards))
    return dict(players=len(results), index=str(root/'index.html'), combined=combined, videos=results)
