"""Preserve TrackID3x3 MOT10 boxes and native video frame order."""
import csv
from collections import defaultdict
from fractions import Fraction
import json
import math
from pathlib import Path
import re

from analysis.teamtrack.dataset import save_json, sha256

SPLIT = 'fixed_camera'
SUBSETS = ('Indoor', 'Outdoor')


def contained(root, name, *, require_file=True):
    path = (root / name).resolve()
    if not path.is_relative_to(root) or (require_file and not path.is_file()):
        raise ValueError(f'Missing/unsafe dataset file: {name}')
    return path


def checksums(root):
    result = {}
    for line in (root / 'checksums.sha256').read_text(encoding='utf-8-sig').splitlines():
        if not line.strip():
            continue
        match = re.fullmatch(r'([0-9a-fA-F]{64}) [ *](.+)', line)
        if not match:
            raise ValueError('Invalid checksums.sha256 entry')
        path = contained(root, match[2], require_file=False)
        # Finder/AppleDouble metadata can disappear during cross-platform
        # copying. It is never an input to decoding, conversion or scoring.
        relative = path.relative_to(root)
        if path.name == '.DS_Store' or path.name.startswith('._') or '__MACOSX' in relative.parts:
            print(f'[prepare] Ignoring macOS metadata checksum entry: {relative}', flush=True)
            continue
        if not path.is_file():
            raise ValueError(f'Missing dataset checksum file: {match[2]}')
        if path in result:
            raise ValueError(f'Duplicate checksum: {path}')
        result[path] = match[1].lower()
    return result


def integer(value, label, minimum=0):
    value = float(value)
    if not math.isfinite(value) or value != int(value) or value < minimum:
        raise ValueError(f'Invalid {label}: {value}')
    return int(value)


def parse_gt(path, frames):
    grouped = defaultdict(list)
    seen, count = set(), 0
    with Path(path).open(encoding='utf-8-sig', newline='') as stream:
        for line, row in enumerate(csv.reader(stream), 1):
            if len(row) != 10:
                raise ValueError(f'{path}:{line}: expected ten MOT columns')
            values = list(map(float, row))
            if not all(map(math.isfinite, values)):
                raise ValueError(f'{path}:{line}: nonfinite GT')
            frame, identity = integer(values[0], 'GT frame', 1), integer(values[1], 'GT identity')
            x,y,w,h = values[2:6]
            if frame > frames or w <= 0 or h <= 0 or (frame,identity) in seen:
                raise ValueError(f'{path}:{line}: frame/duplicate ID/geometry mismatch')
            # Source TrackID MOT10 has four world/unused placeholders. Never
            # apply MOT17 confidence/class filters to -1 and erase the players.
            if values[6:] != [-1.,-1.,-1.,-1.]:
                raise ValueError(f'{path}:{line}: unexpected tail semantics; inspect before evaluating')
            grouped[frame].append((identity,x,y,w,h))
            seen.add((frame,identity)); count += 1
    if not count:
        raise ValueError(f'Empty GT: {path}')
    return grouped


def sources(root, *, subset='all', limit_per_view=0):
    root = Path(root).resolve()
    if subset not in ('all', *SUBSETS) or type(limit_per_view) is not int or limit_per_view < 0:
        raise ValueError('Invalid subset or limit_per_view')
    with (root / 'manifest.csv').open(encoding='utf-8-sig', newline='') as stream:
        rows = list(csv.DictReader(stream))
    declared = checksums(root)
    manifest = root / 'manifest.csv'
    if manifest in declared and sha256(manifest) != declared[manifest]:
        raise ValueError('manifest.csv checksum mismatch')
    seen = set()
    for row in rows:
        view, name = row['subset'], row['sequence']
        if view not in SUBSETS or not re.fullmatch(r'[A-Za-z0-9_-]+',name) or (view,name) in seen:
            raise ValueError('Invalid/duplicate TrackID sequence')
        seen.add((view,name))
        if row.get('check_status') != 'verified':
            raise ValueError(f'{view}/{name}: source inspection is not verified')
    selected=[]
    for view in SUBSETS:
        if subset not in ('all',view):
            continue
        group = [r for r in rows if r['subset']==view]
        # Smallest original full sequences for alignment smoke testing, never
        # truncate frames or pretend this selection is the official test split.
        group.sort(key=lambda r:(integer(r['actual_decoded_frames'],'frame count',1),r['sequence']))
        selected.extend(group[:limit_per_view] if limit_per_view else group)
    if not selected:
        raise ValueError('No selected sequences')
    result=[]
    for row in selected:
        video, annotation = (contained(root,row[k]) for k in ('video_path','gt_path'))
        if (not video.is_relative_to(root/'videos'/row['subset'])
                or not annotation.is_relative_to(root/'annotations')
                or video.stem != row['sequence'] or annotation.stem != row['sequence']):
            raise ValueError('Video/GT paths differ from declared sequence or view')
        if video == annotation:
            raise ValueError('Video and GT must be different files')
        hashes={}
        for label,path in (('video',video),('annotation',annotation)):
            if path not in declared or sha256(path) != declared[path]:
                raise ValueError(f'Source checksum mismatch or absent: {path}')
            hashes[label+'_sha256']=declared[path]
        result.append({'view':row['subset'],'source_sequence':row['sequence'],
            'sequence':row['subset']+'__'+row['sequence'],'video_path':str(video),
            'annotation_path':str(annotation),'frames':integer(row['actual_decoded_frames'],'frames',1),
            'width':integer(row['width'],'width',1),'height':integer(row['height'],'height',1),
            'source_manifest_row':row,**hashes})
    return result


def draw(frame, rows, *, color=(0,255,0), prefix='GT', scale=1.):
    import cv2
    for identity,x,y,w,h in rows:
        a,b=(round(x*scale),round(y*scale)),(round((x+w)*scale),round((y+h)*scale))
        cv2.rectangle(frame,a,b,color,2)
        cv2.putText(frame,f'{prefix} {identity}',a,cv2.FONT_HERSHEY_SIMPLEX,.55,color,2)


def prepare(root, output, *, subset='all', limit_per_view=0):
    import cv2
    root,output=Path(root).resolve(),Path(output).resolve()
    if output==root or output.is_relative_to(root) or root.is_relative_to(output):
        raise ValueError('Input and benchmark output trees must be separate')
    folder=output/SPLIT; folder.mkdir(parents=True,exist_ok=True)
    selected=sources(root,subset=subset,limit_per_view=limit_per_view)
    identity={'schema_version':1,'dataset':'TrackID3x3','scope':'annotated fixed-camera sequences; official test split not asserted',
        'root':str(root),'subset':subset,'limit_per_view':limit_per_view,'sources':selected,
        'source_manifest_sha256':sha256(root/'manifest.csv'),'checksum_list_sha256':sha256(root/'checksums.sha256'),
        'converter_sha256':sha256(__file__)}
    path=folder/'dataset.json'
    if path.is_file():
        old=json.loads(path.read_text())
        if any(old.get(k)!=v for k,v in identity.items()):
            raise ValueError('Dataset/selection/converter changed; use a new output directory')
        for seq in old['sequences']:
            if sha256(folder/'gt'/seq['sequence']/'gt/gt.txt')!=seq['gt_sha256']:
                raise ValueError('Prepared GT changed')
        print(f'Reusing verified TrackID dataset: {len(selected)} sequences',flush=True)
        return old
    sequences=[]
    for seq in selected:
        name,frames=seq['sequence'],seq['frames']
        grouped=parse_gt(seq['annotation_path'],frames)
        boxes=sum(map(len,grouped.values()))
        row=seq['source_manifest_row']
        observed={'gt_rows':boxes,'gt_annotated_frames':len(grouped),'identity_count':len({b[0] for a in grouped.values() for b in a})}
        for key,value in observed.items():
            if row.get(key) and integer(row[key],key)!=value:
                raise ValueError(f'{name}: {key} differs from uploaded inspection')
        capture=cv2.VideoCapture(seq['video_path'])
        decoded=0
        preview=folder/'alignment'/name; preview.mkdir(parents=True,exist_ok=True)
        targets={1,(frames+1)//2,frames}
        try:
            if not capture.isOpened():raise ValueError(f'Cannot decode {name}')
            fps=float(capture.get(cv2.CAP_PROP_FPS))
            rotation=float(capture.get(cv2.CAP_PROP_ORIENTATION_META))
            auto_rotation=bool(capture.get(cv2.CAP_PROP_ORIENTATION_AUTO))
            if not math.isfinite(fps) or fps<=0:raise ValueError('Invalid decoder FPS')
            while True:
                ok,image=capture.read()
                if not ok:break
                decoded+=1
                if image.shape[:2]!=(seq['height'],seq['width']):
                    raise ValueError(f'{name}: decoded dimensions/rotation differ from native GT')
                if decoded in targets:
                    scale=min(1.,1600/seq['width'])
                    canvas=cv2.resize(image,(round(seq['width']*scale),round(seq['height']*scale)))
                    draw(canvas,grouped.get(decoded,[]),scale=scale)
                    if not cv2.imwrite(str(preview/f'gt-{decoded:06d}.jpg'),canvas):raise OSError('Preview write failed')
                if decoded%1000==0:print(f'[prepare] {name}: {decoded}/{frames}',flush=True)
        finally:capture.release()
        if decoded!=frames:
            raise ValueError(f'{name}: decoded {decoded} != declared {frames}; no evaluation allowed')
        outside=sum(x<0 or y<0 or x+w>seq['width'] or y+h>seq['height'] for rows in grouped.values() for _,x,y,w,h in rows)
        gt=folder/'gt'/name/'gt/gt.txt'; gt.parent.mkdir(parents=True,exist_ok=True)
        with gt.open('w') as stream:
            for frame in sorted(grouped):
                for identity_id,x,y,w,h in grouped[frame]:
                    stream.write(f'{frame},{identity_id},{x},{y},{w},{h},1,1,1\n')
        (gt.parent.parent/'seqinfo.ini').write_text(f'[Sequence]\nname={name}\nimDir=img1\nframeRate={fps}\nseqLength={frames}\nimWidth={seq["width"]}\nimHeight={seq["height"]}\nimExt=.jpg\n')
        rational=row.get('fps_average') or row.get('fps_nominal')
        if rational and Fraction(rational)<=0:raise ValueError('Invalid source FPS fraction')
        sequences.append({**seq,'fps':fps,'fps_average_source':rational,'gt_boxes':boxes,
            'out_of_bounds_boxes_retained':outside,'gt_sha256':sha256(gt),
            'decoder_rotation_degrees':rotation,'decoder_auto_rotation':auto_rotation,**observed})
        print(f'[prepare] {name}: {frames} native frames, {boxes} GT boxes, {fps:.6f} FPS',flush=True)
    result={**identity,'sequences':sequences,'protocol':{
        'frame_base':1,'coordinate_format':'source pixel xywh, no coordinate offset or clipping',
        'source_gt_tail':'-1,-1,-1,-1 placeholders; not MOT17 confidence/class/visibility',
        'derived_gt_tail':'1,1,1 for evaluable player class only; raw GT unchanged',
        'video':'all original decoded frames, no FPS conversion; decoded size checked, auto-rotation recorded',
        'model_resize':'upstream PIL max_shorter=800,max_longer=1440; output boxes in source coordinates',
        'identity_scope':'per sequence; Indoor/Outdoor identity counts are not unique athlete counts'}}
    save_json(path,result)
    return result


def verify_sources(output):
    folder=Path(output).resolve()/SPLIT
    manifest=json.loads((folder/'dataset.json').read_text())
    root=Path(manifest['root'])
    if sha256(root/'manifest.csv')!=manifest['source_manifest_sha256']:
        raise ValueError('Original manifest changed')
    for seq in manifest['sequences']:
        if sha256(seq['video_path'])!=seq['video_sha256'] or sha256(seq['annotation_path'])!=seq['annotation_sha256']:
            raise ValueError(f'{seq["sequence"]}: original video/GT changed')
    return manifest
