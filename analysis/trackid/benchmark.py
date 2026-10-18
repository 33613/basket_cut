"""TrackID orchestration over the parity-checked MOTIP adapter and TrackEval."""
import csv
from collections import defaultdict
import json
from pathlib import Path

from analysis.teamtrack.dataset import save_json, sha256
from analysis.trackid.dataset import SPLIT, draw, verify_sources


def infer(output, checkpoint, device='cuda:0'):
    from analysis.teamtrack.inference import infer as native_video_infer
    native_video_infer(output,SPLIT,checkpoint,device)


def evaluate(output):
    from analysis.teamtrack.metrics import evaluate as evaluate_mot
    output=Path(output).resolve()
    verify_sources(output)
    summaries=evaluate_mot(output,SPLIT)
    folder=output/SPLIT
    save_json(output/'status.json',{'status':'completed','dataset':'TrackID3x3',
        'scope':'downloaded annotated fixed-camera sequences, not asserted official test split',
        'sequences':sum(r['scope']=='sequence' for r in summaries),
        'dataset_sha256':sha256(folder/'dataset.json'),
        'summary_sha256':sha256(folder/'metrics/summary.csv')})
    return summaries


def status(output):
    output=Path(output).resolve(); folder=output/SPLIT
    path=folder/'dataset.json'
    if not path.is_file():return {'status':'not_prepared'}
    manifest=json.loads(path.read_text())
    protocol_path=folder/'inference_protocol.json'
    protocol_hash=sha256(protocol_path) if protocol_path.is_file() else None
    completed=[]
    for seq in manifest['sequences']:
        name=seq['sequence']; marker=folder/'completed'/f'{name}.json'
        if not marker.is_file():continue
        row=json.loads(marker.read_text()); pred=folder/'predictions/motip/data'/f'{name}.txt'
        if (row['frames']!=seq['frames'] or row['protocol_sha256']!=protocol_hash
                or not pred.is_file() or row['prediction_sha256']!=sha256(pred)):
            raise ValueError(f'{name}: completion evidence changed')
        completed.append(name)
    saved=json.loads((output/'status.json').read_text()) if (output/'status.json').is_file() else {}
    summary=folder/'metrics/summary.csv'; rows=[]
    if saved.get('status')=='completed':
        if (saved['dataset_sha256']!=sha256(path) or not summary.is_file()
                or saved['summary_sha256']!=sha256(summary)):
            raise ValueError('Completed result checksum mismatch')
        with summary.open(newline='') as stream:
            rows=[r for r in csv.DictReader(stream) if r['scope']!='sequence']
    return {**saved,'status':saved.get('status','prepared'),'selected_sequences':len(manifest['sequences']),
        'completed_sequences':len(completed),'total_frames':sum(s['frames'] for s in manifest['sequences']),
        'views':{v:sum(s['view']==v for s in manifest['sequences']) for v in ('Indoor','Outdoor')},
        'results':rows,'summary':str(summary)}


def preview(output, *, max_frames=100):
    """Early native frames, GT green and predictions red; display-only resizing."""
    import cv2
    from analysis.sportsmot.runner import prediction_rows
    output=Path(output).resolve(); folder=output/SPLIT
    manifest=json.loads((folder/'dataset.json').read_text())
    report=[]
    for seq in manifest['sequences']:
        name=seq['sequence']; frames=seq['frames']
        pred=folder/'predictions/motip/data'/f'{name}.txt'
        gt=folder/'gt'/name/'gt/gt.txt'
        predicted=defaultdict(list); truth=defaultdict(list)
        for f,tid,x,y,w,h,_ in prediction_rows(pred,frames):
            if f<=max_frames:predicted[int(f)].append((int(tid),x,y,w,h))
        with gt.open(newline='') as stream:
            for row in csv.reader(stream):
                if int(row[0])<=max_frames:truth[int(row[0])].append((int(row[1]),*map(float,row[2:6])))
        count=min(max_frames,frames); wanted={1,max(1,count//2),count}
        capture=cv2.VideoCapture(seq['video_path']); paths=[]
        try:
            if not capture.isOpened():raise ValueError(f'Cannot decode {name}')
            for frame in range(1,count+1):
                ok,image=capture.read()
                if not ok:raise ValueError('Source ended before preview frame')
                if frame not in wanted:continue
                scale=min(1.,1600/seq['width'])
                canvas=cv2.resize(image,(round(seq['width']*scale),round(seq['height']*scale)))
                draw(canvas,truth[frame],scale=scale)
                draw(canvas,predicted[frame],color=(0,0,255),prefix='MOTIP',scale=scale)
                target=folder/'alignment'/name/f'comparison-{frame:06d}.jpg'
                target.parent.mkdir(parents=True,exist_ok=True)
                if not cv2.imwrite(str(target),canvas):raise OSError('Preview write failed')
                paths.append(str(target))
        finally:capture.release()
        report.append({'sequence':name,'view':seq['view'],'images':paths})
    save_json(folder/'alignment/preview.json',{'meaning':'GT green, MOTIP red; first frames only, not accuracy proof',
        'display_only_resize_max_width':1600,'sequences':report})
    return report
